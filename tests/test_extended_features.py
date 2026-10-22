import base64
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from elspace.chat import EvidenceTools, local_reply, tool_definitions
from elspace.chemistry import molecular_properties, molecule_data_uri, structure
from elspace.evidence import EvidenceStore
from elspace.plots import chemical_map, spectrum_plot
from elspace.pubchem import lookup_pubchem
from elspace.spectra import parse_peaks

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def store():
    return EvidenceStore.load(ROOT / "data/processed")


def test_custom_descriptor_axes_control_both_2d_and_3d():
    record = {"compound_id": "TEST", "name": "Ethanol", "cas_raw": "64-17-5", "coverage": "Measured", **structure("CCO")}
    frame = pd.DataFrame([record])
    chart = chemical_map(frame, frame, record, frame.iloc[:0], "coverage", "2D", "Molecular descriptors", False, ["tpsa", "hba"])
    assert chart.layout.xaxis.title.text == "TPSA (Å²)"
    assert chart.data[0].x[0] == record["tpsa"]
    chart = chemical_map(frame, frame, record, frame.iloc[:0], "coverage", "3D", "Molecular descriptors", False, ["hbd", "logp", "mol_weight"])
    assert chart.data[0].type == "scatter3d"
    assert chart.data[0].z[0] == record["mol_weight"]
    assert chart.layout.scene.zaxis.title.text == "Molecular weight (g/mol)"
    with pytest.raises(ValueError, match="different valid descriptor"):
        chemical_map(frame, frame, record, frame.iloc[:0], "coverage", "2D", "Molecular descriptors", False, ["tpsa", "tpsa"])


def test_structure_image_is_embedded_and_properties_are_calculated():
    image = molecule_data_uri("CCO")
    assert image.startswith("data:image/png;base64,")
    assert base64.b64decode(image.split(",", 1)[1]).startswith(b"\x89PNG")
    properties = molecular_properties("CCO")
    assert properties["Molecular formula"] == "C2H6O"
    assert properties["Heavy atoms"] == 3
    assert properties["Molecular weight (g/mol)"] == pytest.approx(46.069)
    assert "volume" not in properties


@pytest.mark.parametrize("mzs,intensities", [("1,2", "1"), ("nan", "1"), ("0", "1"), ("1", "-1"), ("1", "0")])
def test_invalid_reference_peaks_are_rejected(mzs, intensities):
    with pytest.raises(ValueError):
        parse_peaks(mzs, intensities)


def test_real_massspecgym_peaks_and_precursor_are_plotted(store):
    target = store.compounds[store.compounds.compound_id.eq("CLAP-RM14")].iloc[0].to_dict()
    peaks = store.reference_spectra(target)
    assert len(peaks) == 3
    assert set(peaks.polarity) == {"positive"}
    assert store.spectrum(target)["n_spectra"] == 8
    reference = peaks.iloc[0].to_dict()
    chart = spectrum_plot(reference, "Reference spectrum")
    assert max(chart.data[1].y) == pytest.approx(100)
    assert chart.layout.shapes[0].x0 == reference["precursor_mz"]
    masses, intensities = parse_peaks(reference["mzs"], reference["intensities"])
    assert np.allclose(chart.data[1].x, masses)
    assert np.allclose(chart.data[1].y, intensities / intensities.max() * 100)


def test_classifier_tooling_is_not_published(store):
    tools = EvidenceTools(store, .5, 0, 5)
    assert "method_detectability" not in {tool["name"] for tool in tool_definitions()}
    with pytest.raises(ValueError, match="Unknown evidence tool"):
        tools.dispatch("method_detectability", {"compound_id": "CLAP-RM14", "method": "LCMS_POS", "method_details": ""})


def test_pubchem_requires_exact_cas_synonym_and_full_key(monkeypatch):
    import elspace.pubchem as pubchem

    lookup_pubchem.cache_clear()
    properties = structure("CCO")

    def response(path):
        if "cids" in path:
            return {"IdentifierList": {"CID": [702]}}
        if "synonyms" in path:
            return {"InformationList": {"Information": [{"CID": 702, "Synonym": ["64-17-5"]}]}}
        return {"PropertyTable": {"Properties": [{"CID": 702, "Title": "Ethanol", "SMILES": "CCO", "InChIKey": properties["inchikey"]}]}}

    monkeypatch.setattr(pubchem, "fetch_json", response)
    result = lookup_pubchem("64-17-5")
    assert result["candidates"][0]["source"] == "PubChem lookup"
    assert result["candidates"][0]["pubchem_cid"] == 702
    lookup_pubchem.cache_clear()
    assert lookup_pubchem("67-56-1")["candidates"] == []
    lookup_pubchem.cache_clear()


def test_invalid_cas_does_not_call_pubchem(monkeypatch):
    import elspace.pubchem as pubchem

    monkeypatch.setattr(pubchem, "fetch_json", lambda path: pytest.fail("Invalid CAS must not be sent"))
    with pytest.raises(ValueError, match="checksum"):
        lookup_pubchem("64-17-6")


def test_missing_cas_never_reuses_previous_compound_context(store):
    tools = EvidenceTools(store, .5, 0, 5)
    reply = local_reply(tools, "Can 171599-83-0 be detected? Is it in CLAP?", "CLAP-RM14")
    assert reply.compound_id is None
    assert "Enable" in reply.text
    assert "PubChem" in reply.text


def test_detailed_answers_expand_actual_evidence(store):
    detailed = local_reply(EvidenceTools(store, .5, 0, 5), "Check Bisphenol A")
    concise = local_reply(EvidenceTools(store, .5, 0, 5, detailed=False), "Check Bisphenol A")
    assert len(detailed.text) > len(concise.text) * 2
    assert "Measured RRFs" in detailed.text
    assert "Calculated molecular properties" in detailed.text
    assert "Classifier notebooks are pending" in detailed.text


def test_offline_smiles_evidence_and_followup(store):
    tools = EvidenceTools(store, .5, 0, 5)
    reply = local_reply(tools, "SMILES: CCO using LC/MS positive")
    assert reply.evidence[0]["compound"]["name"] == "Custom SMILES query"
    assert "detectability" not in reply.evidence[0]
    followup = local_reply(tools, "What is its reference evidence?", reply.compound_id)
    assert followup.compound_id == "QUERY"


def test_pubchem_does_not_silently_choose_from_many_identities(monkeypatch):
    import elspace.pubchem as pubchem

    lookup_pubchem.cache_clear()
    monkeypatch.setattr(pubchem, "fetch_json", lambda path: {"IdentifierList": {"CID": list(range(1, 8))}})
    result = lookup_pubchem("ambiguous test name")
    assert result["candidates"] == []
    assert "Ambiguous" in result["status"]
    lookup_pubchem.cache_clear()
