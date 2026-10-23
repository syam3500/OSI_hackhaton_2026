from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from elspace.chemistry import DESCRIPTORS, extract_cas, structure, valid_cas
from elspace.evidence import EvidenceStore, descriptor_ranges, match_rows, search_compounds
from scripts.prepare_data import parse_fda

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def store():
    if not (ROOT / "data/processed/manifest.json").exists():
        pytest.skip("Run scripts/prepare_data.py for source integration checks")
    return EvidenceStore.load(ROOT / "data/processed")


def test_cas_checksum_and_deduplication():
    assert valid_cas("80-05-7")
    assert not valid_cas("80-05-8")
    assert not valid_cas("001-01-3")
    assert extract_cas("001-01-3, 128-37-0, 128-37-0") == ["128-37-0"]


def test_parent_similarity_does_not_replace_original_identity():
    salt = structure("CC(=O)[O-].[Na+]")
    acid = structure("CC(=O)O")
    assert salt["inchikey_std"] == acid["inchikey_std"]
    assert salt["inchikey"] != acid["inchikey"]
    table = pd.DataFrame([{**acid, "cas": "64-19-7"}])
    rows, basis = match_rows({**salt, "valid_cas": "127-09-3"}, table)
    assert rows.empty
    assert basis == "No match"


def test_conflicting_cas_does_not_attach_evidence():
    ethanol, benzene = structure("CCO"), structure("c1ccccc1")
    rows, basis = match_rows({**ethanol, "valid_cas": "64-17-5"}, pd.DataFrame([{**benzene, "cas": "64-17-5"}]))
    assert rows.empty
    assert basis == "CAS / structure conflict"


def test_ambiguous_unresolved_identity_is_not_arbitrarily_resolved():
    table = pd.DataFrame([{**structure("CCO"), "cas": "64-17-5"}, {**structure("CO"), "cas": "67-56-1"}])
    rows, basis = match_rows({"valid_cas": "64-17-5 | 67-56-1"}, table)
    assert rows.empty
    assert basis == "Ambiguous CAS match"


def test_descriptor_margin_keeps_zero_width_charge_fixed():
    target = structure("CCO")
    reference = pd.DataFrame([target, structure("CCCCO")])
    changed = {**target, "formal_charge": 1}
    result = descriptor_ranges(changed, reference, 0.25).set_index("Descriptor")
    assert not result.loc[DESCRIPTORS["formal_charge"], "Inside range"]
    assert result.loc[DESCRIPTORS["mol_weight"], "Inside range"]


def test_all_borton_rows_preserved_and_map_has_no_fabricated_positions(store):
    borton = store.compounds[store.compounds.source.eq("Borton")]
    assert len(borton) == 18266
    assert borton.compound_id.is_unique
    unresolved = borton[borton.structure_status.ne("Resolved")]
    assert unresolved.map_x.isna().all()
    resolved = borton[borton.structure_status.eq("Resolved")]
    assert np.isfinite(resolved[["map_x", "map_y", "map_z"]]).all().all()
    assert len(resolved) + len(unresolved) == len(borton)


def test_cas_search_does_not_match_suffix_of_another_identifier(store):
    results = search_compounds(store.compounds, "80-05-7")
    assert len(results) > 0
    assert not results.cas_raw.str.contains("2080-05-7", regex=False).any()
    assert results.iloc[0].structure_status == "Resolved"


def test_conflict_status_prevents_unverified_layer_attachment(store):
    target = {"valid_cas": "80-05-7", "structure_status": "Conflict", "inchikey": ""}
    rows, basis = store.layer(target, "plastics")
    assert rows.empty
    assert basis == "Identity conflict requires review"


def test_unverified_rm84_structure_is_quarantined_but_method_record_is_retained(store):
    target = store.clap[store.clap.clap_code.eq("RM84")].iloc[0].to_dict()
    assert target["smiles_std"] == ""
    assert "blend" in target["structure_note"]
    assert "RM84" not in store.reference.clap_code.tolist()
    results = store.assess(target, 0.5, 0)
    assert all(len(result["exact_records"]) == 1 for result in results)


def test_catalog_and_evidence_card_agree_for_unresolved_clap_records(store):
    for method in ["GCMS_POS", "LCMS_POS", "LCMS_NEG"]:
        catalog = store.enrich_catalog(method, 0.5, 0)
        unresolved = catalog[catalog.source.eq("FDA CLAP") & catalog.smiles_std.eq("")]
        for target in unresolved.to_dict("records"):
            result = next(item for item in store.assess(target, 0.5, 0) if item["method"] == method)
            assert target["coverage"] == result["status"]


def test_bisphenol_a_numeric_and_non_detected_outcomes_stay_method_specific(store):
    target = store.clap[store.clap.cas.eq("80-05-7")].iloc[0].to_dict()
    results = {result["method"]: result for result in store.assess(target, 0, 0.25)}
    assert results["GCMS_POS"]["status"] == "Measured"
    assert results["LCMS_POS"]["status"] == "Unsupported"
    assert results["LCMS_NEG"]["status"] == "Unsupported"
    assert results["LCMS_POS"]["exact_records"].iloc[0].outcome == "not_detected"
    assert results["GCMS_POS"]["exact_records"].iloc[0].rrf_5 == "0.066"


def test_below_detection_limit_is_not_measured_at_any_cutoff(store):
    target = store.clap[store.clap.cas.eq("128-37-0")].iloc[0].to_dict()
    result = next(item for item in store.assess(target, 0, 0.25) if item["method"] == "LCMS_NEG")
    assert result["status"] == "Unsupported"
    assert result["exact_records"].iloc[0].outcome == "below_DL"


def test_method_reference_panels_require_numeric_rrfs(store):
    target = structure("CCO")
    result = store.assess(target, 0.5, 0)
    for entry in result:
        observations = store.methods[store.methods.method.eq(entry["method"])]
        measured = observations.loc[observations.rrf_min.notna(), "clap_code"]
        assert entry["panel_size"] == store.reference.clap_code.isin(measured).sum()
    assert len({entry["panel_size"] for entry in result}) > 1


def test_no_structure_does_not_become_unsupported_or_zero_similarity(store):
    results = store.assess({"valid_cas": "", "smiles_std": "", "inchikey": ""}, 0.5, 0)
    assert all(entry["status"] == "Unresolved" and entry["similarity"] is None for entry in results)


def test_neighbours_exclude_exact_identity(store):
    target = store.clap[store.clap.cas.eq("80-05-7")].iloc[0].to_dict()
    neighbours = store.neighbours(target, 5)
    assert target["inchikey"] not in neighbours.inchikey.tolist()
    assert neighbours.Tanimoto.between(0, 1).all()
    assert neighbours.Tanimoto.is_monotonic_decreasing


def test_massspecgym_does_not_guess_negative_polarity(store):
    target = store.clap[store.clap.cas.eq("80-05-7")].iloc[0].to_dict()
    spectrum = store.spectrum(target)
    assert spectrum["n_spectra"] == 8
    assert spectrum["positive"] == 8
    assert spectrum["negative"] == 0
    assert "not stereochemistry" in spectrum["match_basis"]


def test_fda_pdf_parser_keeps_partial_values_and_non_detection():
    path = ROOT.parent / "Relevant BAckground info/FDA_RRF GCMS for CLAP.pdf"
    if not path.exists():
        pytest.skip("Source PDF not available")
    rows = {row["clap_code"]: row for row in parse_fda(path, "GCMS_POS")}
    assert rows["RM26"]["outcome"] == "partial"
    assert rows["RM26"]["rrf_5"] == "below DL"
    assert rows["RM26"]["rrf_20"] == "0.035"
    assert rows["RM1"]["outcome"] == "not_detected"
    assert np.isnan(rows["RM1"]["rrf_min"])
