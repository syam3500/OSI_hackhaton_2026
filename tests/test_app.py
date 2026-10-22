from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not (ROOT / "data/processed/manifest.json").exists(), reason="Prepared sources required")


def radio(app, label):
    return next(widget for widget in app.radio if widget.label == label)


def selectbox(app, label):
    return next(widget for widget in app.selectbox if widget.label == label)


def test_search_examples_smiles_and_missing_records():
    app = AppTest.from_file(str(ROOT / "app/app.py"), default_timeout=60).run()
    radio(app, "Workspace").set_value("Explore").run()
    assert not app.exception
    assert app.metric[0].value == "18,266"
    next(button for button in app.button if button.label == "Caprolactam").click().run()
    assert not app.exception
    assert any("Evidence for" in heading.value for heading in app.subheader)
    radio(app, "Input").set_value("SMILES").run()
    app.text_input[0].set_value("CCO").run()
    assert not app.exception
    assert any("Custom SMILES query" in heading.value for heading in app.subheader)
    app.text_input[0].set_value("not a structure").run()
    assert not app.exception
    assert app.warning


def test_source_methodology_3d_filters_and_rule_changes():
    app = AppTest.from_file(str(ROOT / "app/app.py"), default_timeout=60).run()
    for page in ["Data & sources", "Methodology", "Explore"]:
        radio(app, "Workspace").set_value(page).run()
        assert not app.exception
    selectbox(app, "Dimensions").set_value("3D").run()
    assert not app.exception
    selectbox(app, "Map axes").set_value("Molecular descriptors").run()
    assert not app.exception
    selectbox(app, "X descriptor").set_value("hbd").run()
    selectbox(app, "Y descriptor").set_value("tpsa").run()
    assert not app.exception
    assert selectbox(app, "Z descriptor").value not in ["hbd", "tpsa"]
    selectbox(app, "Analytical method").set_value("LCMS_NEG").run()
    assert not app.exception
    next(widget for widget in app.slider if widget.label == "Tanimoto cutoff").set_value(0.9).run()
    assert not app.exception
    next(widget for widget in app.multiselect if widget.label == "Coverage status").set_value([]).run()
    assert not app.exception


def test_chat_default_followup_and_open_compound_map():
    app = AppTest.from_file(str(ROOT / "app/app.py"), default_timeout=60).run()
    assert not app.exception
    assert radio(app, "Workspace").value == "Chat"
    next(button for button in app.button if button.label == "Check Bisphenol A").click().run()
    assert not app.exception
    assert len(app.session_state["chat_messages"]) == 2
    assert app.session_state["chat_messages"][-1]["reply"]["evidence"]
    app.chat_input[0].set_value("Is it toxic?").run()
    assert not app.exception
    assert "membership alone does not mean toxic" in app.session_state["chat_messages"][-1]["text"]
    next(button for button in app.button if button.label == "Open compound on map").click().run()
    assert not app.exception
    assert radio(app, "Workspace").value == "Explore"


def test_new_chat_clears_history_and_unconfigured_agent_is_disabled():
    app = AppTest.from_file(str(ROOT / "app/app.py"), default_timeout=60).run()
    next(button for button in app.button if button.label == "Data sources").click().run()
    assert len(app.session_state["chat_messages"]) == 2
    next(button for button in app.button if button.label == "New chat").click().run()
    assert not app.exception
    assert app.session_state["chat_messages"] == []


def test_external_pubchem_candidate_maps_without_changing_borton():
    from elspace.chemistry import structure

    app = AppTest.from_file(str(ROOT / "app/app.py"), default_timeout=60).run()
    radio(app, "Workspace").set_value("Explore").run()
    radio(app, "Input").set_value("PubChem lookup").run()
    assert not app.exception
    app.text_input[0].set_value("CID: 702").run()
    app.session_state["pubchem_lookup"] = {"query": "CID: 702", "candidates": [{"compound_id": "PUBCHEM-702", "name": "Ethanol", "source": "PubChem lookup", "cas_raw": "", "valid_cas": "", "structure_status": "Resolved", "structure_source": "PubChem CID 702", "structure_note": "Test fixture", "pubchem_url": "https://pubchem.ncbi.nlm.nih.gov/compound/702", "pubchem_cid": 702, "retrieved_utc": "2026-10-22", "lookup_query": "CID: 702", **structure("CCO")}]}
    app.run()
    assert not app.exception
    assert app.metric[0].value == "18,266"
    assert any("Evidence for Ethanol" in heading.value for heading in app.subheader)
    assert not any(tab.label == "Detectability" for tab in app.tabs)
    radio(app, "Workspace").set_value("Chat").run()
    app.chat_input[0].set_value("Check PUBCHEM-702").run()
    assert not app.exception
    assert radio(app, "Workspace").value == "Chat"
    assert app.session_state["chat_messages"][-1]["reply"]["evidence"][0]["compound"]["compound_id"] == "PUBCHEM-702"
