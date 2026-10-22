from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from elspace.chat import ChatReply

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


def test_pubchem_online_result_is_separate_from_local_evidence(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-5.6-sol")
    monkeypatch.setattr("elspace.ui_chat.agent_reply", lambda *args, **kwargs: ChatReply(
        "PubChem returned a compound property record.",
        external_evidence=[{
            "provider": "PubChem PUG REST", "retrieved_at": "2026-10-22T00:00:00+00:00",
            "match_status": "Exact identity", "match_basis": "Full InChIKey", "cid": 6623,
            "name": "Bisphenol A", "molecular_formula": "C15H16O2", "molecular_weight": "228.29",
            "smiles": "CC(C)(c1ccc(O)cc1)c1ccc(O)cc1", "inchikey": "IISBACLAFKSPIT-UHFFFAOYSA-N",
            "pubchem_url": "https://pubchem.ncbi.nlm.nih.gov/compound/6623", "warnings": [],
        }],
        engine="OpenAI tool-calling agent",
    ))
    app = AppTest.from_file(str(ROOT / "app/app.py"), default_timeout=60).run()
    next(widget for widget in app.toggle if widget.label == "Use AI agent").set_value(True).run()
    app.chat_input[0].set_value("Get the PubChem formula for Bisphenol A").run()
    assert not app.exception
    assert any("PubChem online result · Exact identity" == expander.label for expander in app.expander)
