import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from elspace.chat import EvidenceTools, agent_reply, local_reply
from elspace.evidence import EvidenceStore

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def tools():
    if not (ROOT / "data/processed/manifest.json").exists():
        pytest.skip("Prepared source data required")
    return EvidenceTools(EvidenceStore.load(ROOT / "data/processed"), 0.5, 0, 5)


def test_local_chat_checks_actual_method_specific_records(tools):
    reply = local_reply(tools, "Check Bisphenol A")
    assert reply.evidence
    evidence = reply.evidence[0]
    methods = {item["method"]: item for item in evidence["methods"]}
    assert methods["GCMS_POS"]["status"] == "Measured"
    assert methods["LCMS_POS"]["status"] == "Unsupported"
    assert "No RRF or toxicity is predicted" in reply.text
    assert not reply.candidates


def test_local_followup_retains_resolved_compound(tools):
    first = local_reply(tools, "Check 80-05-7")
    followup = local_reply(tools, "Is it toxic?", first.compound_id)
    assert followup.compound_id == first.compound_id
    assert "membership alone does not mean toxic" in followup.text


def test_chat_does_not_claim_absence_of_natural_occurrence(tools):
    reply = local_reply(tools, "Does this occur in nature?")
    assert "unknown" in reply.text
    assert "not integrated" in reply.text


def test_chat_unknown_compound_is_not_substituted(tools):
    reply = local_reply(tools, "Check obviously-nonexistent-compound-xyzz")
    assert not reply.evidence
    assert reply.compound_id is None


def test_chat_dataset_summary_uses_actual_source_counts(tools):
    reply = local_reply(tools, "What datasets are loaded?")
    assert "18,266" in reply.text
    assert "Parquet" in reply.text
    assert "CSV/XLSX" in reply.text


def test_chat_requires_disambiguation_for_different_structures(tools):
    reply = local_reply(tools, "phthalate")
    assert reply.candidates
    assert not reply.evidence


def test_chat_smiles_runs_same_chemistry_without_an_llm(tools):
    reply = local_reply(tools, "SMILES: CCO")
    assert reply.evidence[0]["compound"]["name"] == "Custom SMILES query"


def test_agent_executes_read_only_tools_and_returns_their_evidence(tools):
    calls = []
    identifier = tools.search_compounds("80-05-7")["matches"][0]["compound_id"]

    def create(**arguments):
        calls.append(arguments)
        if len(calls) == 1:
            item = SimpleNamespace(type="function_call", name="compound_evidence", arguments=json.dumps({"compound_id": identifier}), call_id="call_1")
            return SimpleNamespace(output=[item], output_text="")
        assert arguments["input"][-1]["type"] == "function_call_output"
        result = json.loads(arguments["input"][-1]["output"])
        assert result["methods"][0]["status"] == "Measured"
        return SimpleNamespace(output=[], output_text="The exact GC/MS record has a numeric RRF.")

    client = SimpleNamespace(responses=SimpleNamespace(create=create))
    reply = agent_reply(tools, "Check Bisphenol A", [], None, client, "configured-model")
    assert reply.engine == "OpenAI tool-calling agent"
    assert reply.evidence[0]["compound"]["compound_id"] == identifier
    assert calls[0]["tool_choice"] == "required"
    assert calls[0]["store"] is False


def test_agent_tool_boundary_rejects_arbitrary_code(tools):
    with pytest.raises(ValueError, match="Unknown evidence tool"):
        tools.dispatch("run_python", {"code": "print('should not execute')"})
