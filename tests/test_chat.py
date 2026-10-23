import json
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError

import pandas as pd
import pytest

import elspace.chat as chat
from elspace.chat import EvidenceTools, PubChemUnavailable, agent_reply, local_reply
from elspace.evidence import EvidenceStore

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def tools():
    if not (ROOT / "data/processed/manifest.json").exists():
        pytest.skip("Prepared source data required")
    return EvidenceTools(EvidenceStore.load(ROOT / "data/processed"), 0.5, 0, 5)


@pytest.fixture
def pubchem_tools():
    compounds = pd.DataFrame([
        {"compound_id": "B00001", "name": "Bisphenol A", "valid_cas": "80-05-7", "cas_raw": "80-05-7",
         "inchikey": "IISBACLAFKSPIT-UHFFFAOYSA-N", "structure_status": "Resolved"},
        {"compound_id": "B00002", "name": "Unresolved ethanol", "valid_cas": "64-17-5", "cas_raw": "64-17-5",
         "inchikey": "", "structure_status": "Unresolved"},
        {"compound_id": "B00003", "name": "Conflicting record", "valid_cas": "80-05-7", "cas_raw": "80-05-7",
         "inchikey": "", "structure_status": "Conflict"},
    ])
    return EvidenceTools(SimpleNamespace(compounds=compounds), 0.5, 0, 5)


def pubchem_payload(inchikey="IISBACLAFKSPIT-UHFFFAOYSA-N"):
    return {"PropertyTable": {"Properties": [{
        "CID": 6623, "Title": "Bisphenol A", "MolecularFormula": "C15H16O2", "MolecularWeight": "228.29",
        "ConnectivitySMILES": "CC(C)(C1=CC=C(C=C1)O)C2=CC=C(C=C2)O", "InChI": "InChI=1S/C15H16O2",
        "InChIKey": inchikey,
    }]}}


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


def test_pubchem_lookup_confirms_a_full_inchikey(pubchem_tools, monkeypatch):
    monkeypatch.setattr(chat, "_pubchem_request", lambda namespace, value: (pubchem_payload(), "2026-10-22T00:00:00+00:00"))
    result = pubchem_tools.pubchem_lookup("B00001")
    assert result["match_status"] == "Exact identity"
    assert result["requested_by"] == "Full InChIKey"
    assert result["cid"] == 6623
    assert result["pubchem_url"].endswith("/6623")


def test_pubchem_lookup_keeps_cas_results_as_candidates(pubchem_tools, monkeypatch):
    payload = pubchem_payload("LFQSCWFLJHTTHZ-UHFFFAOYSA-N")
    monkeypatch.setattr(chat, "_pubchem_request", lambda namespace, value: (payload, "2026-10-22T00:00:00+00:00"))
    result = pubchem_tools.pubchem_lookup("B00002")
    assert result["requested_by"] == "CAS"
    assert result["match_status"] == "Candidate only"
    assert result["warnings"]


def test_pubchem_lookup_does_not_attach_a_conflicting_identity(pubchem_tools, monkeypatch):
    payload = pubchem_payload("LFQSCWFLJHTTHZ-UHFFFAOYSA-N")
    monkeypatch.setattr(chat, "_pubchem_request", lambda namespace, value: (payload, "2026-10-22T00:00:00+00:00"))
    result = pubchem_tools.pubchem_lookup("B00001")
    assert result["match_status"] == "Conflicting identity"
    assert "not attached" in result["warnings"][0]


def test_pubchem_lookup_blocks_a_local_identity_conflict(pubchem_tools, monkeypatch):
    monkeypatch.setattr(chat, "_pubchem_request", lambda namespace, value: pytest.fail("Network lookup must not run"))
    result = pubchem_tools.pubchem_lookup("B00003")
    assert result["match_status"] == "Blocked"
    assert result["cid"] is None


def test_pubchem_lookup_returns_local_safe_failures(pubchem_tools, monkeypatch):
    monkeypatch.setattr(chat, "_pubchem_request", lambda namespace, value: (None, "2026-10-22T00:00:00+00:00"))
    assert pubchem_tools.pubchem_lookup("B00001")["match_status"] == "Not found"

    def unavailable(namespace, value):
        raise PubChemUnavailable("PubChem could not be reached within the request limit.")

    monkeypatch.setattr(chat, "_pubchem_request", unavailable)
    result = pubchem_tools.pubchem_lookup("B00001")
    assert result["match_status"] == "Unavailable"
    assert "could not be reached" in result["warnings"][0]


class FakeHTTPResponse:
    def __init__(self, payload: bytes):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self, limit):
        return self.payload


def test_pubchem_http_retries_rate_limits_and_parses_json(monkeypatch):
    attempts = []

    def fake_urlopen(request, timeout):
        attempts.append(timeout)
        if len(attempts) == 1:
            raise HTTPError(request.full_url, 429, "rate limited", None, None)
        return FakeHTTPResponse(json.dumps(pubchem_payload()).encode())

    chat._pubchem_request.cache_clear()
    monkeypatch.setattr(chat, "urlopen", fake_urlopen)
    monkeypatch.setattr(chat.time, "sleep", lambda seconds: None)
    payload, retrieved_at = chat._pubchem_request("inchikey", "IISBACLAFKSPIT-UHFFFAOYSA-N")
    assert payload["PropertyTable"]["Properties"][0]["CID"] == 6623
    assert retrieved_at
    assert attempts == [10, 10]


def test_pubchem_http_handles_not_found_timeout_and_bad_json(monkeypatch):
    chat._pubchem_request.cache_clear()
    monkeypatch.setattr(chat.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(chat, "urlopen", lambda request, timeout: (_ for _ in ()).throw(HTTPError(request.full_url, 404, "missing", None, None)))
    assert chat._pubchem_request("inchikey", "missing")[0] is None

    chat._pubchem_request.cache_clear()
    monkeypatch.setattr(chat, "urlopen", lambda request, timeout: (_ for _ in ()).throw(TimeoutError()))
    with pytest.raises(PubChemUnavailable, match="could not be reached"):
        chat._pubchem_request("name", "timeout")

    chat._pubchem_request.cache_clear()
    monkeypatch.setattr(chat, "urlopen", lambda request, timeout: FakeHTTPResponse(b"{"))
    with pytest.raises(PubChemUnavailable, match="invalid JSON"):
        chat._pubchem_request("name", "bad-json")


def test_agent_offers_pubchem_only_after_local_lookup():
    calls = []

    class FakeTools:
        def dispatch(self, name, arguments):
            if name == "search_compounds":
                return {"matches": [{"compound_id": "B00001", "structure_status": "Resolved"}]}
            if name == "pubchem_lookup":
                return {"match_status": "Exact identity", "cid": 6623, "warnings": []}
            raise AssertionError(name)

    def create(**arguments):
        calls.append(arguments)
        tool_names = {tool["name"] for tool in arguments["tools"]}
        if len(calls) == 1:
            assert "pubchem_lookup" not in tool_names
            item = SimpleNamespace(type="function_call", name="search_compounds", arguments=json.dumps({"query": "Bisphenol A"}), call_id="local")
            return SimpleNamespace(output=[item], output_text="")
        if len(calls) == 2:
            assert "pubchem_lookup" in tool_names
            item = SimpleNamespace(type="function_call", name="pubchem_lookup", arguments=json.dumps({"compound_id": "B00001"}), call_id="online")
            return SimpleNamespace(output=[item], output_text="")
        return SimpleNamespace(output=[], output_text="PubChem reports formula C15H16O2.")

    client = SimpleNamespace(responses=SimpleNamespace(create=create))
    reply = agent_reply(FakeTools(), "Get the PubChem molecular formula for Bisphenol A", [], None, client, "gpt-5.6-sol")
    assert reply.external_evidence[0]["cid"] == 6623
    assert calls[0]["tool_choice"] == "required"


def test_agent_does_not_offer_pubchem_for_local_only_questions():
    calls = []

    class FakeTools:
        def dispatch(self, name, arguments):
            assert name == "compound_evidence"
            return {"compound": {"compound_id": "B00001", "structure_status": "Resolved"}}

    def create(**arguments):
        calls.append(arguments)
        assert "pubchem_lookup" not in {tool["name"] for tool in arguments["tools"]}
        if len(calls) == 1:
            item = SimpleNamespace(type="function_call", name="compound_evidence", arguments=json.dumps({"compound_id": "B00001"}), call_id="local")
            return SimpleNamespace(output=[item], output_text="")
        return SimpleNamespace(output=[], output_text="The local CLAP record is available.")

    client = SimpleNamespace(responses=SimpleNamespace(create=create))
    reply = agent_reply(FakeTools(), "What is the CLAP status for B00001?", [], None, client, "gpt-5.6-sol")
    assert not reply.external_evidence
