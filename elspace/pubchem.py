from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache
import re
from urllib.parse import quote

import requests

from elspace.chemistry import structure, valid_cas

BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"


def fetch_json(path: str) -> dict:
    try:
        response = requests.get(BASE + path, timeout=(5, 15), headers={"User-Agent": "EL-Atlas-prototype/1.0"})
        if response.status_code == 404:
            return {}
        response.raise_for_status()
        return response.json()
    except (requests.RequestException, ValueError) as error:
        raise ValueError("PubChem is unavailable or rate-limited. Retry later, or supply a reviewed SMILES. No identity was guessed.") from error


@lru_cache(maxsize=256)
def lookup_pubchem(query: str) -> dict:
    if not isinstance(query, str) or not query.strip() or len(query) > 200:
        raise ValueError("Enter a CAS number, compound name, or 'CID: number' (at most 200 characters).")
    query = query.strip()
    cas = bool(re.fullmatch(r"\d{2,7}-\d{2}-\d", query))
    if cas and not valid_cas(query):
        raise ValueError("The CAS checksum is invalid. Check the identifier or provide SMILES.")
    cid_match = re.fullmatch(r"(?:CID|PUBCHEM)\s*[: -]\s*(\d+)", query, re.IGNORECASE)
    ids = [int(cid_match[1])] if cid_match else fetch_json("/compound/name/" + quote(query, safe="") + "/cids/JSON").get("IdentifierList", {}).get("CID", [])
    if not ids:
        return {"query": query, "status": "No PubChem match", "candidates": []}
    if len(ids) > 5:
        return {"query": query, "status": "Ambiguous PubChem query; provide a specific CID or reviewed SMILES", "candidates": [], "candidate_count": len(ids)}
    cid_list = ",".join(str(int(identifier)) for identifier in ids)
    properties = fetch_json(f"/compound/cid/{cid_list}/property/Title,IsomericSMILES,InChIKey,MolecularFormula/JSON").get("PropertyTable", {}).get("Properties", [])
    synonyms = fetch_json(f"/compound/cid/{cid_list}/synonyms/JSON").get("InformationList", {}).get("Information", []) if cas else []
    confirmed_cids = {row["CID"] for row in synonyms if query in row.get("Synonym", [])}
    candidates = []
    retrieved = datetime.now(timezone.utc).isoformat()
    for row in properties:
        smiles = row.get("SMILES", row.get("IsomericSMILES", ""))
        parsed = structure(smiles)
        if parsed is None or parsed["inchikey"] != row.get("InChIKey") or (cas and row["CID"] not in confirmed_cids):
            continue
        candidates.append({"compound_id": f"PUBCHEM-{row['CID']}", "name": row.get("Title", query), "aliases": query,
                           "cas_raw": query if cas else "", "valid_cas": query if cas else "", "clap_code": "",
                           "source": "PubChem lookup", "structure_status": "Resolved", "structure_source": f"PubChem CID {row['CID']}",
                           "structure_note": "PubChem lookup candidate; verify the intended salt, stereochemistry and material identity before use. Not added to Borton.",
                           "pubchem_url": f"https://pubchem.ncbi.nlm.nih.gov/compound/{row['CID']}", "retrieved_utc": retrieved,
                           "lookup_query": query, "pubchem_cid": row["CID"], **parsed})
    return {"query": query, "status": "PubChem candidates found" if candidates else "No identity-verified PubChem candidate; supply reviewed SMILES",
            "candidates": candidates}
