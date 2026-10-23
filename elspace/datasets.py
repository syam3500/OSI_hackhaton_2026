from __future__ import annotations

from copy import deepcopy
from io import BytesIO
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from elspace.chemistry import extract_cas, fingerprint_matrix, structure
from elspace.evidence import EvidenceStore

TARGET_SOURCE = "Uploaded targets"


def read_upload(data: bytes, filename: str, sheet_name: str | None = None) -> pd.DataFrame:
    if len(data) > 25 * 1024 * 1024:
        raise ValueError("Target files must be 25 MB or smaller.")
    suffix = Path(filename).suffix.casefold()
    if suffix == ".csv":
        frame = pd.read_csv(BytesIO(data), dtype=str, keep_default_na=False, nrows=20001)
    elif suffix == ".xlsx":
        frame = pd.read_excel(BytesIO(data), sheet_name=sheet_name or 0, dtype=str, keep_default_na=False, nrows=20001)
    else:
        raise ValueError("Upload a CSV or XLSX file.")
    if frame.empty or len(frame) > 20000:
        raise ValueError("Upload between 1 and 20,000 target rows.")
    frame.columns = frame.columns.map(str)
    if not frame.columns.is_unique:
        raise ValueError("Column names must be unique.")
    return frame


def build_target_store(base: EvidenceStore, frame: pd.DataFrame, filename: str, digest: str,
                       name_column: str, cas_column: str | None, smiles_column: str | None,
                       projection_path: Path) -> EvidenceStore:
    if name_column not in frame or (not cas_column and not smiles_column):
        raise ValueError("Choose a name column and at least one CAS or SMILES column.")
    for column in [cas_column, smiles_column]:
        if column and column not in frame:
            raise ValueError("The selected column is missing from the uploaded table.")
    bridges = {}
    known = base.compounds[base.compounds.smiles_raw.ne("")]
    for row in known.itertuples():
        for number in extract_cas(row.valid_cas):
            bridges.setdefault(number, {})[row.inchikey] = row.smiles_raw
    targets = []
    with np.load(projection_path) as projection:
        center, axes = projection["center"], projection["axes"]
    for index, row in enumerate(frame.fillna("").to_dict("records"), 1):
        name = str(row[name_column]).strip()
        cas_raw = str(row[cas_column]).strip() if cas_column else ""
        smiles = str(row[smiles_column]).strip() if smiles_column else ""
        cas_values = extract_cas(cas_raw)
        candidates = {key: value for number in cas_values for key, value in bridges.get(number, {}).items()}
        properties = structure(smiles) if smiles else None
        note, source, status = "", "Uploaded SMILES" if properties else "", "Unresolved"
        if smiles and properties is None:
            note = "Uploaded SMILES is invalid; no structure has been substituted silently."
        elif properties and candidates and set(candidates) != {properties["inchikey"]}:
            properties, status = None, "Conflict"
            note = "Uploaded SMILES conflicts with source identities for the supplied CAS. Review the input."
        elif not smiles and len(candidates) == 1:
            properties = structure(next(iter(candidates.values())))
            source = "CAS bridge to the loaded reference build"
        elif not smiles and len(candidates) > 1:
            status, note = "Conflict", "CAS identifiers resolve to different source structures; review required."
        if properties:
            status = "Resolved"
        record = {"compound_id": f"UP{index:05d}", "name": name or f"Uploaded row {index}", "aliases": name,
                  "cas_raw": cas_raw, "valid_cas": " | ".join(cas_values), "source": TARGET_SOURCE,
                  "clap_code": "", "structure_status": status, "structure_source": source, "structure_note": note,
                  "smiles_raw": "", "smiles_std": "", "inchikey": "", "inchikey_std": "", **(properties or {})}
        if properties:
            position = np.einsum("ij,kj->ik", fingerprint_matrix([properties["smiles_std"]]) - center, axes, optimize=False)[0]
            record.update(dict(zip(["map_x", "map_y", "map_z"], position)))
        targets.append(record)
    target_frame = pd.DataFrame(targets).reindex(columns=base.compounds.columns)
    compounds = pd.concat([base.compounds[base.compounds.source.eq("FDA CLAP")], target_frame], ignore_index=True)
    manifest = deepcopy(base.manifest)
    manifest["target_source"] = TARGET_SOURCE
    manifest["active_targets"] = {"label": Path(filename).name, "rows": len(target_frame), "resolved": int(target_frame.structure_status.eq("Resolved").sum()),
                                  "conflicts": int(target_frame.structure_status.eq("Conflict").sum()), "sha256": digest, "scope": "Current browser session only"}
    return EvidenceStore(compounds, base.methods, base.plastics, base.stavitskaya, base.massspecgym, manifest)


def upload_digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
