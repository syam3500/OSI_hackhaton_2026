from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
from pypdf import PdfReader
from rdkit import Chem

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from elspace.chemistry import extract_cas, fingerprint_matrix, structure

POLYMERS = [
    "PE", "HDPE", "LDPE", "PP", "PS", "EPS", "HIPS", "PVC", "PET", "PA", "PUR",
    "ABS", "BPC", "EVA", "EVOH", "Melamine", "PAA", "PAN", "PC", "PCT", "PES",
    "PMMA", "PVA", "SAN", "SAP", "Tritan", "Rubber", "Silicone", "Cellulose-based",
    "PBAT", "PBS", "PHA", "PLA", "Starch", "Bioplastics_unspecified", "Multilayer",
    "Unspecified_plastics",
]
HAZARDS = [
    "Persistence_score", "Bioaccumulation_score", "Mobility_score", "Toxicity_score",
    "Carcinogenic", "Mutagenic", "Reproduction", "CMR", "STOT", "EDC", "Aquatic_toxicity",
    "PBT", "vPvB", "PMT", "vPvM",
]
FDA_FILES = {
    "GCMS_POS": "FDA_RRF GCMS for CLAP.pdf",
    "LCMS_POS": "FDA_LCMS RRFs CLAP Positive Ion Detection.pdf",
    "LCMS_NEG": "FDA_LCMS RRFs CLAP Negative Ion Detection.pdf",
}
SOURCES = {
    "Borton": "https://doi.org/10.14573/altex.2504231s2",
    "FDA CLAP": "https://cdrh-rst.fda.gov/chemicals-list-analytical-performance-clap",
    "Stavitskaya": "https://doi.org/10.1016/j.yrtph.2026.106163",
    "PlastChem": "https://doi.org/10.5281/zenodo.10701706",
    "MassSpecGym": "https://huggingface.co/datasets/roman-bushuiev/MassSpecGym",
    "NIST": "https://data.nist.gov/od/id/ark:/88434/mds2-4147",
}


def clean(value) -> str:
    return "" if pd.isna(value) else str(value).strip()


def raw_key(smiles: str) -> str:
    molecule = Chem.MolFromSmiles(smiles) if smiles else None
    return Chem.MolToInchiKey(molecule) if molecule is not None else ""


def parse_fda(path: Path, method: str) -> list[dict]:
    pattern = re.compile(
        r"\b(RM\d+)\s+(.+?)\s+(\d{2,7}-\d{2}-\d)\s+"
        r"((?:(?:\d+\.\d+|n\.d\.|below\s+DL)\s*){3})",
        re.DOTALL,
    )
    records = []
    for page_number, page in enumerate(PdfReader(path).pages, 1):
        for match in pattern.finditer(page.extract_text()):
            values = re.findall(r"\d+\.\d+|n\.d\.|below\s+DL", match[4])
            numeric = [float(value) for value in values if re.fullmatch(r"\d+\.\d+", value)]
            if len(numeric) == 3:
                status = "measured"
            elif numeric:
                status = "partial"
            elif all("below" in value for value in values):
                status = "below_DL"
            else:
                status = "not_detected"
            records.append({
                "clap_code": match[1], "method": method, "cas_reported": match[3],
                "name_reported": re.sub(r"\s+", " ", match[2]), "outcome": status,
                "rrf_5": re.sub(r"\s+", " ", values[0]),
                "rrf_10": re.sub(r"\s+", " ", values[1]),
                "rrf_20": re.sub(r"\s+", " ", values[2]),
                "rrf_min": min(numeric) if numeric else np.nan,
                "rrf_max": max(numeric) if numeric else np.nan,
                "source_file": path.name, "source_page": page_number,
            })
    if not records:
        raise ValueError(f"No FDA rows parsed from {path}")
    return records


def prepare_massspecgym(path: Path, output: Path) -> dict:
    required = ["identifier", "inchikey", "smiles", "adduct", "instrument_type"]
    groups = defaultdict(lambda: {"identifiers": [], "adducts": set(), "instruments": set(), "positive": 0, "negative": 0, "unknown": 0})
    count = 0
    for chunk in pd.read_csv(path, sep="\t", usecols=required, chunksize=20000):
        for row in chunk.fillna("").to_dict("records"):
            key = row["inchikey"].split("-")[0]
            if not re.fullmatch(r"[A-Z]{14}", key):
                continue
            entry = groups[key]
            entry["identifiers"].append(row["identifier"])
            entry["adducts"].add(row["adduct"])
            entry["instruments"].add(row["instrument_type"])
            adduct = row["adduct"].strip()
            mode = "positive" if adduct.endswith("+") else "negative" if adduct.endswith("-") else "unknown"
            entry[mode] += 1
            count += 1
    rows = [{
        "connectivity_key": key, "n_spectra": len(entry["identifiers"]),
        "positive": entry["positive"], "negative": entry["negative"], "unknown": entry["unknown"],
        "adducts": " | ".join(sorted(entry["adducts"])),
        "instruments": " | ".join(sorted(entry["instruments"])),
        "example_identifiers": " | ".join(entry["identifiers"][:5]),
    } for key, entry in groups.items()]
    pd.DataFrame(rows).to_parquet(output / "massspecgym.parquet", index=False)
    return {"loaded": True, "spectra": count, "connectivity_keys": len(rows), "source_file": path.name,
            "match_basis": "InChIKey first block (connectivity only; not stereochemistry)"}


def build(background: Path, draft: Path, output: Path, massspecgym: Path | None):
    output.mkdir(parents=True, exist_ok=True)
    source_files = []

    def read_excel(filename, **kwargs):
        path = background / filename
        source_files.append(path)
        return pd.read_excel(path, **kwargs)

    borton = read_excel("Borton_Sup2_AppendixB_18000.xlsx", sheet_name="Extractables").fillna("")
    plastics = read_excel("plastchem_db_v1.01.xlsx", sheet_name="Full database", header=1)
    stavitskaya = read_excel("1-s2.0-S0273230026001364-mmc1_1285tox.xlsx").fillna("")
    clap_compounds = pd.read_csv(draft / "sample_data/clap_compounds.csv").fillna("")
    clap_structures = pd.read_csv(draft / "sample_data/structures_clap.csv").fillna("")
    source_files.extend([draft / "sample_data/clap_compounds.csv", draft / "sample_data/structures_clap.csv"])
    clap = clap_compounds.merge(clap_structures[["clap_code", "smiles_pubchem", "resolution_status"]], on="clap_code", validate="one_to_one")
    exclusions_path = ROOT / "data/manual/clap_structure_exclusions.csv"
    source_files.append(exclusions_path)
    exclusions = pd.read_csv(exclusions_path).set_index("clap_code")["reason"].to_dict()
    excluded_cas = set(clap.loc[clap.clap_code.isin(exclusions), "cas"])
    method_rows = []
    for method, filename in FDA_FILES.items():
        path = background / filename
        source_files.append(path)
        parsed = parse_fda(path, method)
        actual = {row["clap_code"]: row for row in parsed}
        for row in clap.to_dict("records"):
            reported = row[f"{method}_status"]
            if reported != "not_reported" and row["clap_code"] not in actual:
                raise ValueError(f"FDA parser missed {row['clap_code']} / {method}")
            if row["clap_code"] in actual:
                found = actual[row["clap_code"]]
                for bound in ["min", "max"]:
                    expected = row[f"{method}_rrf_{bound}"]
                    observed = found[f"rrf_{bound}"]
                    if expected != "" and not np.isclose(float(expected), observed):
                        raise ValueError(f"FDA RRF audit mismatch: {row['clap_code']} / {method} / {bound}")
                found["is_internal_standard"] = row[f"{method}_is_internal_standard"]
                method_rows.append(found)
            else:
                method_rows.append({"clap_code": row["clap_code"], "method": method, "outcome": "not_reported", "is_internal_standard": False})
    pd.DataFrame(method_rows).to_parquet(output / "clap_methods.parquet", index=False)

    bridges = defaultdict(list)

    def add_bridge(cas_value, smiles, key, source):
        if smiles and key:
            for number in extract_cas(cas_value):
                if number not in excluded_cas:
                    bridges[number].append({"smiles": smiles, "inchikey": key, "source": source})

    plastics_records = []
    for row in plastics.to_dict("records"):
        smiles = clean(row["isomeric_smiles"]) or clean(row["canonical_smiles"])
        key = clean(row["inchikey"])
        add_bridge(row["cas"], smiles, key, "PlastChem CAS bridge")
        plastics_records.append({
            "plastchem_id": str(row["plastchem_ID"]), "cas": clean(row["cas"]), "inchikey": key,
            "name": clean(row["pubchem_name"]), "functions": clean(row["Harmonized_functions"]),
            "applications": clean(row["industrial_sector_plasticmap"]),
            "chemical_groups": " | ".join(name for name in ["bisphenols", "orthophthalates", "PFASs", "phenolic_antioxidants", "aromatic_amines", "silanes_siloxanes_silicones"] if row.get(name) == 1),
            "hazards": json.dumps({name: float(row[name]) for name in HAZARDS if pd.notna(row[name])}),
            "polymers": json.dumps({name: float(row[name]) for name in POLYMERS if pd.notna(row.get(name))}),
            "lists": clean(row["PlastChem_lists"]),
        })
    pd.DataFrame(plastics_records).to_parquet(output / "plastchem.parquet", index=False)

    stav_records = []
    for row in stavitskaya.to_dict("records"):
        smiles = clean(row["SMILES"])
        key = raw_key(smiles)
        add_bridge(row["Reported CAS"], smiles, key, "Stavitskaya CAS bridge")
        stav_records.append({"source_id": str(row["ID"]), "cas": clean(row["Reported CAS"]),
                             "inchikey": key, "name": clean(row["Reported Chemical Name"]),
                             "tier1": clean(row["Tier 1 clusters"]), "tier2": clean(row["Tier 2 Clusters"])})
    pd.DataFrame(stav_records).to_parquet(output / "stavitskaya.parquet", index=False)

    catalog = []
    for row in clap.to_dict("records"):
        properties = structure(row["smiles_pubchem"]) if row["clap_code"] not in exclusions else None
        if properties:
            add_bridge(row["cas"], row["smiles_pubchem"], properties["inchikey"], "FDA CLAP structure table CAS bridge")
        catalog.append({"compound_id": "CLAP-" + row["clap_code"], "name": row["primary_name"],
                        "aliases": row["all_names"], "cas_raw": row["cas"],
                        "valid_cas": " | ".join(extract_cas(row["cas"])), "source": "FDA CLAP",
                        "clap_code": row["clap_code"], "structure_status": "Resolved" if properties else "Unresolved",
                        "structure_source": "Team draft PubChem structure table" if properties else "",
                        "structure_note": exclusions.get(row["clap_code"], "Source identity retained for exact matches; cleaned parent used only for similarity."),
                        **(properties or {})})

    overrides = pd.read_csv(ROOT / "data/manual/structure_overrides.csv").fillna("")
    source_files.append(ROOT / "data/manual/structure_overrides.csv")
    if overrides["borton_id"].duplicated().any():
        raise ValueError("Duplicate borton_id in manual structure overrides")
    override_lookup = overrides.set_index("borton_id").to_dict("index")
    counts = defaultdict(int)
    for index, row in enumerate(borton.to_dict("records"), 1):
        identifier = f"B{index:05d}"
        name, raw_cas = row.values()
        cas_values = extract_cas(raw_cas)
        candidates = [entry for number in cas_values for entry in bridges[number]]
        candidate_keys = {entry["inchikey"] for entry in candidates}
        properties, provenance, note = None, "", ""
        if identifier in override_lookup:
            override = override_lookup[identifier]
            if not override["review_note"]:
                raise ValueError(f"A review note is required for override {identifier}")
            properties = structure(override["smiles"])
            if properties is None:
                raise ValueError(f"Invalid manual override: {identifier}")
            provenance, note = "Manual reviewed override", override["review_note"]
        elif set(cas_values) & excluded_cas:
            note = "A reported CAS identifies the excluded RM84 commercial blend; a reviewed structure is required."
        elif len(candidate_keys) == 1:
            properties = structure(candidates[0]["smiles"])
            provenance = " | ".join(sorted({entry["source"] for entry in candidates}))
        elif len(candidate_keys) > 1:
            note = "Conflicting source structures for the reported CAS identifiers; manual review required."
        status = "Resolved" if properties else "Conflict" if len(candidate_keys) > 1 or set(cas_values) & excluded_cas else "Unresolved"
        counts[status] += 1
        catalog.append({"compound_id": identifier, "name": clean(name), "aliases": clean(name),
                        "cas_raw": clean(raw_cas), "valid_cas": " | ".join(cas_values), "source": "Borton",
                        "clap_code": "", "structure_status": status, "structure_source": provenance,
                        "structure_note": note, **(properties or {})})
    compounds = pd.DataFrame(catalog).fillna({"smiles_raw": "", "smiles_std": "", "inchikey": "", "inchikey_std": ""})
    resolved = compounds[compounds["smiles_std"].ne("")].drop_duplicates("inchikey_std")
    matrix = fingerprint_matrix(resolved["smiles_std"].tolist())
    center = matrix.mean(axis=0)
    centered = matrix - center
    _, singular_values, components = np.linalg.svd(centered, full_matrices=False)
    axes = components[:3]
    for axis in axes:
        if axis[np.argmax(np.abs(axis))] < 0:
            axis *= -1
    coordinates = np.einsum("ij,kj->ik", centered, axes, optimize=False)
    if not np.isfinite(coordinates).all():
        raise ValueError("Fingerprint projection produced non-finite coordinates")
    explained_variance = singular_values[:3] ** 2 / np.sum(singular_values ** 2)
    np.savez_compressed(output / "projection.npz", center=center, axes=axes)
    projection = pd.DataFrame(coordinates, columns=["map_x", "map_y", "map_z"])
    projection["inchikey_std"] = resolved["inchikey_std"].to_numpy()
    compounds = compounds.merge(projection, on="inchikey_std", how="left", validate="many_to_one")
    compounds.to_parquet(output / "compounds.parquet", index=False)

    msg = {"loaded": False, "reason": "MassSpecGym.tsv was not supplied; spectrum availability is unknown."}
    if massspecgym is not None and massspecgym.exists():
        source_files.append(massspecgym)
        msg = prepare_massspecgym(massspecgym, output)
    elif (output / "massspecgym.parquet").exists():
        (output / "massspecgym.parquet").unlink()
    manifest = {
        "schema_version": 1, "generated_utc": datetime.now(timezone.utc).isoformat(),
        "sample_data": False, "description": "Real supplied source files; no synthetic compounds or inferred hazard labels.",
        "counts": {"borton_rows": len(borton), "borton_resolved": counts["Resolved"],
                   "borton_unresolved": counts["Unresolved"], "borton_conflicts": counts["Conflict"],
                   "borton_unique_structures": compounds.loc[compounds.source.eq("Borton") & compounds.inchikey.ne(""), "inchikey"].nunique(),
                   "clap_rows": len(clap), "clap_resolved": int(compounds.source.eq("FDA CLAP").mul(compounds.inchikey.ne("")).sum()),
                   "plastchem_rows": len(plastics), "stavitskaya_rows": len(stavitskaya)},
        "projection": {"method": "PCA of 2048-bit Morgan fingerprints (radius 2), unique standardized structures",
                       "explained_variance_ratio": explained_variance.tolist()},
        "massspecgym": msg, "nist": {"loaded": False, "reason": "Raw polymer extraction runs are not a curated compound-to-polymer table."},
        "natural_occurrence": {"loaded": False, "reason": "No natural-occurrence field exists in the supplied PlastChem workbook."},
        "source_urls": SOURCES,
        "files": [{"file": path.name, "bytes": path.stat().st_size,
                   "sha256": hash_file(path)} for path in source_files],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest["counts"], indent=2), flush=True)
    print(json.dumps(msg, indent=2), flush=True)


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare traceable local E&L evidence tables.")
    parser.add_argument("--background", type=Path, default=ROOT.parent / "Relevant BAckground info")
    parser.add_argument("--draft", type=Path, default=ROOT.parent / "first draft of prototype")
    parser.add_argument("--output", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--massspecgym", type=Path, default=ROOT / "data/raw/MassSpecGym.tsv")
    arguments = parser.parse_args()
    build(arguments.background, arguments.draft, arguments.output, arguments.massspecgym)
