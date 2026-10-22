from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
from rdkit import DataStructs

from elspace.chemistry import DESCRIPTORS, extract_cas, fingerprint

METHODS = {"GCMS_POS": "GC/MS", "LCMS_POS": "LC/MS positive", "LCMS_NEG": "LC/MS negative"}
OUTCOMES = {
    "measured": "Measured RRF", "partial": "Measured at some concentrations",
    "below_DL": "Below detection limit", "not_detected": "Not detected", "not_reported": "Not reported",
}


def search_compounds(table: pd.DataFrame, query: str) -> pd.DataFrame:
    value = query.strip().casefold()
    if not value:
        value = "80-05-7"
    if re.fullmatch(r"\d{2,7}-\d{2}-\d", value):
        mask = table.cas_raw.str.contains(r"(?<!\d)" + re.escape(value) + r"(?!\d)", regex=True, na=False)
    else:
        mask = pd.Series(False, index=table.index)
        for column in ["name", "aliases", "cas_raw", "inchikey", "compound_id"]:
            mask |= table[column].fillna("").astype(str).str.casefold().str.contains(value, regex=False)
    matches = table[mask].copy()
    matches["_structure_order"] = matches.structure_status.map({"Resolved": 0, "Conflict": 1, "Unresolved": 2})
    return matches.sort_values(["_structure_order", "source", "name"], kind="stable").drop(columns="_structure_order")


def match_rows(target: dict, table: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    if target.get("structure_status") == "Conflict":
        return table.iloc[:0], "Identity conflict requires review"
    key = target.get("inchikey", "")
    if isinstance(key, str) and key:
        exact = table[table["inchikey"].eq(key)]
        if len(exact):
            return exact, "Full InChIKey"
    numbers = set(extract_cas(target.get("valid_cas", target.get("cas_raw", ""))))
    matches = table[table["cas"].isin(numbers)]
    if matches.empty:
        return matches, "No match"
    keys = set(matches["inchikey"].dropna()) - {""}
    if key and keys - {key}:
        return matches.iloc[:0], "CAS / structure conflict"
    if not key and len(keys) > 1:
        return matches.iloc[:0], "Ambiguous CAS match"
    return matches, "CAS fallback (identity unverified)"


def descriptor_ranges(target: dict, reference: pd.DataFrame, margin: float) -> pd.DataFrame:
    rows = []
    for name, label in DESCRIPTORS.items():
        low, high = reference[name].min(), reference[name].max()
        padding = (high - low) * margin
        value = float(target[name])
        rows.append({"Descriptor": label, "Target": value, "Panel min": low, "Panel max": high,
                     "Allowed min": low - padding, "Allowed max": high + padding,
                     "Inside range": bool(np.isfinite(low) and low - padding <= value <= high + padding)})
    return pd.DataFrame(rows)


@dataclass
class EvidenceStore:
    compounds: pd.DataFrame
    methods: pd.DataFrame
    plastics: pd.DataFrame
    stavitskaya: pd.DataFrame
    massspecgym: pd.DataFrame
    manifest: dict
    spectra: pd.DataFrame = field(default_factory=pd.DataFrame)

    def __post_init__(self):
        self.clap = self.compounds[self.compounds.source.eq("FDA CLAP")].reset_index(drop=True).copy()
        self.clap["cas"] = self.clap["cas_raw"]
        self.reference = self.clap[self.clap.smiles_std.ne("")].reset_index(drop=True)
        self.fingerprints = [fingerprint(smiles) for smiles in self.reference.smiles_std]
        self.layer_indices = {}
        for name, frame in [("plastics", self.plastics), ("stavitskaya", self.stavitskaya)]:
            self.layer_indices[name] = (
                {key: group.index.to_numpy() for key, group in frame[frame.inchikey.ne("")].groupby("inchikey")},
                {key: group.index.to_numpy() for key, group in frame.groupby("cas")},
            )
        self.msg_index = self.massspecgym.set_index("connectivity_key") if len(self.massspecgym) else None
        self.peak_indices = self.spectra.groupby("connectivity_key").indices if len(self.spectra) else {}

    @classmethod
    def load(cls, directory: Path):
        manifest = json.loads((directory / "manifest.json").read_text())
        load = lambda name: pd.read_parquet(directory / (name + ".parquet"))
        msg = load("massspecgym") if manifest["massspecgym"]["loaded"] else pd.DataFrame()
        peaks = load("massspecgym_peaks") if manifest["massspecgym"].get("peak_spectra", {}).get("loaded") else pd.DataFrame()
        return cls(load("compounds"), load("clap_methods"), load("plastchem"), load("stavitskaya"), msg, manifest, peaks)

    def reference_spectra(self, target: dict) -> pd.DataFrame:
        if target.get("structure_status") == "Conflict":
            return self.spectra.iloc[:0]
        key = str(target.get("inchikey", "")).split("-")[0]
        return self.spectra.iloc[self.peak_indices.get(key, [])].copy()

    def layer(self, target: dict, name: str) -> tuple[pd.DataFrame, str]:
        frame = self.plastics if name == "plastics" else self.stavitskaya
        if target.get("structure_status") == "Conflict":
            return frame.iloc[:0], "Identity conflict requires review"
        keys, numbers = self.layer_indices[name]
        key = target.get("inchikey", "")
        if key and key in keys:
            return frame.loc[keys[key]], "Full InChIKey"
        indices = [index for number in extract_cas(target.get("valid_cas", "")) for index in numbers.get(number, [])]
        return match_rows(target, frame.loc[sorted(set(indices))])

    def neighbours(self, target: dict, count: int = 5) -> pd.DataFrame:
        if not target.get("smiles_std"):
            return pd.DataFrame()
        similarities = np.array(DataStructs.BulkTanimotoSimilarity(fingerprint(target["smiles_std"]), self.fingerprints))
        frame = self.reference.copy()
        frame["Tanimoto"] = similarities
        frame = frame[frame.inchikey.ne(target.get("inchikey", ""))]
        return frame.sort_values(["Tanimoto", "clap_code"], ascending=[False, True]).head(count)

    def assess(self, target: dict, cutoff: float, margin: float) -> list[dict]:
        if target.get("source") == "FDA CLAP" and target.get("clap_code"):
            exact = self.clap[self.clap.clap_code.eq(target["clap_code"])]
            basis = "CLAP record code" if not target.get("smiles_std") else "Full InChIKey"
        else:
            exact, basis = match_rows(target, self.clap)
        results = []
        for method, label in METHODS.items():
            observations = self.methods[self.methods.method.eq(method)]
            direct = observations[observations.clap_code.isin(exact.clap_code)].copy()
            numeric = direct[pd.to_numeric(direct.rrf_min, errors="coerce").notna()]
            panel_codes = observations.loc[pd.to_numeric(observations.rrf_min, errors="coerce").notna(), "clap_code"]
            panel = self.reference[self.reference.clap_code.isin(panel_codes)]
            check, similarity, closest = pd.DataFrame(), None, None
            if target.get("smiles_std") and len(panel):
                check = descriptor_ranges(target, panel, margin)
                values = DataStructs.BulkTanimotoSimilarity(fingerprint(target["smiles_std"]), [fingerprint(smiles) for smiles in panel.smiles_std])
                best = int(np.argmax(values))
                similarity, closest = float(values[best]), panel.iloc[best]["clap_code"]
            if len(numeric):
                status, reason = "Measured", "A numeric RRF is reported for this identity under the FDA conditions."
            elif not direct.empty and direct.outcome.isin(["not_detected", "below_DL"]).any():
                status = "Unsupported"
                reason = "The exact FDA record reports no numeric RRF (not detected or below the detection limit)."
            elif not target.get("smiles_std"):
                status, reason = "Unresolved", "A reviewed molecular structure is needed for a similarity and descriptor check."
            elif similarity is not None and similarity >= cutoff and check["Inside range"].all():
                status, reason = "Represented", "A measured neighbour passes the similarity cutoff and every method-specific descriptor range."
            else:
                status = "Unsupported"
                failures = []
                if similarity is None or similarity < cutoff:
                    failures.append("no measured neighbour passes the similarity cutoff")
                if not check.empty:
                    outside = check.loc[~check["Inside range"], "Descriptor"].tolist()
                    if outside:
                        failures.append("outside range: " + ", ".join(outside))
                reason = "; ".join(failures).capitalize() + "."
            results.append({"method": method, "label": label, "status": status, "reason": reason,
                            "match_basis": basis, "exact_records": direct, "descriptor_check": check,
                            "similarity": similarity, "closest_clap": closest, "panel_size": len(panel)})
        return results

    def spectrum(self, target: dict) -> dict:
        if self.msg_index is None:
            return {"status": "Not loaded", "match_basis": "Not assessed"}
        key = target.get("inchikey", "")
        if not key:
            return {"status": "Unknown — unresolved identity", "match_basis": "Not assessed"}
        connectivity = key.split("-")[0]
        if connectivity not in self.msg_index.index:
            return {"status": "No reference in loaded MassSpecGym", "match_basis": "InChIKey connectivity block"}
        return {"status": "Reference spectra available", "match_basis": "InChIKey connectivity block (not stereochemistry)",
                **self.msg_index.loc[connectivity].to_dict()}

    def enrich_catalog(self, method: str, cutoff: float, margin: float) -> pd.DataFrame:
        frame = self.compounds.copy()
        observations = self.methods[self.methods.method.eq(method)]
        measured_codes = set(observations.loc[pd.to_numeric(observations.rrf_min, errors="coerce").notna(), "clap_code"])
        panel = self.reference[self.reference.clap_code.isin(measured_codes)]
        panel_fps = [fingerprint(smiles) for smiles in panel.smiles_std]
        unique = frame[frame.smiles_std.ne("")].drop_duplicates("inchikey_std")
        similarities = {row.inchikey_std: max(DataStructs.BulkTanimotoSimilarity(fingerprint(row.smiles_std), panel_fps), default=0)
                        for row in unique.itertuples()}
        frame["nearest_measured_similarity"] = frame.inchikey_std.map(similarities)
        inside = frame.smiles_std.ne("")
        for descriptor in DESCRIPTORS:
            low, high = panel[descriptor].min(), panel[descriptor].max()
            padding = (high - low) * margin
            inside &= frame[descriptor].between(low - padding, high + padding)
        frame["coverage"] = np.where(frame.smiles_std.eq(""), "Unresolved",
                                     np.where(inside & frame.nearest_measured_similarity.ge(cutoff), "Represented", "Unsupported"))
        exact = self.clap[["inchikey", "clap_code"]].merge(observations[["clap_code", "outcome", "rrf_min"]], on="clap_code")
        measured_keys = set(exact.loc[pd.to_numeric(exact.rrf_min, errors="coerce").notna(), "inchikey"]) - {""}
        nondetected = set(exact.loc[exact.outcome.isin(["not_detected", "below_DL"]), "inchikey"]) - {""}
        frame.loc[frame.inchikey.isin(nondetected), "coverage"] = "Unsupported"
        frame.loc[frame.inchikey.isin(measured_keys), "coverage"] = "Measured"
        unresolved_reference = self.clap[self.clap.smiles_std.eq("")].merge(observations, on="clap_code")
        for record in unresolved_reference.to_dict("records"):
            status = "Measured" if pd.notna(record["rrf_min"]) else "Unsupported" if record["outcome"] in ["not_detected", "below_DL"] else "Unresolved"
            own_record = frame.source.eq("FDA CLAP") & frame.clap_code.eq(record["clap_code"])
            cas_fallback = frame.smiles_std.eq("") & frame.structure_status.ne("Conflict") & frame.valid_cas.eq(record["cas"])
            frame.loc[own_record | cas_fallback, "coverage"] = status
        for column, name in [("hazard_evidence", "plastics"), ("chemical_class", "stavitskaya")]:
            mapping = {}
            for key, group in frame.groupby("inchikey", dropna=False):
                if not key:
                    continue
                records, _ = self.layer(group.iloc[0].to_dict(), name)
                if name == "plastics":
                    values = [float(value) for text in records.hazards for value in json.loads(text).values()]
                    mapping[key] = "Reported flags" if any(value > 0 for value in values) else "No positive flags in source" if len(records) else "No source match"
                else:
                    mapping[key] = " | ".join(sorted(set(records.tier1) - {""})) or "Not reported"
            frame[column] = frame.inchikey.map(mapping).fillna("Unknown — unresolved identity")
        if self.msg_index is not None:
            frame["ms_reference"] = np.where(frame.inchikey.eq(""), "Unknown — unresolved identity",
                                            np.where(frame.inchikey.str[:14].isin(self.msg_index.index), "Reference spectra available", "No reference in loaded MassSpecGym"))
        else:
            frame["ms_reference"] = "Not loaded"
        return frame
