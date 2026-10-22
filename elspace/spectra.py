from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


def parse_peaks(mzs: str, intensities: str) -> tuple[np.ndarray, np.ndarray]:
    masses = np.asarray([float(value) for value in str(mzs).split(",")], dtype=float)
    values = np.asarray([float(value) for value in str(intensities).split(",")], dtype=float)
    if not len(masses) or len(masses) != len(values) or not np.isfinite(masses).all() or not np.isfinite(values).all():
        raise ValueError("A spectrum needs equal-length finite m/z and intensity arrays.")
    if (masses <= 0).any() or (values < 0).any() or values.max() <= 0:
        raise ValueError("Spectral m/z must be positive and intensities non-negative with a positive maximum.")
    order = np.argsort(masses)
    return masses[order], values[order]


def prepare_reference_peaks(source: Path, destination: Path, per_polarity: int = 3) -> dict:
    columns = ["identifier", "inchikey", "mzs", "intensities", "adduct", "instrument_type", "collision_energy", "precursor_mz", "formula", "smiles"]
    rows, counts = [], Counter()
    invalid = 0
    for chunk in pd.read_csv(source, sep="\t", usecols=columns, chunksize=10000):
        for row in chunk.to_dict("records"):
            connectivity = str(row["inchikey"]).split("-")[0]
            polarity = "positive" if str(row["adduct"]).endswith("+") else "negative" if str(row["adduct"]).endswith("-") else "unspecified"
            group = (connectivity, polarity)
            if counts[group] >= per_polarity:
                continue
            try:
                parse_peaks(row["mzs"], row["intensities"])
            except (ValueError, TypeError):
                invalid += 1
                continue
            rows.append({**row, "connectivity_key": connectivity, "polarity": polarity})
            counts[group] += 1
    pd.DataFrame(rows).to_parquet(destination, index=False, compression="zstd")
    return {"loaded": True, "spectra": len(rows), "connectivity_keys": len({key for key, _ in counts}),
            "selection": f"First {per_polarity} valid source spectra per connectivity key and polarity, in source order; not an averaged or simulated spectrum",
            "invalid_skipped": invalid, "source_file": source.name, "file": destination.name}
