from __future__ import annotations

import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

DATA_FILES = ["compounds.parquet", "clap_methods.parquet", "plastchem.parquet", "stavitskaya.parquet",
              "massspecgym.parquet", "massspecgym_peaks.parquet", "projection.npz", "manifest.json"]


def required_files(manifest: dict) -> set[str]:
    required = set(DATA_FILES) - {"massspecgym.parquet", "massspecgym_peaks.parquet"}
    if manifest["massspecgym"]["loaded"]:
        required.add("massspecgym.parquet")
    if manifest["massspecgym"].get("peak_spectra", {}).get("loaded"):
        required.add("massspecgym_peaks.parquet")
    return required


def create_bundle(directory: Path, destination: Path):
    manifest = json.loads((directory / "manifest.json").read_text())
    required = sorted(required_files(manifest))
    payloads = {name: (directory / name).read_bytes() for name in required}
    checksums = {name: hashlib.sha256(value).hexdigest() for name, value in payloads.items()}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(destination, "w", compression=8) as archive:
        for name, payload in payloads.items():
            archive.writestr(name, payload)
        archive.writestr("bundle_checksums.json", json.dumps(checksums))


def ensure_dataset(directory: Path, bundle: Path) -> Path:
    stamp = directory / ".deployment_bundle_sha256"
    if (directory / "manifest.json").exists() and not stamp.exists():
        return directory
    if not bundle.exists():
        return directory
    bundle_hash = hashlib.sha256(bundle.read_bytes()).hexdigest()
    if (directory / "manifest.json").exists() and stamp.exists() and stamp.read_text() == bundle_hash:
        return directory
    with ZipFile(bundle) as archive:
        if set(archive.namelist()) - {*DATA_FILES, "bundle_checksums.json"}:
            raise ValueError("Unexpected entries in the deployment dataset bundle")
        checksums = json.loads(archive.read("bundle_checksums.json"))
        payloads = {}
        for name, checksum in checksums.items():
            if name not in DATA_FILES:
                raise ValueError("Unsupported dataset bundle member")
            payload = archive.read(name)
            if hashlib.sha256(payload).hexdigest() != checksum:
                raise ValueError("Dataset bundle checksum mismatch")
            payloads[name] = payload
        manifest = json.loads(payloads["manifest.json"])
        required = required_files(manifest)
        if set(payloads) != required:
            raise ValueError("The deployment dataset bundle is incomplete")
    directory.mkdir(parents=True, exist_ok=True)
    for name, payload in payloads.items():
        if name != "manifest.json":
            (directory / name).write_bytes(payload)
    (directory / "manifest.json").write_bytes(payloads["manifest.json"])
    stamp.write_text(bundle_hash)
    return directory
