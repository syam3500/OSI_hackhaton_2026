from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
import pytest

from elspace.bootstrap import create_bundle, ensure_dataset
from elspace.datasets import build_target_store, read_upload, upload_digest
from elspace.evidence import EvidenceStore

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def store():
    if not (ROOT / "data/processed/manifest.json").exists():
        pytest.skip("Prepared data required")
    return EvidenceStore.load(ROOT / "data/processed")


def test_csv_upload_keeps_identifiers_as_text():
    frame = read_upload(b"Name,CAS,SMILES\nEthanol,64-17-5,CCO\n", "targets.csv")
    assert frame.iloc[0].CAS == "64-17-5"
    assert frame.iloc[0].SMILES == "CCO"


def test_xlsx_upload_reads_the_selected_source_worksheet():
    path = ROOT.parent / "Relevant BAckground info/Borton_Sup2_AppendixB_18000.xlsx"
    if not path.exists():
        pytest.skip("Source workbook not supplied")
    frame = read_upload(path.read_bytes(), path.name, "Extractables")
    assert len(frame) == 18266
    assert "Chemical Name" in frame.columns


def test_upload_rejects_empty_and_oversized_tables():
    with pytest.raises(ValueError, match="between 1 and 20,000"):
        read_upload(b"Name,CAS\n", "empty.csv")
    with pytest.raises(ValueError, match="25 MB"):
        read_upload(b" " * (25 * 1024 * 1024 + 1), "large.csv")


def test_target_swap_preserves_reference_layers_and_flags_conflicts(store):
    frame = pd.DataFrame([{"Name": "Ethanol", "CAS": "64-17-5", "SMILES": "CCO"},
                          {"Name": "Wrong structure", "CAS": "80-05-7", "SMILES": "CCO"},
                          {"Name": "Invalid", "CAS": "", "SMILES": "not smiles"}])
    replaced = build_target_store(store, frame, "my_targets.csv", "test-digest", "Name", "CAS", "SMILES", ROOT / "data/processed/projection.npz")
    targets = replaced.compounds[replaced.compounds.source.eq("Uploaded targets")]
    assert len(targets) == 3
    assert targets.structure_status.tolist() == ["Resolved", "Conflict", "Unresolved"]
    assert targets.iloc[0].map_x == pytest.approx(targets.iloc[0].map_x)
    assert len(replaced.clap) == len(store.clap)
    assert len(replaced.methods) == len(store.methods)
    assert len(store.compounds) == 18372
    assert replaced.manifest["active_targets"]["rows"] == 3
    assert replaced.manifest["active_targets"]["label"] == "my_targets.csv"


def test_cas_only_upload_uses_loaded_structures(store):
    frame = pd.DataFrame([{"Name": "BPA", "CAS": "80-05-7"}])
    replaced = build_target_store(store, frame, "targets.csv", "cas-only", "Name", "CAS", None, ROOT / "data/processed/projection.npz")
    target = replaced.compounds[replaced.compounds.compound_id.eq("UP00001")].iloc[0]
    assert target.structure_status == "Resolved"


def test_deployment_bundle_loads_without_source_files(tmp_path, store):
    archive = tmp_path / "evidence.zip"
    create_bundle(ROOT / "data/processed", archive)
    directory = ensure_dataset(tmp_path / "fresh_deployment", archive)
    deployed = EvidenceStore.load(directory)
    assert len(deployed.compounds) == len(store.compounds)
    assert deployed.manifest["counts"] == store.manifest["counts"]
    assert (directory / "projection.npz").exists()
    assert (directory / ".deployment_bundle_sha256").exists()


def test_deployment_bundle_update_replaces_default_data(tmp_path, store):
    source = ROOT / "data/processed"
    archive = tmp_path / "evidence.zip"
    create_bundle(source, archive)
    directory = ensure_dataset(tmp_path / "deployed", archive)
    original = json.loads((directory / "manifest.json").read_text())
    with ZipFile(archive) as package:
        payloads = {name: package.read(name) for name in package.namelist()}
    manifest = json.loads(payloads["manifest.json"])
    manifest["deployment_revision"] = "updated"
    payloads["manifest.json"] = json.dumps(manifest).encode()
    import hashlib

    checksums = json.loads(payloads["bundle_checksums.json"])
    checksums["manifest.json"] = hashlib.sha256(payloads["manifest.json"]).hexdigest()
    payloads["bundle_checksums.json"] = json.dumps(checksums).encode()
    with ZipFile(archive, "w") as package:
        for name, payload in payloads.items():
            package.writestr(name, payload)
    ensure_dataset(directory, archive)
    assert json.loads((directory / "manifest.json").read_text())["deployment_revision"] == "updated"
    assert "deployment_revision" not in original


def test_deployment_bundle_rejects_unexpected_paths(tmp_path):
    archive = tmp_path / "untrusted.zip"
    with ZipFile(archive, "w") as package:
        package.writestr("../unexpected.txt", "not allowed")
    with pytest.raises(ValueError, match="Unexpected entries"):
        ensure_dataset(tmp_path / "data", archive)
