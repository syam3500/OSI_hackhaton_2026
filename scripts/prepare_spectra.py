from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.prepare_data import hash_file, prepare_massspecgym


if __name__ == "__main__":
    source = ROOT / "data/raw/MassSpecGym.tsv"
    directory = ROOT / "data/processed"
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    expected = next(item["sha256"] for item in manifest["files"] if item["file"] == source.name)
    if hash_file(source) != expected:
        raise ValueError("The spectral source differs from the manifest; rebuild the reference dataset instead.")
    manifest["massspecgym"] = prepare_massspecgym(source, directory)
    manifest["generated_utc"] = datetime.now(timezone.utc).isoformat()
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest["massspecgym"], indent=2))
