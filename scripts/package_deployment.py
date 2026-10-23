from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from elspace.bootstrap import create_bundle


if __name__ == "__main__":
    output = ROOT / "deployment/evidence.zip"
    create_bundle(ROOT / "data/processed", output)
    print(f"Prepared {output.relative_to(ROOT)} ({output.stat().st_size:,} bytes).")
