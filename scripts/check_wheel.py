"""Reject wheels that omit curated assets or include runtime/private directories."""
import argparse
from pathlib import Path
import sys
import tomllib
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from build_support import PUBLIC_RESOURCES

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("wheel", type=Path)
args = parser.parse_args()
with zipfile.ZipFile(args.wheel) as archive:
    names = set(archive.namelist())
    expected = {"eternity_resources/" + name for name in PUBLIC_RESOURCES}
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    modules = {name + ".py" for name in metadata["tool"]["setuptools"]["py-modules"]}
    missing = sorted((expected | modules) - names)
    if missing:
        raise SystemExit("Wheel resources/modules missing: " + ", ".join(missing))
    forbidden = {"runtime", "library", "results", ".venv", ".git", ".env"}
    bad = sorted(name for name in names if forbidden.intersection(Path(name).parts)
                 or name.endswith((".sqlite", ".npz", ".log")))
    if bad:
        raise SystemExit("Private/runtime files in wheel: " + ", ".join(bad))
print(f"Wheel checked: {len(expected)} public resources; no runtime/database/checkpoint files")
