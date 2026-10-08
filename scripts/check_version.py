"""Require a release tag to match the version embedded in project metadata."""
from pathlib import Path
import sys
import tomllib
root = Path(__file__).resolve().parents[1]
version = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
if len(sys.argv) != 2 or sys.argv[1] != "v" + version:
    raise SystemExit(f"Expected release tag v{version}")
print(f"Release version verified: {version}")
