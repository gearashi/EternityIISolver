"""Build a native app archive; never starts a solver or downloads board data."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from build_support import resource_files, documentation_files

def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()

def write_metadata(flavor, version):
    destination = ROOT / "build" / "release-metadata"
    destination.mkdir(parents=True, exist_ok=True)
    licenses = destination / "THIRD_PARTY_LICENSES"
    licenses.mkdir(exist_ok=True)
    versions = []
    for dist in sorted(metadata.distributions(), key=lambda item: item.metadata.get("Name", "").lower()):
        name = dist.metadata.get("Name", "unknown")
        versions.append({"name": name, "version": dist.version})
        safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", name)
        for item in dist.files or ():
            if not any(token in str(item).lower() for token in ("license", "licence", "copying", "notice")):
                continue
            source = Path(dist.locate_file(item))
            if not source.is_file() or source.suffix.lower() in (".pyc", ".so", ".dll", ".pyd"):
                continue
            # Metadata file paths vary; flatten them within each distribution.
            target = licenses / safe_name / str(item).replace("\\", "_").replace("/", "_")
            target.parent.mkdir(exist_ok=True)
            shutil.copyfile(source, target)
    python_license = Path(sys.base_prefix) / "LICENSE.txt"
    if python_license.is_file():
        shutil.copyfile(python_license, licenses / "PYTHON-LICENSE.txt")
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    manifest = {"schema_version": 1, "version": version, "flavor": flavor,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": revision.stdout.strip() if revision.returncode == 0 else None,
        "system": platform.system(), "architecture": platform.machine(),
        "python_version": platform.python_version(), "distributions": versions,
        "public_resources": [{"path": relative, "sha256": sha256(source)} for relative, source in resource_files(ROOT)],
        "documentation": [{"path": relative, "sha256": sha256(source)} for relative, source in documentation_files(ROOT)],
        "hardware_validation": "Packaging and CPU validation do not certify a GPU/driver. See release test reports.",
        "private_runtime_data_included": False}
    (destination / "BUILD_INFO.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flavor", choices=("cuda-opencl", "opencl"), required=True)
    parser.add_argument("--skip-smoke", action="store_true", help="Build-only debugging; CI never uses this")
    args = parser.parse_args()
    version = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
    if sys.platform == "darwin" and args.flavor != "opencl":
        parser.error("macOS builds use OpenCL")
    write_metadata(args.flavor, version)
    environment = os.environ.copy()
    environment["ETERNITY_BUILD_CUDA"] = "1" if args.flavor == "cuda-opencl" else "0"
    subprocess.run([sys.executable, "-m", "PyInstaller", "--clean", "--noconfirm", "EternityIISolver.spec"], cwd=ROOT, env=environment, check=True)
    app_name = "EternityIISolver.app" if sys.platform == "darwin" else "EternityIISolver"
    app = ROOT / "dist" / app_name
    executable = app / "Contents" / "MacOS" / "EternityIISolver" if sys.platform == "darwin" else app / ("EternityIISolver.exe" if sys.platform == "win32" else "EternityIISolver")
    if not args.skip_smoke:
        with tempfile.TemporaryDirectory(prefix="eternity-frozen-smoke-") as temporary:
            checked = subprocess.run([str(executable), "validate"], cwd=temporary, capture_output=True, text=True, check=True, timeout=90)
            report = json.loads(checked.stdout)
            if not report.get("valid") or report.get("score") != 466:
                raise RuntimeError(f"Frozen resource validation failed: {report}")
            subprocess.run([str(executable), "status", "--json", "--state-dir", temporary], cwd=temporary, check=True, timeout=30)
            for command in ("diagnose", "boinc"):
                subprocess.run([str(executable), command, "--help"], cwd=temporary,
                               capture_output=True, text=True, check=True, timeout=30)
    # Top-level readable documentation accompanies the app and its dependency notices.
    if sys.platform != "darwin":
        for relative, source in documentation_files(ROOT):
            destination = app / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
    system = {"win32": "windows", "darwin": "macos"}.get(sys.platform, "linux")
    architecture = {"AMD64": "x64", "x86_64": "x64", "aarch64": "arm64", "arm64": "arm64"}.get(platform.machine(), platform.machine())
    release = ROOT / "release-assets"
    release.mkdir(exist_ok=True)
    base = release / f"EternityIISolver-{version}-{system}-{architecture}-{args.flavor}"
    format_name = "zip" if sys.platform == "win32" else "gztar"
    archive = Path(shutil.make_archive(str(base), format_name, root_dir=ROOT / "dist", base_dir=app_name))
    Path(str(archive) + ".sha256").write_text(sha256(archive) + "  " + archive.name + "\n", encoding="utf-8")
    print(json.dumps({"archive": str(archive), "sha256": sha256(archive), "frozen_smoke_passed": not args.skip_smoke}))

if __name__ == "__main__":
    main()
