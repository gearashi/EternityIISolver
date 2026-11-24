"""Build a native app archive and run isolated, bounded solver smoke checks offline."""
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
from cuda_bundle import check_windows_cuda_archive
from build_radius_helper import build as build_radius_helper
from build_fast_helper import build as build_fast_helper

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
    build_radius_helper(ROOT)
    build_fast_helper(ROOT)
    write_metadata(args.flavor, version)
    environment = os.environ.copy()
    environment["ETERNITY_BUILD_CUDA"] = "1" if args.flavor == "cuda-opencl" else "0"
    subprocess.run([sys.executable, "-m", "PyInstaller", "--clean", "--noconfirm", "EternityIISolver.spec"], cwd=ROOT, env=environment, check=True)
    app_name = "EternityIISolver.app" if sys.platform == "darwin" else "EternityIISolver"
    app = ROOT / "dist" / app_name
    executable = app / "Contents" / "MacOS" / "EternityIISolver" if sys.platform == "darwin" else app / ("EternityIISolver.exe" if sys.platform == "win32" else "EternityIISolver")
    if sys.platform.startswith("linux"):
        shadowed = [str(path.relative_to(app)) for path in app.rglob("*")
                    if any(path.name == name or path.name.startswith(name + ".")
                           for name in ("libstdc++.so.6", "libgcc_s.so.1"))]
        if shadowed:
            raise RuntimeError(f"Bundled host C++ runtimes would shadow system GPU drivers: {shadowed}")
    if not args.skip_smoke:
        with tempfile.TemporaryDirectory(prefix="eternity-frozen-smoke-") as temporary:
            reported = subprocess.check_output([str(executable), "--version"], cwd=temporary, text=True, timeout=30).strip()
            if reported != version:
                raise RuntimeError(f"Frozen CLI reports {reported!r}, expected {version!r}")
            helper_name = "eternity-radius.exe" if sys.platform == "win32" else "eternity-radius"
            helpers = {path.resolve() for path in app.rglob(helper_name) if path.parent.name == "native"}
            if len(helpers) != 1:
                raise RuntimeError("Expected one packaged radius helper")
            # One fixed tile with four gray faces, no legal changes and no work to charge.
            radius_input = "1 4 1\n" + "0 0 0 0\n" * 4 + "0\n0\n1 0\n"
            native = subprocess.run([str(helpers.pop())], input=radius_input, cwd=temporary,
                                    capture_output=True, text=True, check=True, timeout=30)
            if native.stdout.split() != ["complete"] + ["0"] * 8:
                raise RuntimeError(f"Frozen native radius helper failed its fixed-board check: {native.stdout}")
            fast_name = "eternity-fast.exe" if sys.platform == "win32" else "eternity-fast"
            fast_helpers = {path.resolve() for path in app.rglob(fast_name) if path.parent.name == "native"}
            if len(fast_helpers) != 1:
                raise RuntimeError("Expected one packaged fast-search helper")
            # One all-gray piece on a 1x1 board: solved at once, exit code 0, board [state] written.
            fast_dir = Path(temporary) / "fast-helper"
            fast_dir.mkdir()
            (fast_dir / "pieces.txt").write_text("0 0 0 0\n", encoding="ascii")
            fast = subprocess.run([str(fast_helpers.pop()), "--pieces", str(fast_dir / "pieces.txt"), "--size", "1", "--out", str(fast_dir)],
                                  stdin=subprocess.DEVNULL, cwd=temporary, capture_output=True, text=True, timeout=30)
            fast_board = json.loads((fast_dir / "best.json").read_text(encoding="utf-8")) if fast.returncode == 0 else {}
            if fast_board.get("claimed_score") != 0 or len(fast_board.get("board", ())) != 1:
                raise RuntimeError(f"Frozen fast-search helper failed its one-piece check: {fast.stdout} {fast.stderr}")
            fast_run = subprocess.run([str(executable), "exact", "--engine", "fast-dfs", "--workers", "2", "--seconds", "15",
                                       "--state-dir", str(fast_dir / "state")], cwd=temporary, capture_output=True, text=True, timeout=120)
            fast_status = json.loads((fast_dir / "state" / "runtime" / "status.json").read_text(encoding="utf-8"))
            if fast_run.returncode != 0 or fast_status.get("state") != "finished_bounded_run" or not fast_status.get("branches"):
                raise RuntimeError(f"Frozen fast-dfs run failed: {fast_run.stdout} {fast_run.stderr}")
            checked = subprocess.run([str(executable), "validate"], cwd=temporary, capture_output=True, text=True, check=True, timeout=90)
            report = json.loads(checked.stdout)
            if not report.get("valid") or report.get("score") != 466:
                raise RuntimeError(f"Frozen resource validation failed: {report}")
            repair = subprocess.run([str(executable), "repair-radius", str(ROOT / "data" / "record466.json"),
                                     "--radius", "8", "--seconds", "600"], cwd=temporary,
                                    capture_output=True, text=True, check=True, timeout=630)
            radius_report = json.loads(repair.stdout)
            if (radius_report.get("termination") != "exhausted" or radius_report.get("max_exhausted_radius") != 8
                    or radius_report.get("score") != 466 or not radius_report.get("native_phases")):
                raise RuntimeError(f"Frozen radius-eight verification failed: {radius_report.get('termination')}")
            subprocess.run([str(executable), "status", "--json", "--state-dir", temporary], cwd=temporary, check=True, timeout=30)
            exported = subprocess.run([str(executable), "export-best", "--state-dir", temporary],
                                      cwd=temporary, capture_output=True, text=True, check=True, timeout=30)
            files = json.loads(exported.stdout)
            validation = json.loads(Path(files["validation_path"]).read_text(encoding="utf-8"))
            if not files.get("exported") or files.get("score") != 466 or not validation.get("valid"):
                raise RuntimeError("Frozen manual candidate export failed validation")
            exact_check = subprocess.run([str(executable), "diagnose-exact", "--seconds", "5"],
                                         cwd=temporary, capture_output=True, text=True, check=True, timeout=60)
            if not json.loads(exact_check.stdout).get("passed"):
                raise RuntimeError("Frozen exact-engine diagnostics failed")
            subprocess.run([sys.executable, str(ROOT / "scripts" / "smoke_exact_app.py"),
                            "--executable", str(executable), "--output", str(ROOT / "build" / "exact-app-smoke.json")],
                           cwd=temporary, check=True, timeout=180)
            for command in ("diagnose", "boinc", "inspect-cpu", "export-best", "exact", "diagnose-exact", "diagnose-hybrid", "diagnose-assignment", "repair-radius", "repair-patches", "repair-parent-pair", "perfect-distance"):
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
    if sys.platform == "win32" and args.flavor == "cuda-opencl":
        check_windows_cuda_archive(archive)
    Path(str(archive) + ".sha256").write_text(sha256(archive) + "  " + archive.name + "\n", encoding="utf-8")
    print(json.dumps({"archive": str(archive), "sha256": sha256(archive), "frozen_smoke_passed": not args.skip_smoke}))

if __name__ == "__main__":
    main()
