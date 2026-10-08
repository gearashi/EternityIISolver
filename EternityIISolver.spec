# Native PyInstaller build. Run scripts/build_release.py to prepare notices/metadata.
import importlib.metadata
import os
from pathlib import Path
import platform
import sys
import tomllib
from PyInstaller.utils.hooks import collect_all, copy_metadata

root = Path(SPECPATH).resolve()
sys.path.insert(0, str(root))
from build_support import resource_files

with_cuda = os.environ.get("ETERNITY_BUILD_CUDA", "0") == "1"
if with_cuda and sys.platform == "darwin":
    raise RuntimeError("The macOS build uses OpenCL; CUDA wheels are not available for macOS")
datas = [(str(source), str(Path(relative).parent)) for relative, source in resource_files(root)]
datas += [(str(root / name), ".") for name in ("LICENSE", "README.md", "THIRD_PARTY_NOTICES.md", "DATA_PROVENANCE.md")]
metadata_dir = root / "build" / "release-metadata"
if not (metadata_dir / "BUILD_INFO.json").is_file():
    raise RuntimeError("Run python scripts/build_release.py instead of invoking the spec directly")
datas.append((str(metadata_dir), "."))
binaries = []
hiddenimports = ["launcher", "app_paths", "process_control", "dashboard_server", "solver", "gpu_engine", "gpu_backends", "kernel_port", "validator", "library_cache"]

def include_package(name):
    package_data, package_binaries, package_imports = collect_all(name, filter_submodules=lambda value: ".tests" not in value and ".testing" not in value)
    datas.extend(package_data)
    binaries.extend(package_binaries)
    hiddenimports.extend(package_imports)

include_package("pyopencl")
if with_cuda:
    include_package("cupy")
    include_package("cupy_backends")
    include_package("cuda.pathfinder")
    datas.extend(copy_metadata("cupy-cuda12x"))
    datas.extend(copy_metadata("cuda-pathfinder"))
    # Keep wheel-relative paths: CUDA Pathfinder uses distribution metadata to
    # find NVIDIA libraries and headers in their nvidia/<component>/ layout.
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name", "")
        if not name.lower().startswith("nvidia-"):
            continue
        datas.extend(copy_metadata(name))
        for relative in distribution.files or ():
            if not relative.parts or relative.parts[0] != "nvidia" or relative.suffix == ".pyc":
                continue
            source = Path(distribution.locate_file(relative))
            if not source.is_file():
                continue
            entry = (str(source), str(relative.parent))
            if source.suffix.lower() in (".dll", ".dylib", ".so") or ".so." in source.name:
                binaries.append(entry)
            else:
                datas.append(entry)

analysis = Analysis([str(root / "launcher.py")], pathex=[str(root)], binaries=binaries,
    datas=datas, hiddenimports=sorted(set(hiddenimports)),
    runtime_hooks=[str(root / "scripts" / "frozen_runtime.py")],
    excludes=[] if with_cuda else ["cupy", "cupy_backends"], noarchive=False)
archive = PYZ(analysis.pure)
executable = EXE(archive, analysis.scripts, [], exclude_binaries=True,
    name="EternityIISolver", debug=False, bootloader_ignore_signals=False,
    strip=False, upx=False, console=True,
    target_arch=platform.machine() if sys.platform == "darwin" else None)
collection = COLLECT(executable, analysis.binaries, analysis.datas,
    strip=False, upx=False, name="EternityIISolver")
if sys.platform == "darwin":
    application = BUNDLE(collection, name="EternityIISolver.app",
        bundle_identifier="org.gearashi.eternityiisolver",
        info_plist={"CFBundleName": "Eternity II Solver", "CFBundleShortVersionString": tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"],
                    "NSHighResolutionCapable": True})
