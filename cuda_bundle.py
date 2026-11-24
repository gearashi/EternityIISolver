"""Windows CUDA wheel paths used by the frozen solver and its archive check."""
from fnmatch import fnmatchcase
from pathlib import Path
import csv
import io
import zipfile


# The component wheels have separate roots; flattening their DLLs breaks
# Pathfinder discovery and can hide missing files on toolkit-equipped hosts.
CUDA_FILES = (
    "cupy/_core/core.*.pyd",
    "cupy_backends/cuda/api/runtime.*.pyd",
    "cupy_backends/cuda/libs/nvrtc.*.pyd",
    "cupy/_core/include/cupy/complex.cuh",
    "cuda/pathfinder/__init__.py",
    "nvidia/cuda_runtime/bin/cudart64_12.dll",
    "nvidia/cuda_runtime/include/cuda_fp16.h",
    "nvidia/cuda_nvrtc/bin/nvrtc64_120_0.dll",
    "nvidia/cuda_nvrtc/bin/nvrtc-builtins64_129.dll",
    "nvidia/cuda_nvrtc/include/nvrtc.h",
)
DISTRIBUTIONS = ("cupy_cuda12x", "cuda_pathfinder", "nvidia_cuda_runtime_cu12", "nvidia_cuda_nvrtc_cu12")
SEARCH_KERNELS = ("kernels.cu", "fast_score.cu")


def missing_windows_cuda_file(root):
    """Cheap preflight before loading DLLs or falling back to a host toolkit."""
    for pattern in CUDA_FILES:
        if not any(p.is_file() and p.stat().st_size for p in Path(root).glob(pattern)):
            return pattern
    return None


def check_windows_cuda_archive(path):
    """Verify the archive's own wheel records and runtime files on a GPU-less builder."""
    prefix = "EternityIISolver/_internal/"
    with zipfile.ZipFile(path) as archive:
        files = {entry.filename.removeprefix(prefix): entry for entry in archive.infolist()
                 if entry.filename.startswith(prefix) and not entry.is_dir()}
        missing = [p for p in CUDA_FILES + SEARCH_KERNELS
                   if not any(fnmatchcase(n, p) and f.file_size for n, f in files.items())]
        for dist in DISTRIBUTIONS:
            records = [n for n in files if fnmatchcase(n, dist + "-*.dist-info/RECORD")]
            if len(records) != 1:
                raise ValueError(f"Expected one wheel RECORD for {dist} in the Windows CUDA archive")
            record = records[0]
            metadata = record.removesuffix("RECORD") + "METADATA"
            if metadata not in files or not files[metadata].file_size:
                missing.append(metadata)
            with archive.open(files[record]) as raw:
                for name, _, size in csv.reader(io.TextIOWrapper(raw, encoding="utf-8", newline="")):
                    # Check all shipped component headers/DLLs and CuPy's
                    # native modules, including extensionless C++ headers.
                    package = name.startswith(("cupy/", "cupy_backends/", "nvidia/"))
                    if not package or not (name.endswith((".pyd", ".dll")) or "/include/" in name):
                        continue
                    if name not in files or (size and files[name].file_size != int(size)):
                        missing.append(name)
        if missing:
            raise ValueError("Windows CUDA archive has missing or incomplete files: " + ", ".join(sorted(set(missing))))
