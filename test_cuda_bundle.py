"""Exercise the Windows release archive gate without CUDA, a GPU, or a download."""
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from cuda_bundle import CUDA_FILES, DISTRIBUTIONS, SEARCH_KERNELS, check_windows_cuda_archive, missing_windows_cuda_file


def archive_fixture(omit=None, move=None):
    data = io.BytesIO()
    recorded_header = "nvidia/cuda_runtime/include/vector_types.h"
    files = {p.replace(".*.pyd", ".cp312-win_amd64.pyd"): b"x" for p in CUDA_FILES}
    files.update({name: b"// sample kernel" for name in SEARCH_KERNELS})
    files[recorded_header] = b"header"
    for dist in DISTRIBUTIONS:
        metadata = dist + "-1.dist-info/"
        files[metadata + "METADATA"] = b"Metadata-Version: 2.1\n"
        files[metadata + "RECORD"] = ((f"{recorded_header},,6\n" if dist == "nvidia_cuda_runtime_cu12"
                                      else f"{metadata}RECORD,,\n").encode())
    with zipfile.ZipFile(data, "w") as archive:
        for path, content in files.items():
            if path == omit:
                continue
            if path == move:
                path = path.rsplit("/", 1)[-1]
            archive.writestr("EternityIISolver/_internal/" + path, content)
    return data


class WindowsCudaArchiveTests(unittest.TestCase):
    def test_complete_archive(self):
        check_windows_cuda_archive(archive_fixture())

    def test_missing_application_kernel_fails(self):
        for path in ("kernels.cu", "fast_score.cu"):
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, path):
                check_windows_cuda_archive(archive_fixture(omit=path))

    def test_extracted_metadata_failure_names_exact_file_and_remedy(self):
        from gpu_backends import BackendUnavailable, _check_windows_cuda_setup

        for missing in ('nvidia_cuda_nvrtc_cu12-1.dist-info/RECORD', 'cupy_cuda12x-1.dist-info/METADATA'):
            with self.subTest(missing=missing), tempfile.TemporaryDirectory() as work:
                with zipfile.ZipFile(archive_fixture(omit=missing)) as archive:
                    archive.extractall(work)
                root = Path(work) / 'EternityIISolver/_internal'
                with patch.object(sys, 'frozen', True, create=True), patch.object(sys, '_MEIPASS', str(root), create=True):
                    with self.assertRaises(BackendUnavailable) as error:
                        _check_windows_cuda_setup('kernels.cu')
                self.assertIn('_internal/' + missing, str(error.exception))
                self.assertIn('Extract the complete Windows x64 cuda-opencl zip', str(error.exception))

    def test_each_worker_requires_only_its_own_cuda_kernel(self):
        from cuda_score_search import run_ladder
        from gpu_backends import BackendUnavailable, CudaBackend

        for missing, other in (('kernels.cu', 'fast_score.cu'), ('fast_score.cu', 'kernels.cu')):
            with self.subTest(missing=missing), tempfile.TemporaryDirectory() as work:
                with zipfile.ZipFile(archive_fixture(omit=missing)) as archive:
                    archive.extractall(work)
                root = Path(work) / 'EternityIISolver/_internal'
                self.assertIsNone(missing_windows_cuda_file(root, other))
                with patch.object(sys, 'platform', 'win32'), patch.object(sys, 'frozen', True, create=True), \
                     patch.object(sys, '_MEIPASS', str(root), create=True):
                    with self.assertRaises(BackendUnavailable) as error:
                        if missing == 'kernels.cu':
                            CudaBackend(Path(work))
                        else:
                            run_ladder(None, {}, 1, 1, 0, 10, Path(work), None)
                self.assertIn('_internal/' + missing, str(error.exception))
                self.assertIn('Extract the complete Windows x64 cuda-opencl zip', str(error.exception))

    def test_missing_or_flattened_component_fails(self):
        for path in ("cupy/_core/core.cp312-win_amd64.pyd", "cuda/pathfinder/__init__.py",
                     "nvidia/cuda_runtime/bin/cudart64_12.dll",
                     "nvidia/cuda_nvrtc/bin/nvrtc64_120_0.dll",
                     "nvidia/cuda_nvrtc/bin/nvrtc-builtins64_129.dll",
                     "nvidia/cuda_runtime/include/vector_types.h",
                     "nvidia_cuda_nvrtc_cu12-1.dist-info/RECORD",
                     "nvidia_cuda_runtime_cu12-1.dist-info/METADATA"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                check_windows_cuda_archive(archive_fixture(omit=path))
        with self.assertRaisesRegex(ValueError, 'cudart64_12.dll'):
            check_windows_cuda_archive(archive_fixture(move="nvidia/cuda_runtime/bin/cudart64_12.dll"))
