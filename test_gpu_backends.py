"""Backend-selection and mechanically shared-kernel checks; no GPU needed."""
import ast
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
from gpu_backends import BackendUnavailable, make_backend
from kernel_port import to_opencl, verified_opencl_source

ROOT = Path(__file__).resolve().parent


class BackendSelectionTests(unittest.TestCase):
    def test_import_does_not_require_gpu_packages(self):
        import gpu_engine
        self.assertTrue(callable(gpu_engine.create_engine))
        for name in ('cupy', 'pyopencl'):
            self.assertNotIn(name, sys.modules)

    def test_auto_windows_and_linux_try_cuda_then_opencl(self):
        for platform in ('win32', 'linux'):
            with self.subTest(platform=platform), patch('gpu_backends.sys.platform', platform), \
                    patch('gpu_backends.CudaBackend', side_effect=BackendUnavailable('unavailable')) as cuda, \
                    patch('gpu_backends.OpenCLBackend', return_value='gpu') as opencl:
                self.assertEqual(make_backend(), 'gpu')
                cuda.assert_called_once()
                opencl.assert_called_once()
                self.assertFalse(opencl.call_args.kwargs['allow_cpu'])

    def test_mac_auto_uses_opencl_gpu(self):
        with patch('gpu_backends.sys.platform', 'darwin'), \
                patch('gpu_backends.CudaBackend') as cuda, \
                patch('gpu_backends.OpenCLBackend', return_value='apple gpu') as opencl:
            self.assertEqual(make_backend(), 'apple gpu')
            cuda.assert_not_called()
            self.assertFalse(opencl.call_args.kwargs['allow_cpu'])

    def test_cpu_cannot_be_automatic_fallback(self):
        with self.assertRaises(ValueError):
            make_backend('auto', allow_opencl_cpu=True)
        with patch('gpu_backends.OpenCLBackend', return_value='diagnostic') as opencl:
            self.assertEqual(make_backend('opencl', allow_opencl_cpu=True), 'diagnostic')
            self.assertTrue(opencl.call_args.kwargs['allow_cpu'])

    def test_explicit_cuda_never_silently_selects_cpu_or_opencl(self):
        with patch('gpu_backends.CudaBackend', side_effect=BackendUnavailable('missing driver')), \
                patch('gpu_backends.OpenCLBackend') as opencl:
            with self.assertRaisesRegex(BackendUnavailable, 'missing driver'):
                make_backend('cuda')
            opencl.assert_not_called()

    def test_unknown_backend_rejected(self):
        with self.assertRaises(ValueError):
            make_backend('bogus')

    def test_mechanical_kernel_matches(self):
        source = verified_opencl_source(ROOT)
        self.assertEqual(source.count('__kernel void '), 3)
        self.assertIn('__private unsigned int *state', source)
        self.assertIn('__global ulong *counters', source)
        self.assertNotIn('atomic_', source)
        self.assertNotIn('unsigned int &state', source)

    def test_translation_does_not_accept_old_reference_rng(self):
        with self.assertRaisesRegex(ValueError, 'pointer convention'):
            to_opencl('unsigned int next_u32(unsigned int &state) {}')

    def test_shared_python_files_parse(self):
        for name in ('gpu_engine.py', 'gpu_backends.py', 'kernel_port.py', 'test_gpu.py'):
            ast.parse((ROOT / name).read_text(encoding='utf-8-sig'), filename=name)


if __name__ == '__main__':
    unittest.main()
