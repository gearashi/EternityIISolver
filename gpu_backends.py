"""Lazy GPU runtimes. OpenCL CPU devices are available only to explicit diagnostics.

Search and score/delta evaluation execute on the selected device. The small
array adapter uses host transfers for indexing/checkpoints, never CPU search.
"""
from __future__ import annotations
import os
from pathlib import Path
import sys
import numpy as np

from app_paths import resource_root
ROOT = resource_root()


class BackendUnavailable(RuntimeError):
    pass


def default_cache_dir() -> Path:
    if sys.platform == 'win32':
        base = Path(os.environ.get('LOCALAPPDATA', Path.home() / 'AppData/Local'))
    elif sys.platform == 'darwin':
        base = Path.home() / 'Library/Caches'
    else:
        base = Path(os.environ.get('XDG_CACHE_HOME', Path.home() / '.cache'))
    return base / 'EternityIISolver' / 'kernels'


class CudaBackend:
    name = 'cuda'
    diagnostic_cpu = False

    def __init__(self, cache_dir: Path):
        directory = cache_dir / 'cuda'
        directory.mkdir(parents=True, exist_ok=True)
        os.environ['CUPY_CACHE_DIR'] = str(directory)
        try:
            import cupy as cp
            if cp.cuda.runtime.getDeviceCount() < 1:
                raise BackendUnavailable('CUDA did not report a GPU')
            self.xp = cp
            self.module = cp.RawModule(code=(ROOT / 'kernels.cu').read_text(encoding='utf-8'),
                                       options=('--std=c++17',),
                                       name_expressions=('search_moves', 'test_delta', 'score_boards'))
            self.search_kernel = self.module.get_function('search_moves')
            self.delta_kernel = self.module.get_function('test_delta')
            self.score_kernel = self.module.get_function('score_boards')
            device_id = cp.cuda.runtime.getDevice()
            name = cp.cuda.runtime.getDeviceProperties(device_id)['name']
            self.device = name.decode() if isinstance(name, bytes) else str(name)
            self.details = {'backend': self.name, 'device': self.device, 'cupy': cp.__version__,
                            'driver_cuda_version': cp.cuda.runtime.driverGetVersion(), 'device_type': 'GPU'}
        except Exception as exc:
            raise BackendUnavailable(f'CUDA unavailable: {exc}. Install the optional CUDA dependencies and a compatible NVIDIA driver.') from exc

    def timed(self, kernel, grid, block, args):
        cp = self.xp
        start, end = cp.cuda.Event(), cp.cuda.Event()
        start.record()
        kernel(grid, block, args)
        end.record()
        end.synchronize()
        return cp.cuda.get_elapsed_time(start, end)


class HostResult:
    """A downloaded indexing/reduction result with the shared .get() contract."""
    def __init__(self, value):
        self.value = np.asarray(value)

    def get(self):
        return self.value.copy()

    def sum(self):
        return HostResult(self.value.sum())

    @property
    def shape(self):
        return self.value.shape

    @property
    def dtype(self):
        return self.value.dtype


class OpenCLArray:
    """Contiguous device buffer with explicit host access for orchestration."""
    def __init__(self, backend, shape, dtype, value=None):
        self.backend = backend
        self.shape = (shape,) if isinstance(shape, int) else tuple(shape)
        self.dtype = np.dtype(dtype)
        self.nbytes = int(np.prod(self.shape)) * self.dtype.itemsize
        self.buffer = backend.cl.Buffer(backend.context, backend.cl.mem_flags.READ_WRITE,
                                        size=max(1, self.nbytes))
        if value is not None:
            self.set(value)

    def get(self):
        result = np.empty(self.shape, self.dtype)
        if self.nbytes:
            self.backend.cl.enqueue_copy(self.backend.queue, result, self.buffer, is_blocking=True)
        return result

    def set(self, value):
        if hasattr(value, 'get'):
            value = value.get()
        array = np.asarray(value, dtype=self.dtype)
        if array.shape != self.shape:
            raise ValueError(f'Array shape {array.shape} does not match device buffer {self.shape}')
        array = np.ascontiguousarray(array)
        if self.nbytes:
            self.backend.cl.enqueue_copy(self.backend.queue, self.buffer, array, is_blocking=True)

    def fill(self, value):
        if self.nbytes:
            self.backend.cl.enqueue_fill_buffer(self.backend.queue, self.buffer,
                                                 np.asarray(value, dtype=self.dtype), 0, self.nbytes).wait()

    def copy(self):
        result = OpenCLArray(self.backend, self.shape, self.dtype)
        if self.nbytes:
            self.backend.cl.enqueue_copy(self.backend.queue, result.buffer, self.buffer, byte_count=self.nbytes).wait()
        return result

    def __getitem__(self, key):
        return HostResult(self.get()[key])

    def __setitem__(self, key, value):
        host = self.get()
        host[key] = value.get() if hasattr(value, 'get') else value
        self.set(host)

    def sum(self):
        return HostResult(self.get().sum())


class OpenCLArrays:
    def __init__(self, backend):
        self.backend = backend

    def asarray(self, value, dtype=None):
        if hasattr(value, 'get'):
            value = value.get()
        value = np.asarray(value, dtype=dtype)
        return OpenCLArray(self.backend, value.shape, value.dtype, value)

    def empty(self, shape, dtype):
        return OpenCLArray(self.backend, shape, dtype)

    def zeros(self, shape, dtype):
        value = self.empty(shape, dtype)
        value.fill(0)
        return value

    def ones(self, shape, dtype):
        value = self.empty(shape, dtype)
        value.fill(1)
        return value


class OpenCLKernel:
    def __init__(self, backend, name):
        self.backend = backend
        self.kernel = backend.cl.Kernel(backend.module, name)

    def __call__(self, grid, block, args):
        # No shared local memory or barriers: any local size preserves semantics.
        global_size = int(grid[0]) * int(block[0])
        limit = self.kernel.get_work_group_info(self.backend.cl.kernel_work_group_info.WORK_GROUP_SIZE,
                                                self.backend.device_object)
        requested = min(int(block[0]), int(limit), 128)
        local_size = requested if global_size % requested == 0 else None
        converted = tuple(value.buffer if isinstance(value, OpenCLArray) else value for value in args)
        return self.kernel(self.backend.queue, (global_size,),
                           (local_size,) if local_size is not None else None, *converted)


class OpenCLBackend:
    name = 'opencl'

    def __init__(self, cache_dir: Path, *, allow_cpu=False, device_name=None):
        directory = cache_dir / 'opencl'
        directory.mkdir(parents=True, exist_ok=True)
        try:
            import pyopencl as cl
            from kernel_port import verified_opencl_source
            self.cl = cl
            platform_filter = os.environ.get('ETERNITY_OPENCL_PLATFORM', '').casefold()
            platforms = [platform for platform in cl.get_platforms() if platform_filter in platform.name.casefold()]
            devices = [device for platform in platforms for device in platform.get_devices()]
            device_name = device_name or os.environ.get('ETERNITY_OPENCL_DEVICE')
            kind = cl.device_type.CPU if allow_cpu else cl.device_type.GPU
            devices = [device for device in devices if device.type & kind]
            if device_name:
                devices = [device for device in devices if device_name.casefold() in device.name.casefold()]
            if not devices:
                label = 'CPU diagnostic device' if allow_cpu else 'GPU'
                raise BackendUnavailable(f'No OpenCL {label} found. Install an OpenCL-capable GPU driver; CPU fallback is disabled.')
            self.device_object = max(devices, key=lambda device: (device.max_compute_units, device.global_mem_size))
            self.diagnostic_cpu = bool(allow_cpu)
            self.context = cl.Context([self.device_object])
            self.queue = cl.CommandQueue(self.context, properties=cl.command_queue_properties.PROFILING_ENABLE)
            self.module = cl.Program(self.context, verified_opencl_source(ROOT)).build(
                options=['-cl-std=CL1.2'], cache_dir=str(directory))
            self.xp = OpenCLArrays(self)
            self.search_kernel = OpenCLKernel(self, 'search_moves')
            self.delta_kernel = OpenCLKernel(self, 'test_delta')
            self.score_kernel = OpenCLKernel(self, 'score_boards')
            self.device = self.device_object.name.strip()
            self.details = {'backend': self.name, 'device': self.device, 'pyopencl': cl.VERSION_TEXT,
                            'platform': self.device_object.platform.name,
                            'opencl_version': self.device_object.version,
                            'device_type': 'CPU diagnostic' if allow_cpu else 'GPU'}
        except BackendUnavailable:
            raise
        except Exception as exc:
            raise BackendUnavailable(f'OpenCL unavailable: {exc}. Install pyopencl and an OpenCL-capable GPU driver. On macOS use the system OpenCL GPU runtime.') from exc

    def timed(self, kernel, grid, block, args):
        event = kernel(grid, block, args)
        event.wait()
        return (event.profile.end - event.profile.start) / 1_000_000.0


def make_backend(name='auto', cache_dir=None, *, allow_opencl_cpu=False, device_name=None):
    if name not in ('auto', 'cuda', 'opencl'):
        raise ValueError('backend must be auto, cuda, or opencl')
    if allow_opencl_cpu and name != 'opencl':
        raise ValueError('CPU OpenCL diagnostics require explicit backend=opencl')
    directory = Path(cache_dir).expanduser() if cache_dir is not None else default_cache_dir()
    order = ('opencl',) if sys.platform == 'darwin' else ('cuda', 'opencl')
    if name != 'auto':
        order = (name,)
    errors = []
    for candidate in order:
        try:
            if candidate == 'cuda':
                return CudaBackend(directory)
            return OpenCLBackend(directory, allow_cpu=allow_opencl_cpu, device_name=device_name)
        except (BackendUnavailable, OSError) as exc:
            errors.append(str(exc))
    raise BackendUnavailable('No usable GPU backend. ' + ' | '.join(errors))
