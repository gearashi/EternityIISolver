"""Read-only peak process memory accounting; never sends process signals."""
from functools import lru_cache
import os
import sys

@lru_cache(maxsize=1)
def _windows_reader():
    # ctypes caches pointer types. Define this structure once, not per sample.
    import ctypes
    from ctypes import wintypes
    class Counters(ctypes.Structure):
        _fields_ = [('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)] + [(k,ctypes.c_size_t) for k in ('PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage')]
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.GetCurrentProcess.restype=wintypes.HANDLE
    psapi=ctypes.WinDLL('psapi',use_last_error=True)
    psapi.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.POINTER(Counters),wintypes.DWORD]
    psapi.GetProcessMemoryInfo.restype=wintypes.BOOL
    return ctypes, kernel, psapi, Counters

def peak_memory_mb():
    if os.name != 'nt':
        import resource
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return value / (1024*1024 if sys.platform == 'darwin' else 1024)
    ctypes,kernel,psapi,Counters=_windows_reader()
    counters=Counters();counters.cb=ctypes.sizeof(counters)
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(),ctypes.byref(counters),counters.cb):
        raise OSError(ctypes.get_last_error(),'Cannot read process memory')
    return max(counters.PeakWorkingSetSize,counters.PeakPagefileUsage)/(1024*1024)
