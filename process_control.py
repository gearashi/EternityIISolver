"""Portable single-instance locking and read-only process checks."""
import json
import os
from pathlib import Path


def process_alive(pid):
    if not isinstance(pid, int) or isinstance(pid, bool) or not 0 < pid <= 0x7FFFFFFF:
        return False
    if os.name != 'nt':
        try:
            os.kill(pid, 0)  # POSIX existence check; never used on Windows.
        except (ProcessLookupError, OverflowError):
            return False
        except PermissionError:
            return True
        return True
    import ctypes
    from ctypes import wintypes
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    kernel.GetExitCodeProcess.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.OpenProcess(0x1000, False, pid)
    if not handle:
        return ctypes.get_last_error() != 87
    try:
        code = wintypes.DWORD()
        return not kernel.GetExitCodeProcess(handle, ctypes.byref(code)) or code.value == 259
    finally:
        kernel.CloseHandle(handle)


class RunLock:
    """Hold an OS lock for the process lifetime; crashes release it automatically.

    The lock file remains on disk. Unlinking a lock file permits concurrent
    instances to lock different inodes on POSIX, so it must not be removed.
    """
    def __init__(self, path):
        self.path = Path(path)
        self.file = None

    def acquire(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        stream = self.path.open('a+b')
        try:
            stream.seek(0)
            if os.name == 'nt':
                import msvcrt
                if self.path.stat().st_size == 0:
                    stream.write(b' ')
                    stream.flush()
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            stream.close()
            raise RuntimeError('A solver already owns this state folder. Use stop first.') from exc
        self.file = stream
        return self

    def close(self):
        if self.file:
            self.file.close()
            self.file = None

    def __enter__(self):
        return self.acquire()

    def __exit__(self, *_):
        self.close()


def read_status(root):
    try:
        value = json.loads((Path(root) / 'runtime' / 'status.json').read_text(encoding='utf-8'))
        value['process_alive'] = process_alive(value.get('pid'))
        return value
    except FileNotFoundError:
        return {'state': 'not_started', 'process_alive': False}
