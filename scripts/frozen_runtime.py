"""Register packaged Windows DLL directories without loading or starting a GPU."""
import os
from pathlib import Path
import sys

if sys.platform == "win32" and getattr(sys, "frozen", False):
    root = Path(sys._MEIPASS)
    candidates = {root}
    for directory in (root / "nvidia", root / "cupy", root / "cupy_backends", root / "pyopencl"):
        if directory.is_dir():
            candidates.update(path.parent for path in directory.rglob("*.dll"))
    handles = []
    for directory in sorted(candidates):
        if directory.is_dir():
            handles.append(os.add_dll_directory(str(directory)))
    # Windows closes search-directory registrations when these handles die.
    sys._eternity_dll_directories = handles
