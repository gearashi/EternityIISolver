"""Read-only application resources and writable per-user search state."""
import os
from pathlib import Path
import sys


def resource_root():
    if getattr(sys, 'frozen', False):
        return Path(sys._MEIPASS)
    root = Path(__file__).resolve().parent
    if (root / 'data' / 'puzzle.json').is_file():
        return root
    import eternity_resources
    return Path(eternity_resources.__file__).resolve().parent


def state_root(override=None):
    override = override or os.environ.get('ETERNITY_SOLVER_HOME')
    if override:
        return Path(override).expanduser().resolve()
    if sys.platform == 'win32':
        return Path(os.environ.get('LOCALAPPDATA', Path.home() / 'AppData' / 'Local')) / 'EternityIISolver'
    if sys.platform == 'darwin':
        return Path.home() / 'Library' / 'Application Support' / 'EternityIISolver'
    return Path(os.environ.get('XDG_DATA_HOME', Path.home() / '.local' / 'share')) / 'eternity-ii-solver'
