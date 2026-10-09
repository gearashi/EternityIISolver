"""Run the same bounded sampling diagnostic exposed by diagnose-hybrid."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hybrid_sampling import main

if __name__ == '__main__':
    raise SystemExit(main())
