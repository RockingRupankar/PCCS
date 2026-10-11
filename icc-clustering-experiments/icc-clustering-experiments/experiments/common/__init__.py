"""Shared helpers for the experiments (baselines, data generators, metrics, ICC adapter)."""
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "src"      # makes `import icc` work without installation
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
