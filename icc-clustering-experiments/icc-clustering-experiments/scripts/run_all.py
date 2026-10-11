#!/usr/bin/env python3
"""Run all five experiment suites in order (about 35 minutes on one CPU core), then summarise the results."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for exp in ("exp0", "exp1", "exp2", "exp3", "exp4"):
    print("=" * 20, exp, flush=True)
    subprocess.run([sys.executable, str(ROOT / "scripts" / "run_experiment.py"), exp], check=True)
subprocess.run([sys.executable, str(ROOT / "scripts" / "summarize_results.py")], check=True)
