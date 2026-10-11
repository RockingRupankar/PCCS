#!/usr/bin/env python3
"""Run one experiment suite:  python scripts/run_experiment.py exp2   (or --list)."""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUITES = {
    "exp0": "exp0_falsification",
    "exp1": "exp1_generalization",
    "exp2": "exp2_ablation",
    "exp3": "exp3_real_world_benchmarks",
    "exp4": "exp4_scalability_and_sensitivity",
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("experiment", nargs="?", choices=sorted(SUITES), help="experiment id")
    ap.add_argument("--list", action="store_true", help="list available experiments")
    args = ap.parse_args()
    if args.list or not args.experiment:
        for k, v in SUITES.items():
            print(f"{k}\t{v}")
        return 0
    runner = ROOT / "experiments" / SUITES[args.experiment] / "run_all.py"
    return subprocess.run([sys.executable, str(runner)], check=False).returncode


if __name__ == "__main__":
    sys.exit(main())
