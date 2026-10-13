#!/usr/bin/env python
"""
Standalone runner for Experiment 2: Perturbation-Family and Intensity Sweeps.
Usage:
    python experiments/experiment_02_perturbation_sweeps/run.py
"""
import sys
import os

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
src_dir = os.path.join(root_dir, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from pccs.experiment_runner import run_experiment_02_perturbation_sweeps

if __name__ == "__main__":
    results_dir = os.path.join(root_dir, "results", "raw")
    print(f"Running Experiment 2 (outputs -> {results_dir})...")
    df = run_experiment_02_perturbation_sweeps(results_dir=results_dir)
    print(f"Done. Processed {len(df)} sweep evaluation records.")
