#!/usr/bin/env python
"""
Standalone runner for Experiment 4: Factorial Perturbation Interaction Analysis.
Usage:
    python experiments/experiment_04_interaction_analysis/run.py
"""
import sys
import os

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
src_dir = os.path.join(root_dir, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from pccs.experiment_runner import run_experiment_04_interaction_analysis

if __name__ == "__main__":
    results_dir = os.path.join(root_dir, "results", "raw")
    print(f"Running Experiment 4 (outputs -> {results_dir})...")
    df = run_experiment_04_interaction_analysis(results_dir=results_dir)
    print(f"Done. Processed {len(df)} interaction analysis records.")
