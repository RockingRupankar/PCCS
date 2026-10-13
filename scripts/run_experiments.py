#!/usr/bin/env python
"""
Main executable script to run all 5 PCCS experiments.
Usage:
    python scripts/run_experiments.py
"""
import sys
import os

# Ensure src is on python path
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from pccs.experiment_runner import run_all_pccs_experiments

if __name__ == "__main__":
    run_all_pccs_experiments()
