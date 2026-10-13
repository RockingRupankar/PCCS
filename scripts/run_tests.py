#!/usr/bin/env python
"""
Executable script to run pytest automated test suites for PCCS.
Usage:
    python scripts/run_tests.py
"""
import sys
import os
import subprocess

if __name__ == "__main__":
    src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
    env = os.environ.copy()
    env["PYTHONPATH"] = src_dir + os.pathsep + env.get("PYTHONPATH", "")

    cmd = [sys.executable, "-m", "pytest", "tests/"]
    print("Running PCCS automated test suite:", " ".join(cmd))
    res = subprocess.run(cmd, env=env)
    sys.exit(res.returncode)
