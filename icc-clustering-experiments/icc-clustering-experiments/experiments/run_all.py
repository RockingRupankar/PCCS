"""Run all five experiments in order. Each experiment also has its own run_all.py.
Total ~25-40 minutes on one CPU core."""
import subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
for e in ["exp0_falsification", "exp1_generalization", "exp2_ablation",
          "exp3_real_world_benchmarks", "exp4_scalability_and_sensitivity"]:
    print("#" * 20, e, flush=True)
    subprocess.run([sys.executable, str(ROOT / e / "run_all.py")], check=True)
