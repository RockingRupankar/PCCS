"""Run exp0 end to end (about 5-10 min on one CPU)."""
import subprocess, sys
from pathlib import Path
SRC = Path(__file__).resolve().parent / "src"
for s in ["exp0_make_data.py", "exp0_run.py", "exp0_finegrid.py", "exp0_noisetypes.py",
          "exp0_scaled_run.py", "exp0_diagnose.py", "plot_all_figures.py",
          "exp0_pccs_fingerprint.py", "plot_pccs_figures.py"]:
    print("==>", s, flush=True)
    subprocess.run([sys.executable, str(SRC / s)], check=True, cwd=SRC)
