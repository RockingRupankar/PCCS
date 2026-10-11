"""Run exp3 end to end (about 3-6 min on one CPU)."""
import subprocess, sys
from pathlib import Path
SRC = Path(__file__).resolve().parent / "src"
for s in ["exp3_make_data.py", "exp3_run.py", "plot_all_figures.py", "exp3_pccs.py", "plot_pccs_figures.py"]:
    print("==>", s, flush=True)
    subprocess.run([sys.executable, str(SRC / s)], check=True, cwd=SRC)
