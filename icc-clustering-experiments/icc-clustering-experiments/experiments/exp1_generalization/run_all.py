"""Run exp1 end to end (about 3-5 min on one CPU)."""
import subprocess, sys
from pathlib import Path
SRC = Path(__file__).resolve().parent / "src"
for s in ["exp1_make_data.py", "exp1_run.py", "plot_all_figures.py"]:
    print("==>", s, flush=True)
    subprocess.run([sys.executable, str(SRC / s)], check=True, cwd=SRC)
