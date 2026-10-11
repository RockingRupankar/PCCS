"""Run exp2 end to end (about 5-8 min on one CPU)."""
import subprocess, sys
from pathlib import Path
SRC = Path(__file__).resolve().parent / "src"
for s in ["exp2_make_data.py", "exp2_run.py", "plot_all_figures.py", "exp2_pccs_ablation.py", "plot_pccs_figures.py"]:
    print("==>", s, flush=True)
    subprocess.run([sys.executable, str(SRC / s)], check=True, cwd=SRC)
