"""Run exp4 end to end (about 8-12 min on one CPU)."""
import subprocess, sys
from pathlib import Path
SRC = Path(__file__).resolve().parent / "src"
for s in ["exp4_make_data.py", "exp4_scalability.py", "exp4_sensitivity.py", "plot_all_figures.py",
          "exp4_pccs_response.py", "plot_pccs_figures.py"]:
    print("==>", s, flush=True)
    subprocess.run([sys.executable, str(SRC / s)], check=True, cwd=SRC)
