"""Prepare exp4 data. Sensitivity study uses PCCS 'noisy' (N=800, 3 generator seeds).
Scalability data (Gaussian mixture, variable N and d) is generated on the fly in exp4_scalability.py;
one example (N=1000, d=100) is written for inspection."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
from common.datasets import make_pccs, make_scalable

OUT = ROOT / "exp4_scalability_and_sensitivity" / "data"
for seed in (42, 43, 44):
    _, _, _, df = make_pccs("noisy", 800, seed=seed)
    df.to_csv(OUT / f"exp4_sensitivity_noisy_seed{seed}.csv", index=False, float_format="%.4f")
X, y = make_scalable(1000, 100, seed=0)
pd.DataFrame(X, columns=[f"x{i + 1:03d}" for i in range(100)]).assign(true_cluster=y).to_csv(
    OUT / "exp4_scalable_example_N1000_d100.csv", index=False, float_format="%.4f")
print("done")
