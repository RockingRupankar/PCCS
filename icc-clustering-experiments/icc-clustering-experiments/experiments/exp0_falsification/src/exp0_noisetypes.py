"""Noise-type sweep: does the stability gap survive heavy tails, outliers, imputation, shuffling?"""
import pandas as pd
from exp0_core import *
levels = [0.5, 1.0, 2.0]
rows = [run_cell(s, nt, lv, sd) for s in STRUCTURES for nt in NOISE_TYPES for lv in levels for sd in range(10)]
df = pd.DataFrame(rows)
df.to_csv(ROOT / "exp0_falsification" / "data" / "exp0_noisetypes.csv", index=False)
p = df.groupby(["noise_type", "level", "structure"]).stab_boot.mean().unstack()
p["gap"] = p["structured"] - p[["null_iso", "null_aniso"]].max(axis=1)
print(p.round(3).to_string())
