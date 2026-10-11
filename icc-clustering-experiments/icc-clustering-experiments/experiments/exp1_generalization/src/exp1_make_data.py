"""Prepare exp1 data: three PCCS-style profiles (clean / noisy / severe), N=2000, with a FIXED stratified test split (800 rows)."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from common.datasets import make_pccs

OUT = ROOT / "exp1_generalization" / "data"
for prof in ("clean", "noisy", "severe"):
    X, y, names, df = make_pccs(prof, 2000, seed=42)
    tr, te = train_test_split(np.arange(len(y)), test_size=800, stratify=y, random_state=0)
    split = np.array(["pool"] * len(y), dtype=object)
    split[te] = "test"
    out = df[["sample_id", "true_cluster"] + names].copy()
    out.insert(2, "split", split)
    out.to_csv(OUT / f"exp1_{prof}.csv", index=False, float_format="%.4f")
    print(prof, out.shape, (split == "test").sum())
