"""Prepare exp2 data: noisy and severe PCCS-style datasets, N=1200, three generator seeds."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common.datasets import make_pccs

OUT = ROOT / "exp2_ablation" / "data"
for prof in ("noisy", "severe"):
    for seed in (42, 43, 44):
        X, y, names, df = make_pccs(prof, 1200, seed=seed)
        df.to_csv(OUT / f"exp2_{prof}_seed{seed}.csv", index=False, float_format="%.4f")
        print(prof, seed, df.shape)
