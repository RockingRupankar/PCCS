#!/usr/bin/env python3
"""Write a PCCS-style synthetic dataset (CSV) with hidden ground truth.

Example:  python scripts/generate_synthetic_data.py --profile noisy --n 3000 --seed 42 --out data/synthetic_noisy.csv
Profiles: clean | noisy | severe. Columns: sample_id, true_cluster, observed_cluster (blank), noise_flag,
outlier_flag, missing_value_count, feature_001 ... feature_100.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from common.datasets import make_pccs  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", choices=["clean", "noisy", "severe"], default="noisy")
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="synthetic.csv")
    a = ap.parse_args()
    _, _, _, df = make_pccs(a.profile, a.n, a.seed)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False, float_format="%.4f")
    print(f"wrote {a.out}: {df.shape}")


if __name__ == "__main__":
    main()
