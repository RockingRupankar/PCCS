#!/usr/bin/env python3
"""Minimal end-to-end demonstration of ICC and PCCS (about 15 seconds).

Generates noisy high-dimensional data with hidden labels, clusters it with ICC and with k-means, labels held-out rows with
the inductive rule, and then runs PCCS to print the per-family stability fingerprint, reliability categories and interactions.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

import numpy as np  # noqa: E402
from icc import ICC, ICCConfig, PCCS, PCCSConfig  # noqa: E402
from common.baselines import run_method  # noqa: E402
from common.datasets import make_pccs  # noqa: E402
from common.utils import scores  # noqa: E402


def main():
    X, y, _, _ = make_pccs("noisy", 1200, seed=42)           # 100 features: 15 informative, 15 redundant, 70 noise
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(y))
    train, test = idx[:900], idx[900:]

    model = ICC(ICCConfig(n_clusters=4, random_state=0)).fit(X[train])
    km_train, km_test = run_method("kmeans", X[train], X[test], 4, 0)

    print("Training rows (900):")
    print(f"  ICC     ARI = {scores(y[train], model.labels_)['ARI']:.3f}")
    print(f"  k-means ARI = {scores(y[train], km_train)['ARI']:.3f}")
    print("Held-out rows (300), inductive assignment:")
    print(f"  ICC     ARI = {scores(y[test], model.predict(X[test]))['ARI']:.3f}")
    print(f"  k-means ARI = {scores(y[test], km_test)['ARI']:.3f}")
    print("\nStability fingerprint (co-association within cluster, per perturbation family):")
    print(model.fingerprint_.round(2).to_string(index=False))

    print("\n--- PCCS on 500 rows (6 perturbation families, 4 base methods) ---")
    p = PCCS(PCCSConfig(random_state=0)).fit(X[:500])
    print(f"consensus ARI = {scores(y[:500], p.labels_)['ARI']:.3f}   reference ARI = {scores(y[:500], p.ref_)['ARI']:.3f}")
    print("active base methods:", ", ".join(p.active_methods_))
    cols = ["cluster", "size", "algorithmic", "sampling", "feature", "measurement", "technical", "distribution", "PSI", "reliability"]
    print(p.fingerprint_[cols].round(2).to_string(index=False))
    top = p.interaction_[p.interaction_.cluster == "global"].sort_values("interaction", ascending=False).head(3)
    print("largest pairwise interactions (expected - observed stability):")
    print(top[["pair", "S_pq_expected", "S_pq_observed", "interaction"]].round(2).to_string(index=False))
    print(f"{p.n_clusterings_} clusterings in {p.timings_['total']:.1f} s")


if __name__ == "__main__":
    main()
