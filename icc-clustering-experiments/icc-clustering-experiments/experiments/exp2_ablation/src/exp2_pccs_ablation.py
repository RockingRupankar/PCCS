"""PCCS ablation and validity of the reliability profile.

Per dataset (PCCS-style synthetic noisy / severe, N=500, 5 generator seeds) one PCCS model is fitted (all perturbation
families, 4 base methods, interactions). Variants rebuild only the consensus from the SAME stored partitions, so
differences are due to the consensus construction, not to different random clusterings.
Validity: do per-cluster PSI / fingerprints / categories predict how wrong a consensus cluster is
(1 - best Jaccard with a true cluster)?
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "src"))
import numpy as np, pandas as pd
from common.datasets import make_pccs
from common.baselines import run_method
from common.utils import scores
from icc import PCCS, PCCSConfig, FAMILIES

DATA = ROOT / "exp2_ablation" / "data"
SEEDS = (42, 43, 44, 45, 46)
VARIANTS = {
    "full_pccs": {},
    "pooled_no_family_tensor": dict(per_family=False),
    "no_reliability_weights": dict(reliability_weights=False),
    "equal_family_weights": dict(family_weighting="equal"),
    "no_consistency_term (lambda=0)": dict(lam=0.0),
    "consistency_only (lambda=1)": dict(lam=1.0),
    "algorithmic_family_only": dict(families=("algorithmic",)),
    "kmeans_only": dict(methods=("kmeans",)),
    "kmeans+fcm": dict(methods=("kmeans", "fcm")),
    "kmeans+hierarchical": dict(methods=("kmeans", "hierarchical")),
    "without_dbscan": dict(methods=("kmeans", "fcm", "hierarchical")),
    **{f"leave_out_{f}": dict(families=tuple(x for x in FAMILIES if x != f)) for f in FAMILIES},
}


def best_jaccard(labels, y, K=4):
    out = []
    for k in np.unique(labels):
        m = labels == k
        out.append(max(((m & (y == t)).sum() / ((m | (y == t)).sum())) for t in np.unique(y)))
    return out


def main():
    rows, valid, example = [], [], None
    for prof in ("noisy", "severe"):
        for seed in SEEDS:
            X, y, names, df = make_pccs(prof, 500, seed)
            df.to_csv(DATA / f"exp2_pccs_{prof}_seed{seed}.csv", index=False, float_format="%.4f")
            m = PCCS(PCCSConfig(random_state=seed)).fit(X)
            for name, kw in VARIANTS.items():
                lab = m.build_consensus(**kw)
                rows.append(dict(profile=prof, data_seed=seed, variant=name, **scores(y, lab)))
            rows.append(dict(profile=prof, data_seed=seed, variant="reference_partition(baseline ensemble)", **scores(y, m.ref_)))
            for b in ("kmeans", "eac"):
                lab, _ = run_method(b, X, None, 4, seed)
                rows.append(dict(profile=prof, data_seed=seed, variant=f"baseline_{b}", **scores(y, lab)))
            bj = best_jaccard(m.labels_, y)
            fp = m.fingerprint_.copy()
            fp.insert(0, "profile", prof); fp.insert(1, "data_seed", seed)
            fp["best_jaccard_with_truth"] = bj
            fp["cluster_error"] = 1 - np.array(bj)
            valid.append(fp)
            if prof == "noisy" and seed == SEEDS[0]:
                example = m
            print(prof, seed, "done", f"ARI full={scores(y, m.labels_)['ARI']:.3f}", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(DATA / "exp2_pccs_ablation_results.csv", index=False)
    pd.concat(valid).to_csv(DATA / "exp2_pccs_fingerprint_validity.csv", index=False)
    example.interaction_.to_csv(DATA / "exp2_pccs_example_interactions.csv", index=False)
    print(res.groupby(["variant", "profile"]).ARI.mean().unstack().round(3))


if __name__ == "__main__":
    main()
