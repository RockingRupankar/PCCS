"""One-at-a-time sensitivity of ICC v1 hyper-parameters (and of the assumed K) on PCCS-noisy, N=800."""
import sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
from common.icc_adapter import ICCv1, ICCConfig
from common.baselines import run_method
from common.utils import scores

DATA = ROOT / "exp4_scalability_and_sensitivity" / "data"
GRID = {
    "n_views": [4, 8, 12, 20, 32],
    "row_frac": [0.5, 0.6, 0.7, 0.8, 0.9],
    "feat_frac": [0.2, 0.35, 0.5, 0.7, 0.9],
    "noise_sd": [0.0, 0.1, 0.25, 0.5, 1.0],
    "weight_power": [0.0, 1.0, 2.0, 4.0],
    "invariance_lambda": [0.0, 0.25, 0.5, 0.75, 1.0],
    "n_clusters_assumed": [3, 4, 5, 6, 8],
}
default = ICCConfig()
rows = []
for seed in (42, 43, 44):
    df = pd.read_csv(DATA / f"exp4_sensitivity_noisy_seed{seed}.csv")
    feats = [c for c in df.columns if c.startswith("feature_")]
    X, y = df[feats].to_numpy(float), df.true_cluster.to_numpy()
    for p, vals in GRID.items():
        for v in vals:
            kw = {"n_clusters": v} if p == "n_clusters_assumed" else {p: v}
            t = time.time()
            m = ICCv1(ICCConfig(random_state=seed, **kw)).fit(X)
            sc = scores(y, m.labels_)
            row = dict(param=p, value=v, data_seed=seed, is_default=(v == 4 if p == "n_clusters_assumed" else v == getattr(default, p)),
                       seconds=time.time() - t, **sc)
            if p == "n_clusters_assumed":     # k-means comparator at the same assumed K
                lab, _ = run_method("kmeans", X, None, v, seed)
                row["kmeans_ARI"] = scores(y, lab)["ARI"]
            rows.append(row)
    print("seed", seed, "done", flush=True)
res = pd.DataFrame(rows)
res.to_csv(DATA / "exp4_sensitivity_results.csv", index=False)
print(res.groupby(["param", "value"])[["ARI", "NMI", "seconds"]].mean().round(3))
