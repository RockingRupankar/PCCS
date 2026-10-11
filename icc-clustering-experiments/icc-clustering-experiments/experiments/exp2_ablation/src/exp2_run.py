"""Ablation of ICC v1 components (transductive: cluster all 1200 rows, score vs true_cluster)."""
import sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
from common.icc_adapter import ICCv1, ICCConfig
from common.baselines import run_method
from common.utils import scores

DATA = ROOT / "exp2_ablation" / "data"
ALL = ("seed", "row", "feature", "noise", "combined")
VARIANTS = {
    "full": {},
    "no_stability_weights": dict(use_stability_weights=False),
    "no_family_fusion (pooled)": dict(use_family_fusion=False),
    "no_invariance (lambda=0)": dict(invariance_lambda=0.0),
    "invariance_only (lambda=1)": dict(invariance_lambda=1.0),
    "no_row_views": dict(families=("seed", "feature", "noise", "combined")),
    "no_feature_views": dict(families=("seed", "row", "noise")),
    "no_noise_views": dict(families=("seed", "row", "feature")),
    "no_combined_views": dict(families=("seed", "row", "feature", "noise")),
    "seed_only_views": dict(families=("seed",)),
    "kmeans_only": dict(base_methods=("kmeans",)),
    "ward_only": dict(base_methods=("ward",)),
    "kmeans+ward+gmm": dict(base_methods=("kmeans", "ward", "gmm")),
}
rows, fps = [], []
for prof in ("noisy", "severe"):
    for seed in (42, 43, 44):
        df = pd.read_csv(DATA / f"exp2_{prof}_seed{seed}.csv")
        feats = [c for c in df.columns if c.startswith("feature_")]
        X, y = df[feats].to_numpy(float), df.true_cluster.to_numpy()
        for name, kw in VARIANTS.items():
            t = time.time()
            m = ICCv1(ICCConfig(n_clusters=4, random_state=seed, **kw)).fit(X)
            sc = scores(y, m.labels_)
            rows.append(dict(profile=prof, data_seed=seed, variant=name, **sc,
                             min_cluster_frac=float(np.bincount(m.labels_, minlength=4).min() / len(y)),
                             used_spectral_fallback=m.used_fallback_, mean_view_agreement=m.mean_agreement_,
                             seconds=time.time() - t))
            if name == "full":
                f = m.fingerprint_.copy(); f.insert(0, "profile", prof); f.insert(1, "data_seed", seed)
                f["ARI_of_run"] = sc["ARI"]; fps.append(f)
        for b in ("kmeans", "eac"):
            t = time.time()
            lab, _ = run_method(b, X, None, 4, seed)
            rows.append(dict(profile=prof, data_seed=seed, variant=f"baseline_{b}", **scores(y, lab),
                             min_cluster_frac=float(np.bincount(lab, minlength=4).min() / len(y)),
                             used_spectral_fallback=False, mean_view_agreement=np.nan, seconds=time.time() - t))
        print(prof, seed, "done", flush=True)
res = pd.DataFrame(rows)
res.to_csv(DATA / "exp2_ablation_results.csv", index=False)
pd.concat(fps).to_csv(DATA / "exp2_fingerprints.csv", index=False)
print(res.groupby(["profile", "variant"]).ARI.mean().unstack(0).round(3))
