"""Inductive generalisation: fit on n_train rows, label 800 held-out rows.
ARI_test uses the method's own out-of-sample rule (see common/baselines.py)."""
import sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
from common.baselines import METHODS, run_method
from common.utils import scores

DATA = ROOT / "exp1_generalization" / "data"
SIZES = [100, 200, 400, 800, 1200]
SEEDS = range(3)
rows = []
for prof in ("clean", "noisy", "severe"):
    df = pd.read_csv(DATA / f"exp1_{prof}.csv")
    feats = [c for c in df.columns if c.startswith("feature_")]
    pool, test = df[df.split == "pool"], df[df.split == "test"]
    Xte, yte = test[feats].to_numpy(float), test.true_cluster.to_numpy()
    for seed in SEEDS:
        for n in SIZES:
            sub = pool.sample(n=n, random_state=seed * 100 + n)
            Xtr, ytr = sub[feats].to_numpy(float), sub.true_cluster.to_numpy()
            for m in METHODS:
                t = time.time()
                ltr, lte = run_method(m, Xtr, Xte, 4, seed)
                rows.append(dict(profile=prof, seed=seed, n_train=n, method=m,
                                 ARI_train=scores(ytr, ltr)["ARI"], **{f"{k}_test": v for k, v in scores(yte, lte).items()},
                                 seconds=time.time() - t))
        print(prof, "seed", seed, "done", flush=True)
res = pd.DataFrame(rows)
res["generalization_gap"] = res.ARI_train - res.ARI_test
res.to_csv(DATA / "exp1_generalization_results.csv", index=False)
summ = res.groupby(["profile", "method", "n_train"])[["ARI_train", "ARI_test", "NMI_test", "generalization_gap", "seconds"]].agg(["mean", "std"])
summ.columns = ["_".join(c) for c in summ.columns]
summ.reset_index().to_csv(DATA / "exp1_summary.csv", index=False)
print(res[res.n_train == 800].groupby(["profile", "method"]).ARI_test.mean().unstack().round(3))
