"""Benchmark all methods on 4 real datasets x {original, corrupted}; K = true number of classes; 5 seeds."""
import sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
from common.baselines import METHODS, run_method
from common.utils import scores, pairwise_ari

DATA = ROOT / "exp3_real_world_benchmarks" / "data"
rows = []
for name in ("iris", "wine", "breast_cancer", "digits"):
    for variant in ("original", "corrupted"):
        df = pd.read_csv(DATA / (f"exp3_{name}.csv" if variant == "original" else f"exp3_{name}_corrupted.csv"))
        y = df.target.to_numpy()
        X = df.drop(columns="target").to_numpy(float)
        k = len(np.unique(y))
        for m in METHODS:
            labs, t0 = [], time.time()
            for seed in range(5):
                lab, _ = run_method(m, X, None, k, seed)
                labs.append(lab)
                rows.append(dict(dataset=name, variant=variant, method=m, seed=seed, n=len(y), d=X.shape[1], k=k,
                                 **scores(y, lab)))
            stab = pairwise_ari(labs)
            for r in rows[-5:]:
                r["seed_stability"] = stab
                r["seconds_per_run"] = (time.time() - t0) / 5
        print(name, variant, "done", flush=True)
res = pd.DataFrame(rows)
res.to_csv(DATA / "exp3_real_world_results.csv", index=False)
s = res.groupby(["dataset", "variant", "method"])[["ARI", "NMI", "VI", "seed_stability", "seconds_per_run"]].mean().reset_index()
s["rank_ARI"] = s.groupby(["dataset", "variant"]).ARI.rank(ascending=False)
s.to_csv(DATA / "exp3_summary.csv", index=False)
print(s.pivot_table(index=["dataset", "variant"], columns="method", values="ARI").round(3))
print(s.groupby(["variant", "method"]).rank_ARI.mean().unstack(0).round(2))
