"""Wall-clock scaling of ICC v1 vs baselines in N (d=100) and in d (N=800). Run sequentially (timing).
Memory is reported analytically: ICC holds (n_families + 3) float32 n x n matrices at peak."""
import sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
from common.datasets import make_scalable
from common.baselines import run_method
from common.utils import scores

DATA = ROOT / "exp4_scalability_and_sensitivity" / "data"
ICC_KW = dict(n_views=12)
rows = []


def bench(n, d, sweep):
    X, y = make_scalable(n, d, seed=n + d)
    for m in ("kmeans", "ward", "eac", "icc_v1"):
        for rep in range(2):
            t = time.time()
            lab, _ = run_method(m, X, None, 4, rep, icc_kw=ICC_KW if m == "icc_v1" else None)
            rows.append(dict(sweep=sweep, n=n, d=d, method=m, rep=rep, seconds=time.time() - t,
                             ARI=scores(y, lab)["ARI"], est_icc_peak_MB=n * n * 4 * (5 + 3) / 1e6 if m == "icc_v1" else np.nan))
    print(sweep, n, d, flush=True)


for n in (250, 500, 1000, 1500, 2000):
    bench(n, 100, "N")
for d in (20, 50, 100, 200, 400):
    bench(800, d, "d")
res = pd.DataFrame(rows)
res.to_csv(DATA / "exp4_scalability_results.csv", index=False)
fits = []
for sweep, var in (("N", "n"), ("d", "d")):
    for m, g in res[res.sweep == sweep].groupby("method"):
        a = g.groupby(var).seconds.mean()
        slope, icpt = np.polyfit(np.log(a.index.values.astype(float)), np.log(a.values), 1)
        fits.append(dict(sweep=sweep, method=m, loglog_slope=slope, seconds_at_max=a.iloc[-1]))
fits = pd.DataFrame(fits)
fits.to_csv(DATA / "exp4_scaling_fits.csv", index=False)
print(fits.round(3))
