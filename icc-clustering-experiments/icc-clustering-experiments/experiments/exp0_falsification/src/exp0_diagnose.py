"""Diagnostics: why can null data look stable? Variance concentration, k-means partition balance,
and the seed-vs-bootstrap stability gap at sigma=1."""
import numpy as np, pandas as pd
from exp0_core import *
rows = []
for s in STRUCTURES:
    for sd in range(10):
        rng = np.random.default_rng(500 + sd)
        X, y = make_structure(s, rng=rng)
        X = add_noise(X, "gaussian", 1.0, rng)
        ev = np.linalg.eigvalsh(np.cov(X.T))[::-1]
        st, lab = stability(X, 4, 8, rng)
        sizes = np.bincount(lab, minlength=4) / len(lab)
        rows.append(dict(structure=s, seed=sd, top1_var_share=float(ev[0] / ev.sum()),
                         top3_var_share=float(ev[:3].sum() / ev.sum()), min_cluster_share=float(sizes.min()),
                         seed_minus_boot=st["stab_seed"] - st["stab_boot"], **st))
df = pd.DataFrame(rows)
out = df.groupby("structure").mean(numeric_only=True).drop(columns="seed").reset_index()
out.to_csv(ROOT / "exp0_falsification" / "data" / "exp0_diagnose.csv", index=False)
print(out.round(3).to_string())
