"""fig20: PCCS versus the other methods on the three smaller real datasets (digits excluded for compute reasons)."""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parents[1]
D, F = BASE / "data", BASE / "figures"
s = pd.read_csv(D / "exp3_summary.csv")
p = pd.read_csv(D / "exp3_pccs_results.csv").groupby(["dataset", "variant"])[["ARI", "NMI", "VI", "seed_stability"]].mean().reset_index()
p["method"] = "pccs"
p["seconds_per_run"] = np.nan
a = pd.concat([s[s.dataset.isin(p.dataset.unique())][["dataset", "variant", "method", "ARI", "NMI", "VI", "seed_stability"]], p[["dataset", "variant", "method", "ARI", "NMI", "VI", "seed_stability"]]])
a["rank_ARI"] = a.groupby(["dataset", "variant"]).ARI.rank(ascending=False)
a.to_csv(D / "exp3_pccs_vs_others.csv", index=False)
M = ["kmeans", "ward", "gmm_diag", "spectral_knn", "eac", "icc_v1", "pccs"]
C = ["#1f77b4", "#2ca02c", "#9467bd", "#8c564b", "#7f7f7f", "#ff7f0e", "#d62728"]
D3 = ["iris", "wine", "breast_cancer"]
fig, axs = plt.subplots(1, 2, figsize=(13, 4.2), sharey=True)
for ax, v in zip(axs, ("original", "corrupted")):
    g = a[a.variant == v]
    for i, m in enumerate(M):
        ax.bar(np.arange(3) + (i - 3) * 0.12, [g[(g.dataset == d) & (g.method == m)].ARI.iloc[0] for d in D3], 0.12, color=C[i], label=m)
    ax.set_xticks(range(3)); ax.set_xticklabels(D3); ax.set_title(v)
axs[0].set_ylabel("ARI vs labels"); axs[0].legend(fontsize=7, ncol=2)
fig.suptitle("Fig 20  PCCS on real data (5 seeds)"); fig.tight_layout(); fig.savefig(F / "fig20_pccs_real_world.png", dpi=150)
print(a.pivot_table(index=["dataset", "variant"], columns="method", values="ARI").round(3).to_string())
print(a.groupby(["variant", "method"]).rank_ARI.mean().unstack(0).round(2))
