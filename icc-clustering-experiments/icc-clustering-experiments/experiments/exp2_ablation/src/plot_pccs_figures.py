"""fig16 (PCCS ablation) and fig17 (does the reliability profile predict cluster errors?) + validity statistics."""
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parents[1]
D, F = BASE / "data", BASE / "figures"
r = pd.read_csv(D / "exp2_pccs_ablation_results.csv")
order = r[r.profile == "noisy"].groupby("variant").ARI.mean().sort_values().index.tolist()
fig, axs = plt.subplots(1, 2, figsize=(14, 7), gridspec_kw=dict(width_ratios=[1.2, 1]))
w = 0.38
for i, prof in enumerate(("noisy", "severe")):
    g = r[r.profile == prof].groupby("variant").ARI.agg(["mean", "std"]).loc[order]
    axs[0].barh(np.arange(len(order)) + (i - .5) * w, g["mean"], w, xerr=g["std"], label=prof, color=["#1f77b4", "#d62728"][i], capsize=2)
axs[0].set_yticks(range(len(order))); axs[0].set_yticklabels(order, fontsize=7)
axs[0].set_xlabel("ARI vs true_cluster (mean +- sd, 5 data seeds)"); axs[0].legend(); axs[0].set_title("ARI by variant")
full = r[r.variant == "full_pccs"].set_index(["profile", "data_seed"]).ARI
r["delta_vs_full"] = r.apply(lambda x: x.ARI - full[(x.profile, x.data_seed)], axis=1)
o2 = [o for o in order if o != "full_pccs"]
for i, prof in enumerate(("noisy", "severe")):
    g = r[(r.profile == prof) & (r.variant != "full_pccs")].groupby("variant").delta_vs_full.agg(["mean", "std"]).loc[o2]
    axs[1].barh(np.arange(len(o2)) + (i - .5) * w, g["mean"], w, xerr=g["std"], color=["#1f77b4", "#d62728"][i], capsize=2)
axs[1].set_yticks(range(len(o2))); axs[1].set_yticklabels(o2, fontsize=7); axs[1].axvline(0, color="k", lw=.8)
axs[1].set_xlabel("paired ARI change vs full PCCS"); axs[1].set_title("Paired delta")
fig.suptitle("Fig 16  PCCS consensus ablation"); fig.tight_layout(); fig.savefig(F / "fig16_pccs_ablation.png", dpi=150); plt.close(fig)
d = r[r.variant != "full_pccs"].groupby(["profile", "variant"]).delta_vs_full.mean().unstack(0).round(3)
d.to_csv(D / "exp2_pccs_delta_vs_full.csv"); print(d)

v = pd.read_csv(D / "exp2_pccs_fingerprint_validity.csv")
stats = []
for name, g in list(v.groupby("profile")) + [("pooled", v)]:
    for col in ("PSI", "max_interaction"):
        rho, p = spearmanr(g[col], g.cluster_error)
        stats.append(dict(group=name, predictor=col, spearman_rho=rho, p_value=p, n_clusters=len(g)))
pd.DataFrame(stats).to_csv(D / "exp2_pccs_validity_stats.csv", index=False)
print(pd.DataFrame(stats).round(3).to_string(index=False))
cat = v.groupby(["profile", "reliability"]).agg(n=("cluster", "count"), mean_error=("cluster_error", "mean")).reset_index()
cat.to_csv(D / "exp2_pccs_category_error.csv", index=False); print(cat.round(3).to_string(index=False))

fig, axs = plt.subplots(1, 2, figsize=(12, 4.6))
for prof, c in (("noisy", "#1f77b4"), ("severe", "#d62728")):
    g = v[v.profile == prof]
    axs[0].scatter(g.PSI, g.cluster_error, color=c, alpha=.7, label=prof)
axs[0].set_xlabel("perturbation sensitivity index PSI (equal family weights)"); axs[0].set_ylabel("cluster error = 1 - best Jaccard with a true cluster")
axs[0].legend(); axs[0].set_title("PSI vs error of each consensus cluster")
cats = ["globally_invariant", "conditionally_invariant", "perturbation_fragile", "ambiguous"]
data = [v[v.reliability == k].cluster_error.to_numpy() for k in cats]
axs[1].boxplot([x for x in data if len(x)], labels=[f"{k.replace('_', ' ')}\n(n={len(x)})" for k, x in zip(cats, data) if len(x)])
axs[1].tick_params(axis="x", labelsize=7); axs[1].set_ylabel("cluster error"); axs[1].set_title("Error by reliability category")
fig.suptitle("Fig 17  Validity of the PCCS reliability profile (10 datasets, 40 clusters)"); fig.tight_layout()
fig.savefig(F / "fig17_pccs_fingerprint_validity.png", dpi=150)
