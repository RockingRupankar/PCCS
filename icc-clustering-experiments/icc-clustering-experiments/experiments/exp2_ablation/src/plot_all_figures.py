"""fig10_icc_v1_performance.png: ablation bars (ARI) and paired delta vs the full model."""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parents[1]
r = pd.read_csv(BASE / "data" / "exp2_ablation_results.csv")
order = r[r.profile == "noisy"].groupby("variant").ARI.mean().sort_values().index.tolist()
fig, axs = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw=dict(width_ratios=[1.2, 1]))
w = 0.38
for i, prof in enumerate(("noisy", "severe")):
    g = r[r.profile == prof].groupby("variant").ARI.agg(["mean", "std"]).loc[order]
    axs[0].barh(np.arange(len(order)) + (i - .5) * w, g["mean"], w, xerr=g["std"], label=prof,
                color=["#1f77b4", "#d62728"][i], capsize=2)
axs[0].set_yticks(range(len(order))); axs[0].set_yticklabels(order, fontsize=8)
axs[0].set_xlabel("ARI vs true_cluster (mean +- sd over 3 data seeds)"); axs[0].legend()
axs[0].set_title("ARI by variant")
full = r[r.variant == "full"].set_index(["profile", "data_seed"]).ARI
r["delta_vs_full"] = r.apply(lambda x: x.ARI - full[(x.profile, x.data_seed)], axis=1)
for i, prof in enumerate(("noisy", "severe")):
    g = r[(r.profile == prof) & (r.variant != "full")].groupby("variant").delta_vs_full.agg(["mean", "std"]).loc[[o for o in order if o != "full"]]
    axs[1].barh(np.arange(len(g)) + (i - .5) * w, g["mean"], w, xerr=g["std"], color=["#1f77b4", "#d62728"][i], capsize=2)
axs[1].set_yticks(range(len(order) - 1)); axs[1].set_yticklabels([o for o in order if o != "full"], fontsize=8)
axs[1].axvline(0, color="k", lw=.8); axs[1].set_xlabel("ARI change vs full ICC v1"); axs[1].set_title("Paired delta")
fig.suptitle("Fig 10  ICC v1 ablation"); fig.tight_layout()
fig.savefig(BASE / "figures" / "fig10_icc_v1_performance.png", dpi=150)
d = r[r.variant != "full"].groupby(["profile", "variant"]).delta_vs_full.mean().unstack(0).round(3)
d.to_csv(BASE / "data" / "exp2_delta_vs_full.csv"); print(d)
