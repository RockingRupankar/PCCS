"""fig13_scalability.png and fig14_sensitivity.png"""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parents[1]
D = BASE / "data"
sc = pd.read_csv(D / "exp4_scalability_results.csv")
fits = pd.read_csv(D / "exp4_scaling_fits.csv")
C = dict(kmeans="#1f77b4", ward="#2ca02c", eac="#7f7f7f", icc_v1="#d62728")
fig, axs = plt.subplots(1, 3, figsize=(14, 4))
for ax, (sweep, var) in zip(axs[:2], (("N", "n"), ("d", "d"))):
    for m, g in sc[sc.sweep == sweep].groupby("method"):
        a = g.groupby(var).seconds.mean()
        sl = fits[(fits.sweep == sweep) & (fits.method == m)].loglog_slope.iloc[0]
        ax.plot(a.index, a.values, marker="o", color=C[m], label=f"{m} (slope {sl:.2f})")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel(var.upper()); ax.set_ylabel("seconds")
    ax.set_title(f"Runtime vs {var.upper()}"); ax.legend(fontsize=8)
for m, g in sc.groupby("method"):
    a = g[g.sweep == "N"].groupby("n").ARI.mean()
    axs[2].plot(a.index, a.values, marker="o", color=C[m], label=m)
axs[2].set_xlabel("N"); axs[2].set_ylabel("ARI (Gaussian mixture, d=100)"); axs[2].set_title("Accuracy vs N"); axs[2].legend(fontsize=8)
fig.suptitle("Fig 13  Scalability"); fig.tight_layout(); fig.savefig(BASE / "figures" / "fig13_scalability.png", dpi=150); plt.close(fig)

se = pd.read_csv(D / "exp4_sensitivity_results.csv")
params = list(se.param.unique())
fig, axs = plt.subplots(2, 4, figsize=(16, 7))
for ax, p in zip(axs.ravel(), params):
    g = se[se.param == p].groupby("value").ARI.agg(["mean", "std"])
    ax.errorbar(g.index, g["mean"], g["std"], marker="o", capsize=2, color="#d62728", label="ICC v1")
    if p == "n_clusters_assumed":
        k = se[se.param == p].groupby("value").kmeans_ARI.mean()
        ax.plot(k.index, k.values, "s--", color="#1f77b4", label="k-means")
        ax.legend(fontsize=8)
    dv = se[(se.param == p) & se.is_default].value.iloc[0]
    ax.axvline(dv, color="gray", ls=":"); ax.set_title(p, fontsize=10); ax.set_ylabel("ARI")
for ax in axs.ravel()[len(params):]:
    ax.axis("off")
fig.suptitle("Fig 14  One-at-a-time sensitivity (dotted = default; mean +- sd over 3 data seeds)")
fig.tight_layout(); fig.savefig(BASE / "figures" / "fig14_sensitivity.png", dpi=150)
rng = se.groupby(["param", "value"]).ARI.mean().groupby("param").agg(lambda v: v.max() - v.min())
rng.rename("ARI_range_over_grid").to_csv(D / "exp4_sensitivity_ranges.csv"); print(rng.round(3))
