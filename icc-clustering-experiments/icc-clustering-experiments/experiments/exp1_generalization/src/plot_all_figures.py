"""fig9_inductive_generalization.png: test ARI and train-test gap vs training size."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parents[1]
s = pd.read_csv(BASE / "data" / "exp1_summary.csv")
colors = dict(zip(["kmeans", "ward", "gmm_diag", "spectral_knn", "eac", "icc_v1"],
                  ["#1f77b4", "#2ca02c", "#9467bd", "#8c564b", "#7f7f7f", "#d62728"]))
fig, axs = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
for j, prof in enumerate(("clean", "noisy", "severe")):
    for m, g in s[s.profile == prof].groupby("method"):
        axs[0, j].errorbar(g.n_train, g.ARI_test_mean, g.ARI_test_std, marker="o", capsize=2, color=colors[m],
                           lw=2.2 if m == "icc_v1" else 1.2, label=m)
        axs[1, j].plot(g.n_train, g.generalization_gap_mean, marker="o", color=colors[m], lw=2.2 if m == "icc_v1" else 1.2)
    axs[0, j].set_title(prof); axs[1, j].set_xlabel("training rows"); axs[1, j].axhline(0, color="k", lw=.5)
    axs[0, j].set_xscale("log"); axs[1, j].set_xscale("log")
axs[0, 0].set_ylabel("ARI on held-out rows"); axs[1, 0].set_ylabel("ARI train - ARI test")
axs[0, 0].legend(fontsize=8)
fig.suptitle("Fig 9  Inductive generalisation (mean +- sd over 3 seeds)"); fig.tight_layout()
fig.savefig(BASE / "figures" / "fig9_inductive_generalization.png", dpi=150)
print("saved fig9")
