"""fig18 (stability vs perturbation intensity per family) and fig19 (interaction effects)."""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parents[1]
D, F = BASE / "data", BASE / "figures"
r = pd.read_csv(D / "exp4_pccs_response.csv")
it = pd.read_csv(D / "exp4_pccs_interactions.csv")
FAM = ["algorithmic", "sampling", "feature", "measurement", "technical", "distribution"]
MC = dict(kmeans="#1f77b4", fcm="#9467bd", hierarchical="#2ca02c", dbscan="#7f7f7f")
LS = dict(clean="-", noisy="--")
s = r[r.kind == "single"]
fig, axs = plt.subplots(1, 6, figsize=(20, 3.8), sharey=True)
for ax, f in zip(axs, FAM):
    for (prof, m), g in s[s.family == f].groupby(["profile", "method"]):
        a = g.groupby("level").ari_ref.mean()
        ax.plot(a.index, a.values, LS[prof], marker="o", ms=3, color=MC[m], label=f"{m} ({prof})")
    ax.set_title(f); ax.set_xlabel("intensity")
    if f in ("measurement", "technical", "distribution"):
        ax.set_xscale("log")
axs[0].set_ylabel("ARI to unperturbed reference"); axs[0].legend(fontsize=6)
fig.suptitle("Fig 18  Stability as a function of perturbation type and intensity (solid: clean data, dashed: noisy data)")
fig.tight_layout(); fig.savefig(F / "fig18_pccs_intensity_response.png", dpi=150); plt.close(fig)

# summary table: area under response (mean over levels) per family, active methods
act = s[s.active_method]
tab = act.groupby(["profile", "family", "method"]).ari_ref.mean().unstack("method").round(3)
tab.to_csv(D / "exp4_pccs_family_stability.csv"); print(tab)

g = it[it.cluster == "global"]
agg = g.groupby(["profile", "pair"]).interaction.agg(["mean", "std"]).reset_index()
agg.to_csv(D / "exp4_pccs_interaction_summary.csv", index=False)
order = agg[agg.profile == "noisy"].sort_values("mean").pair.tolist()
fig, ax = plt.subplots(figsize=(8, 6))
w = 0.38
for i, (prof, c) in enumerate((("clean", "#1f77b4"), ("noisy", "#d62728"))):
    a = agg[agg.profile == prof].set_index("pair").loc[order]
    ax.barh(np.arange(len(order)) + (i - .5) * w, a["mean"], w, xerr=a["std"], color=c, label=prof, capsize=2)
ax.axvline(0.15, color="k", ls=":", label="interaction threshold 0.15"); ax.axvline(0, color="gray", lw=.5)
ax.set_yticks(range(len(order))); ax.set_yticklabels(order, fontsize=8)
ax.set_xlabel("interaction = expected - observed combined stability (positive = extra degradation)")
ax.set_title("Fig 19  Pairwise perturbation interactions (global, mean +- sd over 3 seeds)"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(F / "fig19_pccs_interactions.png", dpi=150)
print(agg.sort_values("mean", ascending=False).head(8).round(3).to_string(index=False))
