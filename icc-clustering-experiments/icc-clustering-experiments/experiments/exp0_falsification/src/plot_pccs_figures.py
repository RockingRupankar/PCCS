"""fig15: per-family stability profiles and detector AUCs (fingerprint vs scalar aggregate)."""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parents[1]
D, F = BASE / "data", BASE / "figures"
feat = pd.read_csv(D / "exp0_pccs_fingerprint_features.csv")
auc = pd.read_csv(D / "exp0_pccs_auc.csv")
FAM = ["algorithmic", "sampling", "feature", "measurement", "technical", "distribution"]
COL = {"structured": "#1f77b4", "null_iso": "#7f7f7f", "null_aniso": "#d62728"}

fig, axs = plt.subplots(1, 2, figsize=(14, 4.8), gridspec_kw=dict(width_ratios=[1, 1.3]))
g = feat[feat.level == 1.0]
for i, s in enumerate(["structured", "null_iso", "null_aniso"]):
    mu = g[g.structure == s][FAM].mean()
    sd = g[g.structure == s][FAM].std()
    axs[0].bar(np.arange(6) + (i - 1) * 0.27, mu, 0.27, yerr=sd, color=COL[s], label=s, capsize=2)
axs[0].set_xticks(range(6)); axs[0].set_xticklabels(FAM, rotation=25, fontsize=8)
axs[0].set_ylabel("stability under perturbation family (mean ARI to reference)")
axs[0].set_title("Stability fingerprints at noise sd 1.0"); axs[0].legend(fontsize=8)

order = ["scalar_mean", "min_family", *FAM, "fingerprint_logreg_cv"]
H = auc.pivot_table(index="predictor", columns=["task", "level"], values="auc").loc[order]
im = axs[1].imshow(H.values, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
axs[1].set_yticks(range(len(order))); axs[1].set_yticklabels(order, fontsize=8)
axs[1].set_xticks(range(H.shape[1])); axs[1].set_xticklabels([f"{t.replace('structured_vs_', 'vs ')}\nsd {l}" for t, l in H.columns], fontsize=7)
for i in range(H.shape[0]):
    for j in range(H.shape[1]):
        axs[1].text(j, i, f"{H.values[i, j]:.2f}", ha="center", va="center", fontsize=7)
axs[1].set_title("Directed AUC: structured vs null (0.5 = chance, <0.5 = null looks more stable)")
fig.colorbar(im, ax=axs[1], fraction=0.03)
fig.suptitle("Fig 15  Fingerprint versus aggregate stability as a detector of structure (PCCS, k-means members)")
fig.tight_layout(); fig.savefig(F / "fig15_pccs_fingerprint_vs_scalar.png", dpi=150)
print(H.round(2).to_string())
