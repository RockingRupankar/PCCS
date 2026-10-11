"""fig11_real_world_benchmarks.png (ARI) and fig12_real_world_stability.png (seed stability)."""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parents[1]
s = pd.read_csv(BASE / "data" / "exp3_summary.csv")
M = ["kmeans", "ward", "gmm_diag", "spectral_knn", "eac", "icc_v1"]
C = ["#1f77b4", "#2ca02c", "#9467bd", "#8c564b", "#7f7f7f", "#d62728"]
D = ["iris", "wine", "breast_cancer", "digits"]
for fname, col, title, fig_id in (("fig11_real_world_benchmarks.png", "ARI", "ARI vs labels", 11),
                                  ("fig12_real_world_stability.png", "seed_stability", "Run-to-run stability (mean pairwise ARI over 5 seeds)", 12)):
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.2), sharey=True)
    for ax, v in zip(axs, ("original", "corrupted")):
        g = s[s.variant == v]
        for i, m in enumerate(M):
            vals = [g[(g.dataset == d) & (g.method == m)][col].iloc[0] for d in D]
            ax.bar(np.arange(4) + (i - 2.5) * 0.13, vals, 0.13, color=C[i], label=m)
        ax.set_xticks(range(4)); ax.set_xticklabels(D); ax.set_title(v)
    axs[0].set_ylabel(title); axs[0].legend(fontsize=7, ncol=2)
    fig.suptitle(f"Fig {fig_id}  Real-world benchmarks: {title}"); fig.tight_layout()
    fig.savefig(BASE / "figures" / fname, dpi=150)
print("saved")
