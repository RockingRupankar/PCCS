"""Figures 1-8 for exp0 + exp0_verdict.csv (H1 supported / falsified per condition)."""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parents[1]
D, F = BASE / "data", BASE / "figures"
COL = {"structured": "#1f77b4", "null_iso": "#7f7f7f", "null_aniso": "#d62728"}
GAP_THR = 0.10          # H1 counts as supported when structured - max(null) stability >= 0.10


def gap_frame(df, idx):
    p = df.pivot_table(index=idx, columns="structure", values="stab_boot")
    p["gap"] = p["structured"] - p[["null_iso", "null_aniso"]].max(axis=1)
    return p


def main():
    agg = pd.read_csv(D / "exp0_agg.csv")
    raw = pd.read_csv(D / "exp0_raw.csv")
    fine = pd.read_csv(D / "exp0_finegrid.csv")
    nt = pd.read_csv(D / "exp0_noisetypes.csv")
    sc = pd.read_csv(D / "exp0_scaled_agg.csv")
    fc = pd.read_csv(D / "exp0_fcm_full_curve.csv")
    verdict = []

    # fig1
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for s, g in agg.groupby("structure"):
        ax.errorbar(g.level, g.stab_boot, g.stab_boot_sd, color=COL[s], marker="o", capsize=2, label=f"{s} (bootstrap)")
        ax.plot(g.level, g.stab_seed, color=COL[s], ls="--", alpha=.6, label=f"{s} (seed)")
    ax.set_xlabel("Gaussian noise sd"); ax.set_ylabel("mean pairwise ARI (stability)")
    ax.set_title("Fig 1  K-means stability vs noise"); ax.legend(fontsize=7); fig.tight_layout()
    fig.savefig(F / "fig1_ari_stability.png", dpi=150); plt.close(fig)

    # fig2 gap
    pf, pa = gap_frame(fine, "level"), gap_frame(agg, "level")
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(pf.index, pf.gap, "-", color="k", label="fine grid")
    ax.plot(pa.index, pa.gap, "o", color="#ff7f0e", label="main sweep")
    ax.axhline(GAP_THR, color="r", ls=":", label=f"gap threshold {GAP_THR}")
    ax.axhline(0, color="gray", lw=.5)
    ax.set_xlabel("Gaussian noise sd"); ax.set_ylabel("stability gap (structured - max null)")
    ax.set_title("Fig 2  Stability gap"); ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(F / "fig2_gap.png", dpi=150); plt.close(fig)
    below = pf[pf.gap < GAP_THR]
    verdict.append(dict(experiment="gaussian_finegrid", condition=f"first sigma with gap<{GAP_THR}",
                        value=float(below.index.min()) if len(below) else np.nan,
                        h1_supported=not len(below)))
    for lv, r in pa.iterrows():
        verdict.append(dict(experiment="gaussian_main", condition=f"sigma={lv}", value=float(r.gap), h1_supported=bool(r.gap >= GAP_THR)))

    # fig3 within (seed) vs between (bootstrap)
    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    for s, g in raw.groupby("structure"):
        ax.scatter(g.stab_seed, g.stab_boot, s=12, alpha=.5, color=COL[s], label=s)
    ax.plot([0, 1], [0, 1], "k:", lw=.8)
    ax.set_xlabel("seed-only stability (algorithmic perturbation)")
    ax.set_ylabel("bootstrap stability (sampling perturbation)")
    ax.set_title("Fig 3  Within-data vs across-sample stability"); ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(F / "fig3_within_between.png", dpi=150); plt.close(fig)

    # fig4 noise-type heatmap
    g = nt.groupby(["noise_type", "level", "structure"]).stab_boot.mean().unstack()
    g["gap"] = g["structured"] - g[["null_iso", "null_aniso"]].max(axis=1)
    H = g["gap"].unstack("level")
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    im = ax.imshow(H.values, cmap="RdYlGn", vmin=-0.2, vmax=0.8, aspect="auto")
    ax.set_xticks(range(H.shape[1])); ax.set_xticklabels(H.columns)
    ax.set_yticks(range(H.shape[0])); ax.set_yticklabels(H.index)
    for i in range(H.shape[0]):
        for j in range(H.shape[1]):
            ax.text(j, i, f"{H.values[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_xlabel("noise intensity"); ax.set_title("Fig 4  Stability gap by noise type")
    fig.colorbar(im, label="gap"); fig.tight_layout(); fig.savefig(F / "fig4_noisetype_heatmap.png", dpi=150); plt.close(fig)
    for (t, lv), v in g["gap"].items():
        verdict.append(dict(experiment="noise_types", condition=f"{t}@{lv}", value=float(v), h1_supported=bool(v >= GAP_THR)))

    # fig5 / fig6 scaled
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
    for ax, lv in zip(axs, sorted(sc.level.unique())):
        for s, gg in sc[sc.level == lv].groupby("structure"):
            ax.errorbar(gg.d, gg.stab_boot, gg.stab_boot_sd, color=COL[s], marker="o", capsize=2, label=s)
        ax.set_xscale("log"); ax.set_title(f"noise sd {lv}"); ax.set_xlabel("dimension d")
    axs[0].set_ylabel("bootstrap stability"); axs[0].legend(fontsize=8)
    fig.suptitle("Fig 5  Scaled regime: stability vs dimension"); fig.tight_layout()
    fig.savefig(F / "fig5_scaled_ari_stability.png", dpi=150); plt.close(fig)

    ps = gap_frame(sc, ["level", "d"])
    fig, ax = plt.subplots(figsize=(6, 4))
    for lv, gg in ps.groupby(level=0):
        ax.plot(gg.index.get_level_values(1), gg.gap, marker="o", label=f"noise sd {lv}")
    ax.axhline(GAP_THR, color="r", ls=":"); ax.set_xscale("log")
    ax.set_xlabel("dimension d"); ax.set_ylabel("gap"); ax.set_title("Fig 6  Scaled regime: stability gap"); ax.legend()
    fig.tight_layout(); fig.savefig(F / "fig6_scaled_gap.png", dpi=150); plt.close(fig)
    for (lv, d), r in ps.iterrows():
        verdict.append(dict(experiment="scaled", condition=f"sigma={lv},d={d}", value=float(r.gap), h1_supported=bool(r.gap >= GAP_THR)))

    # fig7 regime comparison (small d=20 vs scaled d=200) at matched noise
    rows = []
    for lv in (0.5, 1.0):
        a = agg[(agg.level == lv)].set_index("structure").stab_boot
        b = sc[(sc.level == lv) & (sc.d == 200)].set_index("structure").stab_boot
        rows.append((f"small d=20\nsd={lv}", a))
        rows.append((f"scaled d=200\nsd={lv}", b))
    fig, ax = plt.subplots(figsize=(7.5, 4))
    w = 0.25
    for i, s in enumerate(["structured", "null_iso", "null_aniso"]):
        ax.bar(np.arange(len(rows)) + (i - 1) * w, [r[1][s] for r in rows], w, color=COL[s], label=s)
    ax.set_xticks(range(len(rows))); ax.set_xticklabels([r[0] for r in rows], fontsize=8)
    ax.set_ylabel("bootstrap stability"); ax.set_title("Fig 7  Regime comparison"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(F / "fig7_regime_comparison.png", dpi=150); plt.close(fig)

    # fig8 FCM
    f0 = fc[fc.sigma == 0.0]
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.8))
    for d, g in f0.groupby("d"):
        axs[0].plot(g.m, g.ari, marker="o", label=f"d={d}")
        axs[1].plot(g.m, g.center_spread, marker="o", label=f"d={d}")
        axs[2].plot(g.m, g.membership_entropy, marker="o", label=f"d={d}")
    for ax, t in zip(axs, ["ARI vs truth", "mean centre spread", "normalised membership entropy"]):
        ax.set_xscale("log"); ax.set_xlabel("fuzzifier m"); ax.set_title(t)
    axs[0].legend(); fig.suptitle("Fig 8  FCM collapse as fuzzifier m grows (sigma=0)"); fig.tight_layout()
    fig.savefig(F / "fig8_fcm_phase_transition.png", dpi=150); plt.close(fig)

    pd.DataFrame(verdict).to_csv(D / "exp0_verdict.csv", index=False)
    v = pd.DataFrame(verdict)
    print(v.groupby("experiment").h1_supported.agg(["sum", "count"]))


if __name__ == "__main__":
    main()
