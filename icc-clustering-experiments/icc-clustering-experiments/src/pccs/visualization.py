"""
Publication-quality visualization module for PCCS:
- Stability degradation curves across families and intensities
- Cluster stability fingerprint heatmaps
- Pairwise interaction bar plots and heatmaps
- Consensus matrix heatmaps
- Comparative benchmark bar charts with error bars
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_robustness_curves(
    robustness_df: pd.DataFrame,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot stability degradation curves across perturbation families and levels."""
    fig, ax = plt.subplots(figsize=(8, 5))
    families = robustness_df["family"].unique()
    markers = ["o", "s", "^", "D", "v", "P"]

    for i, fam in enumerate(families):
        sub = robustness_df[robustness_df["family"] == fam].sort_values("level_index")
        ax.plot(
            sub["level_index"],
            sub["mean_ari"],
            marker=markers[i % len(markers)],
            label=fam,
            linewidth=2,
            markersize=6,
        )
        if "std_ari" in sub.columns:
            ax.fill_between(
                sub["level_index"],
                sub["mean_ari"] - sub["std_ari"].fillna(0),
                sub["mean_ari"] + sub["std_ari"].fillna(0),
                alpha=0.15,
            )

    ax.set_title("Cluster Stability Degradation Across Perturbation Families", fontsize=12, fontweight="bold")
    ax.set_xlabel("Perturbation Intensity Level Index", fontsize=11)
    ax.set_ylabel("Adjusted Rand Index (ARI)", fontsize=11)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(title="Perturbation Family", frameon=True, fontsize=10)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300)
    return fig


def plot_fingerprint_heatmap(
    fingerprint_df: pd.DataFrame,
    families: Tuple[str, ...],
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot cluster stability fingerprints across perturbation families."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    fams = [f for f in families if f in fingerprint_df.columns]
    data = fingerprint_df[fams].to_numpy(dtype=float)
    clusters = [f"Cluster {c} ({r})" for c, r in zip(fingerprint_df["cluster"], fingerprint_df["reliability"])]

    im = ax.imshow(data, cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(fams)))
    ax.set_xticklabels(fams, rotation=30, ha="right", fontsize=10)
    ax.set_yticks(range(len(clusters)))
    ax.set_yticklabels(clusters, fontsize=10)

    # Add numeric annotations
    for i in range(len(clusters)):
        for j in range(len(fams)):
            val = data[i, j]
            color = "black" if val > 0.65 else "white"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontweight="bold", fontsize=9)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Stability Score $S_{k,p}$", fontsize=10)
    ax.set_title("Cluster-Level Stability Fingerprints", fontsize=12, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300)
    return fig


def plot_interactions(
    interaction_df: pd.DataFrame,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot observed vs expected stability and interaction effects."""
    fig, ax = plt.subplots(figsize=(9, 5))
    sub = interaction_df[interaction_df["cluster"] == "global"].sort_values("interaction", ascending=False)
    if len(sub) == 0:
        sub = interaction_df.sort_values("interaction", ascending=False).head(10)

    x = np.arange(len(sub))
    width = 0.35

    ax.bar(x - width/2, sub["S_pq_expected"], width, label="Expected $S_{pq}$ (Independent)", color="#3498db", alpha=0.85)
    ax.bar(x + width/2, sub["S_pq_observed"], width, label="Observed $S_{pq}$ (Joint)", color="#e74c3c", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(sub["pair"], rotation=40, ha="right", fontsize=9)
    ax.set_ylabel("Stability (ARI)", fontsize=11)
    ax.set_title("Perturbation Interactions: Expected vs Observed Joint Stability", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=True)
    ax.grid(True, linestyle="--", alpha=0.4, axis="y")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300)
    return fig


def plot_consensus_matrix(
    consensus_matrix: np.ndarray,
    labels: np.ndarray,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot sorted pairwise consensus co-association matrix."""
    fig, ax = plt.subplots(figsize=(6, 5))
    sort_idx = np.argsort(labels)
    sorted_C = consensus_matrix[np.ix_(sort_idx, sort_idx)]

    im = ax.imshow(sorted_C, cmap="Blues", vmin=0.0, vmax=1.0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Consensus Affinity", fontsize=10)
    ax.set_title("Perturbation-Aware Consensus Matrix", fontsize=12, fontweight="bold")
    ax.set_xlabel("Sorted Observations", fontsize=10)
    ax.set_ylabel("Sorted Observations", fontsize=10)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300)
    return fig


def plot_baseline_comparison(
    comparison_df: pd.DataFrame,
    metric: str = "ARI",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot method comparison bar chart with standard error bars."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    grouped = comparison_df.groupby("method").agg(
        mean_val=(metric, "mean"),
        std_val=(metric, "std"),
    ).reset_index().sort_values("mean_val", ascending=False)

    x = np.arange(len(grouped))
    ax.bar(
        x,
        grouped["mean_val"],
        yerr=grouped["std_val"].fillna(0),
        capsize=4,
        color="#2ecc71",
        edgecolor="#27ae60",
        alpha=0.85,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(grouped["method"], rotation=30, ha="right", fontsize=10)
    ax.set_ylabel(metric, fontsize=11)
    ax.set_title(f"Comparative Benchmark Performance ({metric})", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.grid(True, linestyle="--", alpha=0.4, axis="y")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300)
    return fig
