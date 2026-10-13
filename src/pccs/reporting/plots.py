"""
Publication-quality figure generation for PCCS using pure Matplotlib.
All figures load data exclusively from saved raw CSV files.
"""
import os
from typing import Optional
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def set_plot_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.titlesize": 13,
        "grid.color": "#e0e0e0",
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,
    })


def plot_fig1_synthetic_performance(
    csv_path: str = "results/raw/synthetic_benchmark_results.csv",
    output_path: str = "results/figures/fig1_synthetic_noisy_performance.png",
) -> None:
    """Figure 1: Benchmark clustering ARI across structural regimes on synthetic ground truth."""
    if not os.path.exists(csv_path):
        return
    df = pd.read_csv(csv_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    set_plot_style()

    methods = ["kmeans", "fcm", "dbscan", "hierarchical", "ordinary_consensus", "pccs_full"]
    df_sub = df[df["method"].isin(methods)].copy()
    regimes = df_sub["regime"].unique()

    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    x = np.arange(len(regimes))
    width = 0.13

    palette = {
        "kmeans": "#4C72B0",
        "fcm": "#55A868",
        "dbscan": "#C44E52",
        "hierarchical": "#8172B2",
        "ordinary_consensus": "#CCB974",
        "pccs_full": "#D95F02",
    }

    for i, m in enumerate(methods):
        sub_m = df_sub[df_sub["method"] == m]
        means = [sub_m[sub_m["regime"] == r]["ari"].mean() for r in regimes]
        stds = [sub_m[sub_m["regime"] == r]["ari"].std() for r in regimes]
        ax.bar(x + i * width, means, width, yerr=stds, capsize=3, label=m, color=palette.get(m, "#333333"))

    ax.set_ylabel("Adjusted Rand Index (ARI)")
    ax.set_title("Synthetic Ground-Truth Clustering Accuracy Across Structural Regimes", fontweight="bold")
    ax.set_xticks(x + width * (len(methods) - 1) / 2)
    ax.set_xticklabels([r.replace("_", " ").title() for r in regimes], rotation=20, ha="right")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, axis="y")
    ax.legend(title="Algorithm", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_fig2_real_world_benchmarks(
    csv_path: str = "results/raw/real_world_benchmark_results.csv",
    output_path: str = "results/figures/fig2_real_world_benchmarks.png",
) -> None:
    """Figure 2: Real-world benchmark performance across public datasets."""
    if not os.path.exists(csv_path):
        return
    df = pd.read_csv(csv_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    set_plot_style()

    methods = ["kmeans", "fcm", "dbscan", "hierarchical", "ordinary_consensus", "pccs_full"]
    df_sub = df[df["method"].isin(methods)].copy()
    datasets = df_sub["dataset"].unique()

    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    x = np.arange(len(datasets))
    width = 0.13

    for i, m in enumerate(methods):
        sub_m = df_sub[df_sub["method"] == m]
        means = [sub_m[sub_m["dataset"] == d]["ari"].mean() for d in datasets]
        stds = [sub_m[sub_m["dataset"] == d]["ari"].std() for d in datasets]
        ax.bar(x + i * width, means, width, yerr=stds, capsize=3, label=m)

    ax.set_ylabel("Adjusted Rand Index (ARI)")
    ax.set_title("Clustering Accuracy (ARI) on Public Benchmark Datasets", fontweight="bold")
    ax.set_xticks(x + width * (len(methods) - 1) / 2)
    ax.set_xticklabels([d.replace("_", " ").title() for d in datasets])
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, axis="y")
    ax.legend(title="Algorithm", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_fig3_cluster_fingerprints(
    csv_path: str = "results/raw/synthetic_fingerprints.csv",
    output_path: str = "results/figures/fig3_cluster_stability_fingerprints.png",
) -> None:
    """Figure 3: Cluster stability fingerprint heatmaps across 6 perturbation dimensions."""
    if not os.path.exists(csv_path):
        return
    df = pd.read_csv(csv_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    set_plot_style()

    fams = ["algorithmic", "sampling", "feature", "measurement", "technical", "distribution"]
    avail_fams = [f for f in fams if f in df.columns]
    if not avail_fams:
        return

    sub = df[df["regime"] == "rare_cluster"] if "regime" in df.columns else df
    if sub.empty:
        sub = df

    matrix = sub[avail_fams].values
    cluster_labels = [f"Cluster {int(row['cluster'])} ({row.get('reliability_category', '')})" for _, row in sub.iterrows()]

    fig, ax = plt.subplots(figsize=(8, max(3.5, len(sub) * 0.8)), dpi=300)
    cax = ax.imshow(matrix, cmap="YlGnBu", vmin=0.0, vmax=1.0, aspect="auto")

    # Annotate numbers
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            color = "white" if val > 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontweight="bold")

    ax.set_xticks(np.arange(len(avail_fams)))
    ax.set_xticklabels([f.capitalize() for f in avail_fams])
    ax.set_yticks(np.arange(len(cluster_labels)))
    ax.set_yticklabels(cluster_labels)
    ax.set_title(r"Cluster Stability Fingerprints ($F_k$) Across Perturbation Families", fontweight="bold", pad=12)
    fig.colorbar(cax, ax=ax, label=r"Stability Score $S_{kp}$")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_fig4_interactions(
    csv_path: str = "results/raw/synthetic_interactions.csv",
    output_path: str = "results/figures/fig4_perturbation_interactions.png",
) -> None:
    """Figure 4: Factorial perturbation interactions showing observed vs expected stability."""
    if not os.path.exists(csv_path):
        return
    df = pd.read_csv(csv_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    set_plot_style()

    pairs = df["pair"].unique()
    means = [df[df["pair"] == p]["interaction"].mean() for p in pairs]
    stds = [df[df["pair"] == p]["interaction"].std() for p in pairs]

    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    x = np.arange(len(pairs))
    ax.bar(x, means, yerr=stds, capsize=4, color="#4C72B0", edgecolor="#333333")

    ax.axhline(0.15, color="red", linestyle="--", linewidth=1.5, label="Synergistic Threshold (0.15)")
    ax.axhline(0.00, color="black", linestyle="-", linewidth=0.8, alpha=0.5)

    ax.set_ylabel(r"Excess Degradation ($\Delta_{pq} = S_{exp} - S_{obs}$)")
    ax.set_title(r"Factorial Perturbation Interactions ($\Delta_{pq}$)", fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels([p.replace("+", " + ").title() for p in pairs], rotation=20, ha="right")
    ax.grid(True, axis="y")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_fig5_falsification_psi(
    csv_path: str = "results/raw/falsification_results.csv",
    output_path: str = "results/figures/fig5_falsification_psi.png",
) -> None:
    """Figure 5: Falsification null study: PSI distinguishes structured data from null noise."""
    if not os.path.exists(csv_path):
        return
    df = pd.read_csv(csv_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    set_plot_style()

    types = df["dataset_type"].unique()
    means = [df[df["dataset_type"] == t]["mean_psi"].mean() for t in types]
    stds = [df[df["dataset_type"] == t]["mean_psi"].std() for t in types]

    colors = ["#2CA02C" if "Structured" in t else "#D62728" for t in types]

    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    x = np.arange(len(types))
    ax.bar(x, means, yerr=stds, capsize=4, color=colors, edgecolor="#333333", width=0.5)

    ax.axhline(0.35, color="black", linestyle=":", linewidth=1.2, label="Structure Separation Boundary")
    ax.set_ylabel("Mean Perturbation Sensitivity Index (PSI)")
    ax.set_title("Unsupervised Falsification: PSI Discriminates Structure From Nulls", fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(types)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, axis="y")
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def generate_all_publication_figures(
    results_dir: str = "results/raw",
    figures_dir: str = "results/figures",
) -> None:
    """Generate all 5 publication figures from executed raw CSV result files."""
    plot_fig1_synthetic_performance(
        os.path.join(results_dir, "synthetic_benchmark_results.csv"),
        os.path.join(figures_dir, "fig1_synthetic_noisy_performance.png"),
    )
    plot_fig2_real_world_benchmarks(
        os.path.join(results_dir, "real_world_benchmark_results.csv"),
        os.path.join(figures_dir, "fig2_real_world_benchmarks.png"),
    )
    plot_fig3_cluster_fingerprints(
        os.path.join(results_dir, "synthetic_fingerprints.csv"),
        os.path.join(figures_dir, "fig3_cluster_stability_fingerprints.png"),
    )
    plot_fig4_interactions(
        os.path.join(results_dir, "synthetic_interactions.csv"),
        os.path.join(figures_dir, "fig4_perturbation_interactions.png"),
    )
    plot_fig5_falsification_psi(
        os.path.join(results_dir, "falsification_results.csv"),
        os.path.join(figures_dir, "fig5_falsification_psi.png"),
    )
