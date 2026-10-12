#!/usr/bin/env python3
"""
Full reproducible execution entry point for PCCS.
Executes the complete experimental pipeline end-to-end and generates all deliverables.
"""
from pathlib import Path
import sys
import time

# Ensure src and experiments directories are in sys.path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pccs.runner import (
    run_synthetic_experiments,
    run_real_world_experiments,
    run_falsification_experiments,
    compute_statistical_summaries,
)
from pccs.evaluation import compute_robustness_curves
from pccs.classification import threshold_sensitivity_analysis
from pccs.visualization import (
    plot_robustness_curves,
    plot_fingerprint_heatmap,
    plot_interactions,
    plot_consensus_matrix,
    plot_baseline_comparison,
)
from pccs import FAMILIES


def main():
    start_time = time.time()
    workspace_root = ROOT.parents[1] if (ROOT.parents[1] / "README.md").exists() else ROOT
    results_dir = workspace_root / "results"
    raw_dir = results_dir / "raw"
    reports_dir = results_dir / "reports"
    figures_dir = results_dir / "figures"

    for d in (results_dir, raw_dir, reports_dir, figures_dir):
        d.mkdir(parents=True, exist_ok=True)

    print("================================================================================")
    print("STARTING FULL PCCS EXPERIMENTAL REPRODUCTION & VALIDATION PIPELINE")
    print(f"Results directory: {results_dir}")
    print("================================================================================")

    # -------------------------------------------------------------------------
    # STAGE 4 & 6: Synthetic Benchmarks (clean, noisy, severe across 5 seeds)
    # -------------------------------------------------------------------------
    print("\n[STAGE 1/4] Executing Synthetic Benchmark Suite across 3 Profiles and 5 Seeds...")
    synth_results, synth_fps, synth_inters = run_synthetic_experiments(
        output_dir=raw_dir,
        profiles=("clean", "noisy", "severe"),
        seeds=(42, 43, 44, 45, 46),
        n_samples=400,
    )

    # -------------------------------------------------------------------------
    # STAGE 4 & 6: Real-World Benchmarks (Iris, Wine, Breast Cancer, Digits)
    # -------------------------------------------------------------------------
    print("\n[STAGE 2/4] Executing Real-World Benchmark Suite (Iris, Wine, Breast Cancer, Digits)...")
    rw_results, rw_fps = run_real_world_experiments(
        output_dir=raw_dir,
        seeds=(42, 43, 44),
    )

    # -------------------------------------------------------------------------
    # STAGE 6: Falsification Study (Structured vs Isotropic and Anisotropic Nulls)
    # -------------------------------------------------------------------------
    print("\n[STAGE 3/4] Executing Falsification & Null Hypothesis Study...")
    fals_results = run_falsification_experiments(output_dir=raw_dir)

    # -------------------------------------------------------------------------
    # STAGE 7: Statistical Analysis & Inference
    # -------------------------------------------------------------------------
    print("\n[STAGE 4/4] Conducting Statistical Inference & Sensitivity Analysis...")
    stat_summary = compute_statistical_summaries(synth_results, output_dir=reports_dir)
    thresh_sens = threshold_sensitivity_analysis(synth_fps, FAMILIES)
    thresh_sens.to_csv(reports_dir / "threshold_sensitivity_summary.csv", index=False)

    # Summary tables
    synth_summary = synth_results.groupby(["profile", "method"]).agg(
        mean_ARI=("ARI", "mean"),
        std_ARI=("ARI", "std"),
        mean_NMI=("NMI", "mean"),
        mean_runtime=("runtime", "mean"),
    ).reset_index()
    synth_summary.to_csv(reports_dir / "synthetic_summary_table.csv", index=False)

    rw_summary = rw_results.groupby(["dataset", "variant", "method"]).agg(
        mean_ARI=("ARI", "mean"),
        std_ARI=("ARI", "std"),
        mean_NMI=("NMI", "mean"),
        mean_runtime=("runtime", "mean"),
    ).reset_index()
    rw_summary.to_csv(reports_dir / "real_world_summary_table.csv", index=False)

    # -------------------------------------------------------------------------
    # STAGE 8: Independent Result Verification & Figure Generation
    # -------------------------------------------------------------------------
    print("\nGenerating publication-quality figures from saved raw results...")

    # Figure 1: Synthetic Performance Comparison
    plot_baseline_comparison(
        synth_results[synth_results["profile"] == "noisy"],
        metric="ARI",
        save_path=figures_dir / "fig1_synthetic_noisy_performance.png",
    )

    # Figure 2: Real-World Performance
    fig, ax = plt.subplots(figsize=(10, 5))
    rw_plot_data = rw_summary.pivot(index=["dataset", "variant"], columns="method", values="mean_ARI")
    rw_plot_data[["PCCS_Full", "KMeans", "Ward", "EAC"]].plot(kind="bar", ax=ax, colormap="tab10")
    ax.set_title("Real-World Benchmark Evaluation: PCCS vs Baselines (Mean ARI)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Adjusted Rand Index (ARI)", fontsize=11)
    ax.set_xlabel("Dataset & Condition", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.grid(True, linestyle="--", alpha=0.4, axis="y")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    fig.savefig(figures_dir / "fig2_real_world_benchmarks.png", dpi=300)
    plt.close(fig)

    # Figure 3: Cluster Stability Fingerprints
    if len(synth_fps) > 0:
        example_fp = synth_fps[(synth_fps["profile"] == "noisy") & (synth_fps["seed"] == 42)]
        plot_fingerprint_heatmap(
            example_fp,
            FAMILIES,
            save_path=figures_dir / "fig3_cluster_stability_fingerprints.png",
        )
        plt.close()

    # Figure 4: Interactions
    if len(synth_inters) > 0:
        example_inter = synth_inters[(synth_inters["profile"] == "noisy") & (synth_inters["seed"] == 42)]
        plot_interactions(
            example_inter,
            save_path=figures_dir / "fig4_perturbation_interactions.png",
        )
        plt.close()

    # Figure 5: Falsification Study
    fig, ax = plt.subplots(figsize=(8, 4.5))
    fals_grouped = fals_results.groupby(["noise_level", "structure"])["mean_psi"].mean().unstack()
    fals_grouped.plot(marker="o", linewidth=2, ax=ax)
    ax.set_title("Perturbation Sensitivity Index (PSI): Structured Data vs Nulls", fontsize=12, fontweight="bold")
    ax.set_xlabel("Noise Level", fontsize=11)
    ax.set_ylabel("Mean Perturbation Sensitivity Index (PSI)", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    fig.savefig(figures_dir / "fig5_falsification_psi.png", dpi=300)
    plt.close(fig)

    total_time = time.time() - start_time
    print("================================================================================")
    print(f"PIPELINE COMPLETED SUCCESSFULLY in {total_time:.1f} seconds!")
    print(f"Outputs written to: {results_dir}")
    print("================================================================================")


if __name__ == "__main__":
    main()
