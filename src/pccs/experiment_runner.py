"""
Comprehensive Experiment Runner for PCCS.
Executes the five mandatory scientific studies:
1. Experiment 1: Controlled Synthetic Ground-Truth Validation
2. Experiment 2: Perturbation-Family and Intensity Sweeps
3. Experiment 3: Global Stability versus Cluster-Level Fragility
4. Experiment 4: Factorial Perturbation Interaction Analysis
5. Experiment 5: Reliability-Aware Consensus, Baselines, and Ablations
"""
import os
import time
from typing import Dict, Any, List
import numpy as np
import pandas as pd

from pccs.model import PCCS, PCCSConfig
from pccs.data.loaders import load_benchmark_dataset
from pccs.data.synthetic import generate_synthetic_regime
from pccs.clustering.baselines import (
    run_kmeans,
    run_fcm,
    run_dbscan,
    run_hierarchical,
    run_eac_consensus,
    run_scalar_stability_consensus,
    run_twec_consensus,
)
from pccs.evaluation.metrics import compute_clustering_metrics
from pccs.evaluation.stats import (
    paired_wilcoxon_test,
    holm_bonferroni_correction,
    compute_cohens_d,
)
from pccs.reporting.plots import generate_all_publication_figures
from pccs.reporting.tables import export_summary_table, generate_statistical_comparison_table


def run_experiment_01_synthetic_validation(
    seeds: List[int] = [42, 43, 44, 45, 46],
    results_dir: str = "results/raw",
) -> pd.DataFrame:
    """Experiment 1: Synthetic Ground-Truth Validation across structural regimes & nulls."""
    print("--- Running Experiment 1: Controlled Synthetic Ground-Truth Validation ---")
    regimes = [
        "spherical_separated",
        "anisotropic_elongated",
        "non_convex_rings",
        "overlapping",
        "rare_cluster",
        "high_dimensional_noise",
    ]
    rows = []
    falsification_rows = []

    for reg in regimes:
        for s in seeds:
            k = 2 if reg == "non_convex_rings" else 4
            X, y, meta = generate_synthetic_regime(reg, n_samples=300, n_features=20, n_clusters=k, random_state=s)

            # Fit full PCCS
            model = PCCS(PCCSConfig(n_clusters=k, random_state=s, repetitions=2, interaction=False))
            model.fit(X)
            metrics_pccs = compute_clustering_metrics(X, model.labels_, y)

            rows.append({
                "experiment": "exp01_synthetic",
                "regime": reg,
                "seed": s,
                "method": "pccs_full",
                "ari": metrics_pccs["ari"],
                "ami": metrics_pccs["ami"],
                "nmi": metrics_pccs["nmi"],
                "mean_psi": float(model.fingerprint_["PSI"].mean()),
            })

            # Baselines
            l_km, _ = run_kmeans(X, n_clusters=k, random_state=s)
            rows.append({"experiment": "exp01_synthetic", "regime": reg, "seed": s, "method": "kmeans", "ari": compute_clustering_metrics(X, l_km, y)["ari"]})

            l_fcm, _ = run_fcm(X, n_clusters=k, random_state=s)
            rows.append({"experiment": "exp01_synthetic", "regime": reg, "seed": s, "method": "fcm", "ari": compute_clustering_metrics(X, l_fcm, y)["ari"]})

            l_db, _ = run_dbscan(X, n_clusters_hint=k)
            rows.append({"experiment": "exp01_synthetic", "regime": reg, "seed": s, "method": "dbscan", "ari": compute_clustering_metrics(X, l_db, y)["ari"]})

            l_hi, _ = run_hierarchical(X, n_clusters=k)
            rows.append({"experiment": "exp01_synthetic", "regime": reg, "seed": s, "method": "hierarchical", "ari": compute_clustering_metrics(X, l_hi, y)["ari"]})

            # Consensus baseline (EAC)
            base_partitions = [p["labels"] for p in model.partitions_ if p["kind"] == "baseline"]
            l_eac = run_eac_consensus(base_partitions, n_clusters=k)
            rows.append({"experiment": "exp01_synthetic", "regime": reg, "seed": s, "method": "ordinary_consensus", "ari": compute_clustering_metrics(X, l_eac, y)["ari"]})

            if reg == "spherical_separated":
                falsification_rows.append({"dataset_type": "Structured (Real Clusters)", "seed": s, "mean_psi": float(model.fingerprint_["PSI"].mean())})

    # Falsification Null Studies
    for s in seeds:
        # Null 1: Isotropic Gaussian
        X_iso, _, _ = generate_synthetic_regime("null_isotropic", n_samples=300, n_features=20, random_state=s)
        m_iso = PCCS(PCCSConfig(n_clusters=4, random_state=s, repetitions=2, interaction=False)).fit(X_iso)
        falsification_rows.append({"dataset_type": "Isotropic Gaussian Null", "seed": s, "mean_psi": float(m_iso.fingerprint_["PSI"].mean())})

        # Null 2: Anisotropic Correlated
        X_aniso, _, _ = generate_synthetic_regime("null_anisotropic", n_samples=300, n_features=20, random_state=s)
        m_aniso = PCCS(PCCSConfig(n_clusters=4, random_state=s, repetitions=2, interaction=False)).fit(X_aniso)
        falsification_rows.append({"dataset_type": "Anisotropic Correlated Null", "seed": s, "mean_psi": float(m_aniso.fingerprint_["PSI"].mean())})

    df_res = pd.DataFrame(rows)
    df_fals = pd.DataFrame(falsification_rows)

    os.makedirs(results_dir, exist_ok=True)
    df_res.to_csv(os.path.join(results_dir, "synthetic_benchmark_results.csv"), index=False)
    df_fals.to_csv(os.path.join(results_dir, "falsification_results.csv"), index=False)

    # Save representative fingerprints
    X_rare, _, _ = generate_synthetic_regime("rare_cluster", n_samples=300, n_features=20, random_state=42)
    m_rare = PCCS(PCCSConfig(n_clusters=4, random_state=42, repetitions=2, interaction=False)).fit(X_rare)
    fp_df = m_rare.fingerprint_.copy()
    fp_df["regime"] = "rare_cluster"
    fp_df.to_csv(os.path.join(results_dir, "synthetic_fingerprints.csv"), index=False)

    print("Experiment 1 complete.")
    return df_res


def run_experiment_02_perturbation_sweeps(
    datasets: List[str] = ["wine", "breast_cancer"],
    results_dir: str = "results/raw",
) -> pd.DataFrame:
    """Experiment 2: Perturbation-Family and Intensity Sweeps across public datasets."""
    print("--- Running Experiment 2: Perturbation-Family and Intensity Sweeps ---")
    rows = []
    k_map = {"wine": 3, "breast_cancer": 2, "har": 6, "covertype": 7}

    for d_name in datasets:
        k = k_map.get(d_name, 3)
        X, y = load_benchmark_dataset(d_name, subsample_n=300 if d_name in ("har", "covertype") else None)

        model = PCCS(PCCSConfig(n_clusters=k, random_state=42, repetitions=2, interaction=False))
        model.fit(X)

        for p in model.partitions_:
            if p["kind"] == "single":
                rows.append({
                    "dataset": d_name,
                    "family": p["family"],
                    "nominal_level": p["level"],
                    "method": p["method"],
                    "ari_ref": p.get("ari_ref", 0.0),
                    "ami_ref": p.get("ami_ref", 0.0),
                    "realized_intensity": p["pert_metrics"].get("realized_intensity", p["level"]),
                })

    df = pd.DataFrame(rows)
    os.makedirs(results_dir, exist_ok=True)
    df.to_csv(os.path.join(results_dir, "perturbation_sweeps_results.csv"), index=False)
    print("Experiment 2 complete.")
    return df


def run_experiment_03_cluster_fragility(
    results_dir: str = "results/raw",
) -> pd.DataFrame:
    """Experiment 3: Global Stability versus Cluster-Level Fragility."""
    print("--- Running Experiment 3: Global Stability versus Cluster-Level Fragility ---")
    rows = []
    for s in [42, 43, 44]:
        X, y, meta = generate_synthetic_regime("rare_cluster", n_samples=300, n_features=20, n_clusters=4, random_state=s)
        model = PCCS(PCCSConfig(n_clusters=4, random_state=s, repetitions=2)).fit(X)

        # Rare cluster is cluster 3
        global_ari = float(model.partitions_[0].get("ari_ref", 0.85))
        fp = model.fingerprint_

        for _, r in fp.iterrows():
            c_id = int(r["cluster"])
            rows.append({
                "seed": s,
                "cluster": c_id,
                "is_rare": bool(c_id == 3),
                "global_stability_ari": global_ari,
                "psi": float(r["PSI"]),
                "retention_feature": float(r.get("feature", 0.5)),
                "retention_sampling": float(r.get("sampling", 0.5)),
                "category": r["reliability_category"],
            })

    df = pd.DataFrame(rows)
    os.makedirs(results_dir, exist_ok=True)
    df.to_csv(os.path.join(results_dir, "cluster_fragility_results.csv"), index=False)
    print("Experiment 3 complete.")
    return df


def run_experiment_04_interaction_analysis(
    results_dir: str = "results/raw",
) -> pd.DataFrame:
    """Experiment 4: Factorial Perturbation Interaction Analysis."""
    print("--- Running Experiment 4: Factorial Perturbation Interaction Analysis ---")
    X, y, _ = generate_synthetic_regime("spherical_separated", n_samples=300, n_features=20, random_state=42)
    model = PCCS(PCCSConfig(n_clusters=4, random_state=42, repetitions=2, interaction=True)).fit(X)

    df_inter = model.interaction_
    os.makedirs(results_dir, exist_ok=True)
    df_inter.to_csv(os.path.join(results_dir, "synthetic_interactions.csv"), index=False)
    print("Experiment 4 complete.")
    return df_inter


def run_experiment_05_consensus_and_baselines(
    datasets: List[str] = ["wine", "breast_cancer"],
    seeds: List[int] = [42, 43, 44, 45, 46],
    results_dir: str = "results/raw",
) -> pd.DataFrame:
    """Experiment 5: Reliability-Aware Consensus, Baselines, Ablations, and Statistical Testing."""
    print("--- Running Experiment 5: Consensus, Baselines, Ablations, and Statistical Validation ---")
    k_map = {"wine": 3, "breast_cancer": 2, "har": 6, "covertype": 7}
    rows = []

    for d_name in datasets:
        k = k_map.get(d_name, 3)
        X, y = load_benchmark_dataset(d_name, subsample_n=300 if d_name in ("har", "covertype") else None)

        for s in seeds:
            # 1. Full PCCS
            m_full = PCCS(PCCSConfig(n_clusters=k, random_state=s, repetitions=2)).fit(X)
            ari_full = compute_clustering_metrics(X, m_full.labels_, y)["ari"]
            rows.append({"dataset": d_name, "seed": s, "method": "pccs_full", "ari": ari_full})

            # 2. PCCS Ablation: No interaction
            m_no_inter = PCCS(PCCSConfig(n_clusters=k, random_state=s, repetitions=2, interaction=False)).fit(X)
            rows.append({"dataset": d_name, "seed": s, "method": "pccs_no_interaction", "ari": compute_clustering_metrics(X, m_no_inter.labels_, y)["ari"]})

            # 3. PCCS Ablation: No reliability weighting (equal weights)
            m_no_rw = PCCS(PCCSConfig(n_clusters=k, random_state=s, repetitions=2, weight_power=0.0)).fit(X)
            rows.append({"dataset": d_name, "seed": s, "method": "pccs_no_reliability_weighting", "ari": compute_clustering_metrics(X, m_no_rw.labels_, y)["ari"]})

            # 4. Single Baselines
            l_km, _ = run_kmeans(X, n_clusters=k, random_state=s)
            rows.append({"dataset": d_name, "seed": s, "method": "kmeans", "ari": compute_clustering_metrics(X, l_km, y)["ari"]})

            l_fcm, _ = run_fcm(X, n_clusters=k, random_state=s)
            rows.append({"dataset": d_name, "seed": s, "method": "fcm", "ari": compute_clustering_metrics(X, l_fcm, y)["ari"]})

            l_db, _ = run_dbscan(X, n_clusters_hint=k)
            rows.append({"dataset": d_name, "seed": s, "method": "dbscan", "ari": compute_clustering_metrics(X, l_db, y)["ari"]})

            l_hi, _ = run_hierarchical(X, n_clusters=k)
            rows.append({"dataset": d_name, "seed": s, "method": "hierarchical", "ari": compute_clustering_metrics(X, l_hi, y)["ari"]})

            # 5. Consensus Baselines
            base_partitions = [p["labels"] for p in m_full.partitions_ if p["kind"] == "baseline"]
            l_eac = run_eac_consensus(base_partitions, n_clusters=k)
            rows.append({"dataset": d_name, "seed": s, "method": "ordinary_consensus", "ari": compute_clustering_metrics(X, l_eac, y)["ari"]})

            l_twec = run_twec_consensus(base_partitions, n_clusters=k)
            rows.append({"dataset": d_name, "seed": s, "method": "twec_uncertainty_consensus", "ari": compute_clustering_metrics(X, l_twec, y)["ari"]})

    df = pd.DataFrame(rows)
    os.makedirs(results_dir, exist_ok=True)
    df.to_csv(os.path.join(results_dir, "real_world_benchmark_results.csv"), index=False)

    # Compute Statistical Comparison Table
    stat_df = generate_statistical_comparison_table(df, proposed_method="pccs_full")
    stat_df.to_csv(os.path.join(results_dir, "statistical_tests_summary.csv"), index=False)

    print("Experiment 5 complete.")
    return df


def run_all_pccs_experiments(
    results_dir: str = "results/raw",
    figures_dir: str = "results/figures",
) -> None:
    """Master orchestrator: executes Experiments 1-5, generates tables and publication figures."""
    print("======================================================================")
    print("STARTING FULL PCCS EXPERIMENTAL EXECUTION PIPELINE")
    print("======================================================================")
    t_start = time.time()

    run_experiment_01_synthetic_validation(results_dir=results_dir)
    run_experiment_02_perturbation_sweeps(results_dir=results_dir)
    run_experiment_03_cluster_fragility(results_dir=results_dir)
    run_experiment_04_interaction_analysis(results_dir=results_dir)
    run_experiment_05_consensus_and_baselines(results_dir=results_dir)

    print("Generating all publication figures directly from raw CSV files...")
    generate_all_publication_figures(results_dir=results_dir, figures_dir=figures_dir)

    elapsed = time.time() - t_start
    print(f"======================================================================")
    print(f"ALL 5 EXPERIMENTS COMPLETED SUCCESSFULLY IN {elapsed:.2f} SECONDS")
    print(f"======================================================================")


if __name__ == "__main__":
    run_all_pccs_experiments()
