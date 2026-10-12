"""
Master experiment runner and orchestration pipeline for PCCS.
Executes the full research protocol:
1. Synthetic benchmark suites across clean, noisy, and severe corruption profiles.
2. Complete heterogeneous baselines (K-Means, FCM, DBSCAN, Ward, EAC) and PCCS ablations.
3. Real-world benchmark evaluation on Iris, Wine, Breast Cancer, and Digits.
4. Falsification study on structured vs isotropic and anisotropic nulls.
5. Multi-level perturbation interaction analysis with bootstrap uncertainty.
6. Statistical significance tests, effect sizes, and sensitivity analyses.
7. Generates raw CSVs, summary tables, and publication-ready figures.
"""
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits

from .model import PCCS, PCCSConfig
from .ensemble import cluster_single
from .data import validate_and_preprocess
from .matching import partition_metrics
from .evaluation import evaluate_clustering, compute_robustness_curves
from .stats import bootstrap_ci, paired_comparison_test, holm_bonferroni
from .classification import threshold_sensitivity_analysis
from .visualization import (
    plot_robustness_curves,
    plot_fingerprint_heatmap,
    plot_interactions,
    plot_consensus_matrix,
    plot_baseline_comparison,
)

# Import synthetic generators from experiments/common if available
from common.datasets import make_pccs, corrupt
from common.baselines import run_method


def run_synthetic_experiments(
    output_dir: Path,
    profiles: Tuple[str, ...] = ("clean", "noisy", "severe"),
    seeds: Tuple[int, ...] = (42, 43, 44, 45, 46),
    n_samples: int = 400,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Execute synthetic benchmark suite comparing Full PCCS against all baselines and ablations.
    """
    raw_results = []
    fingerprint_records = []
    interaction_records = []

    ablation_variants = {
        "full_pccs": {},
        "no_reliability_weights": dict(reliability_weights=False),
        "no_invariance_core (lambda=0)": dict(lam=0.0),
        "invariance_only (lambda=1)": dict(lam=1.0),
        "pooled_unweighted": dict(per_family=False),
        "equal_family_weights": dict(family_weighting="equal"),
        "kmeans_only": dict(methods=("kmeans",)),
        "without_dbscan": dict(methods=("kmeans", "fcm", "hierarchical")),
    }

    for prof in profiles:
        print(f"--> Running synthetic suite: profile={prof} across {len(seeds)} seeds...")
        for sd in seeds:
            t0 = time.time()
            X, y, names, df = make_pccs(prof, n_samples, seed=sd)
            cfg = PCCSConfig(
                n_clusters=4,
                repetitions=2,
                baseline_seeds=2,
                interaction_repetitions=2,
                random_state=sd,
            )
            m = PCCS(cfg).fit(X)
            elapsed = time.time() - t0

            # 1. Evaluate Full PCCS
            eval_full = evaluate_clustering(m.Z_, m.labels_, y_true=y)
            raw_results.append(dict(
                dataset="synthetic_pccs",
                profile=prof,
                seed=sd,
                method="PCCS_Full",
                runtime=elapsed,
                **eval_full
            ))

            # 2. Evaluate Baseline Reference Partition
            eval_ref = evaluate_clustering(m.Z_, m.ref_, y_true=y)
            raw_results.append(dict(
                dataset="synthetic_pccs",
                profile=prof,
                seed=sd,
                method="Baseline_Reference_Partition",
                runtime=m.timings_.get("baseline", 0.0),
                **eval_ref
            ))

            # 3. Evaluate Base Methods Independently
            for b_name in ("kmeans", "fcm", "dbscan", "hierarchical"):
                t_b = time.time()
                res = cluster_single(b_name, m.Z_, n_clusters=4, seed=sd)
                eval_b = evaluate_clustering(m.Z_, res.labels, y_true=y)
                raw_results.append(dict(
                    dataset="synthetic_pccs",
                    profile=prof,
                    seed=sd,
                    method=f"Base_{b_name}",
                    runtime=time.time() - t_b,
                    **eval_b
                ))

            # 4. Evaluate EAC (Evidence Accumulation Clustering)
            t_eac = time.time()
            lab_eac, _ = run_method("eac", X, None, 4, seed=sd)
            eval_eac = evaluate_clustering(m.Z_, lab_eac, y_true=y)
            raw_results.append(dict(
                dataset="synthetic_pccs",
                profile=prof,
                seed=sd,
                method="EAC_Consensus",
                runtime=time.time() - t_eac,
                **eval_eac
            ))

            # 5. Evaluate PCCS Ablations
            for var_name, var_kw in ablation_variants.items():
                if var_name == "full_pccs":
                    continue
                lab_var = m.build_consensus(**var_kw)
                eval_var = evaluate_clustering(m.Z_, lab_var, y_true=y)
                raw_results.append(dict(
                    dataset="synthetic_pccs",
                    profile=prof,
                    seed=sd,
                    method=f"Ablation_{var_name}",
                    runtime=elapsed,
                    **eval_var
                ))

            # Store fingerprints and interactions
            fp = m.fingerprint_.copy()
            fp.insert(0, "profile", prof)
            fp.insert(1, "seed", sd)
            fingerprint_records.append(fp)

            if len(m.interaction_) > 0:
                inter = m.interaction_.copy()
                inter.insert(0, "profile", prof)
                inter.insert(1, "seed", sd)
                interaction_records.append(inter)

    results_df = pd.DataFrame(raw_results)
    fp_df = pd.concat(fingerprint_records, ignore_index=True)
    inter_df = pd.concat(interaction_records, ignore_index=True) if interaction_records else pd.DataFrame()

    results_df.to_csv(output_dir / "synthetic_benchmark_results.csv", index=False)
    fp_df.to_csv(output_dir / "synthetic_fingerprints.csv", index=False)
    if len(inter_df) > 0:
        inter_df.to_csv(output_dir / "synthetic_interactions.csv", index=False)

    return results_df, fp_df, inter_df


def run_real_world_experiments(
    output_dir: Path,
    seeds: Tuple[int, ...] = (42, 43, 44),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Execute real-world benchmark suite across Iris, Wine, Breast Cancer, and Digits.
    Both original and corrupted versions are evaluated.
    """
    datasets = {
        "iris": load_iris(return_X_y=True),
        "wine": load_wine(return_X_y=True),
        "breast_cancer": load_breast_cancer(return_X_y=True),
        "digits": load_digits(return_X_y=True),
    }

    raw_results = []
    fps = []

    for name, (X_raw, y_raw) in datasets.items():
        k = len(np.unique(y_raw))
        # For digits (N=1797), use PCA retaining 90% variance to guarantee reasonable execution time
        pca_var = 0.90 if name == "digits" else None

        for variant in ("original", "corrupted"):
            print(f"--> Running real-world suite: dataset={name}, variant={variant}...")
            if variant == "corrupted":
                # Inject standardized noise and 20 pure noise dimensions
                rng = np.random.default_rng(7)
                noise_feats = rng.normal(size=(len(X_raw), 20))
                X_mat = np.hstack([X_raw, noise_feats])
                # Add 3% missingness
                mask = rng.random(X_mat.shape) < 0.03
                X_mat = np.where(mask, np.nan, X_mat)
            else:
                X_mat = X_raw.copy()

            for sd in seeds:
                # 1. PCCS
                t0 = time.time()
                cfg = PCCSConfig(
                    n_clusters=k,
                    repetitions=2,
                    baseline_seeds=2,
                    interaction_repetitions=1,
                    pca_variance=pca_var,
                    random_state=sd,
                )
                m = PCCS(cfg).fit(X_mat)
                elapsed = time.time() - t0

                eval_pccs = evaluate_clustering(m.Z_, m.labels_, y_true=y_raw)
                raw_results.append(dict(
                    dataset=name,
                    variant=variant,
                    seed=sd,
                    n_samples=len(y_raw),
                    n_features=X_mat.shape[1],
                    k=k,
                    method="PCCS_Full",
                    runtime=elapsed,
                    **eval_pccs
                ))

                if sd == seeds[0]:
                    fp = m.fingerprint_.copy()
                    fp.insert(0, "dataset", name)
                    fp.insert(1, "variant", variant)
                    fps.append(fp)

                # 2. K-Means baseline
                t_km = time.time()
                res_km = cluster_single("kmeans", m.Z_, n_clusters=k, seed=sd)
                eval_km = evaluate_clustering(m.Z_, res_km.labels, y_true=y_raw)
                raw_results.append(dict(
                    dataset=name,
                    variant=variant,
                    seed=sd,
                    n_samples=len(y_raw),
                    n_features=X_mat.shape[1],
                    k=k,
                    method="KMeans",
                    runtime=time.time() - t_km,
                    **eval_km
                ))

                # 3. Hierarchical Ward baseline
                t_w = time.time()
                res_w = cluster_single("hierarchical", m.Z_, n_clusters=k, seed=sd, hierarchical_linkage="ward")
                eval_w = evaluate_clustering(m.Z_, res_w.labels, y_true=y_raw)
                raw_results.append(dict(
                    dataset=name,
                    variant=variant,
                    seed=sd,
                    n_samples=len(y_raw),
                    n_features=X_mat.shape[1],
                    k=k,
                    method="Ward",
                    runtime=time.time() - t_w,
                    **eval_w
                ))

                # 4. Evidence Accumulation Consensus (EAC)
                t_eac = time.time()
                lab_eac, _ = run_method("eac", m.Z_, None, k, seed=sd)
                eval_eac = evaluate_clustering(m.Z_, lab_eac, y_true=y_raw)
                raw_results.append(dict(
                    dataset=name,
                    variant=variant,
                    seed=sd,
                    n_samples=len(y_raw),
                    n_features=X_mat.shape[1],
                    k=k,
                    method="EAC",
                    runtime=time.time() - t_eac,
                    **eval_eac
                ))

    rw_df = pd.DataFrame(raw_results)
    rw_fp_df = pd.concat(fps, ignore_index=True) if fps else pd.DataFrame()

    rw_df.to_csv(output_dir / "real_world_benchmark_results.csv", index=False)
    if len(rw_fp_df) > 0:
        rw_fp_df.to_csv(output_dir / "real_world_fingerprints.csv", index=False)

    return rw_df, rw_fp_df


def run_falsification_experiments(output_dir: Path) -> pd.DataFrame:
    """
    Falsification study comparing structured clusters vs isotropic and anisotropic nulls.
    Evaluates whether multi-dimensional stability profiles distinguish real structure from noise
    without requiring supervised classifiers!
    """
    print("--> Running falsification study (structured vs nulls)...")
    records = []
    noise_levels = (0.25, 0.5, 1.0, 1.5)

    for lv in noise_levels:
        for sd in range(5):
            rng = np.random.default_rng(2000 + sd + int(lv * 100))

            # 1. Structured data (4 clusters)
            c_centers = rng.normal(scale=5.0, size=(4, 10))
            y_struct = rng.integers(0, 4, size=200)
            X_struct = c_centers[y_struct] + rng.normal(scale=lv, size=(200, 10))

            # 2. Isotropic null (unimodal Gaussian)
            X_iso = rng.normal(scale=lv, size=(200, 10))

            # 3. Anisotropic elongated null (correlated Gaussian without clusters)
            cov = np.eye(10)
            cov[0, 1] = cov[1, 0] = 0.85
            cov[0, 2] = cov[2, 0] = 0.70
            X_aniso = rng.multivariate_normal(np.zeros(10), cov, size=200) * lv

            for name, X_cand in [("structured", X_struct), ("null_iso", X_iso), ("null_aniso", X_aniso)]:
                m = PCCS(PCCSConfig(n_clusters=4, repetitions=2, baseline_seeds=2, interaction=False, random_state=sd)).fit(X_cand)
                fp = m.fingerprint_
                mean_stab = float(fp["mean_stability"].mean())
                min_stab = float(fp["min_stability"].mean())
                psi_mean = float(fp["PSI"].mean())
                sil = m.summary_["consensus_quality"].get("consensus_silhouette", np.nan)

                records.append(dict(
                    structure=name,
                    noise_level=lv,
                    seed=sd,
                    mean_stability=mean_stab,
                    min_stability=min_stab,
                    mean_psi=psi_mean,
                    consensus_silhouette=sil,
                    pct_globally_or_conditionally_invariant=float((fp["reliability"].isin(["globally_invariant", "conditionally_invariant"])).mean()),
                ))

    df_fals = pd.DataFrame(records)
    df_fals.to_csv(output_dir / "falsification_results.csv", index=False)
    return df_fals


def compute_statistical_summaries(
    synthetic_results: pd.DataFrame,
    output_dir: Path,
) -> pd.DataFrame:
    """
    Perform statistical significance tests (paired comparisons against PCCS) with Holm-Bonferroni correction.
    """
    methods = [m for m in synthetic_results["method"].unique() if m != "PCCS_Full"]
    records = []

    for prof in synthetic_results["profile"].unique():
        sub_prof = synthetic_results[synthetic_results["profile"] == prof]
        pccs_scores = sub_prof[sub_prof["method"] == "PCCS_Full"].sort_values("seed")["ARI"].to_numpy()

        p_vals = []
        comp_data = []

        for m_name in methods:
            m_scores = sub_prof[sub_prof["method"] == m_name].sort_values("seed")["ARI"].to_numpy()
            comp = paired_comparison_test(pccs_scores, m_scores)
            comp_data.append((m_name, comp))
            p_vals.append(comp["p_val_wilcoxon"])

        adj_p_vals = holm_bonferroni(p_vals)

        for (m_name, comp), adj_p in zip(comp_data, adj_p_vals):
            records.append(dict(
                profile=prof,
                baseline_method=m_name,
                mean_pccs=comp["mean_a"],
                mean_baseline=comp["mean_b"],
                difference=comp["mean_diff"],
                cohens_d=comp["cohens_d"],
                p_value_raw=comp["p_val_wilcoxon"],
                p_value_holm_adjusted=adj_p,
                significant_alpha_05=(adj_p < 0.05),
            ))

    stat_df = pd.DataFrame(records)
    stat_df.to_csv(output_dir / "statistical_tests_summary.csv", index=False)
    return stat_df
