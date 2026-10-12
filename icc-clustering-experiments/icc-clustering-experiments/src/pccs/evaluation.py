"""
Comprehensive evaluation suite for PCCS:
- External metrics: ARI, AMI, NMI, VI
- Internal metrics: Silhouette, Calinski-Harabasz, Davies-Bouldin
- Robustness degradation curves
- Baseline comparisons and ablation studies
"""
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.metrics import (
    adjusted_rand_score,
    adjusted_mutual_info_score,
    normalized_mutual_info_score,
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
from .matching import partition_metrics


def evaluate_clustering(
    X: np.ndarray,
    labels: np.ndarray,
    y_true: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute full battery of external and internal clustering metrics.
    """
    res: Dict[str, float] = {}
    n = len(labels)
    unique_c = np.unique(labels[labels >= 0])
    k_found = len(unique_c)
    res["k_discovered"] = k_found

    # External evaluation
    if y_true is not None:
        pm = partition_metrics(y_true, labels, mask_noise=True)
        res.update(pm)

    # Internal evaluation (requires at least 2 clusters and non-trivial cluster sizes)
    valid_mask = labels >= 0
    if valid_mask.sum() > k_found and k_found > 1:
        X_val = X[valid_mask]
        lab_val = labels[valid_mask]
        try:
            res["silhouette"] = float(silhouette_score(X_val, lab_val))
        except Exception:
            res["silhouette"] = np.nan
        try:
            res["calinski_harabasz"] = float(calinski_harabasz_score(X_val, lab_val))
        except Exception:
            res["calinski_harabasz"] = np.nan
        try:
            res["davies_bouldin"] = float(davies_bouldin_score(X_val, lab_val))
        except Exception:
            res["davies_bouldin"] = np.nan
    else:
        res["silhouette"] = np.nan
        res["calinski_harabasz"] = np.nan
        res["davies_bouldin"] = np.nan

    return res


def compute_robustness_curves(run_table_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate stability degradation as a function of perturbation family and intensity level.
    """
    singles = run_table_df[run_table_df["kind"] == "single"].copy()
    grouped = singles.groupby(["family", "level_index", "level"]).agg(
        mean_ari=("ari_ref", "mean"),
        std_ari=("ari_ref", "std"),
        mean_ami=("ami_ref", "mean"),
        mean_realized=("realized_intensity", "mean"),
        count=("ari_ref", "count"),
    ).reset_index()
    return grouped
