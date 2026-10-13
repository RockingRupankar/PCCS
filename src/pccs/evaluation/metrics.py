"""
Evaluation metrics for clustering solutions: external validity, internal cohesion, and morphology.
"""
from typing import Dict, Any, Optional
import numpy as np
from sklearn.metrics import (
    adjusted_rand_score,
    adjusted_mutual_info_score,
    normalized_mutual_info_score,
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    mutual_info_score,
)


def compute_clustering_metrics(
    X: np.ndarray,
    labels: np.ndarray,
    labels_true: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Compute comprehensive clustering metrics:
    - External: ARI, AMI, NMI, Variation of Information (VI) if reference labels are available.
    - Internal: Silhouette, Calinski-Harabasz, Davies-Bouldin.
    - Structural: number of clusters, size entropy, noise points.
    """
    metrics: Dict[str, Any] = {}
    N = len(labels)
    unique_labels = np.unique(labels)
    n_clusters = len([lbl for lbl in unique_labels if lbl != -1])
    metrics["n_clusters"] = n_clusters
    metrics["n_noise"] = int(np.sum(labels == -1))
    metrics["noise_fraction"] = float(metrics["n_noise"] / N)

    # 1. External Evaluation (Strictly when labels_true is provided)
    if labels_true is not None:
        valid = (labels != -1) & (labels_true != -1)
        if np.sum(valid) > 1:
            y_t = labels_true[valid]
            y_p = labels[valid]
            metrics["ari"] = float(adjusted_rand_score(y_t, y_p))
            metrics["ami"] = float(adjusted_mutual_info_score(y_t, y_p))
            metrics["nmi"] = float(normalized_mutual_info_score(y_t, y_p))
            
            # VI
            mi = mutual_info_score(y_t, y_p)
            h_t = mutual_info_score(y_t, y_t)
            h_p = mutual_info_score(y_p, y_p)
            metrics["vi"] = float(max(0.0, h_t + h_p - 2.0 * mi))
        else:
            metrics["ari"] = 0.0
            metrics["ami"] = 0.0
            metrics["nmi"] = 0.0
            metrics["vi"] = 0.0

    # 2. Internal Evaluation (Only when 2 <= n_clusters < N and no noise issues)
    valid_idx = (labels != -1)
    if n_clusters >= 2 and np.sum(valid_idx) > n_clusters:
        X_val = X[valid_idx]
        l_val = labels[valid_idx]
        try:
            metrics["silhouette"] = float(silhouette_score(X_val, l_val))
        except Exception:
            metrics["silhouette"] = np.nan
        try:
            metrics["calinski_harabasz"] = float(calinski_harabasz_score(X_val, l_val))
        except Exception:
            metrics["calinski_harabasz"] = np.nan
        try:
            metrics["davies_bouldin"] = float(davies_bouldin_score(X_val, l_val))
        except Exception:
            metrics["davies_bouldin"] = np.nan
    else:
        metrics["silhouette"] = np.nan
        metrics["calinski_harabasz"] = np.nan
        metrics["davies_bouldin"] = np.nan

    return metrics
