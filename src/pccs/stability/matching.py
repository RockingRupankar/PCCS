"""
Bipartite cluster matching and permutation-invariant partition metrics.
"""
from typing import Tuple, Dict, Any, Optional
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (
    adjusted_rand_score,
    adjusted_mutual_info_score,
    normalized_mutual_info_score,
)


def hungarian_match(
    ref_labels: np.ndarray,
    cand_labels: np.ndarray,
    n_clusters: int,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Optimal bipartite Hungarian matching between candidate and reference clusters.
    
    Returns:
    - aligned_cand_labels: candidate labels permuted to match reference cluster IDs
    - col_ind: permutation mapping
    - mean_jaccard: mean Jaccard similarity across matched clusters
    """
    K = n_clusters
    cost_matrix = np.zeros((K, K), dtype=float)
    jaccard_matrix = np.zeros((K, K), dtype=float)

    for r in range(K):
        r_mask = (ref_labels == r)
        r_size = np.sum(r_mask)
        for c in range(K):
            c_mask = (cand_labels == c)
            c_size = np.sum(c_mask)
            intersect = np.sum(r_mask & c_mask)
            union = r_size + c_size - intersect
            jaccard = intersect / union if union > 0 else 0.0
            jaccard_matrix[r, c] = jaccard
            cost_matrix[r, c] = 1.0 - jaccard

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Permute candidate labels
    mapping = {c: r for r, c in zip(row_ind, col_ind)}
    aligned_labels = np.array([mapping.get(lbl, -1) if lbl != -1 else -1 for lbl in cand_labels])
    matched_jaccards = [jaccard_matrix[r, c] for r, c in zip(row_ind, col_ind)]

    return aligned_labels, col_ind, float(np.mean(matched_jaccards))


def partition_metrics(
    labels_true: np.ndarray,
    labels_pred: np.ndarray,
) -> Dict[str, float]:
    """Compute permutation-invariant partition comparison metrics."""
    # Exclude noise points (-1) from both if present for clean partition comparison
    valid = (labels_true != -1) & (labels_pred != -1)
    if np.sum(valid) < 2:
        return {"ari": 0.0, "ami": 0.0, "nmi": 0.0, "vi": 0.0}

    y_t = labels_true[valid]
    y_p = labels_pred[valid]

    ari = float(adjusted_rand_score(y_t, y_p))
    ami = float(adjusted_mutual_info_score(y_t, y_p))
    nmi = float(normalized_mutual_info_score(y_t, y_p))

    # Variation of Information (VI = H(X) + H(Y) - 2 * I(X; Y))
    from sklearn.metrics import mutual_info_score
    mi = mutual_info_score(y_t, y_p)
    h_t = mutual_info_score(y_t, y_t)
    h_p = mutual_info_score(y_p, y_p)
    vi = float(max(0.0, h_t + h_p - 2.0 * mi))

    return {"ari": ari, "ami": ami, "nmi": nmi, "vi": vi}


def cluster_retention(
    ref_labels: np.ndarray,
    cand_labels: np.ndarray,
    n_clusters: int,
) -> np.ndarray:
    """Compute per-cluster Jaccard membership retention after Hungarian alignment."""
    aligned, _, _ = hungarian_match(ref_labels, cand_labels, n_clusters)
    ret = np.zeros(n_clusters, dtype=float)
    for k in range(n_clusters):
        r_mask = (ref_labels == k)
        c_mask = (aligned == k)
        intersect = np.sum(r_mask & c_mask)
        union = np.sum(r_mask | c_mask)
        ret[k] = intersect / union if union > 0 else 0.0
    return ret
