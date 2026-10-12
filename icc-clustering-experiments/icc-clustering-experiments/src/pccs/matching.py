"""
Partition comparison and cluster matching for PCCS:
- Label-invariant metrics: ARI, AMI, NMI, VI on shared observation subsets.
- Optimal Hungarian matching (maximum overlap / Jaccard).
- Explicit cluster split, merge, and vanishing detection.
- Centroid drift estimation.
- DBSCAN noise-aware handling.
"""
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, normalized_mutual_info_score, mutual_info_score


def variation_of_information(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """Compute Variation of Information (VI)."""
    def entropy(labels):
        _, counts = np.unique(labels, return_counts=True)
        probs = counts / counts.sum()
        return -float((probs * np.log(np.maximum(probs, 1e-12))).sum())

    h_true = entropy(labels_true)
    h_pred = entropy(labels_pred)
    mi = mutual_info_score(labels_true, labels_pred)
    return float(max(0.0, h_true + h_pred - 2.0 * mi))


def partition_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask_noise: bool = True,
) -> Dict[str, float]:
    """
    Compute label-invariant partition metrics on valid observations.
    If mask_noise is True, observations with label -1 in either partition are excluded.
    """
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)
    if mask_noise:
        valid = (y_t >= 0) & (y_p >= 0)
    else:
        valid = np.ones(len(y_t), dtype=bool)

    if valid.sum() < 2:
        return dict(ARI=0.0, AMI=0.0, NMI=0.0, VI=0.0)

    y_tv = y_t[valid]
    y_pv = y_p[valid]

    # Handle single-cluster degenerate case
    if len(np.unique(y_tv)) == 1 or len(np.unique(y_pv)) == 1:
        ari = 0.0 if len(np.unique(y_tv)) != len(np.unique(y_pv)) else 1.0
        ami = 0.0 if len(np.unique(y_tv)) != len(np.unique(y_pv)) else 1.0
        nmi = 0.0 if len(np.unique(y_tv)) != len(np.unique(y_pv)) else 1.0
        vi = variation_of_information(y_tv, y_pv)
        return dict(ARI=float(ari), AMI=float(ami), NMI=float(nmi), VI=float(vi))

    return dict(
        ARI=float(adjusted_rand_score(y_tv, y_pv)),
        AMI=float(adjusted_mutual_info_score(y_tv, y_pv)),
        NMI=float(normalized_mutual_info_score(y_tv, y_pv)),
        VI=float(variation_of_information(y_tv, y_pv)),
    )


def hungarian_match(
    ref_labels: np.ndarray,
    test_labels: np.ndarray,
    K_ref: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Find optimal 1-to-1 matching from reference clusters 0..K_ref-1 to test clusters.
    Uses cost matrix C_kl = -Jaccard(ref_k, test_l).
    Returns (matched_test_cluster_per_ref, jaccard_per_ref, diagnostics).
    """
    valid = (ref_labels >= 0) & (test_labels >= 0)
    if valid.sum() == 0:
        return np.full(K_ref, -1, dtype=int), np.zeros(K_ref, dtype=float), {}

    ref_v = ref_labels[valid]
    test_v = test_labels[valid]

    test_unique = np.unique(test_v)
    L_test = len(test_unique)
    test_map = {c_val: idx for idx, c_val in enumerate(test_unique)}
    test_mapped = np.array([test_map[c] for c in test_v])

    # Contingency matrix: K_ref x L_test
    contingency = np.zeros((K_ref, L_test), dtype=int)
    for r, t in zip(ref_v, test_mapped):
        if 0 <= r < K_ref:
            contingency[r, t] += 1

    ref_sizes = contingency.sum(axis=1)
    test_sizes = contingency.sum(axis=0)

    # Jaccard overlap matrix: |A cap B| / |A cup B|
    denom = ref_sizes[:, None] + test_sizes[None, :] - contingency
    denom = np.maximum(denom, 1)
    jaccard_matrix = contingency / denom

    cost_matrix = -jaccard_matrix
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matched_test = np.full(K_ref, -1, dtype=int)
    matched_jaccard = np.zeros(K_ref, dtype=float)

    for r_idx, c_idx in zip(row_ind, col_ind):
        if r_idx < K_ref and c_idx < L_test:
            matched_test[r_idx] = test_unique[c_idx]
            matched_jaccard[r_idx] = jaccard_matrix[r_idx, c_idx]

    # Split / merge diagnostics
    # A reference cluster is considered split if its members are distributed across multiple test clusters
    split_info = {}
    for r_idx in range(K_ref):
        row_c = contingency[r_idx]
        active_in_test = (row_c > 0.15 * max(1, ref_sizes[r_idx])).sum()
        split_info[r_idx] = int(active_in_test > 1)

    diagnostics = {
        "contingency": contingency,
        "jaccard_matrix": jaccard_matrix,
        "split_clusters": split_info,
    }
    return matched_test, matched_jaccard, diagnostics


def cluster_retention(
    ref_labels_full: np.ndarray,
    Z_full: np.ndarray,
    perturbed_rows: np.ndarray,
    perturbed_cols: np.ndarray,
    perturbed_labels: np.ndarray,
    K_ref: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute cluster-level retention (Hungarian-matched Jaccard) and centroid drift.
    Returns (retention_per_cluster, drift_per_cluster).
    """
    ref_sub = ref_labels_full[perturbed_rows]
    matched_test, ret, _ = hungarian_match(ref_sub, perturbed_labels, K_ref)

    Z_c = Z_full[:, perturbed_cols]
    Z_sub = Z_c[perturbed_rows]

    # Reference centroids on full feature space (restricted to perturbed columns)
    ref_centroids = np.zeros((K_ref, Z_c.shape[1]), dtype=float)
    ref_counts = np.bincount(np.clip(ref_sub, 0, K_ref - 1), minlength=K_ref)
    for k in range(K_ref):
        mask_k = (ref_sub == k)
        if mask_k.any():
            ref_centroids[k] = Z_sub[mask_k].mean(axis=0)

    # Inter-cluster reference separation scale
    sep_list = []
    for i in range(K_ref):
        for j in range(i + 1, K_ref):
            sep_list.append(np.linalg.norm(ref_centroids[i] - ref_centroids[j]))
    sep_scale = float(np.mean(sep_list)) if len(sep_list) > 0 else 1.0
    sep_scale = max(sep_scale, 1e-6)

    drift = np.full(K_ref, np.nan, dtype=float)
    for k in range(K_ref):
        t_cluster = matched_test[k]
        if t_cluster >= 0 and ref_counts[k] > 0:
            mask_t = (perturbed_labels == t_cluster)
            if mask_t.any():
                t_centroid = Z_sub[mask_t].mean(axis=0)
                drift[k] = float(np.linalg.norm(ref_centroids[k] - t_centroid) / sep_scale)

    return ret, drift
