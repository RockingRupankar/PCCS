"""
Cluster-level stability fingerprints and Perturbation Sensitivity Index (PSI).
F_k = [S_{k,1}, S_{k,2}, ..., S_{k,P}]
"""
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from pccs.stability.tensor import StabilityTensor
from pccs.stability.matching import hungarian_match, cluster_retention


def compute_cluster_fingerprints(
    reference_labels: np.ndarray,
    tensor: StabilityTensor,
    partitions: List[Dict[str, Any]],
    n_clusters: int,
    X: Optional[np.ndarray] = None,
    weights: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """
    Compute cluster-level stability fingerprints across all perturbation families.
    
    For each cluster k and family p:
    - Retention R_k(p): mean Hungarian Jaccard retention
    - Internal co-association A_k(p): within-cluster mean co-association from tensor
    - Combined stability S_kp: 0.5 * (R_k(p) + A_k(p))
    - Across-run variability: standard deviation of retention
    - Drift: centroid shift in feature space (if X is provided)
    - PSI_k: Perturbation Sensitivity Index sum_p w_p (1 - S_kp)
    """
    K = n_clusters
    families = tensor.families
    weights = weights or {f: 1.0 / len(families) for f in families}

    # Reference cluster sizes
    ref_sizes = [int(np.sum(reference_labels == k)) for k in range(K)]

    # Compute reference centroids if X provided
    ref_centroids = None
    if X is not None and len(X) == len(reference_labels):
        ref_centroids = np.zeros((K, X.shape[1]), dtype=float)
        for k in range(K):
            mask = (reference_labels == k)
            if np.sum(mask) > 0:
                ref_centroids[k] = np.mean(X[mask], axis=0)

    rows = []
    for k in range(K):
        k_mask = (reference_labels == k)
        k_indices = np.where(k_mask)[0]
        n_k = len(k_indices)

        row: Dict[str, Any] = {
            "cluster": k,
            "size": n_k,
        }

        s_vals = []
        sd_vals = []

        for f in families:
            # 1. Retention from partitions belonging to this family
            f_parts = [p for p in partitions if p["family"] == f]
            ret_runs = []
            drift_runs = []

            for p in f_parts:
                p_labels = p["labels"]
                p_rows = p["row_indices"]
                # Align on intersection
                sub_ref = reference_labels[p_rows]
                ret_vector = cluster_retention(sub_ref, p_labels, K)
                ret_runs.append(ret_vector[k])

                if ref_centroids is not None and "features" in p:
                    # Centroid drift
                    k_sub = (p_labels == k)
                    if np.sum(k_sub) > 0:
                        p_cent = np.mean(p["features"][k_sub], axis=0)
                        # Project onto shared columns
                        cols = p.get("col_indices", np.arange(X.shape[1]))
                        drift = np.linalg.norm(ref_centroids[k, cols] - p_cent)
                        drift_runs.append(float(drift))

            mean_ret = float(np.mean(ret_runs)) if ret_runs else 0.5
            std_ret = float(np.std(ret_runs)) if len(ret_runs) > 1 else 0.0

            # 2. Internal co-association from tensor slice
            S_f = tensor.get_slice(f)
            if n_k > 1:
                sub_coassoc = S_f[np.ix_(k_indices, k_indices)]
                # Off-diagonal mean
                sum_off = np.sum(sub_coassoc) - n_k
                mean_co = float(sum_off / (n_k * (n_k - 1)))
            else:
                mean_co = 1.0

            # 3. Combined stability score S_kp
            S_kp = float(0.5 * (mean_ret + mean_co))
            row[f] = S_kp
            row[f"retention_{f}"] = mean_ret
            row[f"coassoc_{f}"] = mean_co
            row[f"sd_{f}"] = std_ret
            if drift_runs:
                row[f"drift_{f}"] = float(np.mean(drift_runs))

            s_vals.append(S_kp)
            sd_vals.append(std_ret)

        row["max_response_sd"] = float(np.max(sd_vals)) if sd_vals else 0.0

        # 4. PSI computation
        psi_k = sum(weights[f] * (1.0 - row[f]) for f in families)
        row["PSI"] = float(psi_k)
        rows.append(row)

    df_fp = pd.DataFrame(rows)
    return df_fp


def compute_psi(
    stability_vector: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> float:
    """Compute Perturbation Sensitivity Index from a vector of family stabilities."""
    P = len(stability_vector)
    if weights is None:
        weights = np.full(P, 1.0 / P)
    else:
        weights = weights / weights.sum()
    return float(np.sum(weights * (1.0 - stability_vector)))
