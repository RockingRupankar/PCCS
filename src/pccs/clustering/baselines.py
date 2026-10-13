"""
Established baseline clustering and consensus methods for comparative evaluation:
1. Single K-Means
2. Single Fuzzy C-Means (FCM)
3. Single DBSCAN
4. Single Hierarchical Agglomerative Clustering
5. Ordinary Consensus (Evidence Accumulation Clustering, EAC; Fred & Jain 2005)
6. Scalar Stability-Weighted Consensus
7. Two-tier Uncertainty-Weighted Ensemble Clustering (TWEC; Gu et al. 2024)
"""
from typing import Dict, Any, List, Optional
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from pccs.clustering.ensemble import run_kmeans, run_fcm, run_dbscan, run_hierarchical


def run_eac_consensus(
    partitions: List[np.ndarray],
    n_clusters: int,
    linkage_method: str = "average",
) -> np.ndarray:
    """
    Evidence Accumulation Clustering (EAC; Fred & Jain 2005).
    Computes unweighted pairwise co-association matrix and applies hierarchical clustering.
    """
    R = len(partitions)
    if R == 0:
        raise ValueError("Partitions list cannot be empty.")
    N = len(partitions[0])
    C = np.zeros((N, N), dtype=float)

    for p in partitions:
        mask = (p[:, None] == p[None, :]) & (p[:, None] != -1)
        C += mask.astype(float)

    C /= float(R)
    np.fill_diagonal(C, 1.0)

    # Hierarchical clustering on dissimilarity
    dist = np.clip(1.0 - C, 0.0, 1.0)
    np.fill_diagonal(dist, 0.0)
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method=linkage_method)
    labels = fcluster(Z, t=n_clusters, criterion="maxclust") - 1
    return labels


def run_scalar_stability_consensus(
    partitions: List[np.ndarray],
    stability_scores: List[float],
    n_clusters: int,
    linkage_method: str = "average",
) -> np.ndarray:
    """Consensus weighted by a single scalar stability score per partition."""
    R = len(partitions)
    N = len(partitions[0])
    weights = np.maximum(np.array(stability_scores), 1e-4)
    total_w = weights.sum()

    C = np.zeros((N, N), dtype=float)
    for p, w in zip(partitions, weights):
        mask = (p[:, None] == p[None, :]) & (p[:, None] != -1)
        C += w * mask.astype(float)

    C /= total_w
    np.fill_diagonal(C, 1.0)

    dist = np.clip(1.0 - C, 0.0, 1.0)
    np.fill_diagonal(dist, 0.0)
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method=linkage_method)
    labels = fcluster(Z, t=n_clusters, criterion="maxclust") - 1
    return labels


def run_twec_consensus(
    partitions: List[np.ndarray],
    n_clusters: int,
    linkage_method: str = "average",
) -> np.ndarray:
    """
    Two-tier Uncertainty measurement Weighted Ensemble Clustering (TWEC; Gu et al. 2024).
    Tier 1: Entropy-based cluster uncertainty within base partitions.
    Tier 2: Cluster reliability weighting converted into pairwise affinity.
    """
    R = len(partitions)
    N = len(partitions[0])
    C = np.zeros((N, N), dtype=float)
    W_sum = np.zeros((N, N), dtype=float)

    for p in partitions:
        unique_labels = [lbl for lbl in np.unique(p) if lbl != -1]
        cluster_reliabilities = {}
        for lbl in unique_labels:
            sz = np.sum(p == lbl)
            p_frac = sz / N
            # Tier 1: entropy-based cluster uncertainty
            ent = -p_frac * np.log(p_frac + 1e-10)
            # Tier 2: cluster reliability
            cluster_reliabilities[lbl] = float(np.exp(-ent))

        for lbl in unique_labels:
            rel = cluster_reliabilities[lbl]
            idx = np.where(p == lbl)[0]
            if len(idx) > 1:
                ix_grid = np.ix_(idx, idx)
                C[ix_grid] += rel
                W_sum[ix_grid] += 1.0

    W_denom = np.maximum(W_sum, 1e-8)
    C_norm = np.where(W_sum > 0, C / W_denom, 0.0)
    np.fill_diagonal(C_norm, 1.0)

    dist = np.clip(1.0 - C_norm, 0.0, 1.0)
    np.fill_diagonal(dist, 0.0)
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method=linkage_method)
    labels = fcluster(Z, t=n_clusters, criterion="maxclust") - 1
    return labels
