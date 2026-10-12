"""
Perturbation-aware consensus construction for PCCS.
Fulfills PDF Section 4.9 & Algorithm Section 13:
1. Constructs separate co-association matrices for each perturbation family.
2. Weights pairwise relationships according to cluster- and condition-level reliability.
3. Down-weights relationships that repeatedly disappear or exhibit fragility.
4. Fuses representations using invariance-preserving blending across families.
5. Derives final consensus clusters via hierarchical / spectral consensus clustering.
"""
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from sklearn.cluster import AgglomerativeClustering, SpectralClustering
from .tensor import StabilityTensor


def _cluster_precomputed(
    affinity: np.ndarray,
    n_clusters: int,
    linkage: str = "average",
    min_cluster_frac: float = 0.02,
    random_state: int = 0,
) -> Tuple[np.ndarray, bool]:
    """
    Cluster a precomputed affinity matrix.
    Uses agglomerative clustering on distance (1 - affinity),
    with spectral clustering fallback if severe fragmentation occurs.
    """
    n = affinity.shape[0]
    distance = np.clip(1.0 - affinity, 0.0, 1.0)

    try:
        agg = AgglomerativeClustering(n_clusters=n_clusters, metric="precomputed", linkage=linkage)
    except TypeError:
        agg = AgglomerativeClustering(n_clusters=n_clusters, affinity="precomputed", linkage=linkage)

    labels = agg.fit_predict(distance)
    fallback = False

    counts = np.bincount(labels, minlength=n_clusters)
    if counts.min() < int(np.ceil(min_cluster_frac * n)):
        # Severe fragmentation: use spectral clustering fallback
        spectral = SpectralClustering(
            n_clusters=n_clusters,
            affinity="precomputed",
            random_state=random_state,
            assign_labels="kmeans",
        )
        labels = spectral.fit_predict(np.clip(affinity, 0.0, 1.0))
        fallback = True

    return labels, fallback


def build_perturbation_aware_consensus(
    tensor: StabilityTensor,
    n_clusters: int,
    families: Tuple[str, ...],
    cluster_reliabilities: Optional[np.ndarray] = None,   # length N or per-cluster reliability
    candidate_labels: Optional[np.ndarray] = None,
    interaction_df: Optional[Any] = None,
    consistency_lambda: float = 0.5,
    family_weighting: str = "reliability",
    linkage: str = "average",
    min_cluster_frac: float = 0.02,
    random_state: int = 0,
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """
    Build the final perturbation-aware consensus matrix and partition.
    Returns (labels, fused_affinity_matrix, fallback_used).
    """
    n = tensor.n
    valid_fams = [f for f in families if f in tensor.S_family]

    if not valid_fams:
        if tensor.baseline_matrix is not None:
            labels, fb = _cluster_precomputed(tensor.baseline_matrix, n_clusters, linkage, min_cluster_frac, random_state)
            return labels, tensor.baseline_matrix, fb
        eye = np.eye(n, dtype=np.float32)
        labels, fb = _cluster_precomputed(eye, n_clusters, linkage, min_cluster_frac, random_state)
        return labels, eye, fb

    # Family weights alpha_p
    alphas = []
    for f in valid_fams:
        mat = tensor.S_family[f]
        # Inherent signal strength of family: mean within-cluster coassociation vs baseline
        if family_weighting == "reliability":
            # Stability score based on tensor matrix density and low variance
            var_f = tensor.Var_family.get(f, np.zeros((n, n), dtype=np.float32))
            mean_stab = float(mat.mean())
            mean_var = float(var_f.mean())
            alpha_f = max(0.01, mean_stab / (1.0 + 2.0 * mean_var))
        else:
            alpha_f = 1.0
        alphas.append(alpha_f)

    alpha_arr = np.array(alphas)
    alpha_arr = alpha_arr / alpha_arr.sum()

    # Step 1: Weighted sum of family matrices
    C_weighted = sum(a * tensor.S_family[f] for a, f in zip(alpha_arr, valid_fams))

    # Step 2: Cross-family invariant core (element-wise minimum across families)
    if consistency_lambda > 0 and len(valid_fams) > 1:
        C_min = np.minimum.reduce([tensor.S_family[f] for f in valid_fams])
        C_fused = (1.0 - consistency_lambda) * C_weighted + consistency_lambda * C_min
    else:
        C_fused = C_weighted

    # Step 3: Reliability-based pairwise modulation (PDF Section 4.9)
    # Down-weight pairs belonging to clusters with low measured reliability
    if cluster_reliabilities is not None and candidate_labels is not None:
        # cluster_reliabilities: array of length K with stability scores (1 - PSI or mean_stability)
        rel_weights = np.ones(n, dtype=np.float32)
        for i in range(n):
            c_id = candidate_labels[i]
            if 0 <= c_id < len(cluster_reliabilities):
                rel_weights[i] = float(np.clip(cluster_reliabilities[c_id], 0.1, 1.0))

        # Modulate pairwise affinity: points both in reliable clusters retain full affinity;
        # points in fragile clusters have down-weighted affinity
        pair_rel = np.sqrt(rel_weights[:, None] * rel_weights[None, :])
        C_fused = C_fused * pair_rel

    # Symmetrize and guarantee diagonal is 1.0
    C_fused = 0.5 * (C_fused + C_fused.T)
    np.fill_diagonal(C_fused, 1.0)
    C_fused = np.clip(C_fused, 0.0, 1.0).astype(np.float32)

    # Derive consensus partition
    labels, fallback = _cluster_precomputed(
        C_fused,
        n_clusters=n_clusters,
        linkage=linkage,
        min_cluster_frac=min_cluster_frac,
        random_state=random_state,
    )

    return labels, C_fused, fallback
