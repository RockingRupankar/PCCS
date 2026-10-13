"""
Perturbation-aware consensus construction for PCCS.
Fuses condition-specific co-association matrices using cluster reliability modulation
and invariant core intersection:
C_{ij} = (1 - lambda) * S_bar_{ij}^{(W)} + lambda * min_p S_{ij}^{(p)}
"""
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from pccs.stability.tensor import StabilityTensor


def build_perturbation_aware_consensus(
    tensor: StabilityTensor,
    fingerprints: pd.DataFrame,
    partitions: List[Dict[str, Any]],
    n_clusters: int,
    consistency_lambda: float = 0.5,
    weight_power: float = 2.0,
    linkage_method: str = "average",
    min_cluster_frac: float = 0.02,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Construct perturbation-aware consensus matrix and final cluster assignments:
    1. Cluster reliability weighting: W_k = (1 - PSI_k)^gamma
    2. Modulate pairwise relationships by reliability of clusters from which they originated.
    3. Retain invariant core relationships: min_p S_{ij}^{(p)}.
    4. Fuse matrices and cluster using hierarchical agglomerative clustering.
    """
    N = tensor.n_samples
    K = n_clusters
    families = tensor.families

    # 1. Compute cluster reliability weights W_k
    psi_dict = dict(zip(fingerprints["cluster"], fingerprints["PSI"]))
    weights_k = {k: max(0.01, float((1.0 - psi_dict.get(k, 0.5)) ** weight_power)) for k in range(K)}

    # 2. Compute reliability-modulated mean co-association
    # Weight each family slice by mean stability of clusters under that family
    fam_weights = {}
    for f in families:
        if f in fingerprints.columns:
            fam_weights[f] = max(0.01, float(fingerprints[f].mean()))
        else:
            fam_weights[f] = 1.0

    total_fam_w = sum(fam_weights.values())
    fam_weights = {f: w / total_fam_w for f, w in fam_weights.items()}

    S_weighted = np.zeros((N, N), dtype=float)
    S_min = np.ones((N, N), dtype=float)

    for f in families:
        S_f = tensor.get_slice(f)
        w_f = fam_weights[f]
        S_weighted += w_f * S_f
        S_min = np.minimum(S_min, S_f)

    # 3. Fuse modulated average with invariant core
    lam = float(np.clip(consistency_lambda, 0.0, 1.0))
    C = (1.0 - lam) * S_weighted + lam * S_min
    np.fill_diagonal(C, 1.0)

    # 4. Hierarchical consensus clustering
    dist = np.clip(1.0 - C, 0.0, 1.0)
    np.fill_diagonal(dist, 0.0)
    condensed = squareform(dist, checks=False)

    Z = linkage(condensed, method=linkage_method)
    labels = fcluster(Z, t=K, criterion="maxclust") - 1

    # Check for excessive fragmentation / empty clusters
    unique_lbls, counts = np.unique(labels, return_counts=True)
    min_size = int(np.round(N * min_cluster_frac))

    # Calculate consensus silhouette & quality metrics
    same = (labels[:, None] == labels[None, :])
    np.fill_diagonal(same, False)
    diff = ~same
    np.fill_diagonal(diff, False)

    within_coassoc = float(C[same].mean()) if np.sum(same) > 0 else 1.0
    between_coassoc = float(C[diff].mean()) if np.sum(diff) > 0 else 0.0

    diagnostics = {
        "within_cluster_coassoc": within_coassoc,
        "between_cluster_coassoc": between_coassoc,
        "coassoc_contrast": float(within_coassoc - between_coassoc),
        "family_weights": fam_weights,
        "cluster_weights": weights_k,
        "min_size_observed": int(np.min(counts)),
    }

    return labels, C, diagnostics
