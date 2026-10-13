"""
Heterogeneous clustering ensemble for PCCS:
1. K-Means: centroid-based compact clusters
2. Fuzzy C-Means (FCM): soft memberships with documented conversion
3. DBSCAN: density-based clusters with adaptive k-NN quantile calibration and explicit noise handling
4. Hierarchical/Agglomerative: multiscale proximity relationships
"""
from typing import Tuple, Dict, Any, Optional
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from scipy.spatial.distance import cdist
from scipy.special import logsumexp
from sklearn.neighbors import NearestNeighbors


def run_kmeans(
    X: np.ndarray,
    n_clusters: int,
    random_state: int = 42,
    n_init: int = 10,
    max_iter: int = 300,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Centroid-based compact clustering."""
    km = KMeans(
        n_clusters=n_clusters,
        init="k-means++",
        n_init=n_init,
        max_iter=max_iter,
        random_state=random_state,
    )
    labels = km.fit_predict(X)
    return labels, {"centers": km.cluster_centers_, "inertia": km.inertia_}


def run_fcm(
    X: np.ndarray,
    n_clusters: int,
    fuzzifier: float = 1.3,
    max_iter: int = 150,
    tol: float = 1e-4,
    random_state: int = 42,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Fuzzy C-Means clustering with stable log-domain updates.
    Returns:
    - hard_labels: argmax of membership matrix U
    - info: contains soft membership matrix U in [0, 1]^(N x K) and centroids
    """
    rng = np.random.RandomState(random_state)
    N, D = X.shape
    K = n_clusters
    m = max(1.05, float(fuzzifier))
    p = 2.0 / (m - 1.0)

    # Initialize centroids via k-means++ style sampling
    init_idx = rng.choice(N, size=K, replace=False)
    centers = X[init_idx].copy() + rng.normal(scale=1e-4, size=(K, D))

    U = np.zeros((N, K))
    for it in range(max_iter):
        dists = cdist(X, centers, metric="euclidean")
        dists = np.maximum(dists, 1e-8)

        log_d = np.log(dists)
        log_term = -p * log_d
        log_denom = logsumexp(log_term, axis=1, keepdims=True)
        U_new = np.exp(log_term - log_denom)

        if np.any(np.isnan(U_new)):
            break

        # Update centroids
        um = U_new ** m
        sums = np.maximum(um.sum(axis=0, keepdims=True).T, 1e-8)
        new_centers = np.dot(um.T, X) / sums

        diff = np.max(np.abs(centers - new_centers))
        centers = new_centers
        U = U_new
        if diff < tol:
            break

    hard_labels = np.argmax(U, axis=1)
    return hard_labels, {"U": U, "centers": centers, "iterations": it + 1}


def run_dbscan(
    X: np.ndarray,
    n_clusters_hint: Optional[int] = None,
    min_samples: int = 5,
    eps_quantile: float = 0.6,
    eps_scale: float = 1.2,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Density-based clustering with adaptive k-NN quantile distance calibration.
    Explicit noise handling: noise points are assigned label -1.
    """
    N, D = X.shape
    k = max(min_samples, min(N - 1, 5))
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(X)
    distances, _ = nn.kneighbors(X)
    k_dist = distances[:, -1]
    
    eps = float(np.quantile(k_dist, eps_quantile) * eps_scale)
    eps = max(eps, 1e-4)

    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(X)
    n_noise = int(np.sum(labels == -1))
    noise_frac = float(n_noise / N)

    return labels, {"eps": eps, "n_noise": n_noise, "noise_fraction": noise_frac}


def run_hierarchical(
    X: np.ndarray,
    n_clusters: int,
    linkage: str = "ward",
    metric: str = "euclidean",
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Multiscale proximity clustering via agglomerative clustering."""
    if linkage == "ward":
        metric = "euclidean"
    agg = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage, metric=metric)
    labels = agg.fit_predict(X)
    return labels, {"linkage": linkage, "metric": metric}


def fit_base_clusterer(
    method: str,
    X: np.ndarray,
    n_clusters: int,
    random_state: int = 42,
    params: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Unified dispatcher for ensemble clustering algorithms with edge case validation."""
    params = params or {}
    if method == "kmeans":
        return run_kmeans(X, n_clusters, random_state=random_state, **params)
    elif method == "fcm":
        return run_fcm(X, n_clusters, random_state=random_state, **params)
    elif method == "dbscan":
        return run_dbscan(X, n_clusters_hint=n_clusters, **params)
    elif method == "hierarchical":
        return run_hierarchical(X, n_clusters, **params)
    else:
        raise ValueError(f"Unknown clustering method: {method}")
