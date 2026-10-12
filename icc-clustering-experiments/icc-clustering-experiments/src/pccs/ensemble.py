"""
Heterogeneous base clustering ensemble for PCCS:
- K-Means: compact centroid-based clusters.
- Fuzzy C-Means (FCM): soft memberships with entropy and hard assignments.
- DBSCAN: density-based clusters with explicit noise handling.
- Hierarchical Clustering: multiscale proximity structure.
"""
from dataclasses import dataclass
import time
from typing import Optional, Tuple, Dict, Any
import numpy as np
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.neighbors import NearestNeighbors

METHODS = ("kmeans", "fcm", "dbscan", "hierarchical")


@dataclass
class BaseClustererResult:
    method: str
    labels: np.ndarray          # 0..K-1, with -1 denoting noise (e.g., in DBSCAN)
    n_clusters: int
    noise_count: int
    noise_fraction: float
    soft_memberships: Optional[np.ndarray] = None
    runtime: float = 0.0
    converged: bool = True
    warnings: str = ""


def run_fcm(
    X: np.ndarray,
    n_clusters: int,
    fuzzifier: float,
    rng: np.random.Generator,
    max_iters: int = 60,
    tol: float = 1e-4,
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """
    Fuzzy C-Means clustering in log-domain to avoid numerical underflow.
    Returns (hard_labels, soft_memberships, converged).
    """
    n, d = X.shape
    c = n_clusters
    m = max(1.05, float(fuzzifier))
    p = 1.0 / (m - 1.0)

    # Initialize centers via random choice
    init_idx = rng.choice(n, c, replace=False)
    centers = X[init_idx].copy()

    converged = False
    U = np.zeros((n, c), dtype=float)
    for it in range(max_iters):
        # Distance squared: n x c
        d2 = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2) + 1e-12
        logit = -p * np.log(d2)
        logit -= logit.max(axis=1, keepdims=True)
        U = np.exp(logit)
        U /= U.sum(axis=1, keepdims=True)

        W = U ** m
        sum_W = W.sum(axis=0)[:, None]
        sum_W = np.maximum(sum_W, 1e-12)
        new_centers = (W.T @ X) / sum_W

        shift = np.abs(new_centers - centers).max()
        centers = new_centers
        if shift < tol:
            converged = True
            break

    hard_labels = U.argmax(axis=1)
    return hard_labels, U, converged


def run_dbscan_adaptive(
    X: np.ndarray,
    min_samples: int,
    eps_quantile: float = 0.5,
    eps_scale: float = 1.0,
    jitter: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, int, int]:
    """
    DBSCAN with adaptive epsilon based on k-NN distance distribution.
    Noise observations retain label -1.
    """
    n, d = X.shape
    ms = max(2, min_samples)
    if jitter > 0 and rng is not None:
        ms = max(2, ms + int(rng.integers(-1, 3)))
        eps_scale = eps_scale * (1.0 + jitter * float(rng.uniform(-0.15, 0.15)))

    # Compute ms-th nearest neighbor distance
    k_eval = min(ms + 1, n - 1) if n > 2 else 1
    nbrs = NearestNeighbors(n_neighbors=k_eval).fit(X)
    distances, _ = nbrs.kneighbors(X)
    k_dist = distances[:, -1]

    eps = float(np.quantile(k_dist, np.clip(eps_quantile, 0.1, 0.95)) * eps_scale)
    eps = max(eps, 1e-6)

    db = DBSCAN(eps=eps, min_samples=ms).fit(X)
    raw_labels = db.labels_

    # Map discovered non-noise clusters to 0..K-1 while preserving -1 for noise
    labels = np.full(n, -1, dtype=int)
    non_noise = raw_labels >= 0
    if non_noise.any():
        unique_c = np.unique(raw_labels[non_noise])
        mapping = {c_val: i for i, c_val in enumerate(unique_c)}
        for c_val, i in mapping.items():
            labels[raw_labels == c_val] = i
        n_clusters = len(unique_c)
    else:
        n_clusters = 0

    noise_count = int((labels == -1).sum())
    return labels, n_clusters, noise_count


def cluster_single(
    method: str,
    X: np.ndarray,
    n_clusters: int,
    seed: int,
    jitter: float = 0.0,
    fuzzifier: float = 1.3,
    dbscan_min_samples: int = 5,
    dbscan_eps_quantile: float = 0.5,
    dbscan_eps_scale: float = 1.0,
    hierarchical_linkage: str = "ward",
) -> BaseClustererResult:
    """
    Execute a single base clustering algorithm with explicit timing, metadata, and validation.
    """
    t0 = time.time()
    rng = np.random.default_rng(seed)
    n = X.shape[0]

    warn = ""
    converged = True
    soft_m = None

    if method == "kmeans":
        n_init = int(rng.choice([1, 3])) if jitter > 0 else 1
        init = "random" if (jitter > 0 and rng.random() < 0.5) else "k-means++"
        km = KMeans(n_clusters=n_clusters, n_init=n_init, init=init, random_state=seed)
        labels = km.fit_predict(X)
        k_found = len(np.unique(labels))
        noise_cnt = 0

    elif method == "fcm":
        m = fuzzifier + (jitter * float(rng.uniform(-0.15, 0.25)) if jitter > 0 else 0.0)
        labels, soft_m, converged = run_fcm(X, n_clusters, m, rng)
        k_found = len(np.unique(labels))
        noise_cnt = 0

    elif method == "dbscan":
        labels, k_found, noise_cnt = run_dbscan_adaptive(
            X,
            min_samples=dbscan_min_samples,
            eps_quantile=dbscan_eps_quantile,
            eps_scale=dbscan_eps_scale,
            jitter=jitter,
            rng=rng,
        )
        if k_found == 0:
            warn = "DBSCAN: all points identified as noise"
        elif k_found == 1 and noise_cnt == 0:
            warn = "DBSCAN: single global cluster discovered"

    elif method == "hierarchical":
        if jitter > 0:
            link = str(rng.choice(["ward", "average", "complete"]))
        else:
            link = hierarchical_linkage
        agg = AgglomerativeClustering(n_clusters=n_clusters, linkage=link)
        labels = agg.fit_predict(X)
        k_found = len(np.unique(labels))
        noise_cnt = 0

    else:
        raise ValueError(f"Unknown clustering method: {method}")

    runtime = time.time() - t0
    return BaseClustererResult(
        method=method,
        labels=labels,
        n_clusters=k_found,
        noise_count=noise_cnt,
        noise_fraction=noise_cnt / max(1, n),
        soft_memberships=soft_m,
        runtime=runtime,
        converged=converged,
        warnings=warn,
    )
