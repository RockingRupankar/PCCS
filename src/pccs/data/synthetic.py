"""
Synthetic dataset generator for PCCS experiments.
Provides controlled geometric clusters and null distributions with ground-truth labels.
"""
from typing import Tuple, Dict, Any, Optional
import numpy as np
from sklearn.datasets import make_blobs, make_moons, make_circles


def generate_synthetic_regime(
    regime: str,
    n_samples: int = 500,
    n_features: int = 20,
    n_clusters: int = 4,
    random_state: int = 42,
    noise_std: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Generate synthetic dataset under a controlled structural regime.
    
    Regimes:
    - 'spherical_separated': Compact isotropic Gaussian clusters.
    - 'anisotropic_elongated': Covariance-skewed elongated clusters.
    - 'non_convex_rings': Concentric circular non-convex structures.
    - 'overlapping': Clusters with substantial class overlap.
    - 'rare_cluster': 3 large clusters (N=150 each) and 1 rare cluster (N=20).
    - 'high_dimensional_noise': 5 informative dimensions + (D-5) uninformative noise dimensions.
    - 'null_isotropic': Pure unclustered Gaussian noise (no genuine clusters).
    - 'null_anisotropic': Pure unclustered correlated multivariate Gaussian noise.
    """
    rng = np.random.RandomState(random_state)
    meta = {
        "regime": regime,
        "n_samples": n_samples,
        "n_features": n_features,
        "random_state": random_state,
        "noise_std": noise_std,
    }

    if regime == "spherical_separated":
        X, y = make_blobs(
            n_samples=n_samples,
            n_features=n_features,
            centers=n_clusters,
            cluster_std=1.0,
            center_box=(-10.0, 10.0),
            random_state=random_state,
        )
        meta["n_clusters"] = n_clusters

    elif regime == "anisotropic_elongated":
        X_base, y = make_blobs(
            n_samples=n_samples,
            n_features=2,
            centers=n_clusters,
            cluster_std=0.8,
            random_state=random_state,
        )
        # Apply transformation matrix to stretch clusters
        transform = rng.normal(size=(2, 2))
        X_2d = np.dot(X_base, transform)
        if n_features > 2:
            extra = rng.normal(scale=noise_std, size=(n_samples, n_features - 2))
            X = np.hstack([X_2d, extra])
        else:
            X = X_2d
        meta["n_clusters"] = n_clusters

    elif regime == "non_convex_rings":
        X_2d, y = make_circles(
            n_samples=n_samples,
            factor=0.5,
            noise=noise_std * 0.3,
            random_state=random_state,
        )
        if n_features > 2:
            extra = rng.normal(scale=noise_std, size=(n_samples, n_features - 2))
            X = np.hstack([X_2d, extra])
        else:
            X = X_2d
        meta["n_clusters"] = 2

    elif regime == "overlapping":
        X, y = make_blobs(
            n_samples=n_samples,
            n_features=n_features,
            centers=n_clusters,
            cluster_std=3.0,
            center_box=(-4.0, 4.0),
            random_state=random_state,
        )
        meta["n_clusters"] = n_clusters

    elif regime == "rare_cluster":
        # 3 clusters of 150, 1 cluster of 20
        sizes = [160, 160, 160, 20] if n_samples == 500 else [int(n_samples * 0.32)] * 3 + [int(n_samples * 0.04)]
        diff = n_samples - sum(sizes)
        sizes[0] += diff
        X, y = make_blobs(
            n_samples=sizes,
            n_features=n_features,
            cluster_std=[1.0, 1.0, 1.0, 0.6],
            center_box=(-8.0, 8.0),
            random_state=random_state,
        )
        meta["n_clusters"] = 4
        meta["cluster_sizes"] = sizes

    elif regime == "high_dimensional_noise":
        d_info = min(5, n_features)
        X_info, y = make_blobs(
            n_samples=n_samples,
            n_features=d_info,
            centers=n_clusters,
            cluster_std=1.2,
            random_state=random_state,
        )
        X_noise = rng.normal(scale=2.0, size=(n_samples, n_features - d_info))
        X = np.hstack([X_info, X_noise])
        meta["n_clusters"] = n_clusters
        meta["informative_features"] = d_info

    elif regime == "null_isotropic":
        X = rng.normal(loc=0.0, scale=1.0, size=(n_samples, n_features))
        y = np.zeros(n_samples, dtype=int)
        meta["n_clusters"] = 1
        meta["is_null"] = True

    elif regime == "null_anisotropic":
        # Correlated multivariate Gaussian without cluster structure
        A = rng.normal(size=(n_features, n_features))
        cov = np.dot(A, A.T) + 0.1 * np.eye(n_features)
        mean = np.zeros(n_features)
        X = rng.multivariate_normal(mean, cov, size=n_samples)
        y = np.zeros(n_samples, dtype=int)
        meta["n_clusters"] = 1
        meta["is_null"] = True

    else:
        raise ValueError(f"Unknown synthetic regime: {regime}")

    return X, y, meta
