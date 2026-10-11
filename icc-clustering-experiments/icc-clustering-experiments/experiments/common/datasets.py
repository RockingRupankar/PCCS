"""Dataset factories shared by the experiments."""
import copy
import numpy as np
import pandas as pd
from . import pccs_datagen as pg


def make_pccs(profile, n_samples, seed=42):
    """PCCS-style synthetic data (see generation_script.py): returns X (NaN allowed), y, feature_names, df."""
    cfg = copy.deepcopy(pg.CONFIG)
    cfg["n_samples"], cfg["random_seed"] = n_samples, seed
    gen = pg.SyntheticClusterGenerator(cfg)
    df, _ = gen.build(profile)
    return df[gen.feature_names].to_numpy(float), df["true_cluster"].to_numpy(), gen.feature_names, df


def make_scalable(n, d, k=4, d_inf=10, sep=4.0, seed=0):
    """Plain Gaussian mixture: d_inf informative dims, d-d_inf pure-noise dims (for scaling studies)."""
    rng = np.random.default_rng(seed)
    d_inf = min(d_inf, d)
    sizes = np.full(k, n // k)
    sizes[: n - sizes.sum()] += 1
    y = rng.permutation(np.repeat(np.arange(k), sizes))
    C = rng.normal(size=(k, d_inf))
    C = (C - C.mean(0)) * sep / np.mean([np.linalg.norm(C[i] - C[j]) for i in range(k) for j in range(i + 1, k)])
    X = rng.normal(size=(n, d))
    X[:, :d_inf] += C[y]
    return X, y


def corrupt(X, n_noise_features=100, meas_sd=0.3, outlier_frac=0.02, miss_row_frac=0.03, seed=0):
    """Apply the noise protocol to a real dataset: standardise, add noise features, measurement noise,
    outlier rows (+-4..8 sd in 5-20 cells), MCAR missing cells in a fraction of rows."""
    rng = np.random.default_rng(seed)
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    n, d = X.shape
    X = X + rng.normal(0, meas_sd, X.shape)
    X = np.hstack([X, rng.normal(size=(n, n_noise_features))])
    D = X.shape[1]
    for r in rng.choice(n, int(outlier_frac * n), replace=False):
        k = rng.integers(min(5, D), min(20, D) + 1)
        c = rng.choice(D, k, replace=False)
        X[r, c] = rng.choice([-1, 1], k) * rng.uniform(4, 8, k)
    for r in rng.choice(n, int(miss_row_frac * n), replace=False):
        c = rng.choice(D, rng.integers(3, 11), replace=False)
        X[r, c] = np.nan
    return X
