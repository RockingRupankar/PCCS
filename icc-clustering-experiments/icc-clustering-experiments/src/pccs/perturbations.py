"""
Controlled perturbation engine for PCCS:
Explicitly separates and quantifies:
A. Algorithmic perturbations
B. Sampling perturbations
C. Feature perturbations
D. Measurement perturbations
E. Technical-condition perturbations
F. Distribution perturbations
"""
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any
import numpy as np

FAMILIES = ("algorithmic", "sampling", "feature", "measurement", "technical", "distribution")

DEFAULT_LEVELS = {
    "algorithmic": (0.0, 1.0),          # 0 = seeds only, 1 = seeds + hyperparameter jitter
    "sampling": (0.1, 0.3, 0.5),        # fraction of rows removed
    "feature": (0.2, 0.5, 0.8),         # fraction of features masked/removed
    "measurement": (0.25, 0.5, 1.0),    # additive noise / outlier / missingness intensity
    "technical": (0.1, 0.25, 0.5),      # batch offset sd
    "distribution": (0.5, 1.0, 2.0),    # covariate shift tilting intensity
}


@dataclass
class PerturbedDataView:
    family: str
    nominal_level: float
    realized_intensity: float
    rep: int
    rows: np.ndarray            # indices of observations included
    cols: np.ndarray            # indices of features included
    X: np.ndarray               # perturbed data matrix
    jitter: float = 0.0         # algorithmic hyperparameter jitter


def make_fresh_view(Z: np.ndarray) -> PerturbedDataView:
    n, d = Z.shape
    return PerturbedDataView(
        family="baseline",
        nominal_level=0.0,
        realized_intensity=0.0,
        rep=0,
        rows=np.arange(n),
        cols=np.arange(d),
        X=Z.copy(),
        jitter=0.0,
    )


def apply_perturbation(
    view: PerturbedDataView,
    family: str,
    level: float,
    rep: int,
    rng: np.random.Generator,
    n_clusters: int,
    batch: Optional[np.ndarray] = None,
    n_batches: int = 3,
) -> PerturbedDataView:
    """
    Apply a controlled perturbation to a data view and measure realized intensity.
    """
    rows = view.rows.copy()
    cols = view.cols.copy()
    X = view.X.copy()
    n, d = X.shape
    jitter = 0.0
    realized = 0.0

    if family == "algorithmic":
        jitter = float(level)
        realized = jitter

    elif family == "sampling":
        keep = max(int(round((1.0 - level) * n)), n_clusters * 4)
        pos = np.sort(rng.choice(n, keep, replace=False))
        rows = rows[pos]
        X = X[pos]
        realized = 1.0 - (len(pos) / n)

    elif family == "feature":
        keep = max(int(round((1.0 - level) * d)), 2)
        pos = np.sort(rng.choice(d, keep, replace=False))
        cols = cols[pos]
        X = X[:, pos]
        realized = 1.0 - (len(pos) / d)

    elif family == "measurement":
        kind = ("gaussian", "heavy_tailed", "outliers", "missing")[rep % 4]
        if kind == "gaussian":
            noise = rng.normal(0, level, X.shape)
            X = X + noise
            realized = float(np.std(noise))
        elif kind == "heavy_tailed":
            noise = level * rng.standard_t(3, X.shape) / np.sqrt(3)
            X = X + noise
            realized = float(np.std(noise))
        elif kind == "outliers":
            n_out = max(int(0.05 * level * n), 1)
            out_idx = rng.choice(n, n_out, replace=False)
            for r in out_idx:
                cc = rng.choice(d, min(3, d), replace=False)
                X[r, cc] = rng.choice([-1.0, 1.0], len(cc)) * rng.uniform(6.0, 10.0, len(cc))
            realized = n_out / n
        else:  # missing
            mask = rng.random(X.shape) < (0.2 * level)
            mu = np.where(mask, np.nan, X)
            col_means = np.nanmean(mu, axis=0)
            col_means = np.where(np.isnan(col_means), 0.0, col_means)
            X = np.where(mask, col_means, X)
            realized = float(mask.mean())

    elif family == "technical":
        if batch is not None:
            b = np.asarray(batch)[rows]
        else:
            b = rng.integers(0, n_batches, n)
        b = np.unique(b, return_inverse=True)[1]
        B = int(b.max()) + 1
        offsets = rng.normal(0, level, (B, d))
        scales = np.clip(1.0 + 0.5 * level * rng.normal(size=(B, d)), 0.2, None)
        X = X * scales[b] + offsets[b]
        realized = float(np.std(offsets))

    elif family == "distribution":
        # Covariate shift via exponential tilting along a projection
        v = rng.normal(size=d)
        norm_v = np.linalg.norm(v)
        v = v / (norm_v + 1e-9)
        u = X @ v
        u = (u - u.mean()) / (u.std() + 1e-9)
        logits = np.clip(level * u, -30.0, 30.0)
        p = np.exp(logits)
        p = p / p.sum()
        keep = max(int(0.7 * n), n_clusters * 4)
        pos = np.sort(rng.choice(n, keep, replace=False, p=p))
        rows = rows[pos]
        X = X[pos]
        # Realized shift quantified as KS statistic or mean shift along u
        realized = float(np.abs(u[pos].mean() - u.mean()))

    else:
        raise ValueError(f"Unknown perturbation family: {family}")

    return PerturbedDataView(
        family=family,
        nominal_level=level,
        realized_intensity=realized,
        rep=rep,
        rows=rows,
        cols=cols,
        X=X,
        jitter=jitter,
    )
