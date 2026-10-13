"""
Controlled Perturbation Engine for PCCS.
Implements 6 distinct perturbation families with nominal and realized intensity measurement:
1. Algorithmic: parameter and seed jitter
2. Sampling: subsampling without replacement & rare cluster stress
3. Feature: feature dropout & masking
4. Measurement: additive Gaussian, heavy-tailed noise, and outliers
5. Technical: simulated batch offsets and variance scales
6. Distribution: covariate shift via importance sampling
"""
from typing import Tuple, Dict, Any, Optional
import numpy as np

FAMILIES = (
    "algorithmic",
    "sampling",
    "feature",
    "measurement",
    "technical",
    "distribution",
)

DEFAULT_LEVELS = {
    "algorithmic": (0.1, 0.25, 0.5),
    "sampling": (0.90, 0.75, 0.50),    # fraction of observations retained
    "feature": (0.10, 0.25, 0.50),     # fraction of features removed
    "measurement": (0.10, 0.30, 0.60), # noise scale in std units
    "technical": (0.20, 0.50, 1.00),   # batch effect magnitude
    "distribution": (0.20, 0.50, 0.80) # importance tilt strength
}


def apply_perturbation(
    X: np.ndarray,
    family: str,
    level: float,
    seed: int,
    batch: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Apply a controlled perturbation to feature matrix X.
    
    Returns:
    - X_perturbed: perturbed feature matrix
    - row_indices: array of retained row indices in [0, N-1]
    - col_indices: array of retained column indices in [0, D-1]
    - metrics: dictionary containing nominal and realized perturbation intensities
    """
    rng = np.random.RandomState(seed)
    N, D = X.shape
    row_idx = np.arange(N)
    col_idx = np.arange(D)
    X_out = X.copy()
    metrics: Dict[str, Any] = {"family": family, "nominal_level": float(level), "seed": int(seed)}

    if family == "algorithmic":
        # Algorithmic perturbations do not alter data matrix; method parameters are jittered in ensemble
        metrics["realized_intensity"] = float(level)

    elif family == "sampling":
        frac = float(np.clip(level, 0.1, 1.0))
        keep_n = max(4, int(np.round(N * frac)))
        row_idx = np.sort(rng.choice(N, size=keep_n, replace=False))
        X_out = X_out[row_idx, :]
        metrics["realized_intensity"] = float(1.0 - (keep_n / N))
        metrics["retained_samples"] = keep_n

    elif family == "feature":
        drop_frac = float(np.clip(level, 0.0, 0.95))
        keep_d = max(1, int(np.round(D * (1.0 - drop_frac))))
        col_idx = np.sort(rng.choice(D, size=keep_d, replace=False))
        X_out = X_out[:, col_idx]
        metrics["realized_intensity"] = float((D - keep_d) / D)
        metrics["retained_features"] = keep_d

    elif family == "measurement":
        scale = float(max(0.0, level))
        # Mix Gaussian noise with occasional heavy-tailed Laplace noise
        noise_g = rng.normal(0.0, scale, size=X_out.shape)
        noise_l = rng.laplace(0.0, scale, size=X_out.shape)
        use_heavy = rng.binomial(1, 0.2, size=X_out.shape)
        noise = np.where(use_heavy, noise_l, noise_g)
        X_out = X_out + noise
        realized_std = float(np.std(noise))
        metrics["realized_intensity"] = realized_std
        metrics["realized_noise_std"] = realized_std

    elif family == "technical":
        shift_scale = float(max(0.0, level))
        if batch is None:
            # Simulate 2 or 3 technical batches
            n_b = 3
            batch = rng.choice(n_b, size=N)
        unique_b = np.unique(batch)
        for b in unique_b:
            b_mask = (batch == b)
            b_shift = rng.normal(0.0, shift_scale, size=(1, D))
            b_mult = 1.0 + rng.normal(0.0, shift_scale * 0.2, size=(1, D))
            X_out[b_mask] = (X_out[b_mask] + b_shift) * b_mult
        metrics["realized_intensity"] = shift_scale
        metrics["n_batches"] = len(unique_b)

    elif family == "distribution":
        tilt_strength = float(np.clip(level, 0.0, 1.0))
        # Covariate shift via hyperplane importance weighting
        direction = rng.normal(size=D)
        direction /= np.maximum(np.linalg.norm(direction), 1e-8)
        projections = np.dot(X, direction)
        projections_norm = (projections - np.mean(projections)) / np.maximum(np.std(projections), 1e-8)
        
        # Logistic tilt
        weights = 1.0 / (1.0 + np.exp(-tilt_strength * projections_norm))
        weights /= weights.sum()
        keep_n = max(4, int(np.round(N * 0.8)))
        row_idx = np.sort(rng.choice(N, size=keep_n, replace=False, p=weights))
        X_out = X_out[row_idx, :]
        metrics["realized_intensity"] = tilt_strength
        metrics["retained_samples"] = keep_n

    else:
        raise ValueError(f"Unknown perturbation family: {family}")

    return X_out, row_idx, col_idx, metrics
