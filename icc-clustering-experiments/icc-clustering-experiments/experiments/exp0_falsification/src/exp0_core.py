"""exp0 core: data generators, noise models, stability measures, fuzzy c-means.

Hypothesis under test (H1): "high cluster stability indicates real cluster structure",
i.e. stability separates structured data from structure-free (null) data.
Falsification: find regimes (noise type / level / dimension) where null data are as stable
as, or more stable than, structured data  -> stability alone is perturbation-blind.
"""
import sys
from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

STRUCTURES = ["structured", "null_iso", "null_aniso"]
NOISE_TYPES = ["gaussian", "student_t3", "outliers", "missing_imputed", "feature_shuffle", "uniform"]


def make_structure(structure, n=400, d=20, d_inf=5, k=4, sep=11.0, rng=None):
    """structured: k Gaussian clusters in d_inf dims (mean centre distance = sep), other dims noise.
    null_iso: one isotropic Gaussian.  null_aniso: one elongated, rotated (correlated) Gaussian.
    Columns are standardised afterwards so all structures have unit marginal variance."""
    rng = rng or np.random.default_rng(0)
    X = rng.normal(size=(n, d))
    y = np.zeros(n, int)
    if structure == "structured":
        y = rng.permutation(np.repeat(np.arange(k), n // k))[:n]
        C = rng.normal(size=(k, d_inf))
        C -= C.mean(0)
        md = np.mean([np.linalg.norm(C[i] - C[j]) for i in range(k) for j in range(i + 1, k)])
        X[:, :d_inf] += C[y] * sep / md
    elif structure == "null_aniso":
        # elongated Gaussian (sds 4 -> 1 in d_inf dims) then a random rotation, so that the elongation
        # becomes correlation between columns and SURVIVES per-column standardisation
        X[:, :d_inf] *= np.linspace(4, 1, d_inf)
        Q, _ = np.linalg.qr(rng.normal(size=(d, d)))
        X = X @ Q
    elif structure != "null_iso":
        raise ValueError(structure)
    X = (X - X.mean(0)) / X.std(0)
    return X, y


def add_noise(X, noise_type, level, rng):
    """level is a unit-free intensity:  gaussian/student_t3/uniform: noise sd = level;
    outliers: 5%*level of rows get 3 random cells set to +-(6..10) sd; missing_imputed: 10%*level of cells
    removed MCAR then mean-imputed; feature_shuffle: 15%*level of values permuted within each column."""
    X = X.copy()
    n, d = X.shape
    if level <= 0:
        return X
    if noise_type == "gaussian":
        X += rng.normal(0, level, X.shape)
    elif noise_type == "student_t3":
        X += level * rng.standard_t(3, X.shape) / np.sqrt(3)
    elif noise_type == "uniform":
        X += level * rng.uniform(-np.sqrt(3), np.sqrt(3), X.shape)
    elif noise_type == "outliers":
        rows = rng.choice(n, int(min(0.5, 0.05 * level) * n), replace=False)
        for r in rows:
            c = rng.choice(d, 3, replace=False)
            X[r, c] = rng.choice([-1, 1], 3) * rng.uniform(6, 10, 3)
    elif noise_type == "missing_imputed":
        M = rng.random(X.shape) < min(0.6, 0.1 * level)
        mu = np.where(M, np.nan, X)
        mu = np.nanmean(mu, 0)
        X = np.where(M, mu, X)
    elif noise_type == "feature_shuffle":
        frac = min(1.0, 0.15 * level)
        for j in range(d):
            r = rng.choice(n, int(frac * n), replace=False)
            X[r, j] = rng.permutation(X[r, j])
    else:
        raise ValueError(noise_type)
    return X


def _pairwise(labs):
    v = []
    for i in range(len(labs)):
        for j in range(i + 1, len(labs)):
            m = (labs[i] >= 0) & (labs[j] >= 0)
            v.append(adjusted_rand_score(labs[i][m], labs[j][m]))
    return float(np.mean(v))


def stability(X, k=4, B=8, rng=None, sub=0.8):
    """Returns dict(stab_seed, stab_boot, ari_truth_ref, silhouette).
    stab_seed: mean pairwise ARI of B k-means restarts on identical data (algorithmic perturbation only).
    stab_boot: mean pairwise ARI of k-means on B random 80% subsamples (sampling perturbation)."""
    rng = rng or np.random.default_rng(0)
    n = len(X)
    Z = (X - X.mean(0)) / (X.std(0) + 1e-9)
    seed_labs = [KMeans(k, n_init=1, random_state=int(rng.integers(1e9))).fit_predict(Z) for _ in range(B)]
    boot_labs = []
    for _ in range(B):
        idx = rng.choice(n, int(sub * n), replace=False)
        full = -np.ones(n, int)
        full[idx] = KMeans(k, n_init=1, random_state=int(rng.integers(1e9))).fit_predict(Z[idx])
        boot_labs.append(full)
    sil = float(silhouette_score(Z, seed_labs[0]))
    return dict(stab_seed=_pairwise(seed_labs), stab_boot=_pairwise(boot_labs), silhouette=sil), seed_labs[0]


def run_cell(structure, noise_type, level, seed, n=400, d=20, d_inf=5, B=8):
    rng = np.random.default_rng(hash((structure, noise_type, round(level, 4), seed)) % (2**32))
    X, y = make_structure(structure, n, d, d_inf, rng=rng)
    X = add_noise(X, noise_type, level, rng)
    s, lab = stability(X, 4, B, rng)
    s.update(structure=structure, noise_type=noise_type, level=level, seed=seed, n=n, d=d,
             ari_truth=float(adjusted_rand_score(y, lab)) if structure == "structured" else np.nan)
    return s


# ----------------------------------------------------------------- fuzzy c-means (numpy)
def fcm(X, c=4, m=2.0, iters=100, tol=1e-5, rng=None):
    """Fuzzy c-means with log-domain memberships (stable for m close to 1)."""
    rng = rng or np.random.default_rng(0)
    cent = X[rng.choice(len(X), c, replace=False)].copy()
    p = 1.0 / (m - 1.0)
    U = None
    for _ in range(iters):
        d2 = ((X[:, None, :] - cent[None]) ** 2).sum(2) + 1e-12
        logit = -p * np.log(d2)
        logit -= logit.max(1, keepdims=True)
        U = np.exp(logit)
        U /= U.sum(1, keepdims=True)
        W = U ** m
        new = (W.T @ X) / W.sum(0)[:, None]
        if np.abs(new - cent).max() < tol:
            cent = new
            break
        cent = new
    return U, cent
