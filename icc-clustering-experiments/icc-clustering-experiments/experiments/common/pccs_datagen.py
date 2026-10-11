#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Synthetic data generator for
"Cluster Stability-Based Ensemble Learning for Noisy High-Dimensional Data"

Produces
  * clean_baseline.csv, noisy_dataset.csv, severe_noise_dataset.csv
  * perturbation_001.csv ... perturbation_020.csv  (perturbed copies of the noisy dataset)
  * feature_description.csv, dataset_metadata.json, qc_report.json,
    perturbation_manifest.csv, README.md

Usage:   python generation_script.py --out ./synthetic_cluster_data
Needs:   numpy, pandas, scipy, scikit-learn

Design summary
--------------
All three datasets share the SAME hidden structure (same rows, same cluster
membership, same feature roles, same pure-noise columns, same per-feature scales
and the same underlying random draws). They differ only in the corruption
profile (separation, measurement noise, feature corruption, outliers, missingness,
ambiguous samples). This makes them directly comparable ("related datasets").

Pipeline per dataset
  latent cluster geometry (15 informative dims, moon-shaped pair + Gaussian blobs)
  -> nonlinear transforms (quadratic, sinusoidal, skewed, signed-power)
  -> standardise with BASELINE statistics
  -> 15 redundant features derived from informative ones
  -> 70 pure-noise features (several distributions, two correlated blocks)
  -> feature-specific measurement noise
  -> (severe only) feature corruption by value shuffling
  -> outliers (extreme but plausible standardised magnitudes)
  -> per-feature scale/offset (different raw scales)
  -> MCAR missing cells in a fraction of rows
"""
import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.metrics import (adjusted_rand_score, mutual_info_score,
                             normalized_mutual_info_score, silhouette_score)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION  (edit here)
# =============================================================================
CONFIG = {
    "random_seed": 42,
    "n_samples": 3000,
    "n_features": 100,
    "n_clusters": 4,
    "n_informative": 15,
    "n_redundant": 15,
    "n_noise": 70,
    # mild imbalance -> sizes 1050 / 900 / 630 / 420
    "cluster_proportions": [0.35, 0.30, 0.21, 0.14],
    "cluster_within_sd": [0.9, 1.1, 1.0, 1.2],      # cluster-specific spread
    "mean_center_distance": 7.5,                    # mean pairwise centre distance (latent units, sep_scale=1)
    "informative_corr_groups": [(0, 5), (5, 10), (10, 15)],   # within-cluster correlated blocks
    "within_group_rho": 0.5,
    "moon_radius": 2.5,                             # crescent size (clusters 1 & 2), scaled by sep_scale
    "moon_noise": 0.5,
    "noise_corr_groups": [(0, 10), (10, 20)],       # correlated pure-noise blocks (cluster-independent)
    "noise_group_rho": 0.4,
    "feature_scale_log10_range": (-1.0, 2.7),       # raw feature scales 0.1 ... 500
    "outlier_k_features": (5, 20),                  # features hit per outlier row
    "outlier_magnitude": (4.0, 8.0),                # |value| in standardised units
    "borderline_lambda": (0.40, 0.60),              # fraction of the way towards another centre
    "mislabel_like_lambda": (0.85, 1.00),
    # measurement-noise sd is expressed in units of the (standardised) feature sd
    "profiles": {
        "clean": dict(sep_scale=1.0, meas_noise=(0.05, 0.15), heavy_tail_frac=0.0,
                      borderline_frac=0.0, mislabel_like_frac=0.0,
                      outlier_frac=0.0, missing_row_frac=0.0, missing_cells=(0, 0),
                      corrupt_feature_frac=0.0, corrupt_value_frac=0.0, profile_id=1),
        "noisy": dict(sep_scale=1.0, meas_noise=(0.30, 0.70), heavy_tail_frac=0.0,
                      borderline_frac=0.02, mislabel_like_frac=0.005,
                      outlier_frac=0.02, missing_row_frac=0.03, missing_cells=(3, 12),
                      corrupt_feature_frac=0.0, corrupt_value_frac=0.0, profile_id=2),
        "severe": dict(sep_scale=0.80, meas_noise=(0.60, 1.20), heavy_tail_frac=0.30,
                       borderline_frac=0.04, mislabel_like_frac=0.01,
                       outlier_frac=0.05, missing_row_frac=0.08, missing_cells=(5, 20),
                       corrupt_feature_frac=0.30, corrupt_value_frac=0.15, profile_id=3),
    },
    # perturbations of the noisy dataset (stability analysis)
    "n_perturbations": 20,
    "perturbation_ranges": {
        "feature_keep_frac": (0.60, 0.95),
        "row_keep_frac": (0.70, 0.95),
        "extra_noise_sd": (0.05, 0.40),            # in units of robust column sd
        "extra_missing_cell_frac": (0.003, 0.02),  # extra MCAR cells
    },
}

FILES = {"clean": "clean_baseline.csv", "noisy": "noisy_dataset.csv",
         "severe": "severe_noise_dataset.csv"}

# informative-dimension specification (natural order 0..14)
INF_SPEC = [
    # (feature_type, distribution text, signal strength)
    ("continuous_nonlinear", "Moon-arc coordinate x: clusters 1&2 form interlocking crescents, clusters 0&3 Gaussian blobs", "strong"),
    ("continuous_nonlinear", "Moon-arc coordinate y: clusters 1&2 form interlocking crescents, clusters 0&3 Gaussian blobs", "strong"),
    ("continuous_gaussian", "Gaussian, cluster-specific mean shift + correlated within-cluster noise", "strong"),
    ("continuous_gaussian", "Gaussian, cluster-specific mean shift + correlated within-cluster noise", "strong"),
    ("continuous_gaussian", "Gaussian, cluster-specific mean shift + correlated within-cluster noise", "strong"),
    ("continuous_gaussian", "Gaussian, cluster-specific mean shift + correlated within-cluster noise", "strong"),
    ("continuous_nonlinear", "Quadratic coupling: z6 + 0.15*z7^2", "strong"),
    ("continuous_nonlinear", "Quadratic coupling: z7 + 0.15*z8^2", "strong"),
    ("continuous_nonlinear", "Non-monotone sinusoid: 1.5*sin(0.7*z8) + 0.3*z8", "strong"),
    ("continuous_skewed", "Log-normal-like skew: exp(0.35*z9)", "moderate"),
    ("continuous_skewed", "Log-normal-like skew: exp(0.35*z10)", "moderate"),
    ("continuous_nonlinear", "Nonlinear mix: z11 + 0.6*tanh(z12)", "moderate"),
    ("continuous_nonlinear", "Signed power transform: sign(z12)*|z12|^1.3", "moderate"),
    ("continuous_gaussian", "Weak Gaussian (cluster-mean weight 0.25)", "weak"),
    ("continuous_gaussian", "Weak Gaussian (cluster-mean weight 0.25)", "weak"),
]
INF_WEIGHTS = np.array([1.0] * 9 + [0.8] * 4 + [0.25] * 2)


# =============================================================================
# GENERATOR
# =============================================================================
class SyntheticClusterGenerator:
    """Draws all random ingredients ONCE (fixed seed) so every profile is a
    deterministic, comparable corruption of the same hidden structure."""

    def __init__(self, cfg):
        self.cfg = cfg
        rng = np.random.default_rng(cfg["random_seed"])
        N, D, K = cfg["n_samples"], cfg["n_features"], cfg["n_clusters"]
        P, R, Q = cfg["n_informative"], cfg["n_redundant"], cfg["n_noise"]
        assert P + R + Q == D
        self.N, self.D, self.K, self.P, self.R, self.Q = N, D, K, P, R, Q

        # ---- labels (mild imbalance), shuffled so ids are not sorted by cluster
        sizes = np.round(np.array(cfg["cluster_proportions"]) * N).astype(int)
        sizes[-1] = N - sizes[:-1].sum()
        y = np.repeat(np.arange(K), sizes)
        self.y = y[rng.permutation(N)]
        self.sizes = sizes

        # ---- correlated within-cluster variation (3 blocks of informative dims)
        corr = np.eye(P)
        for a, b in cfg["informative_corr_groups"]:
            corr[a:b, a:b] = cfg["within_group_rho"]
        np.fill_diagonal(corr, 1.0)
        L = np.linalg.cholesky(corr)
        sd_k = np.array(cfg["cluster_within_sd"])
        self.z0 = (rng.standard_normal((N, P)) @ L.T) * sd_k[self.y][:, None]

        # ---- cluster centres with meaningful separation
        while True:
            C = rng.normal(size=(K, P))
            C -= C.mean(0)
            C *= INF_WEIGHTS
            d = pdist(C)
            C *= cfg["mean_center_distance"] / d.mean()
            d = pdist(C)
            if d.min() / d.mean() > 0.70:       # avoid accidental near-duplicate centres
                break
        self.C = C
        self.t_moon = rng.uniform(0, np.pi, N)

        # ---- ambiguous / mislabel-like rows (fixed priority orders -> nested across profiles)
        self.partner = (self.y + rng.integers(1, K, N)) % K
        self.lam_b = rng.uniform(*cfg["borderline_lambda"], N)
        self.lam_m = rng.uniform(*cfg["mislabel_like_lambda"], N)
        self.perm_ml = rng.permutation(N)
        self.perm_bd = rng.permutation(N)
        self.perm_out = rng.permutation(N)
        self.perm_miss = rng.permutation(N)

        # ---- redundant-feature recipe
        self.src1 = rng.permutation(P)
        self.src2 = (self.src1 + rng.integers(1, P, R)) % P
        self.w1 = rng.uniform(0.70, 0.90, R)
        self.w2 = rng.uniform(0.00, 0.30, R)
        self.w3 = rng.uniform(0.10, 0.25, R)
        self.sgn = rng.choice([-1.0, 1.0], R)
        self.eps_r = rng.standard_normal((N, R))

        # ---- pure-noise block (independent of cluster)
        self.noise_feats, self.noise_meta = self._make_noise_features(rng)

        # ---- fixed noise ingredients shared across profiles
        self.eps = rng.standard_normal((N, D))
        self.eps_t = rng.standard_t(3, (N, D)) / np.sqrt(3.0)
        self.u_noise = rng.uniform(size=D)
        self.u_heavy = rng.uniform(size=D)
        self.corrupt_order = rng.permutation(P + R)   # corruption only hits informative/redundant

        # ---- raw scales / offsets and shuffled column order
        lo, hi = cfg["feature_scale_log10_range"]
        self.scale = 10 ** rng.uniform(lo, hi, D)
        self.offset = self.scale * rng.uniform(-3, 6, D)
        self.out_order = rng.permutation(D)           # output column i <- natural column out_order[i]
        self.feature_names = [f"feature_{i + 1:03d}" for i in range(D)]
        self.name_of_natural = {int(n): self.feature_names[i] for i, n in enumerate(self.out_order)}

        # ---- baseline standardisation statistics (clean geometry, no ambiguity)
        x0 = self._raw_informative(1.0, None)
        self.mu_i, self.sd_i = x0.mean(0), x0.std(0)
        inf0 = (x0 - self.mu_i) / self.sd_i
        r0 = self._raw_redundant(inf0)
        self.mu_r, self.sd_r = r0.mean(0), r0.std(0)

    # ------------------------------------------------------------------ noise cols
    def _make_noise_features(self, rng):
        N, Q, cfg = self.N, self.Q, self.cfg
        cols, meta = [], []
        # 0-29 Gaussian: two correlated blocks (10 each) + 10 independent
        corr = np.eye(30)
        for a, b in cfg["noise_corr_groups"]:
            corr[a:b, a:b] = cfg["noise_group_rho"]
        np.fill_diagonal(corr, 1.0)
        g = rng.standard_normal((N, 30)) @ np.linalg.cholesky(corr).T
        for j in range(30):
            grp = "NG1" if j < 10 else ("NG2" if j < 20 else "none")
            cols.append(g[:, j])
            meta.append(("continuous_noise_gaussian", "Standard Gaussian pure noise" +
                         (f" (correlated block {grp}, rho={cfg['noise_group_rho']})" if grp != "none" else ""), grp))
        for _ in range(15):   # uniform
            cols.append(rng.uniform(-1, 1, N)); meta.append(("continuous_noise_uniform", "Uniform(-1,1) pure noise", "none"))
        for _ in range(15):   # heavy tailed
            cols.append(rng.standard_t(4, N)); meta.append(("continuous_noise_heavy_tail", "Student-t(df=4) pure noise", "none"))
        for _ in range(Q - 60):   # skewed
            cols.append(np.exp(0.6 * rng.standard_normal(N))); meta.append(("continuous_noise_skewed", "Log-normal(sigma=0.6) pure noise", "none"))
        M = np.column_stack(cols)
        M = (M - M.mean(0)) / M.std(0)
        return M, meta

    # ------------------------------------------------------------------ signal
    def _eff_centers(self, sep):
        """Effective centres including the crescent geometry in dims 0,1."""
        Ce = self.C * sep
        Rs = self.cfg["moon_radius"] * sep
        mid = (Ce[1, :2] + Ce[2, :2]) / 2
        Ce = Ce.copy()
        # mean of moon 1 (cos t, sin t) -> (0, 2/pi);  moon 2 (1-cos t, .5-sin t) -> (1, .5-2/pi);  re-centred by (.5,.25)
        Ce[1, :2] = mid + Rs * np.array([-0.5, 2 / np.pi - 0.25])
        Ce[2, :2] = mid + Rs * np.array([0.5, 0.25 - 2 / np.pi])
        return Ce

    def _raw_informative(self, sep, shifts):
        y = self.y
        Ce = self._eff_centers(sep)
        Rs = self.cfg["moon_radius"] * sep
        mid = (self.C[1, :2] + self.C[2, :2]) * sep / 2
        z = Ce[y] + self.z0
        for c in (1, 2):                                       # interlocking crescents
            idx = np.where(y == c)[0]
            t = self.t_moon[idx]
            m = np.c_[np.cos(t), np.sin(t)] if c == 1 else np.c_[1 - np.cos(t), 0.5 - np.sin(t)]
            m = m - np.array([0.5, 0.25])
            z[idx, :2] = mid + Rs * m + self.cfg["moon_noise"] * self.z0[idx, :2]
        if shifts is not None:                                 # ambiguous rows move towards another centre
            rows, partner, lam = shifts
            z[rows] += lam[:, None] * (Ce[partner] - Ce[y[rows]])
        x = np.empty_like(z)
        x[:, 0:6] = z[:, 0:6]
        x[:, 6] = z[:, 6] + 0.15 * z[:, 7] ** 2
        x[:, 7] = z[:, 7] + 0.15 * z[:, 8] ** 2
        x[:, 8] = 1.5 * np.sin(0.7 * z[:, 8]) + 0.3 * z[:, 8]
        x[:, 9] = np.exp(0.35 * z[:, 9])
        x[:, 10] = np.exp(0.35 * z[:, 10])
        x[:, 11] = z[:, 11] + 0.6 * np.tanh(z[:, 12])
        x[:, 12] = np.sign(z[:, 12]) * np.abs(z[:, 12]) ** 1.3
        x[:, 13:15] = z[:, 13:15]
        return x

    def _raw_redundant(self, inf_std):
        return self.sgn * (self.w1 * inf_std[:, self.src1] + self.w2 * inf_std[:, self.src2]) + self.w3 * self.eps_r

    # ------------------------------------------------------------------ build one dataset
    def build(self, name):
        cfg, p = self.cfg, self.cfg["profiles"][name]
        N, D = self.N, self.D
        rng = np.random.default_rng(cfg["random_seed"] + 10 * p["profile_id"])

        # ambiguous rows: mislabel-like first, borderline from the remaining rows
        n_ml = int(round(p["mislabel_like_frac"] * N))
        n_bd = int(round(p["borderline_frac"] * N))
        ml = self.perm_ml[:n_ml]
        bd = self.perm_bd[~np.isin(self.perm_bd, ml)][:n_bd]
        rows = np.concatenate([ml, bd]).astype(int)
        lam = np.concatenate([self.lam_m[ml], self.lam_b[bd]])
        shifts = (rows, self.partner[rows], lam) if len(rows) else None

        x = self._raw_informative(p["sep_scale"], shifts)
        inf = (x - self.mu_i) / self.sd_i
        red = (self._raw_redundant(inf) - self.mu_r) / self.sd_r
        X = np.hstack([inf, red, self.noise_feats])

        # measurement noise (feature-specific sd; optional heavy-tailed subset)
        lo, hi = p["meas_noise"]
        noise_sd = lo + self.u_noise * (hi - lo)
        heavy = self.u_heavy < p["heavy_tail_frac"]
        E = np.where(heavy[None, :], self.eps_t, self.eps)
        X = X + E * noise_sd

        # feature corruption: a share of values in some signal features is shuffled
        n_cf = int(round(p["corrupt_feature_frac"] * (self.P + self.R)))
        corrupted = self.corrupt_order[:n_cf]
        for j in corrupted:
            r = rng.choice(N, int(p["corrupt_value_frac"] * N), replace=False)
            X[r, j] = rng.permutation(X[r, j].copy())

        # outliers
        n_out = int(round(p["outlier_frac"] * N))
        out_rows = self.perm_out[:n_out]
        klo, khi = cfg["outlier_k_features"]
        mlo, mhi = cfg["outlier_magnitude"]
        for r in out_rows:
            k = rng.integers(klo, khi + 1)
            cols = rng.choice(D, k, replace=False)
            X[r, cols] = rng.choice([-1.0, 1.0], k) * rng.uniform(mlo, mhi, k)

        # different raw feature scales
        raw = self.offset + X * self.scale

        # MCAR missingness in a fraction of rows
        n_miss = int(round(p["missing_row_frac"] * N))
        for r in self.perm_miss[:n_miss]:
            k = rng.integers(p["missing_cells"][0], p["missing_cells"][1] + 1)
            raw[r, rng.choice(D, k, replace=False)] = np.nan

        df = pd.DataFrame(raw[:, self.out_order], columns=self.feature_names)
        noise_flag = np.zeros(N, dtype=int)
        noise_flag[bd] = 1
        noise_flag[ml] = 2
        outlier_flag = np.zeros(N, dtype=int)
        outlier_flag[out_rows] = 1
        meta = pd.DataFrame({
            "sample_id": [f"S{i + 1:04d}" for i in range(N)],
            "true_cluster": self.y,
            "observed_cluster": pd.array([pd.NA] * N, dtype="Int64"),
            "noise_flag": noise_flag,
            "outlier_flag": outlier_flag,
            "missing_value_count": df.isna().sum(axis=1).to_numpy(),
        })
        out = pd.concat([meta, df], axis=1)
        info = dict(noise_sd=noise_sd, heavy=heavy, corrupted=[int(c) for c in corrupted],
                    n_borderline=int(len(bd)), n_mislabel_like=int(len(ml)), n_outliers=int(n_out),
                    n_missing_rows=int(n_miss))
        return out, info


# =============================================================================
# PERTURBATIONS OF THE NOISY DATASET
# =============================================================================
def make_perturbations(noisy, gen, cfg):
    feats = gen.feature_names
    rngs = cfg["perturbation_ranges"]
    out, manifest = [], []
    for i in range(1, cfg["n_perturbations"] + 1):
        seed = cfg["random_seed"] + 1000 + i
        rng = np.random.default_rng(seed)
        fk = rng.uniform(*rngs["feature_keep_frac"])
        rk = rng.uniform(*rngs["row_keep_frac"])
        ns = rng.uniform(*rngs["extra_noise_sd"])
        mf = rng.uniform(*rngs["extra_missing_cell_frac"])
        keep_f = np.sort(rng.choice(len(feats), int(round(fk * len(feats))), replace=False))
        keep_r = np.sort(rng.choice(len(noisy), int(round(rk * len(noisy))), replace=False))
        kept = [feats[j] for j in keep_f]
        X = noisy.loc[keep_r, kept].to_numpy(float).copy()
        med = np.nanmedian(X, 0)
        rsd = 1.4826 * np.nanmedian(np.abs(X - med), 0)          # robust column sd
        X = X + rng.standard_normal(X.shape) * ns * rsd           # extra noise (NaN stays NaN)
        X[rng.random(X.shape) < mf] = np.nan                      # extra MCAR cells
        pid = f"perturbation_{i:03d}"
        base = noisy.loc[keep_r, ["sample_id", "true_cluster", "observed_cluster", "noise_flag", "outlier_flag"]].reset_index(drop=True)
        base.insert(1, "perturbation_id", pid)
        d = pd.DataFrame(X, columns=kept)
        base["missing_value_count"] = d.isna().sum(axis=1).to_numpy()
        df = pd.concat([base, d], axis=1)
        out.append((pid, df))
        manifest.append(dict(perturbation_id=pid, seed=seed, n_rows=len(keep_r), n_features=len(kept),
                             feature_keep_frac=round(fk, 4), row_keep_frac=round(rk, 4),
                             extra_noise_sd=round(ns, 4), extra_missing_cell_frac=round(mf, 4),
                             dropped_features=";".join(f for f in feats if f not in set(kept))))
    return out, pd.DataFrame(manifest)


# =============================================================================
# QUALITY CONTROL
# =============================================================================
def preprocess(df, cols):
    """Suggested preprocessing: median impute -> robust winsorise (+-4 MAD-z) -> standardise."""
    X = df[cols].to_numpy(float)
    med = np.nanmedian(X, 0)
    X = np.where(np.isnan(X), med, X)
    mad = 1.4826 * np.median(np.abs(X - med), 0) + 1e-9
    return StandardScaler().fit_transform(np.clip((X - med) / mad, -4, 4))


def variation_of_information(a, b):
    def H(l):
        _, c = np.unique(l, return_counts=True)
        p = c / c.sum()
        return -(p * np.log(p)).sum()
    return float(H(a) + H(b) - 2 * mutual_info_score(a, b))


def cluster_all(Z, k, seed):
    out = {"KMeans": KMeans(k, n_init=10, random_state=seed).fit_predict(Z),
           "Agglomerative-Ward": AgglomerativeClustering(k, linkage="ward").fit_predict(Z),
           "GMM-diag": GaussianMixture(k, covariance_type="diag", n_init=2, random_state=seed).fit_predict(Z)}
    try:
        out["Spectral-kNN"] = SpectralClustering(k, affinity="nearest_neighbors", n_neighbors=15,
                                                 random_state=seed, assign_labels="kmeans").fit_predict(Z)
    except Exception:
        pass
    return out


def evaluate(labels, y, Z):
    return dict(ARI=float(adjusted_rand_score(y, labels)), NMI=float(normalized_mutual_info_score(y, labels)),
                VI=variation_of_information(y, labels), Silhouette=float(silhouette_score(Z, labels)))


def band_ratio(df, feats, y):
    """Between-cluster / within-cluster variance ratio for each feature (NaN-aware)."""
    g = df[feats].groupby(y)
    n, m, v = g.count(), g.mean(), g.var()
    tot = df[feats].mean()
    ssb = (n * (m - tot) ** 2).sum()
    ssw = (v * (n - 1)).sum()
    return (ssb / ssw)


def ensemble_check(perturbed, gen, k, seed):
    """Does perturbation-based consensus beat single runs? (honest diagnostic, not a claim)"""
    N = gen.N
    id_to_idx = {f"S{i + 1:04d}": i for i in range(N)}
    y = gen.y
    runs = []
    for pid, df in perturbed:
        feats = [c for c in df.columns if c.startswith("feature_")]
        Z = preprocess(df, feats)
        rows = df["sample_id"].map(id_to_idx).to_numpy()
        for algo in ("KMeans", "Ward"):
            lab = (KMeans(k, n_init=3, random_state=seed).fit_predict(Z) if algo == "KMeans"
                   else AgglomerativeClustering(k, linkage="ward").fit_predict(Z))
            full = -np.ones(N, int)
            full[rows] = lab
            runs.append(dict(pid=pid, algo=algo, rows=rows, lab=lab, full=full))
    ari_truth = [adjusted_rand_score(y[r["rows"]], r["lab"]) for r in runs]
    n = len(runs)
    agree = np.full((n, n), np.nan)
    for a in range(n):
        for b in range(a + 1, n):
            m = (runs[a]["full"] >= 0) & (runs[b]["full"] >= 0)
            agree[a, b] = agree[b, a] = adjusted_rand_score(runs[a]["full"][m], runs[b]["full"][m])
    stab_w = np.nanmean(agree, axis=1)                       # per-run stability weight
    res = {}
    for tag, w in (("unweighted", np.ones(n)), ("stability_weighted", np.clip(stab_w, 1e-3, None))):
        num = np.zeros((N, N), np.float32)
        den = np.zeros((N, N), np.float32)
        for r, wi in zip(runs, w):
            ix = np.ix_(r["rows"], r["rows"])
            same = (r["lab"][:, None] == r["lab"][None, :]).astype(np.float32)
            num[ix] += wi * same
            den[ix] += wi
        dist = 1.0 - num / np.maximum(den, 1e-6)
        np.fill_diagonal(dist, 0.0)
        try:
            cl = AgglomerativeClustering(k, metric="precomputed", linkage="average").fit_predict(dist)
        except TypeError:
            cl = AgglomerativeClustering(k, affinity="precomputed", linkage="average").fit_predict(dist)
        res[tag] = dict(ARI=float(adjusted_rand_score(y, cl)), NMI=float(normalized_mutual_info_score(y, cl)))
        del num, den, dist
    res["single_run_ARI_mean"] = float(np.mean(ari_truth))
    res["single_run_ARI_min"] = float(np.min(ari_truth))
    res["single_run_ARI_max"] = float(np.max(ari_truth))
    res["mean_pairwise_ARI_between_runs"] = float(np.nanmean(agree))
    res["n_runs"] = n
    return res


def run_qc(datasets, infos, perturbed, manifest, gen, cfg):
    k, N, D, seed = cfg["n_clusters"], cfg["n_samples"], cfg["n_features"], cfg["random_seed"]
    feats = gen.feature_names
    natural_role = np.array(["informative"] * gen.P + ["redundant"] * gen.R + ["noise"] * gen.Q)
    names_by_role = {r: [gen.name_of_natural[j] for j in np.where(natural_role == r)[0]] for r in ("informative", "redundant", "noise")}
    sig_names = names_by_role["informative"] + names_by_role["redundant"]
    qc = {"checks": {}, "datasets": {}}

    # ---- structural checks
    for name, df in datasets.items():
        d = qc["datasets"][name] = {}
        d["shape"] = list(df.shape)
        d["n_true_clusters"] = int(df["true_cluster"].nunique())
        d["cluster_sizes"] = {int(c): int(v) for c, v in df["true_cluster"].value_counts().sort_index().items()}
        d["missing_rows_pct"] = float((df["missing_value_count"] > 0).mean() * 100)
        d["missing_cells_pct"] = float(df[feats].isna().mean().mean() * 100)
        d["outlier_pct"] = float(df["outlier_flag"].mean() * 100)
        d["borderline_n"] = int((df["noise_flag"] == 1).sum())
        d["mislabel_like_n"] = int((df["noise_flag"] == 2).sum())
        d["observed_cluster_all_blank"] = bool(df["observed_cluster"].isna().all())
    qc["checks"]["dimensions_ok"] = all(v["shape"] == [N, 6 + D] for v in qc["datasets"].values())
    qc["checks"]["four_true_clusters"] = all(v["n_true_clusters"] == 4 for v in qc["datasets"].values())
    qc["checks"]["clean_has_no_missing_or_outliers"] = (qc["datasets"]["clean"]["missing_cells_pct"] == 0 and qc["datasets"]["clean"]["outlier_pct"] == 0)
    qc["checks"]["noisy_missing_rows_about_3pct"] = abs(qc["datasets"]["noisy"]["missing_rows_pct"] - 3.0) < 0.5
    qc["checks"]["noisy_outliers_about_2pct"] = abs(qc["datasets"]["noisy"]["outlier_pct"] - 2.0) < 0.5
    qc["checks"]["observed_cluster_blank"] = all(v["observed_cluster_all_blank"] for v in qc["datasets"].values())
    qc["checks"]["perturbation_count_ge_20"] = len(perturbed) >= 20

    # ---- signal-to-noise (between/within variance ratio)
    for name, df in datasets.items():
        r = band_ratio(df, feats, df["true_cluster"])
        d = qc["datasets"][name]
        d["bw_variance_ratio"] = {role: dict(mean=float(r[names_by_role[role]].mean()),
                                             min=float(r[names_by_role[role]].min()),
                                             max=float(r[names_by_role[role]].max()))
                                  for role in names_by_role}
        d["_bw_per_feature"] = r.to_dict()

    # ---- correlations (noisy dataset, pairwise-complete)
    corr = datasets["noisy"][feats].corr().abs()
    red_pairs = [corr.loc[gen.name_of_natural[gen.P + i], gen.name_of_natural[int(gen.src1[i])]] for i in range(gen.R)]
    def block_mean(names):
        sub = corr.loc[names, names].to_numpy()
        return float(sub[np.triu_indices(len(names), 1)].mean())
    inf_groups = {f"IG{g + 1}": [gen.name_of_natural[j] for j in range(a, b)] for g, (a, b) in enumerate(cfg["informative_corr_groups"])}
    noise_groups = {f"NG{g + 1}": [gen.name_of_natural[gen.P + gen.R + j] for j in range(a, b)] for g, (a, b) in enumerate(cfg["noise_corr_groups"])}
    cross = corr.loc[names_by_role["noise"], sig_names].to_numpy()
    iu = np.triu_indices(len(feats), 1)
    allc = corr.to_numpy()[iu]
    qc["correlations_noisy"] = dict(
        redundant_vs_primary_source_abs_r=dict(mean=float(np.mean(red_pairs)), min=float(np.min(red_pairs)), max=float(np.max(red_pairs))),
        informative_groups_mean_abs_r={g: block_mean(n) for g, n in inf_groups.items()},
        noise_groups_mean_abs_r={g: block_mean(n) for g, n in noise_groups.items()},
        noise_vs_signal_mean_abs_r=float(np.nanmean(cross)),
        n_pairs_abs_r_gt_0_7=int((allc > 0.7).sum()), n_pairs_abs_r_gt_0_5=int((allc > 0.5).sum()))

    # ---- separation of true clusters
    for name, df in datasets.items():
        Zs = preprocess(df, sig_names)
        y = df["true_cluster"].to_numpy()
        cent = np.vstack([Zs[y == c].mean(0) for c in range(k)])
        within = np.sqrt(np.mean([Zs[y == c].var(0).sum() for c in range(k)]))
        dist = pdist(cent)
        qc["datasets"][name]["separation"] = dict(
            centroid_dist_min=float(dist.min()), centroid_dist_mean=float(dist.mean()),
            within_cluster_rms_radius=float(within), mean_centroid_dist_over_radius=float(dist.mean() / within),
            silhouette_true_labels_signal_features=float(silhouette_score(Zs, y)))

    # ---- clustering performance
    perf = {}
    for name, df in datasets.items():
        y = df["true_cluster"].to_numpy()
        Zall = preprocess(df, feats)
        Zsig = preprocess(df, sig_names)
        perf[name] = {"all_100_features": {a: evaluate(l, y, Zall) for a, l in cluster_all(Zall, k, seed).items()},
                      "oracle_30_signal_features": {a: evaluate(l, y, Zsig) for a, l in cluster_all(Zsig, k, seed).items()}}
    qc["clustering_performance"] = perf
    ari = lambda nm, algo="KMeans": perf[nm]["all_100_features"][algo]["ARI"]
    algos = list(perf["clean"]["all_100_features"])
    qc["checks"]["clean_beats_severe_all_algorithms"] = all(
        perf["clean"]["all_100_features"][a]["ARI"] > perf["severe"]["all_100_features"][a]["ARI"] for a in algos if a in perf["severe"]["all_100_features"])
    qc["checks"]["clean_ARI_gt_noisy_ARI_gt_severe_ARI_kmeans"] = ari("clean") > ari("noisy") > ari("severe")
    qc["checks"]["structure_recoverable_in_severe_with_signal_features"] = perf["severe"]["oracle_30_signal_features"]["KMeans"]["ARI"] > ari("severe")

    # ---- stability / ensemble suitability
    qc["perturbation_ensemble_check"] = ensemble_check(perturbed, gen, k, seed)
    qc["checks"]["perturbations_create_measurable_instability"] = qc["perturbation_ensemble_check"]["mean_pairwise_ARI_between_runs"] < 0.999
    for v in qc["datasets"].values():
        v.pop("_bw_per_feature", None) if False else None
    return qc


# =============================================================================
# DOCUMENTATION OUTPUTS
# =============================================================================
def feature_description_table(gen, infos, qc):
    cfg = gen.cfg
    rows_by_name = {}
    bw = band_ratio  # alias for readability
    noisy_sd = infos["noisy"]["noise_sd"]
    clean_sd = infos["clean"]["noise_sd"]
    sev_sd = infos["severe"]["noise_sd"]
    ig = {j: f"IG{g + 1}" for g, (a, b) in enumerate(cfg["informative_corr_groups"]) for j in range(a, b)}
    for j in range(gen.D):
        if j < gen.P:
            ftype, dist, strength = INF_SPEC[j]
            grp, role = ig[j], "informative"
            intended = {"strong": "Primary discriminative signal; tests recovery of nonlinear / overlapping structure",
                        "moderate": "Secondary signal with skewed or nonlinear geometry",
                        "weak": "Weakly informative feature; tests whether ensembles exploit marginal signal"}[strength]
        elif j < gen.P + gen.R:
            i = j - gen.P
            s1 = int(gen.src1[i])
            ftype = "continuous_redundant"
            dist = (f"Linear mix {gen.sgn[i]:+.0f}*(%.2f*z[{gen.name_of_natural[s1]}] + %.2f*z[{gen.name_of_natural[int(gen.src2[i])]}]) + %.2f*N(0,1), then standardised"
                    % (gen.w1[i], gen.w2[i], gen.w3[i]))
            grp, role = ig[s1], "redundant"
            strength = "derived (" + INF_SPEC[s1][2] + ")"
            intended = f"Correlated duplicate of {gen.name_of_natural[s1]}; tests redundancy handling / double-counting of one signal"
        else:
            ftype, dist, grp = gen.noise_meta[j - gen.P - gen.R]
            role, strength = "noise", "none"
            intended = "Pure noise independent of cluster membership; dilutes distance-based clustering"
        nm = gen.name_of_natural[j]
        rows_by_name[nm] = dict(feature_name=nm, feature_type=ftype, informative_or_noise=role,
                                generating_distribution=dist, correlation_group=grp,
                                expected_signal_strength=strength, noise_level=round(float(noisy_sd[j]), 4),
                                intended_role=intended,
                                noise_level_clean=round(float(clean_sd[j]), 4), noise_level_severe=round(float(sev_sd[j]), 4),
                                heavy_tailed_noise_in_severe=bool(infos["severe"]["heavy"][j]),
                                corrupted_in_severe=bool(j in infos["severe"]["corrupted"]),
                                raw_scale=round(float(gen.scale[j]), 4), raw_offset=round(float(gen.offset[j]), 4))
    return pd.DataFrame([rows_by_name[n] for n in gen.feature_names])


def md_table(headers, rows):
    s = "| " + " | ".join(headers) + " |\n|" + "|".join(["---"] * len(headers)) + "|\n"
    for r in rows:
        s += "| " + " | ".join(str(x) for x in r) + " |\n"
    return s


def build_metadata(cfg, gen, datasets, infos, manifest, qc):
    return {
        "title": "Cluster Stability-Based Ensemble Learning for Noisy High-Dimensional Data - synthetic benchmark",
        "random_seed": cfg["random_seed"],
        "number_of_samples": cfg["n_samples"],
        "number_of_features": cfg["n_features"],
        "feature_roles": {"informative": cfg["n_informative"], "redundant": cfg["n_redundant"], "pure_noise": cfg["n_noise"]},
        "number_of_clusters": cfg["n_clusters"],
        "cluster_sizes": {str(c): int(s) for c, s in enumerate(gen.sizes)},
        "cluster_proportions": cfg["cluster_proportions"],
        "missing_value_percentage": {n: dict(rows_with_missing_pct=round(qc["datasets"][n]["missing_rows_pct"], 3),
                                             cells_missing_pct=round(qc["datasets"][n]["missing_cells_pct"], 4)) for n in datasets},
        "outlier_percentage": {n: round(qc["datasets"][n]["outlier_pct"], 3) for n in datasets},
        "ambiguous_samples": {n: dict(borderline=qc["datasets"][n]["borderline_n"], mislabel_like=qc["datasets"][n]["mislabel_like_n"]) for n in datasets},
        "noise_flag_coding": {"0": "regular sample", "1": "borderline sample (features generated 40-60% of the way towards another cluster centre)",
                              "2": "mislabel-like sample (features generated 85-100% of the way towards another cluster centre; true_cluster keeps the generating label)"},
        "missingness_mechanism": "MCAR: affected rows and cells are chosen uniformly at random, independent of cluster, feature value or outlier status.",
        "noise_parameters": {n: {k: v for k, v in cfg["profiles"][n].items() if k != "profile_id"} for n in cfg["profiles"]} | {
            "noise_sd_units": "standard deviations of the baseline-standardised feature; sd_j = lo + u_j*(hi-lo) with fixed u_j ~ U(0,1) per feature",
            "outlier_magnitude_abs_standardised": cfg["outlier_magnitude"], "outlier_features_per_row": cfg["outlier_k_features"]},
        "feature_generation_assumptions": [
            "Latent geometry: 4 clusters in 15 informative dimensions; cluster centres drawn from N(0,I), weighted (strong 1.0 / moderate 0.8 / weak 0.25) and rescaled to a mean pairwise distance of %.1f." % cfg["mean_center_distance"],
            "Within-cluster variation: cluster-specific sd %s with three correlated blocks of 5 informative dims (rho=%.2f)." % (cfg["cluster_within_sd"], cfg["within_group_rho"]),
            "Dims 0-1: clusters 1 and 2 are interlocking crescents (moon geometry); clusters 0 and 3 are Gaussian blobs -> non-convex, partially overlapping structure.",
            "Nonlinear transforms (quadratic, sinusoidal, exponential skew, tanh mix, signed power) applied to 11 of 15 informative dims.",
            "Redundant features: linear mixes of 1-2 standardised informative features plus independent noise (observed |r| with the primary source is lower in the noisy data because of measurement noise and nonlinear sources; see qc_report.json).",
            "Pure-noise features: 30 Gaussian (two correlated blocks of 10, rho=%.1f), 15 uniform, 15 Student-t(4), 10 log-normal; all independent of cluster." % cfg["noise_group_rho"],
            "Per-feature raw scale = 10^U(-1,2.7) and offset = scale*U(-3,6), identical across datasets.",
            "Feature column order is randomly shuffled (fixed seed) so position does not reveal feature role; roles are listed in feature_description.csv.",
            "All three datasets share the same latent draws (common random numbers); only the corruption profile differs.",
        ],
        "perturbation_settings": {
            "source_dataset": "noisy_dataset.csv", "n_perturbations": cfg["n_perturbations"],
            "seed_rule": "random_seed + 1000 + perturbation_index",
            "ranges": cfg["perturbation_ranges"],
            "operations": ["random feature subsampling (columns dropped)", "random row subsampling without replacement (sample_id preserved)",
                           "extra additive Gaussian noise scaled by robust column sd", "extra MCAR missing cells"],
            "preserved_columns": ["sample_id", "true_cluster", "noise_flag", "outlier_flag"],
            "manifest_file": "perturbation_manifest.csv",
            "summary": {"rows_min": int(manifest["n_rows"].min()), "rows_max": int(manifest["n_rows"].max()),
                        "features_min": int(manifest["n_features"].min()), "features_max": int(manifest["n_features"].max())},
        },
        "qc_checks": qc["checks"],
        "limitations": [
            "Synthetic data cannot reproduce all dependencies of real measurements; clusters are generated from a simple latent model.",
            "The attached document is a problem statement / methodology (PCCS); it specifies no dataset, domain or cluster count, so those values come from the task prompt or are assumptions.",
            "Missingness is MCAR only; real data often has MAR/MNAR missingness.",
            "No explicit batch/platform covariate or distribution-shift mechanism is simulated (the PCCS document lists them as perturbation families); they can be added as extensions.",
            "Mislabel-like and borderline samples keep their generating label as true_cluster, so ground truth remains the generative label, not an 'annotator' label.",
            "Perturbation files drop columns; analyses requiring identical columns must align on feature_name.",
            "Fixed seed gives ONE realisation; repeat with other seeds for statistical claims.",
        ],
    }


def build_readme(cfg, gen, qc, manifest, meta):
    ds = qc["datasets"]
    perf = qc["clustering_performance"]
    names = {"clean": "clean_baseline.csv", "noisy": "noisy_dataset.csv", "severe": "severe_noise_dataset.csv"}
    summary_rows = []
    for n in ("clean", "noisy", "severe"):
        p = cfg["profiles"][n]
        summary_rows.append([names[n], f"{ds[n]['shape'][0]} x {ds[n]['shape'][1]}", p["sep_scale"],
                             f"{p['meas_noise'][0]}-{p['meas_noise'][1]}", f"{ds[n]['missing_rows_pct']:.2f}% rows / {ds[n]['missing_cells_pct']:.2f}% cells",
                             f"{ds[n]['outlier_pct']:.2f}%", f"{ds[n]['borderline_n']} / {ds[n]['mislabel_like_n']}"])
    perf_rows = []
    for a in perf["clean"]["all_100_features"]:
        perf_rows.append([a] + [f"{perf[n]['all_100_features'][a]['ARI']:.3f}" for n in ("clean", "noisy", "severe")]
                         + [f"{perf[n]['oracle_30_signal_features'][a]['ARI']:.3f}" for n in ("clean", "noisy", "severe")])
    snr_rows = [[n] + [f"{ds[n]['bw_variance_ratio'][r]['mean']:.3f}" for r in ("informative", "redundant", "noise")] for n in ("clean", "noisy", "severe")]
    sep_rows = [[n, f"{ds[n]['separation']['centroid_dist_mean']:.2f}", f"{ds[n]['separation']['within_cluster_rms_radius']:.2f}",
                 f"{ds[n]['separation']['mean_centroid_dist_over_radius']:.3f}", f"{ds[n]['separation']['silhouette_true_labels_signal_features']:.3f}"] for n in ("clean", "noisy", "severe")]
    ens = qc["perturbation_ensemble_check"]
    cor = qc["correlations_noisy"]
    return f"""# Synthetic benchmark: Cluster Stability-Based Ensemble Learning for Noisy High-Dimensional Data

## 1. Research motivation
Cluster stability is usually summarised by one reproducibility number. The reference document supplied with this task
("Perturbation-Conditional Consensus Clustering", PCCS) argues that this is *perturbation-blind*: a cluster may survive random
seeds yet collapse under feature removal, noise, resampling or missingness. This benchmark provides data in which the
**true latent structure is known** while the observed features are noisy, high-dimensional, redundant, partly missing and
contaminated, so stability-based ensemble methods can be tested against a ground truth.

## 2. Paper-derived content vs. synthetic assumptions
**Taken from the reference document (a problem statement / approach description, not a data paper):**
- Central problem: stability should be studied as a function of perturbation type, intensity and interaction, not a single score.
- Data setting: X in R^(N x D), high-dimensional, with missingness, scale differences, extreme values; labels reserved for external evaluation only.
- Perturbation families named: seed/algorithm, sampling (bootstrap/subsampling), feature masking/removal, measurement (additive and heavy-tailed noise, missingness, outliers), batch/platform, distribution shift.
- Base methods named: K-Means, Fuzzy C-Means, DBSCAN, hierarchical; ensemble via co-association consensus; ARI named as partition similarity measure; cluster-level membership retention and co-association consistency.
- Not specified in the document: dataset, application domain, number of clusters, sample size, feature counts, noise magnitudes.

**Newly introduced synthetic assumptions (from the task prompt or chosen here):** N=3000, D=100 (15 informative / 15 redundant / 70 noise), 4 clusters
with proportions {cfg['cluster_proportions']}, all noise magnitudes, the moon/nonlinear geometry, MCAR missingness, outlier magnitudes, the
interpretation of "3% of observations" as 3% of *rows* (each affected row loses 3-12 cells), and the perturbation ranges. No result from the paper is reproduced or claimed.

## 3. Files
| File | Content |
|---|---|
| clean_baseline.csv | Low noise, no outliers, no missing values |
| noisy_dataset.csv | Moderate noise, 3% rows with missing cells, 2% outlier rows, ambiguous samples |
| severe_noise_dataset.csv | Higher noise (30% heavy-tailed features), feature corruption, 8% missing rows, 5% outliers, separation x{cfg['profiles']['severe']['sep_scale']} |
| perturbation_001.csv ... perturbation_020.csv | Perturbed copies of the noisy dataset (own seed each) |
| perturbation_manifest.csv | Seed, rows/features kept and perturbation strengths for each perturbation |
| feature_description.csv | Role, distribution, correlation group, signal strength, noise level of every feature |
| dataset_metadata.json | Machine-readable settings, assumptions and limitations |
| qc_report.json | All quality-control numbers |
| generation_script.py | Complete reproducible generator + QC |

## 4. Columns
- `sample_id` - S0001..S3000, identical across datasets (same row = same latent sample).
- `perturbation_id` - (perturbation files only) e.g. perturbation_007.
- `true_cluster` - hidden ground-truth cluster 0-3 (generating cluster). **Use for evaluation only.**
- `observed_cluster` - intentionally blank; fill with your clustering output.
- `noise_flag` - 0 regular; 1 borderline (features placed 40-60% of the way towards another centre); 2 mislabel-like (85-100% towards another centre, so it sits inside another cluster although `true_cluster` keeps its generating label).
- `outlier_flag` - 1 if the row received extreme values (5-20 features set to +/-(4-8) standardised units).
- `missing_value_count` - number of missing feature values in the row.
- `feature_001`..`feature_100` - continuous features, roles shuffled; see feature_description.csv. Perturbation files contain a *subset* of these columns.

## 5. How noise, outliers and missingness are introduced
1. Latent geometry with 4 clusters (sizes {', '.join(str(int(s)) for s in gen.sizes)}); clusters 1 and 2 are interlocking crescents in two dims; other dims use correlated Gaussian clusters with cluster-specific spread; weak features have centre weight 0.25.
2. Nonlinear transforms, standardisation with baseline statistics, then 15 redundant and 70 pure-noise features.
3. Feature-specific additive measurement noise (sd as fraction of feature sd): clean 0.05-0.15, noisy 0.30-0.70, severe 0.60-1.20 (30% of features Student-t(3)).
4. Severe only: 30% of signal features have 15% of their values shuffled across rows (feature corruption).
5. Outliers: random rows get 5-20 features replaced by +/-(4-8) standardised values.
6. Each feature is mapped to its own raw scale (0.1-500) and offset.
7. Missingness: **MCAR** - rows and cells chosen uniformly at random (3-12 cells in 3% of rows for noisy; 5-20 cells in 8% of rows for severe).
8. Perturbations: random feature subset (60-95%), random row subset (70-95%), extra Gaussian noise (0.05-0.40 robust sd) and 0.3-2% extra MCAR cells; seed = {cfg['random_seed']}+1000+index.

## 6. Reproduce
```bash
pip install numpy pandas scipy scikit-learn
python generation_script.py --out ./synthetic_cluster_data
```
All settings are in the `CONFIG` block at the top of the script; seed = {cfg['random_seed']}.

## 7. Quality-control results (seed {cfg['random_seed']})
{md_table(["Dataset", "Shape", "Separation scale", "Meas. noise sd", "Missing", "Outlier rows", "Borderline / mislabel-like"], summary_rows)}
**Between/within variance ratio (signal-to-noise) per feature group:**

{md_table(["Dataset", "Informative", "Redundant", "Pure noise"], snr_rows)}
**Separation of true clusters (30 signal features, robust-standardised):**

{md_table(["Dataset", "Mean centroid distance", "Within-cluster RMS radius", "Distance / radius", "Silhouette (true labels)"], sep_rows)}
**Adjusted Rand Index vs. true_cluster (k=4, suggested preprocessing):**

{md_table(["Algorithm", "Clean (100 feat.)", "Noisy (100 feat.)", "Severe (100 feat.)", "Clean (30 signal)", "Noisy (30 signal)", "Severe (30 signal)"], perf_rows)}
**Correlations (noisy dataset):** redundant vs primary source mean |r| = {cor['redundant_vs_primary_source_abs_r']['mean']:.2f} (min {cor['redundant_vs_primary_source_abs_r']['min']:.2f});
noise blocks NG1/NG2 mean |r| = {cor['noise_groups_mean_abs_r']['NG1']:.2f}/{cor['noise_groups_mean_abs_r']['NG2']:.2f};
noise-vs-signal mean |r| = {cor['noise_vs_signal_mean_abs_r']:.3f}; pairs with |r|>0.7: {cor['n_pairs_abs_r_gt_0_7']}.

**Perturbation / ensemble diagnostic** ({ens['n_runs']} runs = 20 perturbations x {{K-Means, Ward}}): mean single-run ARI {ens['single_run_ARI_mean']:.3f}
(min {ens['single_run_ARI_min']:.3f}, max {ens['single_run_ARI_max']:.3f}); mean pairwise ARI between runs {ens['mean_pairwise_ARI_between_runs']:.3f};
co-association consensus ARI {ens['unweighted']['ARI']:.3f} (unweighted) and {ens['stability_weighted']['ARI']:.3f} (stability-weighted). This is a quick diagnostic, not a benchmark claim.

**Automated checks:** {', '.join(f"{k}={'PASS' if v else 'FAIL'}" for k, v in qc['checks'].items())}

## 8. Suggested preprocessing
Median (or kNN/iterative) imputation; robust scaling (median/MAD) and winsorisation at about +/-4 robust z (used for the QC above);
optional log or Yeo-Johnson transform for skewed features; variance/correlation filtering or PCA (e.g. 20-30 components); supervised information
(`true_cluster`, `noise_flag`, `outlier_flag`) must not be used to build clusters. Keep one fixed protocol across all datasets.

## 9. Suggested clustering algorithms
K-Means; Agglomerative (Ward/average); Gaussian Mixture Models; Spectral clustering (kNN affinity - useful for the crescent pair); DBSCAN or HDBSCAN
(after dimensionality reduction; treat -1 as noise); Fuzzy C-Means if available.

## 10. Suggested ensemble-learning methods
Evidence-accumulation / co-association consensus; cluster-ensemble hypergraph methods (CSPA, HGPA, MCLA); consensus over feature- or
row-subspace ensembles; stability-weighted consensus (weight members/clusters by agreement across perturbations); perturbation-conditional
weighting as in PCCS; cluster-based ensemble learning (cluster assignments or distances as features for a downstream model).

## 11. Suggested cluster-stability measures
Adjusted Rand Index; Normalized Mutual Information; Variation of Information; Silhouette; per-cluster Jaccard similarity of matched clusters;
pairwise co-association matrix stability (variance/entropy of co-association entries); cluster membership consistency across perturbations;
all of these can be stratified by perturbation type using `perturbation_manifest.csv`.

## 12. Limitations
{chr(10).join('- ' + l for l in meta['limitations'])}
"""


# =============================================================================
# MAIN
# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="synthetic_cluster_data")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    cfg = CONFIG
    gen = SyntheticClusterGenerator(cfg)

    datasets, infos = {}, {}
    for name, fname in FILES.items():
        df, info = gen.build(name)
        datasets[name], infos[name] = df, info
        df.to_csv(os.path.join(args.out, fname), index=False, float_format="%.4f")
        print(f"saved {fname}: {df.shape}")

    perturbed, manifest = make_perturbations(datasets["noisy"], gen, cfg)
    for pid, df in perturbed:
        df.to_csv(os.path.join(args.out, f"{pid}.csv"), index=False, float_format="%.4f")
    manifest.to_csv(os.path.join(args.out, "perturbation_manifest.csv"), index=False)
    print(f"saved {len(perturbed)} perturbation files")

    print("running quality control ...")
    qc = run_qc(datasets, infos, perturbed, manifest, gen, cfg)
    with open(os.path.join(args.out, "qc_report.json"), "w") as f:
        json.dump(qc, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))

    feature_description_table(gen, infos, qc).to_csv(os.path.join(args.out, "feature_description.csv"), index=False)
    meta = build_metadata(cfg, gen, datasets, infos, manifest, qc)
    with open(os.path.join(args.out, "dataset_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    with open(os.path.join(args.out, "README.md"), "w") as f:
        f.write(build_readme(cfg, gen, qc, manifest, meta))

    print("\nQC checks:")
    for k, v in qc["checks"].items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print("\nARI (all 100 features):")
    for n in ("clean", "noisy", "severe"):
        print(" ", n, {a: round(m["ARI"], 3) for a, m in qc["clustering_performance"][n]["all_100_features"].items()})
    print("ARI (30 signal features):")
    for n in ("clean", "noisy", "severe"):
        print(" ", n, {a: round(m["ARI"], 3) for a, m in qc["clustering_performance"][n]["oracle_30_signal_features"].items()})
    print("Ensemble check:", json.dumps(qc["perturbation_ensemble_check"], indent=1))


if __name__ == "__main__":
    main()
