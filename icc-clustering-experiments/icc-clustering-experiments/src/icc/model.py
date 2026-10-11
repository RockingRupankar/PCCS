"""
ICC - Inductive Consensus Clustering (reference implementation used by the experiments).

Pipeline
  1. Build perturbed views of the data under distinct perturbation families
     (seed, row, feature, noise, combined).
  2. Run heterogeneous base clusterers (k-means, Ward, optional GMM) on every view.
  3. Weight each partition by its mean agreement (ARI on shared rows) with all other partitions.
  4. Build one weighted co-association matrix per perturbation family and fuse them
     (mean over families blended with the minimum over families).
  5. Cluster the fused matrix (average linkage; spectral fallback if tiny clusters appear).
  6. Report a per-cluster stability fingerprint over perturbation families and a reliability category.
  7. Inductive step: assign unseen rows with a feature-weighted nearest-centroid rule.

This file is a self-contained reference implementation. If your repository already contains its own ICC
implementation, keep it and adapt experiments/common/icc_adapter.py instead.
"""
from dataclasses import dataclass, field
import time
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture


# ----------------------------------------------------------------- preprocessing
def preprocess_fit(X, clip=4.0):
    """Median impute -> robust winsorise (+-clip MAD-z) -> standardise."""
    X = np.asarray(X, float)
    med = np.nanmedian(X, 0)
    med = np.where(np.isnan(med), 0.0, med)
    Xf = np.where(np.isnan(X), med, X)
    mad = 1.4826 * np.median(np.abs(Xf - med), 0)
    mad = np.where(mad < 1e-9, Xf.std(0) + 1e-9, mad)
    Z = np.clip((Xf - med) / mad, -clip, clip)
    mu, sd = Z.mean(0), Z.std(0) + 1e-9
    return (Z - mu) / sd, dict(med=med, mad=mad, mu=mu, sd=sd, clip=clip)


def preprocess_apply(X, p):
    X = np.asarray(X, float)
    Xf = np.where(np.isnan(X), p["med"], X)
    Z = np.clip((Xf - p["med"]) / p["mad"], -p["clip"], p["clip"])
    return (Z - p["mu"]) / p["sd"]


# ----------------------------------------------------------------- config
@dataclass
class ICCConfig:
    n_clusters: int = 4
    n_views: int = 20
    families: tuple = ("seed", "row", "feature", "noise", "combined")
    row_frac: float = 0.8
    feat_frac: float = 0.5
    noise_sd: float = 0.25               # in standardised units
    base_methods: tuple = ("kmeans", "ward")
    kmeans_n_init: int = 3
    use_stability_weights: bool = True
    weight_power: float = 2.0
    use_family_fusion: bool = True
    invariance_lambda: float = 0.5       # 0 = mean over families, 1 = min over families
    linkage: str = "average"
    min_cluster_frac: float = 0.02       # smaller -> spectral fallback
    feature_weighted_assignment: bool = True
    hi: float = 0.75                     # fingerprint thresholds (assumptions)
    lo: float = 0.50
    random_state: int = 0


# ----------------------------------------------------------------- helpers
def _run_base(method, V, k, seed, n_init):
    if method == "kmeans":
        return KMeans(k, n_init=n_init, random_state=seed).fit_predict(V)
    if method == "ward":
        return AgglomerativeClustering(k, linkage="ward").fit_predict(V)
    if method == "gmm":
        return GaussianMixture(k, covariance_type="diag", random_state=seed).fit_predict(V)
    raise ValueError(method)


def _agglo_precomputed(D, k, linkage):
    try:
        return AgglomerativeClustering(k, metric="precomputed", linkage=linkage).fit_predict(D)
    except TypeError:
        return AgglomerativeClustering(k, affinity="precomputed", linkage=linkage).fit_predict(D)


def feature_weights(Z, labels, k):
    """sqrt(between/within variance ratio), normalised to mean 1."""
    tot = Z.mean(0)
    ssb = np.zeros(Z.shape[1])
    ssw = np.zeros(Z.shape[1])
    for c in range(k):
        m = labels == c
        if m.sum() < 2:
            continue
        mc = Z[m].mean(0)
        ssb += m.sum() * (mc - tot) ** 2
        ssw += ((Z[m] - mc) ** 2).sum(0)
    w = np.sqrt(ssb / (ssw + 1e-9))
    return w / (w.mean() + 1e-12)


# ----------------------------------------------------------------- model
class ICC:
    def __init__(self, cfg=None, **kw):
        self.cfg = cfg or ICCConfig()
        for k, v in kw.items():
            setattr(self.cfg, k, v)

    # ---- views + base clusterings
    def _make_partitions(self, Z, rng):
        c = self.cfg
        n, d = Z.shape
        parts = []
        for v in range(c.n_views):
            fam = c.families[v % len(c.families)]
            rows, cols = np.arange(n), np.arange(d)
            if fam in ("row", "combined"):
                rows = np.sort(rng.choice(n, max(int(c.row_frac * n), c.n_clusters * 5), replace=False))
            if fam in ("feature", "combined"):
                cols = np.sort(rng.choice(d, max(int(c.feat_frac * d), 2), replace=False))
            V = Z[np.ix_(rows, cols)]
            if fam in ("noise", "combined"):
                V = V + rng.normal(0, c.noise_sd, V.shape)
            for m in c.base_methods:
                lab = _run_base(m, V, c.n_clusters, int(rng.integers(1_000_000_000)), c.kmeans_n_init)
                parts.append(dict(family=fam, method=m, rows=rows, labels=lab))
        return parts

    def fit(self, X):
        t0 = time.time()
        c = self.cfg
        rng = np.random.default_rng(c.random_state)
        Z, self.prep_ = preprocess_fit(X)
        self.Z_ = Z
        n = Z.shape[0]
        parts = self._make_partitions(Z, rng)
        P = len(parts)

        # ---- stability weights (agreement of each partition with all others on shared rows)
        full = -np.ones((P, n), int)
        for i, p in enumerate(parts):
            full[i, p["rows"]] = p["labels"]
        A = np.full((P, P), np.nan)
        for i in range(P):
            for j in range(i + 1, P):
                m = (full[i] >= 0) & (full[j] >= 0)
                A[i, j] = A[j, i] = adjusted_rand_score(full[i][m], full[j][m])
        stab = np.nanmean(A, 1)
        w = np.clip(stab, 1e-3, None) ** c.weight_power if c.use_stability_weights else np.ones(P)
        self.partition_stability_ = stab
        self.mean_agreement_ = float(np.nanmean(A))

        # ---- per-family weighted co-association
        fams = list(dict.fromkeys(p["family"] for p in parts))
        num_tot = np.zeros((n, n), np.float32)
        den_tot = np.zeros((n, n), np.float32)
        Cf, fam_w = {}, {}
        for f in fams:
            num = np.zeros((n, n), np.float32)
            den = np.zeros((n, n), np.float32)
            ws = []
            for p, wi in zip(parts, w):
                if p["family"] != f:
                    continue
                ix = np.ix_(p["rows"], p["rows"])
                num[ix] += wi * (p["labels"][:, None] == p["labels"][None, :])
                den[ix] += wi
                ws.append(wi)
            num_tot += num
            den_tot += den
            Cf[f] = np.where(den > 0, num / np.maximum(den, 1e-9), 0).astype(np.float32)
            fam_w[f] = float(np.mean(ws))
            del num, den

        # ---- fusion
        if c.use_family_fusion:
            alpha = np.array([fam_w[f] for f in fams])
            alpha = alpha / alpha.sum()
            Cmean = sum(a * Cf[f] for a, f in zip(alpha, fams))
            if c.invariance_lambda > 0:
                Cmin = np.minimum.reduce([Cf[f] for f in fams])
                C = (1 - c.invariance_lambda) * Cmean + c.invariance_lambda * Cmin
            else:
                C = Cmean
        else:
            C = np.where(den_tot > 0, num_tot / np.maximum(den_tot, 1e-9), 0)
        del num_tot, den_tot
        C = ((C + C.T) / 2).astype(np.float32)
        np.fill_diagonal(C, 1.0)

        # ---- consensus
        D = 1.0 - C
        labels = _agglo_precomputed(D, c.n_clusters, c.linkage)
        self.used_fallback_ = False
        if np.bincount(labels, minlength=c.n_clusters).min() < c.min_cluster_frac * n:
            labels = SpectralClustering(c.n_clusters, affinity="precomputed",
                                        random_state=c.random_state).fit_predict(np.clip(C, 0, 1))
            self.used_fallback_ = True
        self.labels_ = labels
        self.consensus_ = C if getattr(self, "keep_consensus", False) else None

        # ---- fingerprints
        rows = []
        for k in range(c.n_clusters):
            idx = np.where(labels == k)[0]
            r = dict(cluster=k, size=len(idx))
            for f in fams:
                if len(idx) < 2:
                    r[f] = np.nan
                else:
                    sub = Cf[f][np.ix_(idx, idx)]
                    r[f] = float((sub.sum() - len(idx)) / (len(idx) * (len(idx) - 1)))
            rows.append(r)
        fp = pd.DataFrame(rows)
        vals = fp[fams].to_numpy()
        cats = []
        for v in vals:
            if np.all(v >= c.hi):
                cats.append("globally_invariant")
            elif np.any(v >= c.hi) and np.nanmin(v) >= c.lo:
                cats.append("conditionally_invariant")
            elif np.any(v >= c.hi):
                cats.append("perturbation_fragile")
            else:
                cats.append("ambiguous")
        fp["reliability"] = cats
        fp["PSI"] = 1 - np.nanmean(vals, 1)       # equal-weight perturbation sensitivity index
        self.fingerprint_ = fp
        del Cf

        # ---- assignment model for new rows
        self.fw_ = feature_weights(Z, labels, c.n_clusters) if c.feature_weighted_assignment else np.ones(Z.shape[1])
        self.centroids_ = np.vstack([Z[labels == k].mean(0) if np.any(labels == k) else np.full(Z.shape[1], 1e6)
                                     for k in range(c.n_clusters)])
        self.fit_time_ = time.time() - t0
        return self

    def fit_predict(self, X):
        return self.fit(X).labels_

    def predict(self, Xnew):
        Zn = preprocess_apply(Xnew, self.prep_)
        d = (((Zn[:, None, :] - self.centroids_[None, :, :]) ** 2) * self.fw_).sum(2)
        return d.argmin(1)


ICCv1 = ICC  # backward-compatible alias used by older experiment scripts
__all__ = ["ICC", "ICCv1", "ICCConfig", "preprocess_fit", "preprocess_apply", "feature_weights"]
