"""Baseline clusterers with an inductive interface: fit on X_train, label X_test."""
import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from .icc_adapter import ICCv1, ICCConfig, preprocess_fit, preprocess_apply, _agglo_precomputed

METHODS = ["kmeans", "ward", "gmm_diag", "spectral_knn", "eac", "icc_v1"]


def run_method(name, Xtr, Xte, k, seed, icc_kw=None):
    """Return (labels_train, labels_test or None)."""
    if name == "icc_v1":
        m = ICCv1(ICCConfig(n_clusters=k, random_state=seed, **(icc_kw or {})))
        ltr = m.fit_predict(Xtr)
        return ltr, (m.predict(Xte) if Xte is not None else None)
    Ztr, prep = preprocess_fit(Xtr)
    Zte = preprocess_apply(Xte, prep) if Xte is not None else None
    if name == "kmeans":
        m = KMeans(k, n_init=10, random_state=seed).fit(Ztr)
        return m.labels_, (m.predict(Zte) if Zte is not None else None)
    if name == "gmm_diag":
        m = GaussianMixture(k, covariance_type="diag", n_init=2, random_state=seed).fit(Ztr)
        return m.predict(Ztr), (m.predict(Zte) if Zte is not None else None)
    if name == "ward":
        ltr = AgglomerativeClustering(k, linkage="ward").fit_predict(Ztr)
        return ltr, (NearestCentroid().fit(Ztr, ltr).predict(Zte) if Zte is not None else None)
    if name == "spectral_knn":
        ltr = SpectralClustering(k, affinity="nearest_neighbors", n_neighbors=15,
                                 random_state=seed, assign_labels="kmeans").fit_predict(Ztr)
        return ltr, (KNeighborsClassifier(10).fit(Ztr, ltr).predict(Zte) if Zte is not None else None)
    if name == "eac":      # classic evidence accumulation: many k-means with random k, average linkage
        rng = np.random.default_rng(seed)
        n = len(Ztr)
        C = np.zeros((n, n), np.float32)
        B = 20
        for _ in range(B):
            kk = int(rng.integers(k, 2 * k + 1))
            lab = KMeans(kk, n_init=1, random_state=int(rng.integers(1e9))).fit_predict(Ztr)
            C += (lab[:, None] == lab[None, :])
        C /= B
        ltr = _agglo_precomputed(1 - C, k, "average")
        return ltr, (NearestCentroid().fit(Ztr, ltr).predict(Zte) if Zte is not None else None)
    raise ValueError(name)
