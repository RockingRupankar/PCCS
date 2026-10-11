import numpy as np
import pytest
from icc import ICC, ICCConfig


@pytest.fixture(scope="module")
def blobs():
    rng = np.random.default_rng(0)
    centers = rng.normal(scale=6, size=(3, 8))
    y = np.repeat(np.arange(3), 50)
    X = centers[y] + rng.normal(size=(150, 8))
    X = np.hstack([X, rng.normal(size=(150, 12))])        # 12 pure-noise columns
    X[rng.random(X.shape) < 0.01] = np.nan                # a little missingness
    return X, y


def test_fit_labels_and_cluster_count(blobs):
    X, y = blobs
    m = ICC(ICCConfig(n_clusters=3, n_views=10, random_state=0)).fit(X)
    assert m.labels_.shape == (150,)
    assert set(np.unique(m.labels_)) == {0, 1, 2}


def test_recovers_clear_structure(blobs):
    from sklearn.metrics import adjusted_rand_score
    X, y = blobs
    m = ICC(ICCConfig(n_clusters=3, n_views=10, random_state=0)).fit(X)
    assert adjusted_rand_score(y, m.labels_) > 0.9


def test_deterministic_for_fixed_seed(blobs):
    X, _ = blobs
    a = ICC(ICCConfig(n_clusters=3, n_views=10, random_state=3)).fit_predict(X)
    b = ICC(ICCConfig(n_clusters=3, n_views=10, random_state=3)).fit_predict(X)
    assert np.array_equal(a, b)


def test_fingerprint_structure(blobs):
    X, _ = blobs
    cfg = ICCConfig(n_clusters=3, n_views=10, random_state=0)
    fp = ICC(cfg).fit(X).fingerprint_
    assert list(fp.columns[:2]) == ["cluster", "size"]
    for fam in cfg.families:
        assert fam in fp.columns
        assert ((fp[fam].dropna() >= 0) & (fp[fam].dropna() <= 1)).all()
    allowed = {"globally_invariant", "conditionally_invariant", "perturbation_fragile", "ambiguous"}
    assert set(fp.reliability) <= allowed
    assert fp["size"].sum() == 150


def test_inductive_predict(blobs):
    X, _ = blobs
    m = ICC(ICCConfig(n_clusters=3, n_views=10, random_state=0)).fit(X)
    pred = m.predict(X[:20])
    assert pred.shape == (20,)
    assert (pred == m.labels_[:20]).mean() > 0.9


def test_ablation_switches_run(blobs):
    X, _ = blobs
    for kw in (dict(use_stability_weights=False), dict(use_family_fusion=False), dict(invariance_lambda=0.0),
               dict(families=("seed",)), dict(base_methods=("kmeans",))):
        m = ICC(ICCConfig(n_clusters=3, n_views=6, random_state=0, **kw)).fit(X)
        assert len(m.labels_) == 150
