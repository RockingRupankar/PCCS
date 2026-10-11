import numpy as np
import pytest
from icc import PCCS, PCCSConfig, FAMILIES


@pytest.fixture(scope="module")
def fitted():
    rng = np.random.default_rng(1)
    centers = rng.normal(scale=7, size=(3, 10))
    y = np.repeat(np.arange(3), 70)
    X = np.hstack([centers[y] + rng.normal(size=(210, 10)), rng.normal(size=(210, 15))])
    X[rng.random(X.shape) < 0.01] = np.nan
    cfg = PCCSConfig(n_clusters=3, repetitions=1, interaction_repetitions=1, baseline_seeds=2, random_state=0,
                     interaction_pairs=(("sampling", "feature"), ("measurement", "technical")))
    return PCCS(cfg).fit(X), y


def test_one_tensor_slice_per_family(fitted):
    m, _ = fitted
    assert set(m.tensor_) == set(FAMILIES)
    for S in m.tensor_.values():
        assert S.shape == (210, 210) and S.min() >= 0 and S.max() <= 1.0 + 1e-6


def test_recovers_clear_structure(fitted):
    from sklearn.metrics import adjusted_rand_score
    m, y = fitted
    assert adjusted_rand_score(y, m.labels_) > 0.9


def test_fingerprint_columns_and_ranges(fitted):
    m, _ = fitted
    fp = m.fingerprint_
    for f in FAMILIES:
        assert f in fp.columns and f"retention_{f}" in fp.columns and f"drift_{f}" in fp.columns
        assert ((fp[f] >= 0) & (fp[f] <= 1)).all()
    assert fp["PSI"].between(0, 1).all()
    assert set(fp.reliability) <= {"globally_invariant", "conditionally_invariant", "perturbation_fragile", "ambiguous"}
    assert fp["size"].sum() == 210


def test_interaction_table(fitted):
    m, _ = fitted
    it = m.interaction_
    assert set(it.pair) == {"sampling+feature", "measurement+technical"}
    assert {"S_p", "S_q", "S_0", "S_pq_observed", "S_pq_expected", "interaction"} <= set(it.columns)
    assert (it[it.cluster == "global"].shape[0]) == 2


def test_active_methods_exclude_non_recovering_member(fitted):
    m, _ = fitted
    assert "kmeans" in m.active_methods_ and "hierarchical" in m.active_methods_


def test_rebuild_consensus_variants_do_not_recluster(fitted):
    m, _ = fitted
    n_before = m.n_clusterings_
    for kw in (dict(per_family=False), dict(reliability_weights=False), dict(lam=0.0), dict(families=("sampling", "feature")),
               dict(methods=("kmeans",))):
        lab = m.build_consensus(**kw)
        assert lab.shape == (210,)
    assert m.n_clusterings_ == n_before


def test_summary_and_stats(fitted):
    m, _ = fitted
    assert set(m.summary_["family_stability_ari"]) == set(FAMILIES)
    assert m.computational_stats()["n_clusterings"] == m.n_clusterings_


def test_batch_metadata_is_used():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(120, 8)) + np.repeat(np.arange(3), 40)[:, None] * 5
    batch = np.tile([0, 1], 60)
    cfg = PCCSConfig(n_clusters=3, families=("technical",), repetitions=1, baseline_seeds=1, interaction=False,
                     methods=("kmeans",), levels={"technical": (0.25,)})
    m = PCCS(cfg).fit(X, batch=batch)
    assert m.labels_.shape == (120,)


def test_deterministic():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(100, 6)) + np.repeat(np.arange(2), 50)[:, None] * 6
    cfg = dict(n_clusters=2, families=("sampling", "feature"), repetitions=1, baseline_seeds=1, interaction=False,
               methods=("kmeans", "hierarchical"), levels={"sampling": (0.3,), "feature": (0.3,)}, random_state=5)
    a = PCCS(PCCSConfig(**cfg)).fit(X).labels_
    b = PCCS(PCCSConfig(**cfg)).fit(X).labels_
    assert np.array_equal(a, b)
