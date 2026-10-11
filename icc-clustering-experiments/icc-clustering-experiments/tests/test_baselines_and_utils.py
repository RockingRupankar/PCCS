import numpy as np
import pytest
from common.baselines import METHODS, run_method
from common.utils import scores, pairwise_ari, vi


@pytest.mark.parametrize("method", METHODS)
def test_every_method_runs_and_labels_test_rows(method):
    rng = np.random.default_rng(0)
    y = np.repeat(np.arange(3), 40)
    centers = rng.normal(scale=6, size=(3, 10))
    X = centers[y] + rng.normal(size=(120, 10))
    ltr, lte = run_method(method, X[:90], X[90:], 3, 0, icc_kw=dict(n_views=6) if method == "icc_v1" else None)
    assert len(ltr) == 90 and len(lte) == 30


def test_scores_perfect_and_permuted():
    y = np.array([0, 0, 1, 1, 2, 2])
    s = scores(y, np.array([2, 2, 0, 0, 1, 1]))
    assert s["ARI"] == pytest.approx(1.0) and s["NMI"] == pytest.approx(1.0) and s["VI"] == pytest.approx(0.0)


def test_vi_positive_for_different_partitions():
    assert vi(np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1])) > 0


def test_pairwise_ari_ignores_unobserved_rows():
    a = np.array([0, 0, 1, 1, -1])
    b = np.array([1, 1, 0, 0, 0])
    assert pairwise_ari([a, b]) == pytest.approx(1.0)
