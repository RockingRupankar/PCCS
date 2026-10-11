import numpy as np
from common.datasets import make_pccs, make_scalable, corrupt


def test_pccs_profiles_have_expected_structure():
    X, y, names, df = make_pccs("noisy", 400, seed=1)
    assert X.shape == (400, 100) and len(names) == 100
    assert set(np.unique(y)) == {0, 1, 2, 3}
    assert list(df.columns[:6]) == ["sample_id", "true_cluster", "observed_cluster", "noise_flag",
                                    "outlier_flag", "missing_value_count"]
    assert df["observed_cluster"].isna().all()
    assert (df["missing_value_count"] > 0).any() and df["outlier_flag"].sum() > 0


def test_clean_profile_has_no_missing_or_outliers():
    X, y, _, df = make_pccs("clean", 400, seed=1)
    assert not np.isnan(X).any()
    assert df["outlier_flag"].sum() == 0 and (df["noise_flag"] == 0).all()


def test_pccs_reproducible():
    a, *_ = make_pccs("severe", 300, seed=5)
    b, *_ = make_pccs("severe", 300, seed=5)
    assert np.array_equal(a, b, equal_nan=True)


def test_make_scalable_shapes_and_balance():
    X, y = make_scalable(500, 30, k=4, seed=0)
    assert X.shape == (500, 30) and np.bincount(y).min() >= 124


def test_corrupt_adds_noise_features_and_missing():
    X = np.random.default_rng(0).normal(size=(200, 5))
    Xc = corrupt(X, n_noise_features=20, seed=0)
    assert Xc.shape == (200, 25) and np.isnan(Xc).any()
