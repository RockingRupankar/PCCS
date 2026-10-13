"""
Unit tests for PCCS components.
"""
import numpy as np
import pytest

from pccs.preprocessing.pipeline import PreprocessingPipeline, validate_and_preprocess
from pccs.perturbations.engine import apply_perturbation, FAMILIES
from pccs.clustering.ensemble import run_kmeans, run_dbscan, run_hierarchical
from pccs.clustering.baselines import run_eac_consensus, run_twec_consensus


def test_preprocessing_pipeline():
    """Test MAD winsorization, imputation, and standardization."""
    X = np.array([
        [1.0, 2.0, np.nan],
        [1.2, 2.1, 5.0],
        [0.9, 1.9, 5.1],
        [100.0, -100.0, 4.9],  # Outliers
    ])
    X_proc, pipe = validate_and_preprocess(X, clip_mad=2.0)
    assert not np.any(np.isnan(X_proc)), "Preprocessing must impute all NaNs."
    assert X_proc.shape == (4, 3)
    # Check that outlier values were clipped
    assert np.all(np.abs(X_proc) < 10.0)


def test_perturbation_engine_families():
    """Verify that each perturbation family produces valid outputs and tracked metrics."""
    rng = np.random.RandomState(42)
    X = rng.normal(size=(50, 10))

    for fam in FAMILIES:
        X_p, r_idx, c_idx, metrics = apply_perturbation(X, fam, level=0.25, seed=42)
        assert len(r_idx) > 0
        assert len(c_idx) > 0
        assert X_p.shape == (len(r_idx), len(c_idx))
        assert "realized_intensity" in metrics
        assert metrics["family"] == fam


def test_base_clustering_algorithms():
    """Verify that base clustering algorithms return valid partitions."""
    rng = np.random.RandomState(42)
    X = rng.normal(size=(40, 5))

    lbls_km, info_km = run_kmeans(X, n_clusters=3, random_state=42)
    assert len(lbls_km) == 40
    assert len(np.unique(lbls_km)) == 3

    lbls_hi, _ = run_hierarchical(X, n_clusters=3)
    assert len(lbls_hi) == 40
    assert len(np.unique(lbls_hi)) == 3

    lbls_db, info_db = run_dbscan(X)
    assert len(lbls_db) == 40


def test_baseline_consensus():
    """Verify unweighted EAC and TWEC consensus execution."""
    p1 = np.array([0, 0, 1, 1, 2, 2])
    p2 = np.array([0, 0, 1, 1, 2, 2])
    p3 = np.array([1, 1, 0, 0, 2, 2])  # permuted

    cons_eac = run_eac_consensus([p1, p2, p3], n_clusters=3)
    assert len(cons_eac) == 6
    assert cons_eac[0] == cons_eac[1]
    assert cons_eac[2] == cons_eac[3]
    assert cons_eac[4] == cons_eac[5]

    cons_twec = run_twec_consensus([p1, p2, p3], n_clusters=3)
    assert len(cons_twec) == 6
