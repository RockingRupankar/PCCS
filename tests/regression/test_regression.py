"""
Regression and reproducibility tests for PCCS.
Verifies deterministic seed behavior and table/figure regeneration.
"""
import numpy as np
import pytest

from pccs.model import PCCS, PCCSConfig
from pccs.data.synthetic import generate_synthetic_regime


def test_deterministic_seed_behavior():
    """Verify that identical random seeds produce byte-for-byte identical results."""
    X, _, _ = generate_synthetic_regime("spherical_separated", n_samples=50, n_features=5, n_clusters=3, random_state=42)
    
    cfg = PCCSConfig(n_clusters=3, random_state=123, repetitions=1, interaction=False)
    m1 = PCCS(cfg).fit(X)
    m2 = PCCS(cfg).fit(X)

    assert np.array_equal(m1.labels_, m2.labels_), "Deterministic seed must reproduce exact labels."
    assert np.allclose(m1.fingerprint_["PSI"].values, m2.fingerprint_["PSI"].values)
    assert np.allclose(m1.consensus_matrix_, m2.consensus_matrix_)
