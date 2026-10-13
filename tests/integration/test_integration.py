"""
Integration tests for full PCCS estimator pipeline and label isolation.
"""
import numpy as np
import pytest

from pccs.model import PCCS, PCCSConfig
from pccs.data.synthetic import generate_synthetic_regime
from pccs.data.loaders import load_benchmark_dataset


def test_full_pccs_pipeline():
    """Verify that full PCCS estimator fits, clusters, and exposes diagnostics."""
    X, y, meta = generate_synthetic_regime("spherical_separated", n_samples=60, n_features=6, n_clusters=3, random_state=42)
    model = PCCS(PCCSConfig(n_clusters=3, random_state=42, repetitions=1, interaction=True))
    model.fit(X)

    assert hasattr(model, "labels_")
    assert len(model.labels_) == 60
    assert hasattr(model, "fingerprint_")
    assert len(model.fingerprint_) == 3
    assert "PSI" in model.fingerprint_.columns
    assert "reliability_category" in model.fingerprint_.columns
    assert hasattr(model, "interaction_")
    assert hasattr(model, "tensor_")
    assert hasattr(model, "computational_stats_")

    # Test out-of-sample prediction
    preds = model.predict(X[:5])
    assert len(preds) == 5


def test_label_isolation():
    """Verify that ground-truth labels are strictly external and never affect fitting."""
    X, y, _ = generate_synthetic_regime("spherical_separated", n_samples=50, n_features=5, n_clusters=2, random_state=42)
    
    # Fit model without labels
    m1 = PCCS(PCCSConfig(n_clusters=2, random_state=42, repetitions=1, interaction=False)).fit(X)

    # Fit model with labels passed to y
    m2 = PCCS(PCCSConfig(n_clusters=2, random_state=42, repetitions=1, interaction=False)).fit(X, y=y)

    assert np.array_equal(m1.labels_, m2.labels_), "Reference labels must have zero influence on cluster assignments."
    assert np.allclose(m1.fingerprint_["PSI"], m2.fingerprint_["PSI"]), "Reference labels must not affect stability fingerprints."
