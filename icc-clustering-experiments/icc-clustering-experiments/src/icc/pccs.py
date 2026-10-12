"""
PCCS - Perturbation-Conditional Consensus Clustering.

Authoritative implementation adhering to:
"PERTURBATION-CONDITIONAL CONSENSUS CLUSTERING: Reliable Discovery of Invariant Structure in High-Dimensional Data"

Directly interfaces with the modular pccs package while maintaining full backwards compatibility.
"""
from pccs import (
    PCCS,
    PCCSConfig,
    FAMILIES,
    DEFAULT_LEVELS,
    METHODS,
    hungarian_match,
    partition_metrics,
    cluster_retention,
    StabilityTensor,
    compute_cluster_fingerprints,
    compute_psi,
    compute_interactions,
    classify_reliability,
    build_perturbation_aware_consensus,
    PreprocessingPipeline,
    validate_and_preprocess,
)
from pccs.ensemble import run_fcm as fcm, run_dbscan_adaptive

# Backwards compatibility helpers
def _singletonize(lab):
    import numpy as np
    lab = np.asarray(lab).copy()
    neg = lab < 0
    if neg.any():
        lab[neg] = lab.max() + 1 + np.arange(neg.sum())
    return np.unique(lab, return_inverse=True)[1]

__all__ = [
    "PCCS",
    "PCCSConfig",
    "FAMILIES",
    "DEFAULT_LEVELS",
    "METHODS",
    "hungarian_match",
    "partition_metrics",
    "cluster_retention",
    "StabilityTensor",
    "compute_cluster_fingerprints",
    "compute_psi",
    "compute_interactions",
    "classify_reliability",
    "build_perturbation_aware_consensus",
    "PreprocessingPipeline",
    "validate_and_preprocess",
    "fcm",
    "_singletonize",
]
