"""
PCCS - Perturbation-Conditional Consensus Clustering.

Authoritative implementation adhering to:
"PERTURBATION-CONDITIONAL CONSENSUS CLUSTERING: Reliable Discovery of Invariant Structure in High-Dimensional Data"
"""

from .model import PCCS, PCCSConfig
from .perturbations import FAMILIES, DEFAULT_LEVELS
from .ensemble import METHODS
from .matching import hungarian_match, partition_metrics, cluster_retention
from .tensor import StabilityTensor
from .fingerprints import compute_cluster_fingerprints, compute_psi
from .interactions import compute_interactions
from .classification import classify_reliability
from .consensus import build_perturbation_aware_consensus
from .data import PreprocessingPipeline, validate_and_preprocess

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
]
