"""
PCCS: Perturbation-Conditional Consensus Clustering.
Reliable Discovery of Invariant Structure in High-Dimensional Data.
"""
from pccs.model import PCCS, PCCSConfig
from pccs.perturbations.engine import FAMILIES, DEFAULT_LEVELS
from pccs.stability.tensor import StabilityTensor
from pccs.stability.matching import hungarian_match, partition_metrics, cluster_retention
from pccs.stability.fingerprints import compute_cluster_fingerprints, compute_psi
from pccs.stability.classification import classify_reliability, sweep_classification_thresholds
from pccs.interactions.factorial import compute_pairwise_interactions
from pccs.consensus.builder import build_perturbation_aware_consensus
from pccs.preprocessing.pipeline import PreprocessingPipeline, validate_and_preprocess

__all__ = [
    "PCCS",
    "PCCSConfig",
    "FAMILIES",
    "DEFAULT_LEVELS",
    "StabilityTensor",
    "hungarian_match",
    "partition_metrics",
    "cluster_retention",
    "compute_cluster_fingerprints",
    "compute_psi",
    "classify_reliability",
    "sweep_classification_thresholds",
    "compute_pairwise_interactions",
    "build_perturbation_aware_consensus",
    "PreprocessingPipeline",
    "validate_and_preprocess",
]
