"""
Evaluation and statistical testing module for PCCS.
"""
from pccs.evaluation.metrics import compute_clustering_metrics
from pccs.evaluation.stats import (
    compute_cohens_d,
    paired_wilcoxon_test,
    holm_bonferroni_correction,
)

__all__ = [
    "compute_clustering_metrics",
    "compute_cohens_d",
    "paired_wilcoxon_test",
    "holm_bonferroni_correction",
]
