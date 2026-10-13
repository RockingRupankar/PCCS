"""
Statistical hypothesis testing, multiple comparison adjustments, and effect sizes.
"""
from typing import Dict, Any, List, Tuple
import numpy as np
from scipy.stats import wilcoxon


def compute_cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """Calculate Cohen's d effect size for paired or independent samples."""
    diff = x - y
    n = len(diff)
    if n < 2:
        return 0.0
    s_diff = np.std(diff, ddof=1)
    if s_diff < 1e-8:
        return 0.0
    return float(np.mean(diff) / s_diff)


def paired_wilcoxon_test(
    scores_proposed: np.ndarray,
    scores_baseline: np.ndarray,
    alternative: str = "greater",
) -> Tuple[float, float]:
    """
    Paired Wilcoxon signed-rank test.
    Returns:
    - statistic: Wilcoxon test statistic
    - p_value: p-value
    """
    diff = scores_proposed - scores_baseline
    if np.all(diff == 0):
        return 0.0, 1.0
    # Add minor jitter if all values identical to prevent zero differences error
    try:
        res = wilcoxon(scores_proposed, scores_baseline, alternative=alternative)
        return float(res.statistic), float(res.pvalue)
    except Exception:
        return 0.0, 1.0


def holm_bonferroni_correction(
    p_values: List[float],
    alpha: float = 0.05,
) -> Tuple[List[float], List[bool]]:
    """
    Holm-Bonferroni step-down procedure for family-wise error rate control.
    Returns:
    - adjusted_p_values
    - rejected: boolean array indicating statistical significance
    """
    m = len(p_values)
    if m == 0:
        return [], []

    indices = np.argsort(p_values)
    sorted_p = np.array(p_values)[indices]
    
    adj_p = np.zeros(m)
    rejected = np.zeros(m, dtype=bool)

    cum_max = 0.0
    for i, p in enumerate(sorted_p):
        k = i + 1
        adj = (m - k + 1) * p
        adj = min(1.0, max(cum_max, adj))
        cum_max = adj
        adj_p[i] = adj

    # Significance test
    rejected = adj_p <= alpha

    # Restore original order
    inv_indices = np.empty_like(indices)
    inv_indices[indices] = np.arange(m)
    
    final_adj_p = adj_p[inv_indices].tolist()
    final_rejected = rejected[inv_indices].tolist()

    return final_adj_p, final_rejected
