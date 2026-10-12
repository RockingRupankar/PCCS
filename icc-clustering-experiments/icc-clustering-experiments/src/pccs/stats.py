"""
Statistical validation and inference for PCCS:
- Bootstrap confidence intervals (e.g. 95% CI)
- Paired statistical hypothesis tests (Wilcoxon signed-rank / paired t-test)
- Effect size quantification (Cohen's d, rank-biserial correlation)
- Multiple comparison correction (Holm-Bonferroni)
"""
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from scipy import stats


def bootstrap_ci(
    data: np.ndarray,
    stat_fn=np.mean,
    confidence: float = 0.95,
    n_resamples: int = 2000,
    random_state: int = 42,
) -> Tuple[float, float, float]:
    """
    Compute non-parametric bootstrap confidence interval.
    Returns (point_estimate, ci_lower, ci_upper).
    """
    arr = np.asarray(data, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return 0.0, 0.0, 0.0
    point_est = float(stat_fn(arr))
    if len(arr) < 2:
        return point_est, point_est, point_est

    rng = np.random.default_rng(random_state)
    boot_stats = []
    n = len(arr)
    for _ in range(n_resamples):
        sample = rng.choice(arr, size=n, replace=True)
        boot_stats.append(stat_fn(sample))

    alpha = 1.0 - confidence
    lower = float(np.percentile(boot_stats, 100.0 * (alpha / 2.0)))
    upper = float(np.percentile(boot_stats, 100.0 * (1.0 - alpha / 2.0)))
    return point_est, lower, upper


def cohens_d(x: np.ndarray, y: np.ndarray, paired: bool = True) -> float:
    """Compute Cohen's d effect size."""
    x_clean = np.asarray(x, dtype=float)
    y_clean = np.asarray(y, dtype=float)
    if paired:
        diff = x_clean - y_clean
        sd = np.std(diff, ddof=1)
        return float(np.mean(diff) / (sd + 1e-9))
    else:
        nx, ny = len(x_clean), len(y_clean)
        vx, vy = np.var(x_clean, ddof=1), np.var(y_clean, ddof=1)
        pooled_sd = np.sqrt(((nx - 1) * vx + (ny - 1) * vy) / max(1, nx + ny - 2))
        return float((np.mean(x_clean) - np.mean(y_clean)) / (pooled_sd + 1e-9))


def paired_comparison_test(
    scores_method_a: np.ndarray,
    scores_method_b: np.ndarray,
) -> Dict[str, float]:
    """
    Perform paired comparison test (Wilcoxon and paired t-test) and effect size.
    """
    a = np.asarray(scores_method_a, dtype=float)
    b = np.asarray(scores_method_b, dtype=float)
    valid = (~np.isnan(a)) & (~np.isnan(b))
    av, bv = a[valid], b[valid]
    n = len(av)

    if n < 3:
        return dict(
            mean_a=float(np.mean(av)) if n else 0.0,
            mean_b=float(np.mean(bv)) if n else 0.0,
            mean_diff=float(np.mean(av - bv)) if n else 0.0,
            cohens_d=0.0,
            p_val_ttest=1.0,
            p_val_wilcoxon=1.0,
        )

    t_res = stats.ttest_rel(av, bv)
    try:
        w_res = stats.wilcoxon(av, bv)
        w_pval = float(w_res.pvalue)
    except Exception:
        w_pval = 1.0

    d = cohens_d(av, bv, paired=True)

    return dict(
        mean_a=float(np.mean(av)),
        mean_b=float(np.mean(bv)),
        mean_diff=float(np.mean(av - bv)),
        cohens_d=d,
        p_val_ttest=float(t_res.pvalue),
        p_val_wilcoxon=w_pval,
    )


def holm_bonferroni(p_values: List[float]) -> List[float]:
    """Adjust p-values using Holm-Bonferroni step-down procedure."""
    m = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        adj = min(1.0, p * (m - rank))
        running_max = max(running_max, adj)
        adjusted[orig_idx] = running_max
    return adjusted
