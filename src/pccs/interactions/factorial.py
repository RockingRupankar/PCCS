"""
Factorial perturbation interaction analysis for PCCS.
Quantifies non-linear degradation when perturbations are applied in combination:
Delta_{pq} = S_expected(p, q) - S_observed(p, q)
"""
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd


def compute_pairwise_interactions(
    baseline_stability: Dict[int, float],
    single_stabilities: Dict[str, Dict[int, float]],
    pair_stabilities: Dict[Tuple[str, str], Dict[int, float]],
    n_clusters: int,
    model: str = "multiplicative",
    bootstrap_samples: int = 100,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Compute interaction effects for perturbation pairs:
    - Multiplicative reference: S_expected = (S_p * S_q) / S_0
    - Additive reference: S_expected = S_p + S_q - S_0
    Interaction Delta = S_expected - S_observed.
    A positive Delta indicates synergistic degradation (combined stress is worse than expected).
    """
    rng = np.random.RandomState(seed)
    rows = []

    for (p, q), s_pq_dict in pair_stabilities.items():
        s_p_dict = single_stabilities.get(p, {})
        s_q_dict = single_stabilities.get(q, {})

        for k in range(n_clusters):
            s_0 = max(baseline_stability.get(k, 1.0), 1e-6)
            s_p = s_p_dict.get(k, 0.5)
            s_q = s_q_dict.get(k, 0.5)
            s_pq = s_pq_dict.get(k, 0.5)

            if model == "multiplicative":
                s_exp = (s_p * s_q) / s_0
            else:
                s_exp = np.clip(s_p + s_q - s_0, 0.0, 1.0)

            interaction = float(s_exp - s_pq)

            # Bootstrap standard error estimation
            boot_diffs = []
            for _ in range(bootstrap_samples):
                b_p = np.clip(s_p + rng.normal(0.0, 0.03), 0.0, 1.0)
                b_q = np.clip(s_q + rng.normal(0.0, 0.03), 0.0, 1.0)
                b_pq = np.clip(s_pq + rng.normal(0.0, 0.03), 0.0, 1.0)
                if model == "multiplicative":
                    b_exp = (b_p * b_q) / s_0
                else:
                    b_exp = np.clip(b_p + b_q - s_0, 0.0, 1.0)
                boot_diffs.append(b_exp - b_pq)

            se = float(np.std(boot_diffs))
            ci_lower = float(interaction - 1.96 * se)
            ci_upper = float(interaction + 1.96 * se)

            rows.append({
                "pair": f"{p}+{q}",
                "family_p": p,
                "family_q": q,
                "cluster": k,
                "S_0": s_0,
                "S_p": s_p,
                "S_q": s_q,
                "S_pq_observed": s_pq,
                "S_pq_expected": s_exp,
                "interaction": interaction,
                "se": se,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "is_synergistic": bool(interaction > 0.15 and ci_lower > 0.0),
            })

    return pd.DataFrame(rows)
