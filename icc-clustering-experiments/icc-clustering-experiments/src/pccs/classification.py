"""
Defensible reliability classifications for PCCS:
- Globally Invariant: consistently reliable across all major perturbation families, low response variance, no severe interaction collapse.
- Conditionally Invariant: reliable under identifiable conditions, but sensitive to specific perturbations.
- Perturbation-Fragile: substantial degradation under one or more perturbation families.
- Ambiguous: unstable or inconsistent evidence preventing confident structural interpretation.
"""
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd


def classify_reliability(
    fingerprint_df: pd.DataFrame,
    families: Tuple[str, ...],
    interaction_df: Optional[pd.DataFrame] = None,
    hi_threshold: float = 0.75,
    lo_threshold: float = 0.50,
    max_sd_threshold: float = 0.15,
    interaction_threshold: float = 0.15,
) -> pd.DataFrame:
    """
    Assign descriptive reliability classifications based on observed multidimensional fingerprints.
    """
    df = fingerprint_df.copy()
    fams = [f for f in families if f in df.columns]

    # Max interaction per cluster
    inter_max = np.zeros(len(df), dtype=float)
    if interaction_df is not None and len(interaction_df) > 0:
        ik = interaction_df[interaction_df.cluster != "global"].copy()
        if len(ik) > 0:
            ik["cluster"] = ik["cluster"].astype(int)
            grouped = ik.groupby("cluster").interaction.max()
            inter_max = np.array([float(grouped.get(k, 0.0)) for k in df["cluster"]])

    df["max_interaction"] = inter_max

    categories = []
    rationale = []

    for i, row in df.iterrows():
        scores = np.array([float(row[f]) for f in fams])
        min_s = float(np.min(scores)) if len(scores) else 0.0
        max_s = float(np.max(scores)) if len(scores) else 0.0
        n_hi = int((scores >= hi_threshold).sum())
        sd_val = float(row.get("max_response_sd", 0.0))
        inter_val = float(inter_max[i])

        if n_hi == len(fams) and sd_val <= max_sd_threshold and inter_val < interaction_threshold:
            cat = "globally_invariant"
            reason = f"All {len(fams)} families >= {hi_threshold:.2f}, max_sd={sd_val:.2f} <= {max_sd_threshold:.2f}, inter={inter_val:.2f}"
        elif min_s >= lo_threshold and n_hi >= 1:
            cat = "conditionally_invariant"
            reason = f"Min stability={min_s:.2f} >= {lo_threshold:.2f}, with {n_hi} families >= {hi_threshold:.2f}"
        elif max_s >= lo_threshold and min_s < lo_threshold:
            cat = "perturbation_fragile"
            reason = f"Fragile under some families: min={min_s:.2f} < {lo_threshold:.2f}, but max={max_s:.2f} >= {lo_threshold:.2f}"
        else:
            cat = "ambiguous"
            reason = f"Insufficient or unstable evidence across families: min={min_s:.2f}, max={max_s:.2f}, max_sd={sd_val:.2f}"

        categories.append(cat)
        rationale.append(reason)

    df["reliability"] = categories
    df["classification_rationale"] = rationale
    return df


def threshold_sensitivity_analysis(
    fingerprint_df: pd.DataFrame,
    families: Tuple[str, ...],
    hi_range: Tuple[float, ...] = (0.70, 0.75, 0.80),
    lo_range: Tuple[float, ...] = (0.45, 0.50, 0.55),
) -> pd.DataFrame:
    """
    Evaluate how sensitive reliability classifications are to threshold choices.
    """
    records = []
    for hi in hi_range:
        for lo in lo_range:
            cdf = classify_reliability(fingerprint_df, families, hi_threshold=hi, lo_threshold=lo)
            counts = cdf["reliability"].value_counts().to_dict()
            records.append(dict(
                hi=hi,
                lo=lo,
                globally_invariant=counts.get("globally_invariant", 0),
                conditionally_invariant=counts.get("conditionally_invariant", 0),
                perturbation_fragile=counts.get("perturbation_fragile", 0),
                ambiguous=counts.get("ambiguous", 0),
            ))
    return pd.DataFrame(records)
