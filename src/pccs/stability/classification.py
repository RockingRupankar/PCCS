"""
Descriptive reliability classification for PCCS.
Assigns evidence-based categories from measured stability fingerprints:
1. Globally invariant
2. Conditionally invariant
3. Perturbation-fragile
4. Ambiguous
"""
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd


def classify_reliability(
    fingerprints: pd.DataFrame,
    families: List[str],
    hi: float = 0.75,
    lo: float = 0.50,
    max_response_sd: float = 0.15,
    interaction_threshold: float = 0.15,
    interactions_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Classify discovered clusters into 4 descriptive reliability categories:
    - globally_invariant: S_kp >= hi across all families, response_sd <= max_sd, max_interaction < threshold
    - conditionally_invariant: S_kp >= hi on at least one family and min S_kp >= lo
    - perturbation_fragile: S_kp >= hi on at least one family but min S_kp < lo
    - ambiguous: S_kp < hi across all families
    """
    df = fingerprints.copy()
    P = len(families)

    # Extract maximum interaction drop per cluster if available
    inter_max = np.zeros(len(df), dtype=float)
    if interactions_df is not None and not interactions_df.empty:
        ik = interactions_df[interactions_df["cluster"] != "global"].copy()
        if not ik.empty:
            ik["cluster"] = ik["cluster"].astype(int)
            m = ik.groupby("cluster")["interaction"].max()
            inter_max = np.array([m.get(k, 0.0) for k in df["cluster"]])

    df["max_interaction"] = inter_max

    categories = []
    for idx, row in df.iterrows():
        v = np.array([row[f] for f in families if f in row])
        n_hi = int(np.sum(v >= hi))
        min_v = float(np.min(v)) if len(v) > 0 else 0.0
        resp_sd = float(row.get("max_response_sd", 0.0))
        inter_val = float(inter_max[idx])

        if n_hi == P and resp_sd <= max_response_sd and inter_val < interaction_threshold:
            cat = "globally_invariant"
        elif n_hi == P or (n_hi >= 1 and min_v >= lo):
            cat = "conditionally_invariant"
        elif n_hi >= 1:
            cat = "perturbation_fragile"
        else:
            cat = "ambiguous"

        categories.append(cat)

    df["reliability_category"] = categories
    return df


def sweep_classification_thresholds(
    fingerprints: pd.DataFrame,
    families: List[str],
    hi_range: List[float] = [0.70, 0.75, 0.80],
    lo_range: List[float] = [0.45, 0.50, 0.55],
) -> pd.DataFrame:
    """Evaluate stability category sensitivity across threshold sweeps."""
    results = []
    for hi in hi_range:
        for lo in lo_range:
            if lo >= hi:
                continue
            df_classified = classify_reliability(fingerprints, families, hi=hi, lo=lo)
            counts = df_classified["reliability_category"].value_counts().to_dict()
            results.append({
                "hi": hi,
                "lo": lo,
                "globally_invariant": counts.get("globally_invariant", 0),
                "conditionally_invariant": counts.get("conditionally_invariant", 0),
                "perturbation_fragile": counts.get("perturbation_fragile", 0),
                "ambiguous": counts.get("ambiguous", 0),
            })
    return pd.DataFrame(results)
