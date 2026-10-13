"""
Report table formatting and CSV exporting utilities for PCCS.
"""
from typing import Dict, Any, List, Optional
import os
import numpy as np
import pandas as pd


def export_summary_table(
    df: pd.DataFrame,
    output_csv_path: str,
    output_md_path: Optional[str] = None,
) -> None:
    """Save DataFrame as CSV and optional GitHub-flavored markdown table."""
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df.to_csv(output_csv_path, index=False)
    if output_md_path is not None:
        os.makedirs(os.path.dirname(output_md_path), exist_ok=True)
        with open(output_md_path, "w", encoding="utf-8") as fp:
            fp.write(df.to_markdown(index=False))


def generate_statistical_comparison_table(
    results_df: pd.DataFrame,
    proposed_method: str = "pccs_full",
) -> pd.DataFrame:
    """Generate statistical pairwise test summary across baselines."""
    from pccs.evaluation.stats import paired_wilcoxon_test, holm_bonferroni_correction, compute_cohens_d

    baselines = [m for m in results_df["method"].unique() if m != proposed_method]
    rows = []
    p_vals = []

    for b in baselines:
        p_sub = results_df[results_df["method"] == proposed_method].sort_values(by=["dataset", "seed"])
        b_sub = results_df[results_df["method"] == b].sort_values(by=["dataset", "seed"])

        merged = pd.merge(p_sub, b_sub, on=["dataset", "seed"], suffixes=("_pccs", "_base"))
        if len(merged) > 1:
            scores_p = merged["ari_pccs"].values
            scores_b = merged["ari_base"].values
            stat, p_val = paired_wilcoxon_test(scores_p, scores_b, alternative="greater")
            d = compute_cohens_d(scores_p, scores_b)
            p_vals.append(p_val)
            rows.append({
                "baseline": b,
                "n_comparisons": len(merged),
                "mean_pccs_ari": float(np.mean(scores_p)),
                "mean_baseline_ari": float(np.mean(scores_b)),
                "ari_gain": float(np.mean(scores_p) - np.mean(scores_b)),
                "wilcoxon_stat": stat,
                "raw_p_value": p_val,
                "cohens_d": d,
            })

    if p_vals:
        adj_p, rejected = holm_bonferroni_correction(p_vals)
        for i in range(len(rows)):
            rows[i]["adj_p_value"] = adj_p[i]
            rows[i]["significant"] = rejected[i]

    return pd.DataFrame(rows)
