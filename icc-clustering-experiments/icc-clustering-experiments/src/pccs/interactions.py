"""
Perturbation interaction analysis for PCCS:
Quantifies whether combining perturbations induces non-additive / synergistic collapse.
Computes:
- Reference stability S_0 (unperturbed reproducibility).
- Isolated stabilities S_p and S_q.
- Observed joint stability S_{pq}.
- Multiplicative expected stability S_{pq}^{expected} = S_p * S_q / S_0.
- Interaction statistic Delta_{pq} = S_{pq}^{expected} - S_{pq}^{observed}.
- Additive interaction statistic Delta_{pq}^{add} = (S_p + S_q - S_0) - S_{pq}^{observed}.
- Bootstrap uncertainty estimation.
"""
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from .matching import cluster_retention, partition_metrics


def compute_interactions(
    candidate_labels: np.ndarray,
    Z_full: np.ndarray,
    partitions: List[Dict[str, Any]],
    pairs: List[Tuple[str, str]],
    n_bootstrap: int = 100,
    random_state: int = 0,
) -> pd.DataFrame:
    """
    Compute cluster-level and global interaction statistics across specified perturbation pairs.
    """
    K = int(np.max(candidate_labels[candidate_labels >= 0])) + 1
    rng = np.random.default_rng(random_state)

    # Compute unperturbed baseline reproducibility S_0
    base_parts = [p for p in partitions if p.get("kind") == "baseline"]
    if base_parts:
        base_rets = []
        base_aris = []
        for bp in base_parts:
            ret, _ = cluster_retention(
                ref_labels_full=candidate_labels,
                Z_full=Z_full,
                perturbed_rows=bp["rows"],
                perturbed_cols=bp["cols"],
                perturbed_labels=bp["labels"],
                K_ref=K,
            )
            base_rets.append(ret)
            pm = partition_metrics(candidate_labels[bp["rows"]], bp["labels"])
            base_aris.append(pm["ARI"])
        s0_k = np.nanmean(np.array(base_rets), axis=0)
        s0_global = float(np.mean(base_aris))
    else:
        s0_k = np.ones(K, dtype=float)
        s0_global = 1.0

    s0_k = np.maximum(s0_k, 1e-4)
    s0_global = max(s0_global, 1e-4)

    rows = []
    for (p_, q_) in pairs:
        pair_key = f"{p_}+{q_}"
        p_parts = [p for p in partitions if p.get("kind") == "single" and p.get("family") == p_]
        q_parts = [p for p in partitions if p.get("kind") == "single" and p.get("family") == q_]
        joint_parts = [p for p in partitions if p.get("kind") == "pair" and p.get("family") == pair_key]

        if not (p_parts and q_parts and joint_parts):
            continue

        def get_responses(part_list):
            rets = []
            aris = []
            for item in part_list:
                ret, _ = cluster_retention(
                    ref_labels_full=candidate_labels,
                    Z_full=Z_full,
                    perturbed_rows=item["rows"],
                    perturbed_cols=item["cols"],
                    perturbed_labels=item["labels"],
                    K_ref=K,
                )
                rets.append(ret)
                pm = partition_metrics(candidate_labels[item["rows"]], item["labels"])
                aris.append(pm["ARI"])
            return np.array(rets), np.array(aris)

        rets_p, aris_p = get_responses(p_parts)
        rets_q, aris_q = get_responses(q_parts)
        rets_pq, aris_pq = get_responses(joint_parts)

        sp_k = np.nanmean(rets_p, axis=0)
        sq_k = np.nanmean(rets_q, axis=0)
        spq_k = np.nanmean(rets_pq, axis=0)

        sp_g = float(np.mean(aris_p))
        sq_g = float(np.mean(aris_q))
        spq_g = float(np.mean(aris_pq))

        # Cluster-level interactions
        for k in range(K):
            exp_k = float(sp_k[k] * sq_k[k] / s0_k[k])
            obs_k = float(spq_k[k])
            inter_k = float(exp_k - obs_k)
            add_exp_k = float(sp_k[k] + sq_k[k] - s0_k[k])
            add_inter_k = float(add_exp_k - obs_k)

            # Bootstrap uncertainty on interaction
            boot_inters = []
            n_p, n_q, n_pq = len(rets_p), len(rets_q), len(rets_pq)
            for _ in range(n_bootstrap):
                b_p = np.nanmean(rets_p[rng.choice(n_p, n_p, replace=True)], axis=0)[k]
                b_q = np.nanmean(rets_q[rng.choice(n_q, n_q, replace=True)], axis=0)[k]
                b_pq = np.nanmean(rets_pq[rng.choice(n_pq, n_pq, replace=True)], axis=0)[k]
                b_exp = b_p * b_q / s0_k[k]
                boot_inters.append(b_exp - b_pq)
            boot_se = float(np.std(boot_inters))

            rows.append(dict(
                pair=pair_key,
                family_1=p_,
                family_2=q_,
                cluster=k,
                S_0=float(s0_k[k]),
                S_p=float(sp_k[k]),
                S_q=float(sq_k[k]),
                S_pq_observed=obs_k,
                S_pq_expected=exp_k,
                interaction=inter_k,
                interaction_se=boot_se,
                interaction_additive=add_inter_k,
            ))

        # Global partition-level interaction
        exp_g = float(sp_g * sq_g / s0_global)
        obs_g = spq_g
        inter_g = float(exp_g - obs_g)
        rows.append(dict(
            pair=pair_key,
            family_1=p_,
            family_2=q_,
            cluster="global",
            S_0=float(s0_global),
            S_p=float(sp_g),
            S_q=float(sq_g),
            S_pq_observed=obs_g,
            S_pq_expected=exp_g,
            interaction=inter_g,
            interaction_se=0.0,
            interaction_additive=float((sp_g + sq_g - s0_global) - obs_g),
        ))

    return pd.DataFrame(rows)
