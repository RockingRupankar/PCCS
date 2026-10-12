"""
Cluster-level stability fingerprints and Perturbation Sensitivity Index (PSI).
Computes for each candidate cluster C_k:
- Membership retention across realizations (Hungarian matching).
- Within-cluster co-association consistency.
- Centroid / representative drift.
- Realization variability / response standard deviation.
- PSI_k = sum_p w_p (1 - S_{k,p}).
"""
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from .matching import cluster_retention
from .tensor import StabilityTensor


def compute_cluster_fingerprints(
    candidate_labels: np.ndarray,
    Z_full: np.ndarray,
    partitions: List[Dict[str, Any]],
    tensor: StabilityTensor,
    families: Tuple[str, ...],
    weights: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """
    Compute multidimensional stability fingerprint for each cluster.
    """
    K = int(np.max(candidate_labels[candidate_labels >= 0])) + 1
    fams = [f for f in families if f in tensor.S_family]
    n = len(candidate_labels)

    # Precompute responses for each single-family run
    run_responses = []
    for p in partitions:
        if p.get("kind") == "single":
            ret, drift = cluster_retention(
                ref_labels_full=candidate_labels,
                Z_full=Z_full,
                perturbed_rows=p["rows"],
                perturbed_cols=p["cols"],
                perturbed_labels=p["labels"],
                K_ref=K,
            )
            run_responses.append(dict(family=p["family"], method=p["method"], ret=ret, drift=drift))

    rows = []
    for k in range(K):
        c_mask = (candidate_labels == k)
        c_size = int(c_mask.sum())
        c_idx = np.where(c_mask)[0]

        res = dict(cluster=k, size=c_size)
        fam_scores = []
        fam_sds = []

        for f in fams:
            f_runs = [r for r in run_responses if r["family"] == f]
            if f_runs:
                rets = np.array([r["ret"][k] for r in f_runs])
                drifts = np.array([r["drift"][k] for r in f_runs])
                mean_ret = float(np.nanmean(rets))
                sd_ret = float(np.nanstd(rets))
                mean_drift = float(np.nanmean(drifts))
            else:
                mean_ret, sd_ret, mean_drift = 0.5, 0.0, 0.0

            # Within-cluster co-association consistency
            if c_size > 1 and f in tensor.S_family:
                sub_S = tensor.S_family[f][np.ix_(c_idx, c_idx)]
                coas = float((sub_S.sum() - c_size) / (c_size * (c_size - 1)))
            else:
                coas = float(mean_ret)

            # Combined cluster stability for family p
            S_kp = float(np.clip(0.5 * (mean_ret + coas), 0.0, 1.0))
            res[f] = S_kp
            res[f"retention_{f}"] = mean_ret
            res[f"coassoc_{f}"] = coas
            res[f"drift_{f}"] = mean_drift
            res[f"sd_{f}"] = sd_ret

            fam_scores.append(S_kp)
            fam_sds.append(sd_ret)

        res["max_response_sd"] = float(np.max(fam_sds)) if fam_sds else 0.0
        res["min_stability"] = float(np.min(fam_scores)) if fam_scores else 0.0
        res["mean_stability"] = float(np.mean(fam_scores)) if fam_scores else 0.0

        # Compute PSI: sum_p w_p * (1 - S_{k,p})
        if weights is None:
            w_arr = np.ones(len(fams)) / max(1, len(fams))
        else:
            w_arr = np.array([weights.get(f, 1.0 / len(fams)) for f in fams])
            w_arr = w_arr / w_arr.sum()

        res["PSI"] = float(np.sum(w_arr * (1.0 - np.array(fam_scores))))
        rows.append(res)

    return pd.DataFrame(rows)


def compute_psi(
    fingerprint_df: pd.DataFrame,
    families: Tuple[str, ...],
    weights: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    """
    Calculate PSI for arbitrary weighting scheme (useful for sensitivity analysis).
    """
    fams = [f for f in families if f in fingerprint_df.columns]
    if weights is None:
        w_arr = np.ones(len(fams)) / max(1, len(fams))
    else:
        w_arr = np.array([weights.get(f, 1.0 / len(fams)) for f in fams])
        w_arr = w_arr / w_arr.sum()

    scores = fingerprint_df[fams].to_numpy(dtype=float)
    return np.sum(w_arr[None, :] * (1.0 - scores), axis=1)
