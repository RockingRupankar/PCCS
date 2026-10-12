"""
Perturbation-specific stability tensor for PCCS:
Maintains label-invariant pairwise co-association per family p and intensity level l:
  S_{ij}^{(p)} = sum_{m,r} I(i, j co-clustered) / sum_{m,r} I(i, j eligible)
Strictly avoids treating unobserved pairs as zero stability.
Tracks mean stability, realization variance, coverage, and per-intensity slices.
"""
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


class StabilityTensor:
    """
    Multidimensional stability representation storing:
    - S_family: Dict[str, np.ndarray] of shape (N, N)
    - Var_family: Dict[str, np.ndarray] of shape (N, N)
    - Coverage_family: Dict[str, np.ndarray] of shape (N, N) (eligible comparison count)
    - S_level: Dict[Tuple[str, float], np.ndarray] of shape (N, N)
    """
    def __init__(self, n_samples: int):
        self.n = n_samples
        self.families_: List[str] = []
        self.S_family: Dict[str, np.ndarray] = {}
        self.Var_family: Dict[str, np.ndarray] = {}
        self.Coverage_family: Dict[str, np.ndarray] = {}
        self.S_level: Dict[Tuple[str, float], np.ndarray] = {}
        self.baseline_matrix: Optional[np.ndarray] = None

    def __getitem__(self, key: str) -> np.ndarray:
        return self.S_family[key]

    def __iter__(self):
        return iter(self.S_family)

    def __len__(self) -> int:
        return len(self.S_family)

    def __contains__(self, key: str) -> bool:
        return key in self.S_family

    def keys(self):
        return self.S_family.keys()

    def values(self):
        return self.S_family.values()

    def items(self):
        return self.S_family.items()

    def fit_from_partitions(
        self,
        partitions: List[Dict[str, Any]],
        families: Tuple[str, ...],
    ) -> "StabilityTensor":
        self.families_ = list(families)
        n = self.n

        # First compute baseline co-association matrix from baseline runs
        base_parts = [p for p in partitions if p.get("kind") == "baseline"]
        if base_parts:
            b_num = np.zeros((n, n), dtype=np.float32)
            b_den = np.zeros((n, n), dtype=np.float32)
            for bp in base_parts:
                rows = bp["rows"]
                labs = bp["labels"]
                valid = labs >= 0
                v_rows = rows[valid]
                v_labs = labs[valid]
                if len(v_rows) > 0:
                    ix = np.ix_(v_rows, v_rows)
                    b_num[ix] += (v_labs[:, None] == v_labs[None, :]).astype(np.float32)
                    b_den[ix] += 1.0
            self.baseline_matrix = np.where(b_den > 0, b_num / np.maximum(b_den, 1e-9), 0.5).astype(np.float32)
            np.fill_diagonal(self.baseline_matrix, 1.0)
        else:
            self.baseline_matrix = np.full((n, n), 0.5, dtype=np.float32)
            np.fill_diagonal(self.baseline_matrix, 1.0)

        # For each perturbation family, accumulate pairwise co-association
        for fam in families:
            fam_parts = [p for p in partitions if p.get("kind") == "single" and p.get("family") == fam]
            if not fam_parts:
                continue

            num = np.zeros((n, n), dtype=np.float32)
            den = np.zeros((n, n), dtype=np.float32)

            # Store binary co-association matrices across runs to compute variance
            run_coas = []

            for p in fam_parts:
                rows = p["rows"]
                labs = p["labels"]
                valid = labs >= 0
                v_rows = rows[valid]
                v_labs = labs[valid]

                run_mat = np.full((n, n), np.nan, dtype=np.float32)
                if len(v_rows) > 0:
                    ix = np.ix_(v_rows, v_rows)
                    agree = (v_labs[:, None] == v_labs[None, :]).astype(np.float32)
                    num[ix] += agree
                    den[ix] += 1.0
                    run_mat[ix] = agree

                run_coas.append(run_mat)

            # Impute unobserved pairs with baseline matrix to prevent silent zero bias
            S_mat = np.where(den > 0, num / np.maximum(den, 1e-9), self.baseline_matrix)
            S_mat = np.clip(S_mat, 0.0, 1.0).astype(np.float32)
            np.fill_diagonal(S_mat, 1.0)
            self.S_family[fam] = S_mat
            self.Coverage_family[fam] = den

            # Variance across runs
            if len(run_coas) > 1:
                stack = np.array(run_coas)
                valid_counts = np.sum(~np.isnan(stack), axis=0)
                var_mat = np.zeros((n, n), dtype=np.float32)
                has_two = valid_counts >= 2
                if has_two.any():
                    with np.errstate(invalid="ignore", divide="ignore"):
                        var_mat[has_two] = np.nanvar(stack[:, has_two], axis=0)
                var_mat = np.nan_to_num(var_mat, nan=0.0).astype(np.float32)
            else:
                var_mat = np.zeros((n, n), dtype=np.float32)
            self.Var_family[fam] = var_mat

            # Per-intensity slices
            levels = sorted(list({p["level"] for p in fam_parts}))
            for lev in levels:
                lev_parts = [p for p in fam_parts if p["level"] == lev]
                l_num = np.zeros((n, n), dtype=np.float32)
                l_den = np.zeros((n, n), dtype=np.float32)
                for lp in lev_parts:
                    rows = lp["rows"]
                    labs = lp["labels"]
                    valid = labs >= 0
                    v_rows = rows[valid]
                    v_labs = labs[valid]
                    if len(v_rows) > 0:
                        ix = np.ix_(v_rows, v_rows)
                        l_num[ix] += (v_labs[:, None] == v_labs[None, :]).astype(np.float32)
                        l_den[ix] += 1.0
                l_mat = np.where(l_den > 0, l_num / np.maximum(l_den, 1e-9), self.baseline_matrix)
                np.fill_diagonal(l_mat, 1.0)
                self.S_level[(fam, float(lev))] = l_mat.astype(np.float32)

        return self
