"""
Multi-dimensional Perturbation Stability Tensor for PCCS.
Preserves separate pairwise co-association representations for every perturbation family:
S in [0, 1]^(N x N x P) with explicit pairwise observation denominators and variance tracking.
"""
from typing import Dict, Any, List, Optional
import numpy as np


class StabilityTensor:
    """
    Constructs and stores perturbation-specific co-association matrices:
    S_{ij}^{(p)} = sum_r I(P_i^(r) == P_j^(r)) / sum_r I(i, j in X_r)
    """
    def __init__(self, n_samples: int, families: List[str]):
        self.n_samples = n_samples
        self.families = list(families)
        self.tensors: Dict[str, np.ndarray] = {}
        self.counts: Dict[str, np.ndarray] = {}
        self.variances: Dict[str, np.ndarray] = {}
        self.raw_runs: Dict[str, List[np.ndarray]] = {f: [] for f in self.families}

        for f in self.families:
            self.tensors[f] = np.zeros((n_samples, n_samples), dtype=float)
            self.counts[f] = np.zeros((n_samples, n_samples), dtype=float)
            self.variances[f] = np.zeros((n_samples, n_samples), dtype=float)

    def accumulate(
        self,
        family: str,
        labels: np.ndarray,
        row_indices: np.ndarray,
    ) -> None:
        """Accumulate a perturbed clustering run into the stability tensor."""
        if family not in self.families:
            return

        N = self.n_samples
        sub_N = len(row_indices)
        if sub_N < 2:
            return

        # Co-occurrence indicator
        coassoc_sub = (labels[:, None] == labels[None, :]) & (labels[:, None] != -1)
        grid = np.ix_(row_indices, row_indices)
        self.tensors[family][grid] += coassoc_sub.astype(float)
        self.counts[family][grid] += 1.0

        # Store run for variance calculation
        full_run = np.full(N, -2, dtype=int)
        full_run[row_indices] = labels
        self.raw_runs[family].append(full_run)

    def finalize(self, unobserved_prior: float = 0.5) -> None:
        """Normalize accumulated co-occurrences by exact pairwise coverage counts."""
        for f in self.families:
            denom = self.counts[f]
            observed = denom > 0
            
            # Where observed, compute sample proportion; where unobserved, assign prior
            self.tensors[f] = np.where(observed, self.tensors[f] / np.maximum(denom, 1.0), unobserved_prior)
            np.fill_diagonal(self.tensors[f], 1.0)

            # Compute across-run variance if multiple runs exist
            runs = self.raw_runs[f]
            if len(runs) > 1:
                run_mats = []
                for r in runs:
                    valid_mask = (r != -2)
                    co = (r[:, None] == r[None, :]) & (r[:, None] != -1)
                    co = np.where(valid_mask[:, None] & valid_mask[None, :], co.astype(float), np.nan)
                    run_mats.append(co)
                stacked = np.stack(run_mats, axis=0)
                with np.errstate(all="ignore"):
                    import warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        self.variances[f] = np.nanvar(stacked, axis=0)
                self.variances[f] = np.nan_to_num(self.variances[f], nan=0.0)
            else:
                self.variances[f] = np.zeros((self.n_samples, self.n_samples), dtype=float)

    def get_slice(self, family: str) -> np.ndarray:
        """Return the N x N co-association matrix for a given perturbation family."""
        return self.tensors[family]

    def get_variance_slice(self, family: str) -> np.ndarray:
        """Return the N x N variance matrix for a given perturbation family."""
        return self.variances[family]
