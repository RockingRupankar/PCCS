"""
Benchmark dataset loaders for PCCS experiments.
Enforces strict isolation of ground-truth reference labels.
"""
import os
from typing import Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.datasets import load_wine, load_breast_cancer, fetch_covtype, fetch_openml


def load_benchmark_dataset(
    name: str,
    data_dir: str = "data/processed",
    subsample_n: Optional[int] = None,
    random_state: int = 42,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Load a verified benchmark dataset by name.
    
    Returns:
    - X: np.ndarray, feature matrix for unsupervised clustering
    - y: Optional[np.ndarray], reference labels strictly for external evaluation
    """
    name_clean = name.lower().replace(" ", "_").replace("-", "_")

    # Check for processed CSV files first
    feat_path = os.path.join(data_dir, f"{name_clean}.csv")
    label_path = os.path.join(data_dir, f"{name_clean}_labels.csv")

    if os.path.exists(feat_path):
        X = pd.read_csv(feat_path).values
        y = pd.read_csv(label_path).values.ravel() if os.path.exists(label_path) else None
        if subsample_n is not None and subsample_n < len(X):
            rng = np.random.RandomState(random_state)
            idx = rng.choice(len(X), size=subsample_n, replace=False)
            X = X[idx]
            if y is not None:
                y = y[idx]
        return X, y

    # Direct fallback loading via sklearn
    if name_clean in ("wine", "uci_wine"):
        raw = load_wine()
        X, y = raw.data, raw.target
    elif name_clean in ("breast_cancer", "breast_cancer_diagnostic", "cancer"):
        raw = load_breast_cancer()
        X, y = raw.data, raw.target
    elif name_clean in ("har", "human_activity_recognition"):
        raw = fetch_openml("HAR", version=1, as_frame=False)
        X, y = raw.data, raw.target
    elif name_clean in ("covertype", "covtype"):
        raw = fetch_covtype()
        X, y = raw.data, raw.target
        if subsample_n is None:
            subsample_n = 2500  # Default deterministic subsample for Covertype
    else:
        raise ValueError(f"Dataset {name} not recognized.")

    if subsample_n is not None and subsample_n < len(X):
        rng = np.random.RandomState(random_state)
        idx = rng.choice(len(X), size=subsample_n, replace=False)
        X = X[idx]
        y = y[idx]

    return np.asarray(X, dtype=float), np.asarray(y)
