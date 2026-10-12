"""
Data inspection, validation, and reproducible preprocessing for PCCS.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


@dataclass
class DataProfile:
    n_samples: int
    n_features: int
    feature_names: List[str]
    missing_count: int
    missing_fraction: float
    constant_features: List[str]
    binary_features: List[str]
    numerical_features: List[str]
    extreme_outlier_count: int
    mean_scale: float
    max_scale: float
    min_scale: float


class PreprocessingPipeline:
    """
    Deterministic preprocessing pipeline for PCCS:
    1. Median imputation (missing value handling)
    2. Robust clipping / winsorization based on Median Absolute Deviation (MAD)
    3. Standardization (zero mean, unit variance)
    4. Optional PCA dimensionality reduction preserving variance fraction
    """
    def __init__(self, clip_mad: float = 4.0, pca_variance: Optional[float] = None):
        self.clip_mad = clip_mad
        self.pca_variance = pca_variance
        self.fitted_ = False
        self.med_ = None
        self.mad_ = None
        self.mu_ = None
        self.sd_ = None
        self.pca_ = None
        self.profile_: Optional[DataProfile] = None

    def fit_transform(self, X: Any, feature_names: Optional[List[str]] = None) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            if feature_names is None:
                feature_names = list(X.columns)
            X_arr = X.to_numpy(dtype=float)
        else:
            X_arr = np.asarray(X, dtype=float)
            if feature_names is None:
                feature_names = [f"f_{i}" for i in range(X_arr.shape[1])]

        n, d = X_arr.shape
        missing_count = int(np.isnan(X_arr).sum())
        missing_fraction = missing_count / max(1, n * d)

        # Profile features
        constant_features = []
        binary_features = []
        numerical_features = []
        for j in range(d):
            col = X_arr[:, j]
            valid = col[~np.isnan(col)]
            if len(valid) == 0 or np.all(valid == valid[0]):
                constant_features.append(feature_names[j])
            elif len(np.unique(valid)) == 2:
                binary_features.append(feature_names[j])
            else:
                numerical_features.append(feature_names[j])

        med = np.nanmedian(X_arr, axis=0)
        med = np.where(np.isnan(med), 0.0, med)
        X_imp = np.where(np.isnan(X_arr), med, X_arr)

        mad = 1.4826 * np.median(np.abs(X_imp - med), axis=0)
        mad = np.where(mad < 1e-9, X_imp.std(axis=0) + 1e-9, mad)

        # Identify extreme outliers (> 5 MAD before clipping)
        z_raw = np.abs((X_imp - med) / mad)
        extreme_outliers = int((z_raw > 5.0).sum())

        Z = np.clip((X_imp - med) / mad, -self.clip_mad, self.clip_mad)
        mu = Z.mean(axis=0)
        sd = Z.std(axis=0) + 1e-9
        Z = (Z - mu) / sd

        scales = np.nanstd(X_imp, axis=0)
        self.profile_ = DataProfile(
            n_samples=n,
            n_features=d,
            feature_names=feature_names,
            missing_count=missing_count,
            missing_fraction=missing_fraction,
            constant_features=constant_features,
            binary_features=binary_features,
            numerical_features=numerical_features,
            extreme_outlier_count=extreme_outliers,
            mean_scale=float(np.mean(scales)),
            max_scale=float(np.max(scales)) if len(scales) else 0.0,
            min_scale=float(np.min(scales)) if len(scales) else 0.0,
        )

        self.med_ = med
        self.mad_ = mad
        self.mu_ = mu
        self.sd_ = sd

        if self.pca_variance is not None and self.pca_variance < 1.0 and d > 2:
            self.pca_ = PCA(n_components=self.pca_variance, svd_solver="full")
            Z = self.pca_.fit_transform(Z)

        self.fitted_ = True
        return Z

    def transform(self, X: Any) -> np.ndarray:
        if not self.fitted_:
            raise RuntimeError("Pipeline must be fitted before calling transform.")
        if isinstance(X, pd.DataFrame):
            X_arr = X.to_numpy(dtype=float)
        else:
            X_arr = np.asarray(X, dtype=float)

        X_imp = np.where(np.isnan(X_arr), self.med_, X_arr)
        Z = np.clip((X_imp - self.med_) / self.mad_, -self.clip_mad, self.clip_mad)
        Z = (Z - self.mu_) / self.sd_
        if self.pca_ is not None:
            Z = self.pca_.transform(Z)
        return Z


def validate_and_preprocess(X: Any, clip_mad: float = 4.0, pca_variance: Optional[float] = None) -> Tuple[np.ndarray, PreprocessingPipeline]:
    pipeline = PreprocessingPipeline(clip_mad=clip_mad, pca_variance=pca_variance)
    Z = pipeline.fit_transform(X)
    return Z, pipeline
