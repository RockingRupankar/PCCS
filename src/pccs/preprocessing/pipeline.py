"""
Deterministic preprocessing pipeline for PCCS.
Strictly isolates feature transformation from class labels.
"""
from typing import Optional, Tuple
import numpy as np
from sklearn.decomposition import PCA


class PreprocessingPipeline:
    """
    Deterministic feature preprocessing pipeline:
    1. Missing value imputation (column median).
    2. Outlier winsorization (clipping at ±clip_mad * MAD).
    3. Standardization (zero mean, unit variance).
    4. Optional PCA dimensionality reduction (preserving target variance).
    """
    def __init__(self, clip_mad: float = 4.0, pca_variance: Optional[float] = None):
        self.clip_mad = clip_mad
        self.pca_variance = pca_variance
        self.medians_: Optional[np.ndarray] = None
        self.mads_: Optional[np.ndarray] = None
        self.means_: Optional[np.ndarray] = None
        self.stds_: Optional[np.ndarray] = None
        self.pca_: Optional[PCA] = None

    def fit(self, X: np.ndarray) -> "PreprocessingPipeline":
        X = np.asarray(X, dtype=float)
        # 1. Median
        self.medians_ = np.nanmedian(X, axis=0)
        # Impute temporarily for MAD
        X_imp = np.where(np.isnan(X), self.medians_, X)
        
        # 2. MAD
        dev = np.abs(X_imp - self.medians_)
        self.mads_ = np.nanmedian(dev, axis=0)
        self.mads_ = np.where(self.mads_ < 1e-8, 1.0, self.mads_)

        # Winsorize for mean and std
        lower = self.medians_ - self.clip_mad * 1.4826 * self.mads_
        upper = self.medians_ + self.clip_mad * 1.4826 * self.mads_
        X_win = np.clip(X_imp, lower, upper)

        # 3. Standardization parameters
        self.means_ = np.mean(X_win, axis=0)
        self.stds_ = np.std(X_win, axis=0)
        self.stds_ = np.where(self.stds_ < 1e-8, 1.0, self.stds_)

        X_norm = (X_win - self.means_) / self.stds_

        # 4. Optional PCA
        if self.pca_variance is not None and self.pca_variance < 1.0:
            self.pca_ = PCA(n_components=self.pca_variance, random_state=42)
            self.pca_.fit(X_norm)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.medians_ is None or self.means_ is None:
            raise ValueError("Pipeline must be fitted before transform.")
        X = np.asarray(X, dtype=float)
        X_imp = np.where(np.isnan(X), self.medians_, X)
        lower = self.medians_ - self.clip_mad * 1.4826 * self.mads_
        upper = self.medians_ + self.clip_mad * 1.4826 * self.mads_
        X_win = np.clip(X_imp, lower, upper)
        X_norm = (X_win - self.means_) / self.stds_

        if self.pca_ is not None:
            return self.pca_.transform(X_norm)
        return X_norm

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


def validate_and_preprocess(
    X: np.ndarray,
    clip_mad: float = 4.0,
    pca_variance: Optional[float] = None
) -> Tuple[np.ndarray, PreprocessingPipeline]:
    """Validate data shape, handle NaNs/outliers, and return preprocessed matrix with fitted pipeline."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {X.shape}")
    pipe = PreprocessingPipeline(clip_mad=clip_mad, pca_variance=pca_variance)
    X_proc = pipe.fit_transform(X)
    return X_proc, pipe
