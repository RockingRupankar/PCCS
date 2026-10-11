"""Inductive Consensus Clustering (ICC): perturbation-aware consensus clustering with out-of-sample assignment."""
from .model import ICC, ICCv1, ICCConfig, preprocess_fit, preprocess_apply, feature_weights
from .config import load_icc_config, load_pccs_config
from .pccs import PCCS, PCCSConfig, FAMILIES, DEFAULT_LEVELS

__version__ = "0.1.0"
__all__ = ["ICC", "ICCv1", "ICCConfig", "preprocess_fit", "preprocess_apply", "feature_weights",
           "load_icc_config", "load_pccs_config", "PCCS", "PCCSConfig", "FAMILIES", "DEFAULT_LEVELS", "__version__"]
