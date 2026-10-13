# Data and Preprocessing Specification Report

**Project:** Perturbation-Conditional Consensus Clustering (PCCS)  
**Specification Reference:** Section 4.1 & Problem Statement  
**Authoritative Manifest:** `data/manifests/dataset_manifest.csv`, `references/dataset_sources.md`  

---

## 1. Dataset Provenance and Characteristics

All datasets utilized in the PCCS evaluation are documented with official repository sources, licenses, and SHA-256 checksums:

1. **UCI Wine:** $N=178, D=13, K=3$ cultivars. Chemical composition measurements across Italian wines.
2. **Breast Cancer Wisconsin (Diagnostic):** $N=569, D=30, K=2$ diagnostic classes (malignant vs benign). Nuclear feature measurements computed from digitized images.
3. **Human Activity Recognition (HAR):** $N=10,299, D=561, K=6$ activity classes. Triaxial smartphone accelerometer/gyroscope measurements.
4. **Forest Covertype:** Subsampled to $N=2500, D=54, K=7$ cover types. Cartographic forest variables from US Forest Service.
5. **Controlled Synthetic Manifolds:** 6 geometric regimes with known latent clusters embedded in informative, redundant, and noise dimensions.

## 2. Preprocessing Pipeline Protocol

The preprocessing pipeline (`pccs.preprocessing.pipeline.PreprocessingPipeline`) follows a deterministic protocol:

```
Raw High-Dimensional Data X
          │
          ▼
1. Feature Profiling & Validation (detect constant/binary features)
          │
          ▼
2. Missing-Value Imputation (column-wise median)
          │
          ▼
3. Robust Winsorization ([-4.0 MAD, +4.0 MAD] clipping)
          │
          ▼
4. Standardization (zero-mean, unit-variance scaling)
          │
          ▼
Fixed Preprocessed Representation Z
```

**Strict Label Isolation:** Ground-truth class labels are stored in separate `*_labels.csv` files and are never provided to preprocessing, feature scaling, perturbation generation, or consensus clustering. Labels are accessed strictly for post-hoc external validation (ARI, AMI, NMI).
