# Dataset Sources, Attribution, and Integrity Manifest

This document records the official provenance, licenses, DOI citations, dimensionality, and subsampling policies for all benchmark datasets used in the Perturbation-Conditional Consensus Clustering (PCCS) research project.

---

## 1. Primary Benchmark Datasets

### A. UCI Wine
- **Official URL:** https://archive.ics.uci.edu/dataset/109/winedataset
- **DOI / Citation:** 10.24432/C5PC7J (Aeberhard, S., Coomans, D. & de Vel, O., 1992)
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Dimensions:** 178 observations, 13 continuous chemical features.
- **Reference Classes:** 3 cultivars (strictly reserved for external evaluation; excluded from unsupervised clustering).
- **Subsampling Policy:** Full dataset ($N=178$).
- **Integrity SHA-256:** `1599a6ff0a718a5eff727ae12068293f84d7fa083ad51c4877edaf2a7a44a158`

### B. Breast Cancer Wisconsin (Diagnostic)
- **Official URL:** https://archive.ics.uci.edu/dataset/17/breast-cancer
- **DOI / Citation:** 10.24432/C5DW2B (Street, W.N., Wolberg, W.H. & Mangasarian, O.L., 1993)
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Dimensions:** 569 observations, 30 continuous nuclear morphological features.
- **Reference Classes:** 2 diagnostic classes (malignant / benign; external evaluation only).
- **Subsampling Policy:** Full dataset ($N=569$).
- **Integrity SHA-256:** `18ca3f294fe6821ff692d56a53d64bd339f87a4c5bb6f7c2c26d0c4a62031e51`

### C. Human Activity Recognition Using Smartphones (HAR)
- **Official URL:** https://archive.ics.uci.edu/dataset/240/human%2Bactivity%2Brecognition%2Busing%2Bsmartphones
- **DOI / Citation:** 10.24432/C54S42 (Anguita, D., Ghio, A., Oneto, L., Parra, X. & Reyes-Ortiz, J.L., 2013)
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Dimensions:** 10,299 observations, 561 features extracted from smartphone accelerometers and gyroscopes.
- **Reference Classes:** 6 activity classes (walking, walking upstairs, walking downstairs, sitting, standing, laying; external evaluation only).
- **Subsampling Policy:** Full dataset ($N=10,299, d=561$) for high-dimensional feature evaluation, and seeded subsets for factorial sweeps.
- **Integrity SHA-256:** `25cba58a098f5c842dd74b3fe057325f1fb512cd692a1b8bf3565625545de320`

### D. Forest Covertype (Covertype)
- **Official URL:** https://archive.ics.uci.edu/dataset/31/covertype
- **DOI / Citation:** 10.24432/C50W2D (Blackard, J.A. & Denis, D.J., 1998)
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Dimensions:** 581,012 observations, 54 features (10 continuous cartographic variables, 4 wilderness binary areas, 40 soil type binary indicators).
- **Reference Classes:** 7 forest cover types (external evaluation only).
- **Subsampling Policy:** Deterministic stratified subsample of $N=2500$ observations using random seed 42 to permit exact $O(N^2)$ stability tensor computation without memory exhaustion.
- **Integrity SHA-256:** `8112e15e83c3193b234d2bb67017918fbcb6e05f65fa13c70cff3653445315c8`

---

## 2. Controlled Synthetic Regimes

Synthetic benchmarks are generated algorithmically with known latent ground-truth cluster assignments to test mathematical edge cases:
1. **Spherical Separated:** Compact Gaussian clusters with well-separated centroids.
2. **Anisotropic / Elongated:** Clusters with strong covariance alignment and unequal axis variances.
3. **Non-Convex Manifolds:** Two-moons and concentric circles to stress density-based and hierarchical proximity.
4. **Overlapping / Ambiguous:** Gaussian clusters with controlled Bayes classification error.
5. **Imbalanced / Rare Clusters:** 3 major clusters ($N=150$ each) plus 1 fragile rare cluster ($N=15$).
6. **High-Dimensional Noise:** 10 informative latent dimensions embedded inside 90 uninformative Gaussian noise dimensions.
7. **Isotropic and Anisotropic Null Distributions:** Unclustered continuous data used for falsification testing.
