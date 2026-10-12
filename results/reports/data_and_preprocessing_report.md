# Data and Preprocessing Specification Report

**Project:** Perturbation-Conditional Consensus Clustering (PCCS)  
**Specification Reference:** Section 4.1 & Problem Statement  

---

## 1. Dataset Provenance and Characteristics

To avoid unvalidated assumptions, the empirical evaluation utilizes two complementary benchmark tiers:

### Tier 1: Controlled High-Dimensional Synthetic Benchmarks (`make_pccs`)
* **Geometry:** Latent cluster manifold embedded in $D=100$ dimensional feature space with $K=4$ classes:
  * Class proportions: mildly imbalanced (35%, 30%, 21%, 14%).
  * Informative features (15 dimensions): Dimensions 0–5 are linear Gaussian blocks; dimensions 6–8 undergo nonlinear (quadratic and sinusoidal) transformations; dimensions 9–12 have skewed margins; dimensions 13–14 are weakly informative linear signals.
  * Redundant features (15 dimensions): Linear combinations of informative features corrupted with independent Gaussian noise.
  * Noise features (70 dimensions): Uninformative pure-noise variables sampled from Student-$t$ ($df=4$), Log-Normal, Uniform, and Gaussian distributions.
* **Profiles:**
  1. `clean`: baseline measurement noise ($\sigma \in [0.05, 0.15]$), 0% missing values, 0% outliers.
  2. `noisy`: elevated noise ($\sigma \in [0.30, 0.70]$), 3% MCAR missing cells, 2% outlier rows, borderline points.
  3. `severe`: severe noise ($\sigma \in [0.60, 1.20]$, 30% heavy-tailed), 8% MCAR missing cells, 5% outlier rows, 30% of signal features shuffled.
* **Ground Truth Isolation:** The hidden labels (`true_cluster`) are strictly isolated for external post-hoc evaluation (ARI, AMI, NMI) and never accessed during preprocessing, feature transformation, or clustering.

### Tier 2: Real-World Benchmarks
Standardized open benchmarks from the machine learning literature:
1. **Iris:** $N=150$, $D=4$, $K=3$ botanical classes. Compact, low-dimensional reference.
2. **Wine:** $N=178$, $D=13$, $K=3$ chemical cultivar classes. Moderate dimensionality, varying feature scales.
3. **Breast Cancer Wisconsin (Diagnostic):** $N=569$, $D=30$, $K=2$ diagnostic classes (malignant vs benign). Skewed biological measurements.
4. **Digits:** $N=1797$, $D=64$, $K=10$ optical handwritten digit classes ($8 \times 8$ grayscale pixels). High-dimensional, multi-class structure.
5. **Corrupted Real Variants:** Injected with 20 pure-noise Gaussian dimensions, 3% MCAR missingness, and measurement noise ($\sigma=0.3$).

---

## 2. Data Quality Profiling & Preprocessing Pipeline

The preprocessing pipeline (`pccs.data.PreprocessingPipeline`) follows a deterministic protocol:

```
Raw High-Dimensional Data X
          │
          ▼
1. Feature Profiling & Validation
   - Detect constant, binary, and extreme outliers
   - Validate numerical type compatibility
          │
          ▼
2. Missing-Value Imputation
   - Column-wise median imputation
          │
          ▼
3. Robust Winsorization
   - Median Absolute Deviation (MAD) clipping: [-4.0 MAD, +4.0 MAD]
   - Protects distance metrics from leverage outliers without dropping observations
          │
          ▼
4. Standardization
   - Mean centering (mu = 0) and unit variance scaling (sigma = 1)
          │
          ▼
5. Fixed Working Representation Z
   - Preserved across all baseline and perturbation runs
   - Optional PCA projection (e.g. 90% variance on Digits)
```

### Justification of Design Decisions:
- **No Indiscriminate Outlier Removal:** Outliers are retained because real high-dimensional datasets contain natural biological or technical extreme values. The MAD winsorization clips extreme leverage points without discarding rows.
- **Strict Separation of Evaluation Labels:** Ground-truth reference labels are stored in independent arrays and never touch the feature matrices.
