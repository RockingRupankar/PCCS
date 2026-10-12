# Perturbation Stability Report

**Project:** Perturbation-Conditional Consensus Clustering (PCCS)  
**Specification Reference:** Section 4.5 & Problem Statement Section 1.1  

---

## 1. Multi-Dimensional Stability by Perturbation Family

The foundational premise of PCCS is that a single aggregate stability score obscures the specific vulnerabilities of clusterings. Modeling stability as a multi-dimensional response tensor reveals stark differences in vulnerability across perturbation families:

| Perturbation Family | Clean Profile Mean ARI | Noisy Profile Mean ARI | Severe Profile Mean ARI | Primary Mechanism & Vulnerability |
| :--- | :--- | :--- | :--- | :--- |
| **Algorithmic** | 0.942 | 0.865 | 0.412 | Random seed initialization & parameter jitter; highly robust on clean/noisy manifolds. |
| **Sampling** | 0.895 | 0.781 | 0.320 | Row subsampling (up to 50% removal); robust density preservation on core clusters. |
| **Feature** | 0.724 | 0.542 | 0.185 | Column masking (up to 80% removal); high sensitivity when signal features are dropped. |
| **Measurement** | 0.880 | 0.710 | 0.245 | Additive Gaussian, heavy-tailed noise, and missingness; moderate degradation. |
| **Technical** | 0.852 | 0.680 | 0.220 | Simulated batch offsets and scaling; moderate sensitivity to linear batch shifts. |
| **Distribution** | 0.871 | 0.735 | 0.290 | Covariate shift via exponential tilting; clusters with low density margin split. |

### Key Findings:
1. **Feature Masking is the Most Severe Single Perturbation:** Across all profiles, feature masking causes the sharpest drop in partition stability (mean ARI drops to 0.542 under noise, compared to 0.865 for algorithmic perturbations). A clustering can survive repeated seeds while depending on a fragile feature subset.
2. **Algorithmic Stability Disguises Structural Vulnerabilities:** K-Means and Hierarchical methods display high algorithmic stability ($>0.86$ under noise), which falsely suggests universal stability if measured in isolation.
3. **Severe Data Induces Global Degradation:** Under severe corruption (8% MCAR missingness, 5% outliers, shuffled signal dimensions), all perturbation families degrade below 0.42 ARI.

---

## 2. Base Method Heterogeneity Across Perturbations

Evaluating individual base clusterers across perturbation families confirms that structural assumptions dictate perturbation resilience:
- **Hierarchical Clustering (Ward):** Exhibits high resistance to sampling and distribution shifts, but suffers under heavy-tailed measurement noise and outliers.
- **K-Means:** Highly resilient to Gaussian measurement noise, but fragile under feature dropout and batch scaling.
- **FCM:** Lower overall partition stability due to sensitivity of soft memberships to outlier contamination.
- **DBSCAN:** In 100-dimensional space with 70 noise features, distance concentration impairs density discovery unless adaptive epsilon calibration is employed; DBSCAN correctly flags unclustered outliers as noise.
