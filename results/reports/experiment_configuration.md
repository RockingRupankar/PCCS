# Experiment Configuration and Execution Protocol

**Project:** Perturbation-Conditional Consensus Clustering (PCCS)  
**Specification Reference:** Section 5 (Complete PCCS Algorithm) & Section 14 (Baselines)  

---

## 1. Experimental Grid Parameters

### A. Base Clustering Ensemble ($M=4$ Families)
1. **K-Means:** Centroid-based compact clusters ($k$-means++, Euclidean distance).
2. **Fuzzy C-Means (FCM):** Soft memberships with entropy diagnostics ($m=1.3$, hard assignment via argmax).
3. **DBSCAN:** Density-based recovery (adaptive $\varepsilon$ calibrated via k-NN distance quantile $q=0.5$, noise points handled formally).
4. **Hierarchical Agglomerative Clustering:** Multiscale proximity (Ward/Average linkage).

### B. Controlled Perturbation Families ($P=6$)
| Family | Nominal Intensity Levels | Transformation Mechanism | Realized Metric |
| :--- | :--- | :--- | :--- |
| **Algorithmic** | {0.10, 0.25, 0.50} | Seed variation and hyperparameter jitter | Nominal jitter factor |
| **Sampling** | {0.50, 0.75, 0.90} | Uniform row removal without replacement | Fraction of removed rows |
| **Feature** | {0.10, 0.25, 0.50} | Random column masking / subspace projection | Fraction of removed features |
| **Measurement** | {0.10, 0.30, 0.60} | Gaussian noise and Student-$t$ heavy tails | Empirical standard deviation |
| **Technical** | {0.20, 0.50, 1.00} | Simulated batch offsets and scaling | Batch offset standard deviation |
| **Distribution** | {0.20, 0.50, 0.80} | Covariate shift via exponential tilting | Absolute projected mean shift |

### C. Random Seed Protocol
- Synthetic experiments: 5 generator seeds (42, 43, 44, 45, 46).
- Real-world benchmarks: 5 independent seeds (42, 43, 44, 45, 46).
- Repeat realizations: $R=2$ per level per family.
