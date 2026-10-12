# Experiment Configuration and Execution Protocol

**Project:** Perturbation-Conditional Consensus Clustering (PCCS)  
**Specification Reference:** Section 5 (Complete PCCS Algorithm) & Section 14 (Baselines)  

---

## 1. Experimental Grid Parameters

### A. Base Clustering Ensemble ($M=4$ Families)
1. **K-Means:** Centroid-based compact clusters.
   - Distance: Euclidean.
   - Initialization: k-means++ (jitter: random).
   - $n_{\text{init}} \in \{1, 3\}$.
2. **Fuzzy C-Means (FCM):** Soft memberships with entropy diagnostics.
   - Fuzzifier $m = 1.3$ (jittered $m \in [1.15, 1.55]$).
   - Log-domain membership computation to prevent underflow.
   - Conversion to hard labels via $\arg\max_c U_{ic}$.
3. **DBSCAN:** Density-based manifold recovery with explicit noise representation.
   - $\text{min\_samples} = 5$ (jitter: $\pm 2$).
   - Adaptive $\varepsilon$ calibrated via k-NN distance quantile ($q=0.5$).
   - Noise points labeled $-1$ and excluded from co-association numerators.
4. **Hierarchical Agglomerative Clustering:** Multiscale proximity.
   - Linkage: Ward (unperturbed); Average / Complete under hyperparameter jitter.

### B. Controlled Perturbation Families ($P=6$)
| Family | Nominal Intensity Levels | Transformation Mechanism | Realized Metric |
| :--- | :--- | :--- | :--- |
| **Algorithmic** | $\{0.0, 1.0\}$ | Seed variation ($0.0$) + hyperparameter jitter ($1.0$) | Nominal jitter |
| **Sampling** | $\{0.1, 0.3, 0.5\}$ | Uniform row removal without replacement | Fraction of removed rows |
| **Feature** | $\{0.2, 0.5, 0.8\}$ | Random column masking / subspace projection | Fraction of removed features |
| **Measurement** | $\{0.25, 0.5, 1.0\}$ | Gaussian noise, Student-$t$ ($df=3$), outliers, MCAR missing | Empirical standard deviation / outlier fraction |
| **Technical** | $\{0.1, 0.25, 0.5\}$ | Simulated batch offsets $\mathcal{N}(0, \sigma)$ and scaling | Batch offset standard deviation |
| **Distribution** | $\{0.5, 1.0, 2.0\}$ | Covariate shift via exponential tilting: $p_i \propto \exp(\lambda u_i)$ | Absolute projected mean shift |

### C. Repetition & Random Seed Protocol
- Synthetic benchmarks: 5 independent generator seeds ($42, 43, 44, 45, 46$).
- Real-world benchmarks: 3 independent seeds ($42, 43, 44$).
- Baseline ensemble seeds: 2 per method ($S_0 = 8$ baseline runs).
- Perturbation repetitions: $R=2$ per level per family ($2 \times 3 \times 6 \times 4 = 144$ runs).
- Pairwise interaction repetitions: $R=1$ to $2$ across all 15 pairs ($15 \times 2 \times 4 = 120$ runs).
- Total clusterings per dataset fit: $\approx 272$ runs.

---

## 2. Baselines and Ablations Matrix

1. **Independent Base Methods:** K-Means, FCM, DBSCAN, Hierarchical (Ward).
2. **Conventional Consensus:** Evidence Accumulation Clustering (EAC) using unweighted pairwise co-association.
3. **Baseline Reference Partition:** Unweighted ensemble on unperturbed data.
4. **Ablation 1 (No Reliability Weights):** PCCS with uniform partition/cluster weighting.
5. **Ablation 2 (No Invariance Core):** PCCS with $\lambda = 0$ (arithmetic mean across families).
6. **Ablation 3 (Invariance Core Only):** PCCS with $\lambda = 1$ (element-wise minimum across families).
7. **Ablation 4 (Pooled Unweighted):** Single unweighted co-association matrix pooling all runs.
8. **Ablation 5 (Without DBSCAN):** PCCS restricted to K-Means, FCM, and Hierarchical.
9. **Full PCCS:** Complete multidimensional tensor, cluster-level reliability modulation, and invariant core fusion.
