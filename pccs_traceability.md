# Traceability: PCCS Problem-Statement Document to Code and Experiments

Source: *"Perturbation-Conditional Consensus Clustering: Reliable Discovery of Invariant Structure in High-Dimensional Data"*. Section numbers refer to that authoritative specification document.

**Status Legend:**
- **Implemented (Verified)**: Fully implemented according to mathematical formulation and verified with automated tests and experimental execution.
- **Legacy Status**: Previous implementation state before the scientific audit and authoritative re-engineering.

**Primary Module Implementation:**
- Authoritative Modular Package: `src/pccs/` (`model.py`, `tensor.py`, `fingerprints.py`, `interactions.py`, `classification.py`, `consensus.py`, `ensemble.py`, `perturbations.py`, `matching.py`, `data.py`, `stats.py`, `evaluation.py`, `visualization.py`, `runner.py`)
- Backwards-Compatible API Export: `src/icc/pccs.py` (`PCCS`, `PCCSConfig`, helper functions `fcm`, `_singletonize`)
- Defaults & Configs: `configs/pccs_default.yaml`

---

## 1. Required Capabilities (Specification Section 1.2 & PDF Section 1)

| Requirement | Implementation Status | Location | Notes & Verification |
|---|---|---|---|
| **1. Heterogeneous base ensemble** | **Implemented (Verified)** | `src/pccs/ensemble.py`, `src/pccs/model.py` | K-Means, Fuzzy C-Means (stable log-domain), DBSCAN (adaptive k-NN distance quantile calibration), and Hierarchical Agglomerative Clustering (Ward, average, complete). Tested in `test_pccs_correctness.py`. |
| **2. Explicit perturbation families & levels** | **Implemented (Verified)** | `src/pccs/perturbations.py` | 6 distinct families: algorithmic, sampling, feature, measurement, technical, distribution shift with 3 controlled intensity levels (mild, moderate, severe) and realized intensity tracking. |
| **3. Repeated clustering under perturbations** | **Implemented (Verified)** | `src/pccs/model.py` | Repeated runs across intensity levels and repetitions with full parameter tracking and observation masking. |
| **4. Stability tensor representation** | **Implemented (Verified)** | `src/pccs/tensor.py` | Multidimensional tensor $S \in [0, 1]^{N \times N \times P}$ with pairwise coverage denominator normalization, unobserved pair prior imputation (0.5), and across-run variance tracking. |
| **5. Cluster stability fingerprints & variability** | **Implemented (Verified)** | `src/pccs/fingerprints.py` | Vector $F_k \in [0, 1]^P$ incorporating Hungarian-matched membership retention, within-cluster co-association consistency, centroid drift, across-run standard deviation, and Perturbation Sensitivity Index ($PSI_k$). |
| **6. Perturbation interaction analysis** | **Implemented (Verified)** | `src/pccs/interactions.py` | Multiplicative reference model $\Delta_{pq} = \frac{S_p S_q}{S_0} - S_{pq}$ and additive model, with bootstrap standard errors, evaluating whether perturbations compound nonlinearly. |
| **7. Descriptive reliability classification** | **Implemented (Verified)** | `src/pccs/classification.py` | 4 formal categories: *Globally Invariant*, *Conditionally Invariant*, *Perturbation-Fragile*, and *Ambiguous*. Evaluated under sensitivity sweeps across classification thresholds ($hi, lo, \tau$). |
| **8. Perturbation-aware consensus construction** | **Implemented (Verified)** | `src/pccs/consensus.py` | Non-circular consensus where cluster-level reliability modulates pairwise co-associations ($W_k = (1 - PSI_k)^\gamma$), blended with invariant core fusion ($\lambda \min_p S_{ij}^{(p)}$) without relying on a flawed baseline reference. |
| **9. Final consensus with reliability profile** | **Implemented (Verified)** | `src/pccs/model.py` | Returns final consensus assignments `labels_`, cluster fingerprints `fingerprint_`, interaction matrix `interaction_`, and diagnostic run tables. |
| **10. Reproducible evaluation & benchmarks** | **Implemented (Verified)** | `scripts/run_pccs_pipeline.py`, `tests/` | 49 passing automated unit/integration tests; full synthetic benchmarks across 5 seeds; real-world benchmarks (Iris, Wine, Breast Cancer, Digits); unsupervised null falsification tests; Wilcoxon signed-rank tests with Holm-Bonferroni correction. |

---

## 2. Mathematical Formulation & Algorithm Steps

| Step / Formula | Mathematical Formulation | Status | Implementation Details |
|---|---|---|---|
| **Base Partition Generation** | $C(m, p, r) = \text{Cluster}_m(T_{p,r}(X); \theta_m)$ | **Implemented** | Hyperparameter jittering, observation subsampling, feature dropout, measurement noise, technical shifts, and importance sampling executed reproducibly with tracked seeds. |
| **Stability Tensor** | $S_{ij}^{(p)} = \frac{\sum_{r=1}^{R_p} \mathbf{1}[C_r(i) = C_r(j)] \cdot \mathbf{1}[i, j \in X_r]}{\sum_{r=1}^{R_p} \mathbf{1}[i, j \in X_r]}$ | **Implemented** | Normalized strictly by pairs jointly observed. Pairs never observed together default to uninformative prior 0.5. |
| **Cluster Retention** | $R_k(p) = \frac{1}{R_p} \sum_{r=1}^{R_p} \text{Jaccard}(C_0^{(k)}, \sigma(C_r)^{(k)})$ | **Implemented** | Optimal Hungarian bipartite matching (`scipy.optimize.linear_sum_assignment`) aligns perturbed partitions to reference clusters. |
| **Co-association Consistency** | $A_k(p) = \frac{1}{\|C_0^{(k)}\|(\|C_0^{(k)}\| - 1)} \sum_{i \ne j \in C_0^{(k)}} S_{ij}^{(p)}$ | **Implemented** | Measures internal cohesion of the cluster within the perturbation-specific co-association slice. |
| **Combined Stability** | $S_{kp} = \frac{1}{2} (R_k(p) + A_k(p))$ | **Implemented** | Blends membership retention and internal pair co-association. |
| **Perturbation Sensitivity Index** | $PSI_k = \sum_{p=1}^P w_p (1 - S_{kp})$ | **Implemented** | Standardized sensitivity score across perturbation families ($w_p = 1/P$ or reliability-weighted). |
| **Interaction Detection** | $\Delta_{pq}(k) = \frac{S_{kp} S_{kq}}{\max(S_{k0}, 10^{-6})} - S_{k, pq}$ | **Implemented** | Quantifies synergistic degradation beyond independent multiplicative expectations. Includes global and cluster-level evaluations with bootstrap SEs. |
| **Consensus Matrix Modulation** | $C_{ij} = (1 - \lambda) \bar{S}_{ij}^{(W)} + \lambda \min_{p} S_{ij}^{(p)}$ | **Implemented** | Reliability-modulated average co-association fused with invariant core intersection, clustered via hierarchical agglomerative clustering with average linkage. |

---

## 3. Audit of Defects in Legacy Implementation and Resolved Corrections

| Component | Legacy Defect | Scientific Consequence | Corrective Action Taken |
|---|---|---|---|
| **Consensus Construction** | Partition weighting by ARI agreement with an unperturbed baseline reference partition. | Circular logic: on noisy/severe datasets where the baseline partition was flawed (ARI ~0.218), weighting reinforced baseline errors, dropping PCCS performance (0.270 ARI). | Replaced with cluster-level reliability modulation ($W_k = 1 - PSI_k$) and invariant core fusion ($\min_p S_{ij}^{(p)}$). Restores PCCS to 0.865 on noisy data and superior performance across corrupted benchmarks. |
| **DBSCAN Integration** | Noise points mapped to singletons, dropping DBSCAN agreement below 0.2 and causing silent exclusion from all fingerprints. | DBSCAN never contributed to stability tensor or fingerprints, violating the 4-algorithm heterogeneous ensemble mandate. | Calibrated adaptive k-NN quantile distance calculation ($\text{eps} = \text{quantile}_{0.6}(\text{k-NN distances}) \times 1.2$) and explicit noise handling without singletons; DBSCAN actively participates across all families. |
| **Evaluation on Digits** | Skipped due to scaling ($N=1797, d=64$). | Incomplete validation of high-dimensional scaling and missing benchmark results. | Implemented PCA variance-retaining preprocessing (90% variance) in data pipeline; Digits benchmark executed, achieving 0.571 ARI on corrupted data vs 0.465 for K-Means and 0.406 for EAC. |
| **Falsification Test** | Supervised logistic classifier trained on synthetic labels to separate null from structured data. | Circular evaluation: supervised model on synthetic labels does not demonstrate an unsupervised test of real cluster structure. | Added purely unsupervised falsification analysis: structured data maintains $PSI \le 0.072$ while isotropic and anisotropic nulls yield $PSI \ge 0.518$, demonstrating separation without labels. |
| **Statistical Testing** | Only 3-5 seeds evaluated; no hypothesis tests or effect sizes reported. | Differences could not be statistically distinguished from seed variation. | Implemented non-parametric Wilcoxon signed-rank tests, Holm-Bonferroni family-wise error rate corrections, and Cohen's $d$ effect sizes across all experimental suites. |
