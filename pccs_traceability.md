# Traceability Matrix: PCCS Specification to Implementation and Experiments

**Specification Document:** *"Perturbation-Conditional Consensus Clustering: Reliable Discovery of Invariant Structure in High-Dimensional Data"*  
**Project:** Perturbation-Conditional Consensus Clustering (PCCS)  
**Date:** October 2026  
**Status:** Fully Validated, Audited, and Empirically Verified  

---

## 1. Architectural Implementation Mapping

The PCCS codebase is implemented as an authoritative, modular scientific package in `src/pccs/`, fully decoupled from legacy circular references:

| Subsystem | Module Path | Primary Classes / Functions | Monograph Section |
|---|---|---|---|
| **Base Ensemble** | `src/pccs/clustering/ensemble.py` | `HeterogeneousEnsemble`, `fit_base_models()` | Section 3.1, Section 5 (Step 1) |
| **Baselines & Ablations** | `src/pccs/clustering/baselines.py` | `KMeansBaseline`, `FCMBaseline`, `DBSCANBaseline`, `HierarchicalBaseline`, `EACConsensus`, `TWECBaseline`, `PCCSAblation` | Section 14 |
| **Perturbation Engine** | `src/pccs/perturbations/engine.py` | `PerturbationEngine`, `apply_perturbation()`, `get_realized_intensity()` | Section 3.2, Section 7 |
| **Stability Tensor** | `src/pccs/stability/tensor.py` | `StabilityTensor`, `compute_tensor()`, `impute_unobserved_pairs()` | Section 3.2, Section 5 (Steps 4-6) |
| **Cluster Matching** | `src/pccs/stability/matching.py` | `HungarianClusterMatcher`, `match_clusters_hungarian()` | Section 3.3, Section 5 (Step 7) |
| **Stability Fingerprints** | `src/pccs/stability/fingerprints.py` | `ClusterFingerprintCalculator`, `compute_fingerprints()`, `compute_psi()` | Section 3.3, Section 5 (Steps 8-10) |
| **Reliability Classification** | `src/pccs/stability/classification.py` | `ClusterReliabilityClassifier`, `classify_clusters()` | Section 3.5, Section 5 (Step 12) |
| **Factorial Interactions** | `src/pccs/interactions/factorial.py` | `PerturbationInteractionAnalyzer`, `evaluate_interactions()`, `bootstrap_ci()` | Section 3.4, Section 5 (Step 11) |
| **Consensus Builder** | `src/pccs/consensus/builder.py` | `PerturbationConditionalConsensus`, `build_affinity_matrix()`, `extract_consensus()` | Section 3.6, Section 5 (Steps 13-16) |
| **Preprocessing & Data** | `src/pccs/preprocessing/pipeline.py` | `PreprocessingPipeline`, `impute_median()`, `winsorize_mad()`, `standardize()` | Section 4.1 |
| **Evaluation Metrics** | `src/pccs/evaluation/metrics.py` | `adjusted_rand_score`, `adjusted_mutual_info_score`, `normalized_mutual_info_score`, `silhouette_score` | Section 15 |
| **Statistical Analysis** | `src/pccs/evaluation/stats.py` | `paired_wilcoxon_test`, `holm_bonferroni_correction`, `cohens_d` | Section 16 |
| **Visualization & Plots** | `src/pccs/reporting/plots.py` | `generate_all_publication_figures()`, `plot_fig1_synthetic()` through `plot_fig5_falsification()` | Section 19.K |
| **Summary Tables** | `src/pccs/reporting/tables.py` | `generate_markdown_summary_tables()`, `generate_latex_tables()` | Section 19.J |
| **Estimator API** | `src/pccs/model.py` | `PCCS` (scikit-learn compatible estimator) | Section 5 |
| **Experiment Runner** | `src/pccs/experiment_runner.py` | `run_all_experiments()`, `run_experiment_1()` through `run_experiment_5()` | Section 11, Section 12, Section 13, Section 14 |

---

## 2. 10 Required Capabilities (Monograph Section 1.2)

| # | Required Capability | Implementation Status | Code Location | Empirical Verification & Test Evidence |
|---|---|---|---|---|
| **1** | **Heterogeneous Base Ensemble** | **PASS** | `src/pccs/clustering/ensemble.py` | Incorporates 4 distinct clustering paradigms: K-Means (centroids), FCM (soft fuzzy memberships), DBSCAN (density with adaptive k-NN quantile $\varepsilon$ calibration), and Hierarchical (Ward/Average). Verified in `tests/unit/test_unit.py::test_ensemble_generation`. |
| **2** | **Explicit Perturbation Families & Levels** | **PASS** | `src/pccs/perturbations/engine.py` | 6 controlled perturbation families (algorithmic, sampling, feature, measurement, technical, distribution shifts) across 3 calibrated intensity tiers (mild, moderate, severe) with realized intensity tracking. Verified in `tests/unit/test_unit.py::test_perturbation_engine_families`. |
| **3** | **Repeated Clustering under Controlled Perturbations** | **PASS** | `src/pccs/model.py`, `src/pccs/experiment_runner.py` | Executes repeated stochastic realizations ($R \ge 2$) per family and intensity tier with tracked seeds and sample masking. Verified in `tests/integration/test_integration.py::test_pccs_estimator_fit_predict`. |
| **4** | **Stability Tensor Representation** | **PASS** | `src/pccs/stability/tensor.py` | Constructs $S \in [0, 1]^{N \times N \times P}$ with pairwise co-association normalized strictly by joint eligibility denominators $\sum I(i, j \in X_r)$, with uninformative baseline prior imputation for unobserved pairs. Verified in `tests/mathematical_validation/test_math.py::test_tensor_properties`. |
| **5** | **Cluster Stability Fingerprints & Variability** | **PASS** | `src/pccs/stability/fingerprints.py` | Computes $F_k \in [0, 1]^P$ combining optimal bipartite Hungarian membership retention, internal co-association, centroid drift, across-run standard deviation, and Perturbation Sensitivity Index ($PSI_k$). Verified in `tests/mathematical_validation/test_math.py::test_fingerprint_computation`. |
| **6** | **Factorial Perturbation Interaction Analysis** | **PASS** | `src/pccs/interactions/factorial.py` | Multiplicative reference model $\Delta_{pq} = \frac{S_p S_q}{S_0} - S_{pq}$ and additive model with non-parametric bootstrap standard errors ($B=100$) and 95% confidence intervals. Verified in `tests/mathematical_validation/test_math.py::test_interaction_modeling`. |
| **7** | **Descriptive Reliability Classifications** | **PASS** | `src/pccs/stability/classification.py` | Four evidence-based categories: *Globally Invariant*, *Conditionally Invariant*, *Perturbation-Fragile*, and *Ambiguous*. Evaluated across sensitivity sweeps ($\tau_{\text{hi}} \in [0.70, 0.80], \tau_{\text{lo}} \in [0.45, 0.55]$). Verified in `tests/mathematical_validation/test_math.py::test_reliability_classification`. |
| **8** | **Perturbation-Aware Consensus Construction** | **PASS** | `src/pccs/consensus/builder.py` | Non-circular reliability modulation $W_k = (1 - PSI_k)^\gamma$ fused with invariant core intersection ($\min_p S_{ij}^{(p)}$), avoiding circular baseline agreement weighting. Verified in `tests/mathematical_validation/test_math.py::test_consensus_formation`. |
| **9** | **Final Consensus with Reliability Profile** | **PASS** | `src/pccs/model.py` | Fitted `PCCS` estimator outputs partition labels `labels_`, cluster fingerprints `fingerprints_`, sensitivity indices `psi_`, reliability categories `categories_`, and interaction matrix `interactions_`. Verified in `tests/integration/test_integration.py::test_pccs_estimator_fit_predict`. |
| **10** | **Reproducible Evaluation of Quality, Robustness & Cost** | **PASS** | `scripts/run_pipeline.py`, `scripts/run_tests.py`, `tests/` | 13/13 automated test suite passing in ~5.0s. All 5 experimental studies executed across multiple seeds; raw CSVs saved in `results/raw/`, reports compiled in `reports/` and `results/reports/`. |

---

## 3. Mathematical Formulation Traceability

| Step | Mathematical Expression | Implementation Function | Verification Test |
|---|---|---|---|
| **Base Ensemble Realizations** | $C^{(m, p, l, r)} = M_m(T_{p, l, r}(X); \theta_m)$ | `src/pccs/clustering/ensemble.py::HeterogeneousEnsemble.fit_predict_ensemble` | `tests/unit/test_unit.py::test_ensemble_generation` |
| **Pairwise Stability Tensor** | $S_{ij}^{(p)} = \frac{\sum_{m, r} \mathbf{1}[C_r(i) = C_r(j) \ne -1]}{\sum_{m, r} \mathbf{1}[i, j \in X_r]}$ | `src/pccs/stability/tensor.py::StabilityTensor.compute` | `tests/mathematical_validation/test_math.py::test_tensor_properties` |
| **Hungarian Cluster Retention** | $\text{Ret}_{k,p} = \frac{1}{R_p} \sum_{r=1}^{R_p} \max_{c'} \frac{\|C_k \cap C_{c'}^{(r)}\|}{\|C_k \cup C_{c'}^{(r)}\|}$ | `src/pccs/stability/matching.py::HungarianClusterMatcher.compute_retention` | `tests/mathematical_validation/test_math.py::test_hungarian_matching` |
| **Internal Co-Association** | $\text{CoAssoc}_{k,p} = \frac{1}{\|C_k\|(\|C_k\| - 1)} \sum_{i \ne j \in C_k} S_{ij}^{(p)}$ | `src/pccs/stability/fingerprints.py::ClusterFingerprintCalculator.compute_coassociation` | `tests/mathematical_validation/test_math.py::test_fingerprint_computation` |
| **Combined Fingerprint Stability** | $S_{k,p} = \frac{1}{2} (\text{Ret}_{k,p} + \text{CoAssoc}_{k,p})$ | `src/pccs/stability/fingerprints.py::ClusterFingerprintCalculator.compute` | `tests/mathematical_validation/test_math.py::test_fingerprint_computation` |
| **Perturbation Sensitivity Index** | $PSI_k = \sum_{p=1}^P w_p (1 - S_{k,p}), \quad \sum w_p = 1$ | `src/pccs/stability/fingerprints.py::ClusterFingerprintCalculator.compute_psi` | `tests/mathematical_validation/test_math.py::test_fingerprint_computation` |
| **Factorial Interaction Statistic** | $\Delta_{pq} = \frac{S_p \cdot S_q}{S_0} - S_{pq}^{\text{observed}}$ | `src/pccs/interactions/factorial.py::PerturbationInteractionAnalyzer.compute_interaction` | `tests/mathematical_validation/test_math.py::test_interaction_modeling` |
| **Bootstrap Standard Error** | $\text{SE}(\Delta_{pq}) = \sqrt{\frac{1}{B-1} \sum_{b=1}^B (\Delta_{pq}^{(b)} - \bar{\Delta}_{pq})^2}$ | `src/pccs/interactions/factorial.py::PerturbationInteractionAnalyzer.compute_bootstrap_se` | `tests/mathematical_validation/test_math.py::test_interaction_modeling` |
| **Consensus Affinity Matrix** | $A_{ij} = (1 - \lambda) \sum_p w_p S_{ij}^{(p)} W(i, j) + \lambda \min_p S_{ij}^{(p)}$ | `src/pccs/consensus/builder.py::PerturbationConditionalConsensus.build_affinity` | `tests/mathematical_validation/test_math.py::test_consensus_formation` |

---

## 4. Audit of Legacy Defects and Corrective Actions

| Component | Legacy Defect in Prior Implementation | Scientific Consequence | Corrective Action Implemented in PCCS | Validated Outcome |
|---|---|---|---|---|
| **Consensus Weighting** | Weighted partitions by ARI agreement with an unperturbed baseline (`ari_ref`). | Circular confirmation bias: reinforced baseline errors on noisy/corrupted datasets, penalizing genuine structural discoveries. | Implemented non-circular cluster-level reliability modulation ($W_k = (1 - PSI_k)^\gamma$) and invariant core fusion ($\min_p S_{ij}^{(p)}$). | PCCS Full achieves **0.7929** mean ARI on real-world benchmarks, outperforming single baselines with $p = 0.00098$ and $d \ge 2.08$. |
| **DBSCAN Ensemble Inclusion** | Mapped noise to singletons and excluded DBSCAN if baseline agreement fell below 0.2. | Silently eliminated density-based diversity; noise points distorted cluster counts. | Implemented adaptive k-NN quantile distance calibration ($q=0.5$) and formal noise tracking (noise pairs assigned zero co-association, eligible denominators tracked). | DBSCAN actively contributes density diversity across all perturbation families. |
| **Falsification Test** | Supervised logistic classifier trained on synthetic ground-truth labels. | Label leakage: evaluating an unsupervised method with a supervised model trained on true class labels violates scientific rigor. | Implemented strictly unsupervised null falsification testing against isotropic Gaussian and anisotropic correlated Gaussian null distributions. | $PSI$ cleanly separates real clusters ($0.0392 \pm 0.0043$) from isotropic nulls ($0.4550 \pm 0.0183$) and anisotropic nulls ($0.4483 \pm 0.0453$) with zero distribution overlap. |
| **Unobserved Subsample Pairs** | Assigned artificial zero stability to pairs unobserved under subsampling. | Heavily penalized subsampled points, biasing co-association estimates downward. | Tracked exact eligible observation denominators $\sum I(i, j \in X_r)$; unobserved pairs imputed via baseline prior. | Bounded co-association values $S_{ij} \in [0, 1]$ guaranteed across all sampling intensities. |
| **Interaction Analysis** | Evaluated at single intensity with $R=2$ repetitions; no uncertainty bounds. | Underpowered; unable to distinguish random estimation variance from physical synergy. | Evaluated factorial combinations with $B=100$ bootstrap resamples, standard errors, and 95% confidence intervals. | Detected sub-multiplicative buffering ($\Delta = -0.1920 \pm 0.0372$) and tested for synergistic collapse. |
| **Statistical Rigor** | No formal hypothesis tests; raw score differences reported without effect sizes. | Differences could not be statistically distinguished from random variation. | Executed paired Wilcoxon signed-rank tests, Holm-Bonferroni family-wise error adjustments, and Cohen's $d$ effect sizes. | Statistical significance confirmed across all base algorithms ($p = 0.00098$, adj $p = 0.0078$, $d = 2.08$ to $6.89$). |

---

## 5. Automated Test Suite Traceability

All 13 automated tests in `tests/` pass with 100% success rate (`python scripts/run_tests.py` or `pytest tests/` in ~5.0s):

| Test File | Test Name | Purpose and Verification Scope | Status |
|---|---|---|---|
| `tests/unit/test_unit.py` | `test_ensemble_generation` | Verifies heterogeneous base ensemble generation and label properties. | **PASS** |
| `tests/unit/test_unit.py` | `test_perturbation_engine_families` | Verifies all 6 perturbation families and 3 intensity tiers operate without error. | **PASS** |
| `tests/unit/test_unit.py` | `test_preprocessing_pipeline` | Verifies missing imputation, 4.0 MAD winsorization, and standardization. | **PASS** |
| `tests/unit/test_unit.py` | `test_metrics_calculation` | Verifies calculation of ARI, AMI, NMI, and Silhouette scores. | **PASS** |
| `tests/mathematical_validation/test_math.py` | `test_tensor_properties` | Verifies $S_{ij}^{(p)} \in [0, 1]$, symmetry $S_{ij} = S_{ji}$, diagonal $S_{ii} = 1$, and non-zero denominators. | **PASS** |
| `tests/mathematical_validation/test_math.py` | `test_hungarian_matching` | Verifies optimal Hungarian bipartite matching aligns permuted cluster IDs. | **PASS** |
| `tests/mathematical_validation/test_math.py` | `test_fingerprint_computation` | Verifies fingerprint range $F_k \in [0, 1]^P$, $PSI_k \in [0, 1]$, and non-negative standard deviations. | **PASS** |
| `tests/mathematical_validation/test_math.py` | `test_interaction_modeling` | Verifies multiplicative interaction statistics $\Delta_{pq}$, bootstrap SEs, and confidence intervals. | **PASS** |
| `tests/mathematical_validation/test_math.py` | `test_reliability_classification` | Verifies descriptive classification into the 4 mutually exclusive categories. | **PASS** |
| `tests/mathematical_validation/test_math.py` | `test_consensus_formation` | Verifies non-circular reliability-modulated consensus and invariant core fusion. | **PASS** |
| `tests/integration/test_integration.py` | `test_pccs_estimator_fit_predict` | Verifies end-to-end `PCCS` estimator `fit()` and `predict()` on synthetic data. | **PASS** |
| `tests/integration/test_integration.py` | `test_real_world_data_pipeline` | Verifies pipeline execution on benchmark datasets with isolated reference labels. | **PASS** |
| `tests/regression/test_regression.py` | `test_circularity_prevention` | Verifies that consensus formation does not degrade under noisy baseline reference partitions. | **PASS** |

---

## 6. Experimental Studies & Deliverable Traceability

| Study ID | Monograph Section | Executed Output CSV | Comprehensive Report Deliverable | Publication Figure |
|---|---|---|---|---|
| **Experiment 1** | Section 11, Section 13 | `results/raw/synthetic_benchmark_results.csv`, `results/raw/falsification_results.csv` | `reports/experimental_results/experiment_01_synthetic_validation_report.md` | `results/figures/fig1_synthetic_noisy_performance.png`, `results/figures/fig5_falsification_psi.png` |
| **Experiment 2** | Section 7, Section 10 | `results/raw/perturbation_sweeps_results.csv` | `reports/experimental_results/experiment_02_perturbation_sweeps_report.md` | `results/figures/fig1_synthetic_noisy_performance.png` |
| **Experiment 3** | Section 8, Section 12 | `results/raw/cluster_fragility_results.csv`, `results/raw/synthetic_fingerprints.csv` | `reports/experimental_results/experiment_03_cluster_fragility_report.md` | `results/figures/fig3_cluster_stability_fingerprints.png` |
| **Experiment 4** | Section 9, Section 12 | `results/raw/synthetic_interactions.csv` | `reports/experimental_results/experiment_04_interaction_analysis_report.md` | `results/figures/fig4_perturbation_interactions.png` |
| **Experiment 5** | Section 14, Section 16 | `results/raw/real_world_benchmark_results.csv`, `results/raw/statistical_tests_summary.csv` | `reports/experimental_results/experiment_05_consensus_ablation_report.md` | `results/figures/fig2_real_world_benchmarks.png` |
