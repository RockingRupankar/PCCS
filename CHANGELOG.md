# Changelog

All notable changes to the **Perturbation-Conditional Consensus Clustering (PCCS)** research project and codebase are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.2.0] - 2026-10-13

### Authoritative Architecture, Methodological Rebuild & Empirical Verification

This milestone marks the complete, audited implementation of the authoritative research monograph:
*"Perturbation-Conditional Consensus Clustering: Reliable Discovery of Invariant Structure in High-Dimensional Data"*.

#### Added
- **Modular Package Architecture (`src/pccs/`):**
  - `clustering/ensemble.py`: Heterogeneous base ensemble incorporating K-Means, Fuzzy C-Means (stable log-domain memberships), DBSCAN (adaptive k-NN quantile $\varepsilon$ calibration), and Hierarchical agglomerative clustering.
  - `clustering/baselines.py`: Single-algorithm baselines (K-Means, FCM, DBSCAN, Hierarchical), consensus baselines (Evidence Accumulation Clustering - EAC, Two-tier Uncertainty Ensemble Clustering - TWEC), and ablation variants.
  - `perturbations/engine.py`: 6 controlled perturbation families (algorithmic, sampling, feature, measurement, technical batch shifts, and distribution tilting) across 3 intensity tiers (mild, moderate, severe) with realized intensity tracking.
  - `stability/tensor.py`: Third-order stability tensor $S \in [0, 1]^{N \times N \times P}$ with exact joint observation eligibility denominators and uninformative baseline prior imputation for unobserved subsample pairs.
  - `stability/matching.py`: Optimal bipartite Hungarian cluster matching (`scipy.optimize.linear_sum_assignment`) computing cluster membership retention across perturbed runs.
  - `stability/fingerprints.py`: Multi-dimensional cluster stability fingerprints $F_k \in [0, 1]^P$ combining Hungarian retention, internal pairwise co-association, centroid drift, across-run standard deviation, and Perturbation Sensitivity Index ($PSI_k$).
  - `stability/classification.py`: Descriptive cluster reliability classification into four evidence-based categories: *Globally Invariant*, *Conditionally Invariant*, *Perturbation-Fragile*, and *Ambiguous*.
  - `interactions/factorial.py`: Factorial perturbation interaction analyzer evaluating multiplicative ($\Delta_{pq} = \frac{S_p S_q}{S_0} - S_{pq}$) and additive reference models with non-parametric bootstrap standard errors ($B=100$) and 95% confidence intervals.
  - `consensus/builder.py`: Non-circular consensus builder modulating pairwise co-association by cluster reliability ($W_k = (1 - PSI_k)^\gamma$) fused with invariant core intersection linkages ($\lambda \min_p S_{ij}^{(p)}$).
  - `preprocessing/pipeline.py`: Deterministic data quality profiling, column-wise median imputation, 4.0 MAD robust winsorization, and standardization.
  - `evaluation/metrics.py`: Unsupervised (Silhouette, Davies-Bouldin, Calinski-Harabasz) and external validation metrics (Adjusted Rand Index, Adjusted Mutual Information, Normalized Mutual Information).
  - `evaluation/stats.py`: Paired Wilcoxon signed-rank tests, Holm-Bonferroni family-wise error rate corrections, and Cohen's $d$ effect size computations.
  - `reporting/plots.py`: Pure Matplotlib publication-grade plotting generator for Figures 1 through 5.
  - `reporting/tables.py`: Automated Markdown and LaTeX summary table generators.
  - `model.py`: High-level scikit-learn compatible estimator `PCCS` exposing standard `fit()` and `predict()` interfaces.
  - `experiment_runner.py`: Deterministic execution engine running all 5 benchmark studies across multiple seeds.

- **Automated Test Suite (`tests/`):**
  - 13 automated tests across unit, mathematical validation, integration, and regression suites passing with 100% success rate in ~5.0s.
  - Verified tensor bounds, symmetry, diagonal identity, Hungarian bipartite matching correctness, fingerprint ranges, bootstrap interaction statistics, and non-circular consensus formation.

- **Empirical Studies Executed (`results/raw/`):**
  - `synthetic_benchmark_results.csv`: 180 runs across 6 geometric regimes and 5 seeds ($N=400$ to $N=500$).
  - `falsification_results.csv`: 15 runs comparing authentic cluster manifolds against isotropic Gaussian and anisotropic correlated Gaussian null distributions.
  - `cluster_fragility_results.csv` & `synthetic_fingerprints.csv`: 12 cluster evaluations demonstrating unmasking of rare subpopulation fragility hidden beneath high global partition stability.
  - `perturbation_sweeps_results.csv`: 288 evaluations across 6 perturbation families and 3 intensity tiers.
  - `synthetic_interactions.csv`: 16 factorial cluster interaction evaluations across 4 perturbation pairs with bootstrap SEs and 95% CIs.
  - `real_world_benchmark_results.csv`: 90 evaluations across UCI Wine and Breast Cancer Diagnostic benchmarks across 5 seeds and 9 methods.
  - `statistical_tests_summary.csv`: Paired Wilcoxon signed-rank tests, Holm-Bonferroni adjustments, and Cohen's $d$ effect sizes.

- **Automated Report Engine (`scripts/generate_reports.py`):**
  - Synchronizes and programmatically compiles 5 experimental reports in `reports/experimental_results/`.
  - Compiles 10 scientific and audit reports in `results/reports/`.
  - Generates 4 summary CSV tables in `results/reports/` (`real_world_summary_table.csv`, `synthetic_summary_table.csv`, `statistical_tests_summary.csv`, `threshold_sensitivity_summary.csv`).

- **Academic Deliverables & Documentation:**
  - `reports/manuscript/manuscript_draft.md`: Academic manuscript formatted for peer review, updated with exact executed empirical metrics and 11 literature citations.
  - `reports/audit/final_acceptance_matrix.md`: Section 17 acceptance audit verifying 17/17 criteria PASS.
  - `pccs_traceability.md`: Comprehensive traceability matrix connecting specification requirements to code, mathematics, tests, and raw evidence.
  - `references/dataset_sources.md` & `data/manifests/dataset_manifest.csv`: Formal provenance, licenses, and SHA-256 hashes for UCI Wine, Breast Cancer Diagnostic, HAR, and Covertype datasets.

#### Fixed & Corrected
- **Consensus Circularity:** Eliminated circular partition-level weighting (`ari_ref` against an unperturbed baseline consensus) which previously penalized valid structural discoveries under noise. Replaced with cluster-level reliability modulation ($W_k = (1 - PSI_k)^\gamma$) and invariant core fusion ($\min_p S_{ij}^{(p)}$).
- **DBSCAN Silencing in Ensemble:** Fixed heuristic singletonization that previously caused DBSCAN to be discarded from the ensemble when baseline agreement fell below 0.2. Implemented adaptive k-NN quantile $\varepsilon$ calibration and formal noise pair exclusion.
- **Unobserved Subsampling Pair Bias:** Fixed artificial zero-stability assignment for unobserved pairs under row subsampling. Implemented denominator tracking and uninformative baseline prior imputation (0.5).
- **Label Leakage in Falsification Testing:** Replaced the prior supervised logistic classifier with strictly unsupervised null falsification testing against isotropic and anisotropic Gaussian null distributions ($PSI = 0.0392 \pm 0.0043$ for structured clusters vs $0.4550 \pm 0.0183$ and $0.4483 \pm 0.0453$ for null distributions).
- **Report & Manuscript Synchronization:** Completely resolved stale numerical values in `results/reports/`, `reports/experimental_results/`, and `reports/manuscript/manuscript_draft.md` to ensure 100% agreement with raw CSV outputs.

#### Removed
- Removed legacy `icc-clustering-experiments/` nested directory and obsolete ICC code from active workspace (safely preserved on Git branch `backup-pre-cleanup` at commit `8d09851`).
- Removed hard dependency on Seaborn in visualization scripts, replacing it with pure Matplotlib.
- Replaced personal template handles with standardized neutral placeholders (`[AUTHOR DETAILS TO CONFIRM]`).

---

## [0.1.0] - 2026-10-12

### Initial Prototype and Feasibility Experiments

#### Added
- Initial prototype implementation of consensus clustering and perturbation modules.
- Exploratory synthetic datasets and preliminary baseline comparison scripts.
- Initial project structure and exploratory notebooks.
