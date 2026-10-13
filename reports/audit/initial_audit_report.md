# Initial Pre-Cleanup Audit Report
**Project:** Perturbation-Conditional Consensus Clustering (PCCS)  
**Date:** 2026-10-13  
**Operating Mode:** IMPLEMENTER (Antigravity)  
**Authoritative Specification:** *"Perturbation-Conditional Consensus Clustering: Reliable Discovery of Invariant Structure in High-Dimensional Data"*

---

## 1. Executive Summary & Workspace Inventory

An exhaustive inspection of the workspace (`c:\Users\USER\OneDrive\Desktop\PCCS`) was conducted prior to executing any modifications.
- **Git Branching & Backup:** A dedicated Git backup branch `backup-pre-cleanup` was created at commit `8d09851` to ensure full recoverability of all historic files.
- **File Inventory:** 267 files were inventoried with exact file sizes and SHA-256 cryptographic hashes saved in `file_inventory_pre_cleanup.csv`.
- **Existing Layout Defect:** The workspace contained a nested legacy repository structure `PCCS/icc-clustering-experiments/icc-clustering-experiments` containing mixed assets from an unrelated Inductive Consensus Clustering (ICC) project and incomplete PCCS prototypes.
- **Reference Image Status:** As specified in Section 5, the reference directory structure image was not provided in attachments or local directories. As mandated, this was noted and the standardized fallback PCCS layout has been adopted without inventing non-existent reference images.

---

## 2. Inventory of Authoritative vs. Stale & Unrelated Assets

| Asset Category | Location | Classification | Disposition / Action |
|---|---|---|---|
| **Authoritative PDF Spec** | `.user_uploaded/media_1791872959932_4d55c11e.pdf` | **Authoritative Specification** | Retained as ground-truth theoretical specification for PCCS. |
| **PCCS Modular Core** | `icc-clustering-experiments/.../src/pccs/` | **Authoritative Core** | Refactor and elevate to top-level `src/pccs/` according to standard modular architecture. |
| **ICC Core Implementation** | `icc-clustering-experiments/.../src/icc/model.py` | **Unrelated (ICC)** | Remove. Implements Inductive Consensus Clustering (out-of-sample prediction), which is unrelated to PCCS. |
| **ICC Adapter & Config** | `configs/icc_default.yaml`, `experiments/common/icc_adapter.py` | **Unrelated (ICC)** | Remove. PCCS does not require ICC adapters or YAML definitions. |
| **ICC Legacy Suites** | `experiments/exp0` to `exp4` (containing `fig10_icc_v1_performance.png`, etc.) | **Stale / Mixed ICC** | Archive/Remove from active PCCS project. Replace with dedicated PCCS 5-experiment suite. |
| **Legacy Tests** | `tests/test_icc_model.py` | **Unrelated (ICC)** | Remove. Replace with comprehensive unit, integration, mathematical validation, and regression suites for PCCS. |
| **Stale Personal Names** | `README.md`, `CITATION.cff`, `CONTRIBUTING.md` (`Akash Kundu`, `AkashKundu114`) | **Stale Metadata** | Remove and replace with neutral PCCS metadata (`[AUTHOR DETAILS TO CONFIRM]`). |
| **Previous Results** | `results/raw/`, `results/figures/`, `results/reports/` | **Historic Evidence** | Preserve logs and raw data in backup branch; generate fresh verified results under target layout. |

---

## 3. Mathematical and Scientific Correctness Audit (Verification of Checks A–M)

| ID | Component / Property | Existing Implementation | Scientific Defect / Risk | Evidence | Corrective Action | Validation Test |
|---|---|---|---|---|---|---|
| **A** | Label Permutation Invariance | Scipy Hungarian matching (`linear_sum_assignment`) | Cluster ID alignment across runs must be strictly invariant to arbitrary integer permutation. | `src/pccs/matching.py` | Enforce bipartite Hungarian cost minimization on Jaccard/overlap. | `tests/mathematical_validation/test_math.py::test_label_permutation_invariance` |
| **B** | Shared Observation Alignment | Slicing by observation index | Comparisons under subsampling could mismatch rows if indices are not strictly matched. | `src/pccs/tensor.py` | Compute pairwise co-association only over shared row sets: $i, j \in X_r$. | `tests/mathematical_validation/test_math.py::test_shared_observation_masking` |
| **C** | Pairwise Co-association Denominators | Denominator matrix $N_{ij}^{(p)}$ | If denominator is fixed to total runs $R$ rather than co-observed runs $R_{ij}$, stability is artificially depressed. | `src/pccs/tensor.py` | Track exact pairwise coverage count matrix $C_{ij}^{(p)} = \sum_r \mathbf{1}[i, j \in X_r]$. | `tests/mathematical_validation/test_math.py::test_denominator_eligibility` |
| **D** | Undefined Pairwise Comparisons | Prior fill = 0.5 | If a pair is never jointly observed, setting co-association to 0 penalizes the pair unfairly. | `src/pccs/tensor.py` | Explicitly flag unobserved pairs as undefined, imputing uninformative prior 0.5 only when required for distance metrics. | `tests/mathematical_validation/test_math.py::test_undefined_pair_handling` |
| **E** | DBSCAN Noise Treatment | Noise labeled as -1 | Treating noise as cluster ID -1 causes all noise points to co-associate falsely. | `src/pccs/ensemble.py` | Noise points are treated as non-co-associating: $\mathbf{1}[C(i)=C(j) \land C(i) \ne -1]$. | `tests/unit/test_ensemble.py::test_dbscan_noise_handling` |
| **F** | Cluster Morphologies (Split/Merge/Vanish) | Best-match Jaccard retention | A vanished cluster or fragmented cluster may be obscured if only a single best-match score is reported. | `src/pccs/fingerprints.py` | Measure cluster retention, split entropy, and centroid drift simultaneously. | `tests/mathematical_validation/test_math.py::test_split_merge_detection` |
| **G** | Fuzzy C-Means (FCM) Membership | Log-sum-exp soft membership | Hard argmax conversion must be explicit, preserving soft membership matrices for uncertainty weighting. | `src/pccs/ensemble.py` | Store soft memberships $U \in [0, 1]^{N \times K}$ and document hard assignment mapping. | `tests/unit/test_ensemble.py::test_fcm_soft_membership` |
| **H** | Run Tracking & Bookkeeping | Run dictionaries with metadata | Incomplete tracking of failed runs or stochastic variations hinders reproducibility. | `src/pccs/model.py` | Record seed, realized perturbation intensity, convergence status, execution time, and noise proportion per run. | `tests/integration/test_bookkeeping.py` |
| **I** | Cluster-Level vs Global Stability | Cluster retention and co-association | Reporting only global partition ARI conceals localized cluster fragility. | `src/pccs/fingerprints.py` | Compute cluster-specific fingerprint $F_k \in [0, 1]^P$ and standard deviation across runs. | `tests/mathematical_validation/test_math.py::test_local_vs_global_stability` |
| **J** | Perturbation Sensitivity Index (PSI) | $PSI_k = \sum_p w_p (1 - S_{kp})$ | PSI is a compact summary diagnostic and must never substitute for the multi-dimensional fingerprint $F_k$. | `src/pccs/fingerprints.py` | Preserve both $F_k$ vector and $PSI_k$ scalar; perform sensitivity analysis over weights $w_p$. | `tests/mathematical_validation/test_math.py::test_psi_weight_sensitivity` |
| **K** | Consensus Circularity Prevention | Reliability modulation $W_k = (1 - PSI_k)^\gamma$ | Weighting partitions by agreement with a flawed unperturbed baseline partition creates circularity. | `src/pccs/consensus.py` | Weight pairwise links by cluster reliability profiles and invariant core fusion ($\lambda \min_p S_{ij}^{(p)}$) without baseline reference bias. | `tests/mathematical_validation/test_math.py::test_non_circular_consensus` |
| **L** | Unsupervised Integrity & Label Leakage | Evaluator strictly isolated | Ground-truth labels must never influence preprocessing, feature selection, or consensus fitting. | `src/pccs/data.py` | Explicit separation: data loader returns $X$ and optional $y$; $y$ is reserved exclusively for external evaluation metrics. | `tests/integration/test_label_isolation.py` |
| **M** | Executed vs Hard-Coded Results | Raw CSV results pipeline | Visualizations and tables must be generated exclusively from executed, saved raw CSV files. | `scripts/run_pccs_pipeline.py` | All tables and figures load directly from `results/raw/*.csv`. No placeholder or hardcoded arrays. | `tests/regression/test_reproducibility.py` |

---

## 4. Planned Project Reorganization

Following the authoritative specification and fallback directory guidelines, the workspace will be restructured cleanly at the project root:
- `configs/` (`datasets.yaml`, `experiments.yaml`, `clustering.yaml`, `perturbations.yaml`, `consensus.yaml`)
- `data/` (`README.md`, `raw/`, `processed/`, `manifests/`)
- `src/pccs/` (`data/`, `preprocessing/`, `clustering/`, `perturbations/`, `stability/`, `interactions/`, `consensus/`, `evaluation/`, `reporting/`, `experiment_runner.py`)
- `tests/` (`unit/`, `integration/`, `mathematical_validation/`, `regression/`)
- `experiments/` (`experiment_01_synthetic_validation/`, `experiment_02_perturbation_sweeps/`, `experiment_03_cluster_fragility/`, `experiment_04_interaction_analysis/`, `experiment_05_consensus_ablation/`)
- `results/` (`raw/`, `tables/`, `figures/`, `logs/`, `manifests/`)
- `reports/` (`audit/`, `literature_review/`, `experimental_results/`, `manuscript/`, `reproducibility/`)
- `references/` (`references.bib`, `dataset_sources.md`)
- `scripts/` (`run_experiments.py`, `run_tests.py`, `generate_reports.py`)
- Root: `README.md`, `pyproject.toml`, `requirements.txt`, `.gitignore`
