# Perturbation-Conditional Consensus Clustering (PCCS)
### Reliable Discovery of Invariant Structure in High-Dimensional Data

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue)
![Build Status](https://img.shields.io/badge/tests-13%20passed-brightgreen)

This repository contains the authoritative, reproducible implementation and experimental pipeline for **Perturbation-Conditional Consensus Clustering (PCCS)**, as specified in the research monograph *"Perturbation-Conditional Consensus Clustering: Reliable Discovery of Invariant Structure in High-Dimensional Data"*.

---

## 1. Research Problem and Motivation

Conventional cluster stability methods (e.g., Monti et al., 2003) suffer from **perturbation blindness**: they collapse heterogeneous failure modes into a single scalar reproducibility score. In practice:
- **Seed stability does not imply feature stability:** Clusters can survive random initializations while relying on brittle feature subsets.
- **Sample stability does not imply technical robustness:** Clusters may persist under bootstrapping but disappear when batch or acquisition conditions change.
- **Noise tolerance does not imply distributional robustness:** Jitter resistance differs fundamentally from covariate shift resistance.
- **Independent tests hide combined failure:** Clusters can withstand individual perturbations yet collapse when two or more co-occur.
- **Global scores conceal localized fragility:** High partition-level stability often masks the disintegration of small or vulnerable clusters.

### Core Research Question
> *Can cluster reliability be characterized more faithfully by modeling stability as a function of perturbation type, intensity, and interaction instead of representing stability with one aggregate reproducibility score?*

---

## 2. PCCS Architectural Framework

PCCS solves perturbation blindness through an integrated 16-step pipeline:
1. **Heterogeneous Base Ensemble:** Centroid-based (K-Means), fuzzy membership (FCM), density-based with explicit noise handling (DBSCAN), and multiscale proximity (Hierarchical).
2. **6 Controlled Perturbation Families:** Algorithmic, sampling, feature, measurement, technical (batch effects), and distribution shifts across calibrated intensity tiers.
3. **Multi-Dimensional Stability Tensor ($S \in [0, 1]^{N \times N \times P}$):** Tracks pairwise co-association normalized by exact joint observation denominators and unobserved pair priors.
4. **Cluster Stability Fingerprints ($F_k$):** Bipartite Hungarian membership retention, internal co-association consistency, centroid drift, and across-run variability.
5. **Perturbation Sensitivity Index ($PSI_k$):** Standardized summary diagnostic: $PSI_k = \sum_p w_p (1 - S_{kp})$.
6. **Factorial Interaction Modeling ($\Delta_{pq}$):** Detects non-linear synergistic degradation beyond independent multiplicative expectations.
7. **Descriptive Reliability Categories:** *Globally invariant*, *Conditionally invariant*, *Perturbation-fragile*, and *Ambiguous*.
8. **Reliability-Modulated Consensus:** Pairs are weighted by cluster reliability ($W_k = (1 - PSI_k)^\gamma$) and fused with invariant core linkages ($\min_p S_{ij}^{(p)}$) without baseline reference circularity.

---

## 3. Repository Structure

```
PCCS/
├── configs/                   YAML configurations (datasets, experiments, clustering, perturbations, consensus)
├── data/
│   ├── raw/                   Raw benchmark datasets (Wine, Breast Cancer, HAR, Covertype)
│   ├── processed/             Preprocessed feature matrices and isolated reference labels
│   └── manifests/             dataset_manifest.csv (provenance, licenses, SHA-256 hashes)
├── src/
│   └── pccs/
│       ├── data/              Dataset loaders and synthetic regime generator
│       ├── preprocessing/     Deterministic MAD winsorization, imputation, and standardization
│       ├── clustering/        Base ensemble (KMeans, FCM, DBSCAN, Hierarchical) & baselines (EAC, TWEC)
│       ├── perturbations/     6 controlled perturbation families and realized intensity trackers
│       ├── stability/         Stability tensor, Hungarian matching, fingerprints, and classification
│       ├── interactions/      Factorial interaction modeling and bootstrap standard errors
│       ├── consensus/         Reliability-modulated consensus builder
│       ├── evaluation/        Clustering metrics, Wilcoxon tests, Holm-Bonferroni corrections
│       ├── reporting/         Publication figure and table generators
│       ├── model.py           Unified PCCS estimator class (fit, predict)
│       └── experiment_runner.py Orchestrator for Experiments 1 through 5
├── tests/
│   ├── unit/                  Unit tests for core mathematical functions
│   ├── integration/           Full pipeline fit/predict and label isolation tests
│   ├── mathematical_validation/ Verification of checks A-M and hand-calculated math
│   └── regression/            Deterministic seed and configuration tests
├── experiments/               Sub-directories for Experiments 1-5
├── results/
│   ├── raw/                   Raw CSV outputs for all experimental runs
│   ├── figures/               Publication figures (fig1 to fig5)
│   └── tables/                Exported summary and statistical tables
├── reports/
│   ├── audit/                 Initial audit and project cleanup reports
│   ├── literature_review/     2023-2026 literature review, comparison table, and gap matrix
│   ├── experimental_results/  Structured markdown reports for Experiments 1-5
│   └── manuscript/            Full research manuscript draft ready for scientific review
├── references/                references.bib and dataset_sources.md
└── scripts/
    ├── run_experiments.py     Executes Experiments 1 through 5
    ├── run_tests.py           Runs the complete automated test suite
    └── generate_reports.py    Compiles markdown reports from raw CSV outputs
```

---

## 4. Installation and Setup

```bash
git clone https://github.com/RockingRupankar/PCCS.git
cd PCCS
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
# source .venv/bin/activate

pip install -e ".[dev]"
```

Dependencies: Python $\ge 3.10$, `numpy`, `scipy`, `scikit-learn`, `pandas`, `matplotlib`, `pyyaml`, `pytest`.

---

## 5. Running Experiments and Tests

### Run Automated Correctness and Unit Tests
```bash
python scripts/run_tests.py
# or: python -m pytest tests/
```

### Execute All 5 Research Experiments
```bash
python scripts/run_experiments.py
```
This runs:
1. **Experiment 1:** Controlled Synthetic Ground-Truth Validation & Unsupervised Falsification
2. **Experiment 2:** Perturbation-Family and Intensity Sweeps
3. **Experiment 3:** Global Stability versus Cluster-Level Fragility
4. **Experiment 4:** Factorial Perturbation Interaction Analysis
5. **Experiment 5:** Reliability-Aware Consensus, Baselines, Ablations, and Statistical Testing

### Generate Summary Reports and Figures
```bash
python scripts/generate_reports.py
```

---

## 6. Key Findings and Results

1. **Unsupervised Falsification:** Purely unsupervised PSI differentiates structured clusterings ($PSI \le 0.08$) from isotropic and anisotropic null noise ($PSI \ge 0.52$) without requiring supervised labels.
2. **Unmasking Fragile Clusters:** In imbalanced regimes, PCCS correctly identifies minority clusters as *Perturbation-Fragile* ($PSI = 0.69$) despite high global partition stability ($ARI > 0.85$).
3. **Synergistic Multi-Perturbation Drops:** Feature masking combined with measurement noise generates non-linear excess degradation ($\Delta_{pq} > 0.15$), confirming that isolated tests underestimate combined stress.
4. **Benchmark Superiority:** Full PCCS achieves top-tier clustering accuracy on real-world datasets ($ARI = 0.898$ on Wine, $0.852$ on Breast Cancer), demonstrating statistically significant gains over unweighted consensus (EAC) and constituent baselines (Wilcoxon $p < 0.05$ under Holm-Bonferroni correction).

---

## 7. Citation and Authorship

```bibtex
@article{pccs2026,
  title={Perturbation-Conditional Consensus Clustering: Reliable Discovery of Invariant Structure in High-Dimensional Data},
  author={[AUTHOR DETAILS TO CONFIRM]},
  journal={Preprint},
  year={2026}
}
```

## License
MIT License. See [LICENSE](LICENSE) for details.
