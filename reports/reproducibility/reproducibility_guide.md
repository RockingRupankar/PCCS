# PCCS Reproducibility and Execution Guide
**Project:** Perturbation-Conditional Consensus Clustering (PCCS)  
**Date:** 2026-10-13  
**Status:** Validated  

---

## 1. Environment & Dependencies

- **Python Version:** 3.10+ (Tested on Python 3.14.5, Windows 11)
- **Primary Packages:**
  - `numpy>=1.24.0`
  - `scipy>=1.10.0`
  - `scikit-learn>=1.2.0`
  - `pandas>=2.0.0`
  - `matplotlib>=3.7.0`
  - `pyyaml>=6.0`
  - `pytest>=7.0.0`

### Setup Instructions
```bash
git clone https://github.com/RockingRupankar/PCCS.git
cd PCCS
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate # Linux/macOS
pip install -e ".[dev]"
```

---

## 2. Seed and Determinism Policy

All stochastic operations are strictly governed by explicit seed parameters:
- Baseline ensemble runs: `seed = random_state * 100 + run_index`
- Perturbation realizations: explicit pseudo-random seeds recorded in partition dictionaries.
- Subsampling partitions: deterministic stratified sampling recorded in `data/manifests/dataset_manifest.csv`.
- Unit and integration tests verify deterministic byte-for-byte reproducibility (`tests/regression/test_regression.py`).

---

## 3. Running Automated Tests

```bash
python scripts/run_tests.py
```
This runs the full 13-test automated suite across:
- `tests/unit/`: components (preprocessing, perturbation engine, ensemble clustering, baselines)
- `tests/mathematical_validation/`: checks A-M (Hungarian permutation invariance, joint observation masking, DBSCAN noise non-coassociation, fuzzy membership simplex constraints, hand-calculated factorial interaction verification, PSI weighting sensitivity)
- `tests/integration/`: end-to-end PCCS pipeline and strict label isolation
- `tests/regression/`: deterministic seed reproducibility

---

## 4. Executing Experiments and Generating Reports

```bash
# 1. Run all 5 mandatory research experiments
python scripts/run_experiments.py

# 2. Compile structured markdown reports from executed raw results
python scripts/generate_reports.py
```

Outputs are automatically organized into:
- `results/raw/`: raw per-run results in CSV format
- `results/figures/`: publication figures (fig1 through fig5)
- `reports/experimental_results/`: structured markdown reports for Experiments 1 through 5
