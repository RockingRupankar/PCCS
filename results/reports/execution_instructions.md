# PCCS Execution and Reproducibility Instructions

**Project:** Perturbation-Conditional Consensus Clustering (PCCS)  
**Specification Reference:** Section 18 & Section 19.L  

---

## 1. Environment & Dependencies

- **Operating System:** Windows / Linux / macOS
- **Python Version:** Python 3.10+
- **Required Libraries:**
  ```bash
  pip install numpy scipy scikit-learn pandas matplotlib pytest
  ```

## 2. Standard Workspace Directory Layout

```
PCCS/
├── configs/             # Experiment configuration files
├── data/                # Raw and preprocessed benchmark datasets
├── src/pccs/            # Authoritative PCCS implementation
├── tests/               # Automated unit, integration, and mathematical tests
├── experiments/         # Experiment execution runners
├── results/             # Raw CSVs, figures, and synchronized reports
├── reports/             # Publication reports, manuscript draft, and audit matrix
├── references/          # Literature references and BibTeX citations
└── scripts/             # Master execution and report generation scripts
```

## 3. Step-by-Step Reproduction Commands

### Step 1: Run All Automated Correctness Tests
```powershell
python scripts/run_tests.py
```
*(Runs 13 mathematical, logical, and pipeline correctness tests; expected duration: ~5 seconds; 100% pass rate).*  

### Step 2: Execute Master Pipeline & Experiments
```powershell
python scripts/run_pipeline.py
```
*(Executes the complete experimental suite across all regimes and benchmarks, saving raw CSVs to `results/raw/`).*  

### Step 3: Regenerate Figures and Reports
```powershell
python scripts/generate_reports.py
```
*(Compiles all markdown reports, summary CSV tables, and publication figures from raw CSV results).*  
