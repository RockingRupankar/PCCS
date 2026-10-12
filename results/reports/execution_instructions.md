# PCCS Execution and Reproducibility Instructions

**Project:** Perturbation-Conditional Consensus Clustering (PCCS)  
**Specification Reference:** Section 18 & Section 19.L  

---

## 1. Environment & Dependencies

* **OS:** Windows / Linux / macOS
* **Python Version:** Python 3.10, 3.11, 3.12, or 3.14
* **Required Libraries:**
  ```bash
  pip install numpy scipy scikit-learn pandas matplotlib pytest
  ```

---

## 2. Directory Structure

```
PCCS/
├── icc-clustering-experiments/
│   └── icc-clustering-experiments/
│       ├── src/
│       │   ├── pccs/            # Authoritative PCCS implementation
│       │   └── icc/             # Legacy reference adapter
│       ├── experiments/         # Experiment baselines and generators
│       ├── scripts/
│       │   └── run_pccs_pipeline.py  # Master end-to-end execution script
│       └── tests/
│           ├── test_pccs_correctness.py  # Comprehensive correctness tests
│           └── test_pccs.py              # Legacy test suite
└── results/
    ├── raw/                     # Machine-readable per-run CSV outputs
    ├── reports/                 # Structured research reports & summary tables
    └── figures/                 # Publication-quality visual figures
```

---

## 3. Step-by-Step Reproduction Commands

### Step 1: Run the Automated Correctness Test Suite
```powershell
$env:PYTHONPATH="src;experiments"
python -m pytest tests/test_pccs_correctness.py
```
*(Runs 13 mathematical, logical, and pipeline correctness tests; expected duration: ~8 seconds).*

### Step 2: Run All Tests (Legacy + Comprehensive Correctness)
```powershell
$env:PYTHONPATH="src;experiments"
python -m pytest tests/
```
*(Runs all 49 automated unit tests; all pass with 100% success).*

### Step 3: Execute the Complete Experimental Pipeline
```powershell
$env:PYTHONPATH="src;experiments"
python scripts/run_pccs_pipeline.py
```
This single command executes:
1. Synthetic benchmark suites across clean, noisy, and severe regimes (5 seeds).
2. Base methods (K-Means, FCM, DBSCAN, Ward), unweighted consensus (EAC), and 7 PCCS ablations.
3. Real-world benchmarks across Iris, Wine, Breast Cancer, and Digits (original and corrupted).
4. Falsification study on structured data vs isotropic and anisotropic null distributions.
5. Multi-level interaction evaluation with bootstrap confidence intervals.
6. Statistical significance testing with Holm-Bonferroni correction and Cohen's $d$.
7. Threshold sensitivity analysis.
8. Generation of all publication-quality figures (`fig1` through `fig5`).

*(Expected total runtime on a single CPU core: ~7.5 minutes).*

---

## 4. Output Deliverables Locations

* **Raw Experiment Data:** `results/raw/`
* **Generated Figures:** `results/figures/`
  * `fig1_synthetic_noisy_performance.png`: Comparative method performance on noisy synthetic data.
  * `fig2_real_world_benchmarks.png`: Real-world performance across Iris, Wine, Breast Cancer, and Digits.
  * `fig3_cluster_stability_fingerprints.png`: Heatmap of cluster-level stability fingerprints.
  * `fig4_perturbation_interactions.png`: Observed vs expected joint perturbation stability.
  * `fig5_falsification_psi.png`: Discrimination of real clusters from null distributions via PSI.
* **Reports and Summaries:** `results/reports/`
