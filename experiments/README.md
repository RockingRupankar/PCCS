# PCCS Experiments Directory

This directory contains the experimental framework for evaluating **Perturbation-Conditional Consensus Clustering (PCCS)** across 5 scientific studies defined in the research specification:

1. **[experiment_01_synthetic_validation/](experiment_01_synthetic_validation/)**: Controlled Synthetic Ground-Truth Validation and Unsupervised Falsification Study.
2. **[experiment_02_perturbation_sweeps/](experiment_02_perturbation_sweeps/)**: Perturbation-Family and Intensity Sweeps across 6 perturbation operators.
3. **[experiment_03_cluster_fragility/](experiment_03_cluster_fragility/)**: Global Stability versus Cluster-Level Fragility (unmasking rare subpopulation vulnerability).
4. **[experiment_04_interaction_analysis/](experiment_04_interaction_analysis/)**: Factorial Perturbation Interaction Analysis with bootstrap standard errors.
5. **[experiment_05_consensus_ablation/](experiment_05_consensus_ablation/)**: Reliability-Aware Consensus, Constituent Baselines, Consensus Baselines, and Statistical Testing.

---

## How to Execute the Experiments

### Option 1: Run All 5 Experiments Simultaneously (Master Script)
```bash
python scripts/run_experiments.py
```
This executes all 5 experiments end-to-end, writes machine-readable CSVs to `results/raw/`, and generates all 5 publication figures in `results/figures/`.

### Option 2: Run an Individual Experiment Standalone
You can execute any individual experiment independently:

```bash
# Experiment 1: Synthetic Validation & Falsification
python experiments/experiment_01_synthetic_validation/run.py

# Experiment 2: Perturbation Sweeps
python experiments/experiment_02_perturbation_sweeps/run.py

# Experiment 3: Cluster Fragility
python experiments/experiment_03_cluster_fragility/run.py

# Experiment 4: Interaction Analysis
python experiments/experiment_04_interaction_analysis/run.py

# Experiment 5: Consensus Benchmarks & Statistical Tests
python experiments/experiment_05_consensus_ablation/run.py
```

### Option 3: Compile Detailed Reports
After running experiments, compile all markdown reports and summary tables:
```bash
python scripts/generate_reports.py
```

---

## Output Destinations
- **Raw CSV Outputs:** `results/raw/*.csv`
- **Publication Figures:** `results/figures/fig*.png`
- **Experimental Markdown Reports:** `reports/experimental_results/*.md`
- **Scientific Audit Reports:** `results/reports/*.md` and `*.csv`
- **Configuration Specs:** `configs/experiments.yaml`
