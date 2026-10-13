# Experiment 2: Perturbation-Family and Intensity Sweeps

## Objective
Evaluate stability degradation across 6 explicit perturbation families (algorithmic, sampling, feature, measurement, technical batch shifts, and distribution tilting) at 3 intensity tiers (mild, moderate, severe) to reveal multi-dimensional failure modes.

## Execution
```bash
python experiments/experiment_02_perturbation_sweeps/run.py
```

## Generated Artifacts
- `results/raw/perturbation_sweeps_results.csv`
- `results/figures/fig1_synthetic_noisy_performance.png`
- `reports/experimental_results/experiment_02_perturbation_sweeps_report.md`
