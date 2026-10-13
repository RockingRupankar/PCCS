# Experiment 4: Factorial Perturbation Interaction Analysis

## Objective
Quantify combined perturbation interactions ($\Delta_{pq} = \frac{S_p S_q}{S_0} - S_{pq}$) using multiplicative and additive reference models along with non-parametric bootstrap standard errors ($B=100$) and 95% confidence intervals, detecting sub-multiplicative buffering ($\Delta < -0.15$) and testing for synergistic degradation ($\Delta > +0.15$).

## Execution
```bash
python experiments/experiment_04_interaction_analysis/run.py
```

## Generated Artifacts
- `results/raw/synthetic_interactions.csv`
- `results/figures/fig4_perturbation_interactions.png`
- `reports/experimental_results/experiment_04_interaction_analysis_report.md`
