# Experiment 5: Reliability-Aware Consensus, Baselines, Ablations, and Statistical Testing

## Objective
Evaluate PCCS Full against constituent single algorithms (K-Means, FCM, DBSCAN, Hierarchical), consensus baselines (EAC, TWEC), and PCCS ablations across real-world scientific datasets (UCI Wine and Breast Cancer Diagnostic). Conduct paired Wilcoxon signed-rank tests with Holm-Bonferroni corrections and Cohen's $d$ effect sizes.

## Execution
```bash
python experiments/experiment_05_consensus_ablation/run.py
```

## Generated Artifacts
- `results/raw/real_world_benchmark_results.csv`
- `results/raw/statistical_tests_summary.csv`
- `results/figures/fig2_real_world_benchmarks.png`
- `reports/experimental_results/experiment_05_consensus_ablation_report.md`
