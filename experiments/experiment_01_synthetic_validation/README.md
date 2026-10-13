# Experiment 1: Controlled Synthetic Ground-Truth Validation & Falsification

## Objective
1. **Manifold Recovery:** Validate that PCCS recovers ground-truth geometries across 6 synthetic regimes (spherical separated, anisotropic elongated, concentric rings, overlapping, rare cluster, high-dimensional noise features).
2. **Unsupervised Falsification:** Demonstrate that the Perturbation Sensitivity Index ($PSI$) discriminates real structure from isotropic and anisotropic Gaussian null noise without requiring supervised labels.

## Execution
```bash
python experiments/experiment_01_synthetic_validation/run.py
```

## Generated Artifacts
- `results/raw/synthetic_benchmark_results.csv`
- `results/raw/falsification_results.csv`
- `results/raw/synthetic_fingerprints.csv`
- `results/figures/fig1_synthetic_noisy_performance.png`
- `results/figures/fig5_falsification_psi.png`
- `reports/experimental_results/experiment_01_synthetic_validation_report.md`
