# Experiments

Five experiment suites evaluate cluster stability and ensemble (consensus) clustering on noisy high-dimensional data. They test the central question of the PCCS document (stability as a function of perturbation type, intensity and interaction) with PCCS (`src/icc/pccs.py`) and compare it with ICC and standard baselines. Mapping to the document: `docs/pccs_traceability.md`.
Every suite has the same layout: `data/` (inputs and result tables), `figures/`, `src/` (code), `README.md`, `run_all.py`.

| Suite | Question | Main outputs | Figures |
|---|---|---|---|
| [`exp0_falsification`](exp0_falsification) | Does high stability imply real cluster structure? | `exp0_agg.csv`, `exp0_finegrid.csv`, `exp0_noisetypes.csv`, `exp0_scaled_agg.csv`, `exp0_fcm_full_curve.csv`, `exp0_verdict.csv` | fig1-fig8; PCCS fingerprint detector: fig15 |
| [`exp1_generalization`](exp1_generalization) | Do consensus partitions transfer to held-out rows? | `exp1_generalization_results.csv`, `exp1_summary.csv` | fig9 |
| [`exp2_ablation`](exp2_ablation) | Which ICC components contribute? | `exp2_ablation_results.csv`, `exp2_fingerprints.csv` | fig10; PCCS ablation and validity: fig16-fig17 |
| [`exp3_real_world_benchmarks`](exp3_real_world_benchmarks) | Do the findings hold on real data, original and corrupted? | `exp3_real_world_results.csv`, `exp3_summary.csv` | fig11-fig12; PCCS: fig20 |
| [`exp4_scalability_and_sensitivity`](exp4_scalability_and_sensitivity) | How does ICC scale and how sensitive is it to its settings? | `exp4_scalability_results.csv`, `exp4_sensitivity_results.csv` | fig13-fig14; PCCS response and interactions: fig18-fig19 |

## Shared code (`common/`)
- `baselines.py`: K-means, Ward, diagonal GMM, spectral (kNN affinity), EAC, and ICC behind one `run_method(name, X_train, X_test, k, seed)` interface.
- `datasets.py`: PCCS-style synthetic generator (`make_pccs`, built on `pccs_datagen.py`), scalable Gaussian mixtures (`make_scalable`), and the real-data corruption protocol (`corrupt`).
- `utils.py`: ARI, NMI, variation of information, pairwise ARI with unobserved rows.
- `icc_adapter.py`: the **only** place where the experiments import ICC. It points to `src/icc/model.py`.

## Running
From the repository root:
```bash
python scripts/run_experiment.py exp2      # one suite (see --list)
python scripts/run_all.py                  # all suites, ~35 min on one CPU core, then regenerates docs/results_summary.md
```
Each suite can also be run on its own: `python experiments/exp2_ablation/run_all.py`.

## Method labels in result files
The consensus method appears as `icc_v1` in result tables and figures: it is the reference implementation in `src/icc/model.py`.
To benchmark a different ICC implementation (for example ICC-v2), edit `common/icc_adapter.py` so that it exports `ICC`,
`ICCConfig`, `preprocess_fit`, `preprocess_apply`, and `_agglo_precomputed` with the same interface, then rerun the suites.
Result tables will then describe that implementation.

## Conventions
Fixed seeds in every script; result tables are CSV; figures are PNG at 150 dpi; every suite README states data, design, results, and limits.
Synthetic data are produced on the fly or written to the suite's `data/` folder; real data are the scikit-learn bundled datasets.
