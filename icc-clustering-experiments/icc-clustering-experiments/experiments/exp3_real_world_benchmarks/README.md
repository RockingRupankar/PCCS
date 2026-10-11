# exp3 - Real-world benchmarks

**Question:** do the findings transfer to real data, and to real data corrupted with the noise protocol?

## Data (`data/`)
Datasets bundled with scikit-learn (no download): iris (150x4, K=3), wine (178x13, K=3), breast_cancer (569x30, K=2), digits (1797x64, K=10): `exp3_<name>.csv` (column `target`).
`exp3_<name>_corrupted.csv`: features standardised, + measurement noise (sd 0.3), + 100 pure-noise features, 2% outlier rows, MCAR missing cells in 3% of rows (`common/datasets.corrupt`, seed 7).
K is set to the true number of classes. Class labels are used only for scoring.

## Run
`python run_all.py` (~3 min). Outputs: `data/exp3_real_world_results.csv`, `data/exp3_summary.csv` (mean ARI/NMI/VI, seed stability, average rank), `figures/fig11_real_world_benchmarks.png`, `figures/fig12_real_world_stability.png`.

## What was found (mean over 5 seeds)
- **Mean ARI rank (1 = best, 4 datasets):** corrupted data - ICC 1.75, EAC 2.75, GMM 3.0, K-means 3.25, spectral 4.5, Ward 5.75. Original data - GMM/K-means/ICC/EAC all between 2.75 and 3.75, spectral 3.25, Ward 4.75 (ICC has rank 1 on original wine only; average rank 2.75 is the best of the six).
- ICC has the best ARI in 3 of 8 settings (wine original and corrupted; breast_cancer corrupted); GMM is best on both iris variants, spectral on original digits and breast_cancer, EAC on corrupted digits. On corrupted wine ICC reaches 0.66 vs 0.56 for K-means.
- Among the stochastic methods ICC is the most reproducible across seeds on corrupted iris (0.79 vs 0.18 K-means, 0.42 GMM) and corrupted wine (0.78 vs 0.52).
- Ward and spectral show stability 1.0 only because they are deterministic given the data.

## Limits
Four small datasets, one corruption draw, 5 seeds (seeds change initialisation only). No significance testing. Ranks over 4 datasets are noisy.

## PCCS block (fig20)
`src/exp3_pccs.py`, `src/plot_pccs_figures.py`: PCCS on iris, wine, breast cancer (original and corrupted), 5 seeds, true K. Digits is excluded (N=1797 exceeds this study's compute budget).
ARI original / corrupted: iris 0.618 / 0.263, wine 0.894 / 0.409, breast cancer 0.685 / 0.660. PCCS is within 0.01 of its own reference partition in four of six settings and does not beat k-means or ICC overall
(mean rank among seven methods: 3.5 original, 4.0 corrupted; `data/exp3_pccs_vs_others.csv`). On corrupted wine it is clearly below k-means (0.409 vs 0.556).
