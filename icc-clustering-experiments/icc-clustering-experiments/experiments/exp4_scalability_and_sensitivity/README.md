# exp4 - Scalability and sensitivity of ICC

## Data (`data/`)
- Scalability: Gaussian mixtures generated on the fly (`common/datasets.make_scalable`: 4 clusters, 10 informative dims, rest noise); example `exp4_scalable_example_N1000_d100.csv`.
- Sensitivity: `exp4_sensitivity_noisy_seed{42,43,44}.csv` (PCCS-style noisy data, N=800).

## Run
`python run_all.py` (~12 min, 1 CPU; timings are wall-clock on a single core, so run it without other jobs). Outputs: `exp4_scalability_results.csv`, `exp4_scaling_fits.csv` (log-log slopes), `exp4_sensitivity_results.csv`, `exp4_sensitivity_ranges.csv`, `figures/fig13_scalability.png`, `figures/fig14_sensitivity.png`.

## What was found
**Scalability** (ICC with 12 views x 2 base methods; 2 repetitions):
| N (d=100) | K-means | Ward | EAC | ICC |
|---|---|---|---|---|
| 250 | 0.02 s | 0.01 s | 0.06 s | 0.32 s |
| 1000 | 0.09 s | 0.07 s | 0.25 s | 1.26 s |
| 2000 | 0.08 s | 0.21 s | 0.52 s | 4.26 s |
- ICC time grows with log-log slope ~1.2 in N over this range (co-association work is O(n^2) per partition, so the slope should steepen for larger N) and is nearly flat in d (slope 0.18). It is 15-50x slower than K-means.
- Memory: ICC holds about (families + 3) float32 n x n matrices (~128 MB at N=2000); N beyond roughly 5,000-10,000 needs a sparse/landmark co-association. Memory here is analytic, not measured.
- Accuracy on the Gaussian mixtures is equal for K-means, EAC and ICC (0.78-0.79 at N=2000) - this benchmark is easy and does not test the noisy regime.

**Sensitivity** (one-at-a-time around the defaults; ARI of the default is 0.771 +- 0.141 over 3 data seeds):
- Flat (range <= 0.012): feature fraction, invariance lambda, noise sd of the noise views, weight power.
- Small and non-monotone (range 0.05-0.08): number of views (4 views 0.744, 8 views 0.819, 32 views 0.763) and row fraction (0.757-0.809). With 3 seeds this is indistinguishable from noise; more than ~8 views brought no gain but costs time linearly.
- **Large: the assumed number of clusters** (range 0.34). ICC tracks K-means at every K (K=3: 0.68 vs 0.68, K=5: 0.69 vs 0.66, K=8: 0.43 vs 0.45): consensus does not protect against K misspecification.

## PCCS block: stability versus perturbation type, intensity and interaction (fig18, fig19)
`src/exp4_pccs_response.py`, `src/plot_pccs_figures.py`. Clean and noisy data (N=500, 3 seeds); five intensity levels per family, 4 repetitions, 4 base methods; all 15 pairs combined at the middle level.
- k-means agreement with the unperturbed reference (noisy data) is highest under algorithmic perturbations (0.75), then sampling and distribution shift (0.66), and lowest under feature masking (0.44) and batch effects (0.45); hierarchical clustering is the most robust to sampling and distribution shift (0.66, 0.65); fuzzy c-means is the least stable (0.27 to 0.48); DBSCAN does not recover structure here.
- No pair reaches the interaction threshold 0.15 when averaged over seeds (maximum 0.067). Two pairs that looked synergistic in one exploratory fit did not replicate.
Results: `exp4_pccs_response.csv`, `exp4_pccs_interactions.csv`, `exp4_pccs_interaction_summary.csv`, `exp4_pccs_family_stability.csv`.
