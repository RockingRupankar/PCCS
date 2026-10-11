# exp2 - Ablation of ICC

**Question:** which ICC component (stability weights, per-family fusion, invariance term, perturbation families, base methods) contributes to accuracy?

## Data (`data/`)
`exp2_{noisy,severe}_seed{42,43,44}.csv`: PCCS-style data, N=1200 (6 datasets). Transductive: all rows are clustered and scored against `true_cluster`.

## Variants
full; no stability weights; pooled (no family fusion); lambda=0 (mean over families only); lambda=1 (min over families only); remove row / feature / noise / combined views;
seed-only views; K-means only; Ward only; K-means+Ward+GMM. Baselines: K-means, EAC.

## Run
`python run_all.py` (~7 min). Outputs: `data/exp2_ablation_results.csv`, `data/exp2_delta_vs_full.csv`, `data/exp2_fingerprints.csv` (per-cluster stability fingerprints of the full model), `figures/fig10_icc_v1_performance.png`.

## What was found (ARI, mean +- sd over 3 data seeds)
- **Noisy:** full 0.870 +- 0.037; K-means 0.863; EAC 0.858. Every variant except seed-only/Ward-only lies within one sd of full - **no component shows a measurable effect**.
- **Severe:** full 0.211 +- 0.074; K-means 0.176. Removing noise views (0.099), row views (0.141) or feature views (0.158) lowers ARI; adding GMM as a third base method gives the best score (0.264). Variance is large and n=3, so these are indications, not significant effects.
- **Seed-only views = Ward-only (0.684 / 0.099):** agreement-based weights reward a deterministic base method, which agrees perfectly with itself. Stability weighting can therefore *amplify* deterministic members - a design weakness worth addressing in a next ICC version.
- 8 of 78 runs used the spectral fallback (6 of them in the lambda=1 variant).
- `exp2_fingerprints.csv` shows the reliability output: the "combined" (all perturbations at once) view is the least stable family for 22 of 24 clusters (mean 0.51, vs 0.58 for feature views and 0.68-0.69 for seed/row/noise views), i.e. combined perturbations expose fragility that single perturbations do not.

## PCCS block: consensus ablation and validity of the reliability profile (fig16, fig17)
`src/exp2_pccs_ablation.py`, `src/plot_pccs_figures.py`. One PCCS model per dataset (noisy and severe, N=500, 5 seeds); variants rebuild only the consensus from the same stored partitions.
- Noisy: full PCCS 0.859 +- 0.041; k-means 0.833; EAC 0.846; reference partition 0.761. Individual components change ARI by at most 0.014; algorithmic family alone: -0.063.
- Severe: full PCCS 0.270 +- 0.099, below k-means (0.318) and EAC (0.389); PCCS with k-means members only: 0.361. Variants differ within seed variation.
- Validity: PSI versus cluster error, Spearman 0.70 (noisy) and 0.85 (severe) over 20 clusters each; conditionally invariant clusters have mean error 0.09 versus 0.19 for ambiguous clusters on noisy data; no cluster is globally invariant; clusters within a dataset are not independent.
Data: `data/exp2_pccs_*_seed*.csv`; results: `exp2_pccs_ablation_results.csv`, `exp2_pccs_fingerprint_validity.csv`, `exp2_pccs_validity_stats.csv`, `exp2_pccs_category_error.csv`.
