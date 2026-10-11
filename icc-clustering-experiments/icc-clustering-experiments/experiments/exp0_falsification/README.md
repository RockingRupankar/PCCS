# exp0 - Falsification: does high cluster stability imply real structure?

**Hypothesis H1 (to be falsified):** K-means stability (mean pairwise ARI) separates structured data from structure-free data,
whatever the noise type, noise level or dimension. **Criterion:** H1 is supported in a condition if
`stability(structured) - max(stability(null_iso), stability(null_aniso)) >= 0.10`.

## Data (`data/`)
Generated on the fly by `src/exp0_core.py` (n=400, d=20 unless noted, 5 informative dims). Three structures, all column-standardised:
`structured` (4 Gaussian clusters, 15 pure-noise dims), `null_iso` (one isotropic Gaussian), `null_aniso` (one elongated *and rotated* Gaussian, so the
elongation survives standardisation). One example file per structure: `exp0_example_*_sigma1.csv`.
Results: `exp0_agg.csv` (+`exp0_raw.csv`), `exp0_finegrid.csv`, `exp0_noisetypes.csv`, `exp0_scaled_agg.csv` (n=600, d=20..400),
`exp0_fcm_full_curve.csv`, `exp0_diagnose.csv`, `exp0_verdict.csv` (one row per condition: gap and H1 supported / not).

## Run
`python run_all.py` (~8 min, 1 CPU). Figures 1-8 in `figures/` (fig9/fig10 of your layout belong to exp1/exp2 and are saved there).

## What was found (this run; 20 seeds main sweep, 8-10 seeds elsewhere)
- **H1 is falsified.** Stability gap vs the best null (bootstrap stability): +0.32 at sigma=0, +0.15 at 0.5, **-0.04 at 1.0, -0.14 at 1.5** (fine grid: gap drops below 0.10 near sigma~0.6).
  At sigma=1 the structured data score 0.32 and the elongated null 0.37; the isotropic null stays ~0.12 (so stability does flag *isotropic* nulls, not correlated ones).
- **Noise type matters** (`fig4`): gap survives outliers and mean-imputed missingness (+0.14 to +0.33 at all tested levels) but disappears for Gaussian, Student-t(3) and uniform noise at intensity >=1.
- **Dimension matters** (`fig5-7`): at sigma=0.5 the gap goes +0.18 (d=20) -> +0.03 (d=50) -> -0.24 (d=100) -> -0.36 (d=200). At sigma=1, d=200: structured 0.02 vs elongated null 0.33.
- **Seed vs bootstrap** (`fig3`, diagnose): seed-only stability exceeds bootstrap stability by only ~0.03-0.04 here, i.e. both views flag the same data as "stable".
- **FCM** (`fig8`): all fuzzy-c-means centres collapse to the global mean once the fuzzifier m exceeds a threshold that falls with dimension (about 2-2.5 at d=10, ~1.25 at d=50, ~1.1 at d=200).

## Limits
One algorithm (k-means, K=4), one cluster-separation setting (`sep=11`, chosen so that the structured data is clearly stable at sigma=0), 80% subsampling, B=8 runs per stability value.
The elongated null is a deliberately adversarial but common null; other nulls may behave differently. A first version with an axis-aligned null was discarded because standardisation made it identical to the isotropic null.

## PCCS block: fingerprint versus aggregate stability (fig15)
`src/exp0_pccs_fingerprint.py`, `src/plot_pccs_figures.py`. The PCCS perturbation engine (six families, intensity levels, repetitions, k-means members) gives one stability value per family for each simulated dataset
(structured, isotropic null, elongated null; noise sd 0.5, 1.0, 1.5; 20 datasets per cell). Detectors of structure are compared by directed AUC (`data/exp0_pccs_auc.csv`).
- Against the isotropic null every detector has AUC near 1.
- Against the elongated null the scalar mean stability has AUC 0.90, 0.11, 0.02 at noise sd 0.5, 1.0, 1.5; single families and the minimum family behave the same way.
- A logistic regression on the six-family fingerprint (5-fold CV, repeated) reaches 0.86, 0.86, 0.97. It uses labelled simulated examples, so this shows that the fingerprint keeps information that the scalar discards; it is not an unsupervised test.
