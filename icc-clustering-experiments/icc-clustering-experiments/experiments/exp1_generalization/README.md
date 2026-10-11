# exp1 - Inductive generalisation (fit on few rows, label unseen rows)

**Question:** does perturbation-aware consensus (ICC) give partitions that transfer better to held-out rows than single clusterers when training data is small and noisy?

## Data (`data/`)
`exp1_{clean,noisy,severe}.csv`: PCCS-style synthetic data (see `common/pccs_datagen.py`), N=2000, 100 features, 4 clusters, column `split` = `test` (fixed stratified 800 rows) or `pool`.
Training sets of 100/200/400/800/1200 rows are drawn from the pool (3 seeds).

## Methods
K-means (`predict`), Ward + nearest centroid, diagonal GMM (`predict`), spectral (kNN affinity) + kNN classifier, EAC (evidence accumulation, uniform) + nearest centroid, ICC (feature-weighted nearest centroid). Preprocessing is fit on training rows only.

## Run
`python run_all.py` (~5 min). Outputs: `data/exp1_generalization_results.csv` (every run), `data/exp1_summary.csv`, `figures/fig9_inductive_generalization.png`.

## What was found (mean over 3 seeds; ARI on the 800 held-out rows)
| train rows | clean: ICC / K-means / GMM | noisy: ICC / K-means / GMM | severe: ICC / K-means / GMM |
|---|---|---|---|
| 100 | 0.89 / 0.81 / 0.80 | 0.74 / 0.63 / 0.50 | 0.23 / 0.21 / 0.14 |
| 1200 | 0.99 / 0.97 / 0.98 | 0.87 / 0.87 / 0.88 | 0.47 / 0.44 / 0.46 |
- ICC helps most with **small training sets**; from ~400 rows the differences to K-means/GMM/EAC are small and, with 3 seeds and no significance test, should not be read as wins.
- Spectral + kNN is clearly worst here (the kNN graph is dominated by the 70 noise features).
- The train-minus-test gap is near zero or negative for most methods (Ward's negative gap comes from nearest-centroid re-assignment correcting its dendrogram).
- Cost: ICC ~1.5 s per fit vs ~0.04 s for K-means.
