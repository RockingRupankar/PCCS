# Methods

This repository contains two consensus methods: **PCCS**, which implements the PCCS problem-statement document, and **ICC**, a lighter inductive consensus method.

## PCCS: Perturbation-Conditional Consensus Clustering (`src/icc/pccs.py`)

PCCS treats cluster stability as a response to controlled perturbations instead of a single score. Section-by-section status against the document: [pccs_traceability.md](pccs_traceability.md).

1. **Reference.** A baseline ensemble (k-means, fuzzy c-means, DBSCAN, hierarchical) on the unperturbed, preprocessed data is combined by co-association into a reference partition.
2. **Perturbation engine.** Six families, each with intensity levels and repetitions: algorithmic (seeds and hyper-parameter jitter), sampling (rows removed), feature (columns masked), measurement (Gaussian, heavy-tailed, outliers, missingness), technical (batch offsets and scales) and distribution (covariate shift). Pairs of families are also combined at the middle intensity.
3. **Repeated clustering.** Every base method is run on every perturbed dataset; each partition stores method, family, level, repetition, rows and features.
4. **Stability tensor.** One co-association matrix per family, weighted by each partition's agreement with the reference.
5. **Fingerprints.** For every final cluster and family: membership retention (best-match Jaccard), within-cluster co-association, centroid drift and across-run variability; `S_kp` is the mean of retention and co-association, `PSI = mean_p(1 - S_kp)`.
6. **Interactions.** Expected combined stability `S_p S_q / S_0` versus observed; positive differences indicate extra degradation.
7. **Categories.** Globally invariant, conditionally invariant, perturbation-fragile, ambiguous (thresholds in `PCCSConfig`).
8. **Consensus.** `C = (1 - lambda) * sum_p alpha_p S_p + lambda * min_p S_p`, with family weights `alpha_p` from family reliability; average-linkage clustering of `1 - C` (spectral fallback).
9. **Output.** Labels, fingerprints, PSI, interactions, categories, family and method stability, consensus quality, computational statistics.

```python
from icc import PCCS, PCCSConfig
model = PCCS(PCCSConfig(n_clusters=4, random_state=0)).fit(X)     # X may contain NaN; optional batch=vector
model.labels_; model.fingerprint_; model.interaction_; model.summary_; model.computational_stats()
model.build_consensus(per_family=False)                           # rebuild consensus from stored partitions (ablations)
```

# ICC: Inductive Consensus Clustering (reference implementation)

Notation: X is an n x d matrix, K the number of clusters, B the number of views.

## Pipeline
1. **Preprocessing.** Median imputation, robust winsorisation at +-4 MAD-z, standardisation. The fitted parameters are stored for out-of-sample rows.
2. **Perturbed views.** View v belongs to family f(v), cycling through `seed`, `row` (80% of rows), `feature` (50% of columns), `noise` (additive Gaussian, sd 0.25 in standardised units), and `combined` (all three).
3. **Base clusterings.** K-means and Ward (optionally a diagonal GMM) are run on every view.
4. **Stability weights.** Each partition receives the weight `w_p = max(mean_q ARI(p, q), 1e-3)^gamma` where ARI is computed on shared rows and gamma = `weight_power` (default 2).
5. **Per-family co-association.** `C_f(i,j)` is the weighted fraction of partitions in family f that co-cluster i and j among the partitions that observe both.
6. **Fusion.** `C = (1 - lambda) * sum_f alpha_f C_f + lambda * min_f C_f` with alpha_f proportional to the mean weight in family f; lambda = `invariance_lambda` (default 0.5). Pairs that stay together in every family are favoured.
7. **Consensus.** Average-linkage clustering of `1 - C` into K clusters; if a cluster has fewer than 2% of rows, spectral clustering of `C` is used instead.
8. **Fingerprint.** For each final cluster, the mean within-cluster `C_f` per family, the perturbation sensitivity index `PSI = 1 - mean_f C_f`, and a category: globally invariant (all families >= 0.75), conditionally invariant (some >= 0.75 and all >= 0.50), perturbation-fragile, or ambiguous. The thresholds are design choices, not estimated quantities.
9. **Inductive assignment.** New rows are preprocessed with the stored parameters and assigned to the nearest consensus centroid, with features weighted by the square root of their between/within variance ratio.

## Hyper-parameters
See `src/icc/model.py` (`ICCConfig`) and `configs/icc_default.yaml`; a test keeps the two in sync.

## Known properties (from the experiments)
- Cost grows roughly linearly to quadratically in N and is nearly flat in d; memory holds about (families + 3) float32 n x n matrices.
- Agreement-based weighting favours deterministic base methods (seed-only views reproduce Ward alone).
- The method does not correct a wrong K.

## Relation to other implementations
This module is a self-contained reference implementation used to run the experiments. It is not a statement about any other ICC version; to evaluate
another implementation, change only `experiments/common/icc_adapter.py`.
