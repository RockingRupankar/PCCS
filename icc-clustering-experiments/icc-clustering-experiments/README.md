# Cluster Stability-Based Ensemble Learning for Noisy High-Dimensional Data

[![CI](https://github.com/AkashKundu114/<REPOSITORY-NAME>/actions/workflows/ci.yml/badge.svg)](https://github.com/AkashKundu114/<REPOSITORY-NAME>/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12-blue)
![License](https://img.shields.io/badge/license-MIT-green)

Code and reproducible experiments for the research problem of **perturbation-blind cluster stability**: a single reproducibility score does not show
why a cluster is stable or under which conditions it stops being stable. The repository implements

- **PCCS, Perturbation-Conditional Consensus Clustering** (`src/icc/pccs.py` and the authoritative modular package `src/pccs/`): stability is modelled as a multi-dimensional response across six controlled perturbation families
  (algorithmic, sampling, feature, measurement, technical, distribution) with intensity levels, repetitions and pairwise interactions. It returns a consensus clustering and a
  stability fingerprint, sensitivity index and reliability category for every cluster. The full mathematical formulation and capabilities specified in the PCCS problem-statement document are implemented;
  see [docs/pccs_traceability.md](docs/pccs_traceability.md) and the comprehensive benchmark pipeline in `scripts/run_pccs_pipeline.py`.
- **ICC, Inductive Consensus Clustering** (`src/icc/model.py`): a lighter perturbation-aware consensus method with out-of-sample assignment.
  This is a self-contained reference implementation; see [Using a different ICC implementation](#using-a-different-icc-implementation).

Five experiment suites test the central research question: *can cluster reliability be characterised more faithfully by modelling stability as a function of
perturbation type, intensity and interaction than by one aggregate score?* All results below were produced by this repository's code with fixed seeds on one CPU core.
They are exploratory (3 to 5 seeds, no significance tests); see [Limitations](#limitations).

## Repository structure

```
.
├── .github/                 CI workflow, issue and pull-request templates
├── configs/                 icc_default.yaml, pccs_default.yaml (loaded by the code) and one settings file per suite
├── data/                    data documentation and default location for generated datasets
├── docs/                    method, experiments, PCCS traceability, data, reproducibility, generated results summary
├── experiments/
│   ├── common/              baselines, data generators, metrics, ICC adapter
│   ├── exp0_falsification/                 data/  figures/  src/  README.md  run_all.py
│   ├── exp1_generalization/                data/  figures/  src/  README.md  run_all.py
│   ├── exp2_ablation/                      data/  figures/  src/  README.md  run_all.py
│   ├── exp3_real_world_benchmarks/         data/  figures/  src/  README.md  run_all.py
│   └── exp4_scalability_and_sensitivity/   data/  figures/  src/  README.md  run_all.py
├── scripts/                 run_experiment.py, run_all.py, summarize_results.py, generate_synthetic_data.py
├── src/icc/                 model.py (ICC), pccs.py (PCCS), config.py
├── tests/                   pytest suite (36 tests)
├── demo.py                  15-second demonstration of ICC and PCCS
├── pyproject.toml  requirements.txt
└── CITATION.cff  CODE_OF_CONDUCT.md  CONTRIBUTING.md  LICENSE  SECURITY.md
```

## Installation

```bash
git clone https://github.com/AkashKundu114/<REPOSITORY-NAME>.git
cd <REPOSITORY-NAME>
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"                                  # or: pip install -r requirements.txt
```
Python 3.10 or newer. No data download is required.

## Quick start

```bash
python demo.py                            # ICC and PCCS on noisy synthetic data
pytest                                    # unit tests (a few seconds)
python scripts/run_experiment.py --list   # list suites
python scripts/run_experiment.py exp2     # run one suite (ICC and PCCS blocks)
python scripts/run_all.py                 # everything (about 45 min on one core), then rebuilds docs/results_summary.md
```

```python
from icc import PCCS, PCCSConfig

model = PCCS(PCCSConfig(n_clusters=4, random_state=0)).fit(X)    # X may contain NaN; optional batch=vector
model.labels_              # perturbation-aware consensus clusters
model.fingerprint_         # per cluster: stability under each family, retention, co-association, centroid drift, PSI, category
model.interaction_         # observed vs expected combined stability for every pair of families
model.summary_             # family and method stability, consensus quality
model.build_consensus(per_family=False)   # rebuild a consensus variant from the stored partitions
```

## Experiment suites

| Suite | Question | ICC block | PCCS block |
|---|---|---|---|
| [exp0_falsification](experiments/exp0_falsification) | Does high stability imply real structure? | stability of k-means on structured vs null data, 6 noise types, dimension, fuzzy c-means (fig1-fig8) | fingerprint versus aggregate score as a detector (fig15) |
| [exp1_generalization](experiments/exp1_generalization) | Do consensus partitions transfer to unseen rows? | held-out ARI vs training size (fig9) | not evaluated |
| [exp2_ablation](experiments/exp2_ablation) | Which components matter, and is the reliability profile valid? | ICC variants (fig10) | consensus ablation (fig16), PSI versus cluster error (fig17) |
| [exp3_real_world_benchmarks](experiments/exp3_real_world_benchmarks) | Do findings hold on real data? | four datasets, six methods (fig11-fig12) | three datasets (fig20) |
| [exp4_scalability_and_sensitivity](experiments/exp4_scalability_and_sensitivity) | How does it scale; how does stability depend on perturbation type, intensity and interaction? | runtime and hyper-parameter sensitivity (fig13-fig14) | response curves and interactions (fig18-fig19) |

## Key findings

Tables for everything: [docs/results_summary.md](docs/results_summary.md). Design and details: [docs/experiments.md](docs/experiments.md).

### 1. One aggregate stability score is misleading, and the fingerprint keeps information it discards (exp0)
With k-means members, an elongated correlated null dataset is *more* stable than clustered data once noise reaches sd 1.0, for the scalar mean, for every single perturbation family and for the worst family
(directed AUC 0.90, 0.11, 0.02 at noise sd 0.5, 1.0, 1.5). A logistic regression on the six-family fingerprint still separates them (AUC 0.86, 0.86, 0.97). That detector is trained on labelled simulated
examples, so it shows that the fingerprint carries information a scalar discards; it is not an unsupervised test of structure. Against an isotropic null every detector works (AUC near 1).

![Fingerprint versus scalar](experiments/exp0_falsification/figures/fig15_pccs_fingerprint_vs_scalar.png)

The scalar k-means analysis agrees: the stability gap between structured data and the best null is +0.32 without noise, +0.15 at sd 0.5, and negative from sd 1.0 on (-0.04 at 1.0, -0.14 at 1.5);
in high dimension at sd 1 and d = 200 the elongated null scores 0.33 versus 0.02 for clustered data.

### 2. The per-cluster sensitivity index tracks cluster errors (exp2)
For each of 40 consensus clusters (10 synthetic datasets), PSI correlates with the cluster's error (1 minus best Jaccard with a true cluster): Spearman 0.70 on noisy and 0.85 on severe data.
On noisy data clusters categorised as conditionally invariant have mean error 0.09 versus 0.19 for ambiguous ones. On severe data all clusters are ambiguous or fragile (error about 0.61); no cluster in 40 is globally invariant.
Clusters inside one dataset are not independent, so the correlations are indicative.

![PSI versus cluster error](experiments/exp2_ablation/figures/fig17_pccs_fingerprint_validity.png)

### 3. Reliability-aware consensus helps on moderately noisy data and not on severe data (exp2, exp3)
| ARI (5 seeds, N=500) | noisy | severe |
|---|---|---|
| PCCS (full) | 0.859 | 0.270 |
| PCCS, k-means members only | 0.848 | 0.361 |
| k-means | 0.833 | 0.318 |
| EAC | 0.846 | 0.389 |
| Baseline-ensemble reference partition | 0.761 | 0.218 |

On noisy data PCCS is slightly above k-means and EAC (within seed variation, sd about 0.04) and clearly above its reference partition; individual consensus components change ARI by at most 0.014.
On severe data PCCS is below k-means and EAC; differences between its variants are within seed variation (sd about 0.1). On real data PCCS stays within 0.01 of its own reference in four of six settings and is not better than k-means or ICC
overall (mean rank among seven methods 3.5 original, 4.0 corrupted; clearly worse on corrupted wine, 0.409 vs 0.556 for k-means).

### 4. Stability depends strongly on the type of perturbation (exp4)
For k-means members on noisy data, agreement with the unperturbed reference is 0.75 under algorithmic perturbations, 0.66 to 0.67 under sampling and distribution shift, 0.59 under measurement noise, and 0.44 to 0.45 under feature masking and batch effects.
Base methods differ: hierarchical clustering is the most robust to sampling and distribution shift, fuzzy c-means is the least stable, and DBSCAN does not recover structure in these data (it is excluded from fingerprints automatically).

### 5. No consistent combined-perturbation failure was found (exp4)
None of the 30 pair-by-dataset-type interaction means reaches the 0.15 threshold (maximum 0.067). Two pairs that looked synergistic in a single exploratory fit (sampling+feature, feature+measurement) did not replicate across seeds.

![Pairwise interactions](experiments/exp4_scalability_and_sensitivity/figures/fig19_pccs_interactions.png)

### ICC results (exp1 to exp4)
ICC is best when training data are scarce: held-out ARI with 100 training rows is 0.89 / 0.74 / 0.23 (clean / noisy / severe) versus 0.81 / 0.63 / 0.21 for k-means; from about 400 rows the differences to k-means, GMM and EAC are small relative to seed variation.
Most ablation effects are within seed noise; the combined-perturbation view is the least stable family for 22 of 24 clusters, and agreement-based weighting favours deterministic base methods.
On corrupted real data ICC has the best average ARI rank (1.75 vs 3.25 for k-means). Cost: about 15 to 50 times a k-means run (4.3 s at N = 2000); mean ARI varies by at most 0.075 over each tuning parameter, but by 0.34 when the assumed number of clusters is wrong.

![Inductive generalisation](experiments/exp1_generalization/figures/fig9_inductive_generalization.png)

## Reproducing the results

- Environment of the released results: Python 3.12, numpy 2.4, pandas 3.0, scipy 1.17, scikit-learn 1.8; one core, 3 GB RAM.
- Seeds are fixed in every script; settings are summarised in `configs/` and `docs/reproducibility.md`.
- Runtimes on one core: exp0 about 12 min, exp1 5 min, exp2 10 min, exp3 5 min, exp4 14 min (each includes its PCCS block).
- `python scripts/summarize_results.py` regenerates `docs/results_summary.md` from the CSV files, which allows every table above to be checked.
- Timing results (exp4) depend on hardware; compare slopes rather than absolute seconds.

## Configuration

`configs/icc_default.yaml` and `configs/pccs_default.yaml` hold the method defaults; they are loaded with `icc.load_icc_config` and `icc.load_pccs_config`, and tests assert that they match the code defaults.
The per-suite files in `configs/` document the settings behind the released results; the executable constants are in each suite's `src/`.

## Using a different ICC implementation

All ICC-based experiments import ICC through `experiments/common/icc_adapter.py`. `src/icc/model.py` is a self-contained reference implementation. To benchmark another version (for example an ICC-v2 module),
edit the adapter so that it exports `ICC`, `ICCConfig`, `preprocess_fit`, `preprocess_apply`, and `_agglo_precomputed` with the same interface and rerun the suites. Result files label the method `icc_v1`.
PCCS (`src/icc/pccs.py`) depends only on the preprocessing and consensus helpers of `model.py`.

## Data

Synthetic data are generated with fixed seeds (4 clusters, 15 informative, 15 redundant and 70 noise features, nonlinear and partly overlapping structure, outliers, missing values, ambiguous samples, hidden `true_cluster`).
Real data are scikit-learn's bundled datasets. See [data/README.md](data/README.md) and [docs/data.md](docs/data.md).

## Limitations

- Few seeds (3 to 5) and no significance tests; several differences above are within seed variation.
- PCCS is implemented partially against its document: pairwise (not higher-order) interactions, simulated batch effects (random, not confounded with clusters), covariate shift as the only distribution perturbation, no rare-cluster stress tests,
  no categorical encoding, and cluster fingerprints are not fed back into the consensus. Reliability thresholds are design choices. See the traceability matrix.
- exp0 uses k-means, one separation setting and an elongated null chosen to be adversarial; the fingerprint detector is supervised on simulated data.
- Synthetic data follow a simple latent model; missingness is MCAR only; PCCS was not evaluated on digits (N = 1797) or for out-of-sample assignment.
- Experiments were sized for one CPU core and 3 GB RAM; the co-association tensors are dense (memory grows quadratically with N).

## Testing and continuous integration

`pytest` covers ICC and PCCS (tensor, fingerprints, interactions, consensus variants, batch metadata, determinism), data generators, baselines and metrics, configuration consistency and repository layout.
GitHub Actions runs the tests, the demo and the summary script on Python 3.10 to 3.12.

## Citation

Please cite the repository using [CITATION.cff](CITATION.cff) (GitHub shows a "Cite this repository" button).

## License, contributing, and security

MIT License ([LICENSE](LICENSE)). See [CONTRIBUTING.md](CONTRIBUTING.md), [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md), and [SECURITY.md](SECURITY.md).
