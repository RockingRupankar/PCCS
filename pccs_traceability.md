# Traceability: PCCS problem-statement document to code and experiments

Source: "Perturbation-Conditional Consensus Clustering: Problem Statement and Approach" (PCCS document). Section numbers refer to that
document. Status: **Implemented**, **Partial** (deviation or reduced scope, stated), **Not implemented**.
Code: `src/icc/pccs.py` (`PCCS`, `PCCSConfig`); defaults in `configs/pccs_default.yaml`.

## Required properties (section 1.2)
| Requirement | Status | Where |
|---|---|---|
| Multiple explicit perturbation families | Implemented (6 families) | `pccs.py` `_perturb`; exp0, exp4 |
| Stability at partition and cluster level | Implemented | `ari_ref` per partition; `fingerprints()` per cluster |
| Globally invariant / conditionally invariant / fragile / ambiguous | Implemented, thresholds are design choices (hi 0.75, lo 0.50, response sd 0.15, interaction 0.15) | `_classify()`; exp2 validity |
| Measure perturbation interactions | Partial: all pairs at the middle intensity; higher-order combinations not implemented | `interactions()`; exp4 |
| Use reliability information during consensus | Partial: partition weights (agreement with the reference), family weights, and a cross-family consistency term enter the consensus; cluster-level fingerprints are reported but not fed back | `build_consensus()`; exp2 |
| Controlled strength, algorithms, seeds | Implemented: level, repetition, method and family are stored for every partition | `partitions_`, `run_table()` |

## Mathematical formulation (section 2)
| Item | Status | Note |
|---|---|---|
| `C(m,p,r) = Cluster_m(T_{p,r}(X); theta_m)` | Implemented | hyper-parameters theta_m are jittered under the algorithmic family; the jittered values are not stored |
| Co-association tensor `S[i,j,p]` | Partial | one N x N matrix per family, averaged over methods, intensity levels, repetitions; normalised by the weighted number of runs that observed the pair (the document's `1/(MR)` is only valid without row removal) |
| Fingerprint `F_k = [S_k1 ... S_kP]` | Implemented | `S_kp` = mean of membership retention (best-match Jaccard) and within-cluster co-association |
| `PSI_k = sum_p w_p (1 - S_kp)` | Implemented with equal weights `w_p = 1/P` | |

## Methodology
| Section | Status | Note |
|---|---|---|
| 4.1 Preparation | Partial | numeric data only; median imputation, robust winsorisation, standardisation; no categorical encoding; no dimensionality reduction; optional batch vector for technical variables |
| 4.2 Base ensemble | Partial | k-means, fuzzy c-means (hard labels from soft memberships), DBSCAN (noise points become singletons), hierarchical (Ward / average / complete); several seeds and jittered settings |
| 4.3 Perturbation engine | Partial | algorithmic, sampling (subsampling without replacement), feature masking, measurement (Gaussian, heavy-tailed, outliers, missingness), technical (simulated batch offsets and scales), distribution (covariate shift by importance resampling). Not implemented: bootstrap with replacement, rare-cluster stress tests, representation changes, real acquisition metadata in the experiments |
| 4.4 Repeated clustering | Implemented | partitions compared by ARI and Jaccard, which are invariant to label permutation, so no alignment is needed |
| 4.5 Stability tensor | Implemented | mean response and across-run sd of retention are retained |
| 4.6 Fingerprints | Implemented | retention, co-association consistency, centroid drift (reported separately) |
| 4.7 Interaction analysis | Partial | expected combined stability `S_p S_q / S_0` (S_0 = unperturbed reproducibility); interaction = expected - observed; pairs only |
| 4.8 Reliability classification | Implemented | members whose unperturbed ARI to the reference is below 0.2 (DBSCAN on the synthetic data) are excluded from fingerprints and interactions because they carry no perturbation information |
| 4.9 Perturbation-aware consensus | Partial | see "use reliability during consensus" above |
| 4.10 Final output | Implemented | labels, fingerprints, PSI, interactions, categories, family and method stability summaries, consensus quality, computational statistics |
| Algorithm step 16 | Partial | repeated experiments with 3 to 5 seeds (exp2, exp3, exp4) |

## Claims in section 6 and where they are tested
| Document claim | Experiment | Outcome in this repository |
|---|---|---|
| One scalar hides perturbation type | exp0 (fig15) | Supported with a caveat: each scalar or single-family score ranks an elongated null above structured data from noise sd 1.0, while a supervised logistic model on the six-family fingerprint still separates them (AUC 0.86 and 0.97). The model needs labelled simulated examples, so this is evidence that the fingerprint carries information, not an unsupervised test |
| High global stability can hide fragile clusters | exp2 (fig17) | Supported: PSI correlates with cluster error (Spearman 0.70 noisy, 0.85 severe) |
| Independent tests can hide combined failure | exp4 (fig19) | Not observed: no pair reached the 0.15 threshold when averaged over 3 seeds |
| Stability is used during consensus, not only as evaluation | exp2 (fig16) | Effect on accuracy is small on noisy data and negative to mixed on severe data |
| Algorithm agreement is not universal robustness | exp4 (fig18) | Supported: base methods differ strongly by family (k-means falls under feature masking and batch effects; hierarchical clustering resists sampling and distribution shift) |
| Conditional structure is not stable/unstable | exp2 | Categories are produced; no cluster in 40 was globally invariant |
| Perturbations are reproducible | all | seeds, levels and repetitions are recorded in `run_table()` and the result CSVs |
