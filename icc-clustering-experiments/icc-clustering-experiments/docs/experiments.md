# Experiments

The experiments study the central question of the PCCS document: can cluster reliability be characterised more faithfully by modelling stability
as a function of perturbation type, intensity and interaction than by one aggregate score? Each suite adds a block for PCCS and keeps the
earlier ICC block. All tables: [results_summary.md](results_summary.md). Document-to-code mapping: [pccs_traceability.md](pccs_traceability.md).

| Suite | ICC block | PCCS block |
|---|---|---|
| exp0_falsification | scalar stability versus structured and null data (fig1-fig8) | fingerprint versus scalar as a detector of structure (fig15) |
| exp1_generalization | held-out assignment versus training size (fig9) | not evaluated (PCCS has no out-of-sample rule) |
| exp2_ablation | ICC components (fig10) | PCCS consensus ablation (fig16) and validity of the reliability profile (fig17) |
| exp3_real_world_benchmarks | six methods on four real datasets (fig11-fig12) | PCCS on three datasets (fig20); digits excluded for compute reasons |
| exp4_scalability_and_sensitivity | ICC runtime and sensitivity (fig13-fig14) | stability versus perturbation type and intensity (fig18), pairwise interactions (fig19) |

## PCCS findings (mean over the stated seeds; exploratory, no significance tests)

**exp0, fingerprint versus scalar.** Data: structured, isotropic null, elongated null; noise sd 0.5, 1.0, 1.5; 20 datasets per cell; k-means members.
Against the isotropic null every detector has AUC near 1. Against the elongated null the scalar mean stability has AUC 0.90 at noise sd 0.5,
0.11 at 1.0 and 0.02 at 1.5, and no single family or the minimum family does better. A logistic regression on the six-family fingerprint reaches
0.86, 0.86 and 0.97. The relative drop under feature masking, batch and distribution shift is slightly larger for structured data; the detector exploits
these small differences and requires labelled simulated examples, so it shows that the fingerprint carries information that a scalar discards but is not an unsupervised test.

**exp2, consensus ablation (5 data seeds, N=500).** Noisy data: full PCCS 0.859 (sd 0.041) versus k-means 0.833, EAC 0.846 and the baseline-ensemble
reference partition 0.761; removing individual families, the consistency term or reliability weights changes ARI by at most 0.014; using the algorithmic family alone
costs 0.063. Severe data: full PCCS 0.270 (sd 0.099) is below k-means 0.318 and EAC 0.389; PCCS restricted to k-means members reaches 0.361. Differences between variants
on severe data are within seed variation.

**exp2, validity of the reliability profile (40 consensus clusters).** PSI correlates with cluster error (1 minus best Jaccard with a true cluster): Spearman 0.70 on noisy data,
0.85 on severe data (the pooled 0.93 mixes two error regimes and is inflated). On noisy data clusters labelled conditionally invariant have mean error 0.09 versus 0.19 for ambiguous ones; on severe data
all clusters are ambiguous or fragile (mean error about 0.61). No cluster is classified globally invariant. The interaction term shows no relation to error (rho -0.25 pooled, not significant).
Clusters within a dataset are not independent.

**exp3, real data (5 seeds).** PCCS is within 0.01 of its own reference partition in four of six settings (higher by 0.02 and 0.05 on wine) and does not beat k-means or ICC overall: ARI on original data iris 0.618, wine 0.894,
breast cancer 0.685; on corrupted data 0.263, 0.409, 0.660 (k-means 0.260, 0.556, 0.656). Mean rank among seven methods on the three datasets: 3.5 (original) and 4.0 (corrupted).

**exp4, stability versus intensity and interaction (3 seeds, clean and noisy).** Under perturbation of intensity and family, k-means agreement with the reference is highest under algorithmic perturbations (0.75 on noisy data), then sampling and distribution shift (0.66), and lowest
under feature masking and batch effects (0.44 and 0.45); hierarchical clustering is the most robust to sampling and distribution shift (0.66 and 0.65), fuzzy c-means is
least stable (0.27 to 0.48), and DBSCAN does not recover structure on these data (ARI to the reference about 0). None of the 30 pair-by-dataset-type interaction means reaches the 0.15 threshold (maximum 0.067);
two pairs that looked synergistic in a single exploratory fit (sampling+feature, feature+measurement, 0.14 to 0.16) did not replicate.

## ICC findings
See the suite READMEs and [results_summary.md](results_summary.md): ICC is best with small training sets (exp1), most ablation effects are within seed noise (exp2), it ranks first on corrupted real data (exp3), and costs 15 to 50 times a k-means run (exp4).

## Limits
Synthetic data follow one generator; stress families are simulated (batch effects are random, not confounded with clusters; distribution shift is covariate shift only);
seeds are few; thresholds for categories are design choices; PCCS cost is about 270 clusterings per fit (2 to 6 s at N=500, 1.4 to 2 s on breast cancer N=569), and N=1797 (digits) was not run.
