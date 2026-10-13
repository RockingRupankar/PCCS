# Experiment 2: Perturbation-Family and Intensity Sweeps Report

**Project:** Perturbation-Conditional Consensus Clustering (PCCS)  
**Specification Reference:** Section 4.5, Section 7, Section 10  
**Authoritative Data Source:** `results/raw/perturbation_sweeps_results.csv`  

---

## 1. Scientific Objective and Hypothesis

A core claim of PCCS is that conventional stability measurements suffer from perturbation blindness: a cluster may be reproducible under repeated random initializations or row subsampling while collapsing under feature masking, measurement noise, or technical batch shifts. Experiment 2 maps out stability response surfaces across 6 explicitly parameterized perturbation families at 3 distinct intensity tiers.

## 2. Controlled Perturbation Families & Intensities

1. **Algorithmic:** Initializations and hyperparameter jittering (nominal levels: 0.10, 0.25, 0.50).
2. **Sampling:** Subsampling without replacement (nominal fractions: 0.50, 0.75, 0.90).
3. **Feature:** Subspace feature dropout/masking (nominal fractions: 0.10, 0.25, 0.50).
4. **Measurement:** Additive Gaussian and heavy-tailed Student-$t$ noise (nominal standard deviations: 0.10, 0.30, 0.60).
5. **Technical:** Multiplicative batch scaling and mean offset shifts (nominal offsets: 0.20, 0.50, 1.00).
6. **Distribution:** Covariate shift via exponential tilting (nominal tilting factors: 0.20, 0.50, 0.80).

## 3. Quantitative Results by Family and Intensity Tier

| family       |   nominal_level |   ari_ref_mean |   ari_ref_std |   ari_ref_min |   ari_ref_max |   ami_ref_mean |   ami_ref_std |   ami_ref_min |   ami_ref_max |
|:-------------|----------------:|---------------:|--------------:|--------------:|--------------:|---------------:|--------------:|--------------:|--------------:|
| algorithmic  |            0.1  |      0.68318   |     0.419052  |    0          |      1        |      0.66648   |     0.417247  |     0         |      1        |
| algorithmic  |            0.25 |      0.683831  |     0.41957   |    0          |      1        |      0.668891  |     0.419177  |     0         |      1        |
| algorithmic  |            0.5  |      0.683041  |     0.418945  |    0          |      1        |      0.666881  |     0.417558  |     0         |      1        |
| distribution |            0.2  |      0.682021  |     0.407777  |    0          |      1        |      0.656221  |     0.398501  |     0         |      1        |
| distribution |            0.5  |      0.690269  |     0.411749  |    0          |      1        |      0.659315  |     0.398656  |     0         |      1        |
| distribution |            0.8  |      0.671043  |     0.415122  |    0          |      0.976955 |      0.646513  |     0.402571  |     0         |      0.967503 |
| feature      |            0.1  |      0.649668  |     0.406092  |    0          |      0.971664 |      0.621096  |     0.394168  |     0         |      0.95468  |
| feature      |            0.25 |      0.650077  |     0.402164  |    0          |      0.971694 |      0.613587  |     0.385558  |     0         |      0.934843 |
| feature      |            0.5  |      0.564537  |     0.366426  |    0          |      0.957624 |      0.531642  |     0.33908   |     0         |      0.920006 |
| measurement  |            0.1  |      0.696067  |     0.419614  |    0          |      1        |      0.67573   |     0.409375  |     0         |      1        |
| measurement  |            0.3  |      0.645997  |     0.395811  |    0          |      0.964275 |      0.610908  |     0.376165  |     0         |      0.946592 |
| measurement  |            0.6  |      0.582627  |     0.361905  |    0          |      0.923103 |      0.541564  |     0.335084  |     0         |      0.851779 |
| sampling     |            0.5  |      0.668997  |     0.407412  |    0          |      1        |      0.643501  |     0.395096  |     0         |      1        |
| sampling     |            0.75 |      0.698194  |     0.418382  |    0          |      1        |      0.675556  |     0.40714   |     0         |      1        |
| sampling     |            0.9  |      0.704998  |     0.420081  |    0          |      1        |      0.68286   |     0.411284  |     0         |      1        |
| technical    |            0.2  |      0.677544  |     0.419784  |    0          |      0.971694 |      0.65008   |     0.404094  |     0         |      0.947615 |
| technical    |            0.5  |      0.647576  |     0.392648  |    0          |      0.971723 |      0.614768  |     0.37439   |     0         |      0.942646 |
| technical    |            1    |      0.0281002 |     0.0586443 |   -0.00272824 |      0.19865  |      0.0229797 |     0.0479688 |    -0.0032209 |      0.150084 |

## 4. Key Empirical Findings

1. **Non-Uniform Degradation:** The degradation of cluster stability is highly non-uniform across families:
   - **Algorithmic Stability:** Remains consistently elevated ($ARI_{ref} = 0.683 \pm 0.419$ across all levels), confirming that measuring only random seed stability gives a false sense of security.
   - **Feature Dropout:** Decays sharply as feature masking increases from 0.10 ($ARI_{ref} = 0.650$) to 0.50 ($ARI_{ref} = 0.565$).
   - **Technical Batch Shifts:** Produces near-total catastrophic collapse at extreme intensity 1.00 ($ARI_{ref} = 0.0281 \pm 0.0586$), demonstrating extreme vulnerability to technical domain shifts.
   - **Measurement Noise:** Displays systematic decline from $ARI_{ref} = 0.6961$ (level 0.10) to $ARI_{ref} = 0.5826$ (level 0.60).
2. **Implications for Consensus Design:** Because different clusters fail under different stress mechanisms, a single scalar stability score is inadequate. Consensus affinity matrices must be conditioned on multi-dimensional stability tensors to separate true invariants from perturbation-specific artifacts.
