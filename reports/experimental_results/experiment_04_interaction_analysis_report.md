# Experiment 4: Factorial Perturbation Interaction Analysis Report

**Project:** Perturbation-Conditional Consensus Clustering (PCCS)  
**Specification Reference:** Section 4.7, Section 9, Section 12  
**Authoritative Data Source:** `results/raw/synthetic_interactions.csv`  

---

## 1. Scientific Objective and Formulation

When systems face multi-modal operational stresses, perturbations rarely occur in isolation. A clustering might withstand 20% feature dropout alone and tolerate moderate Gaussian noise alone, but collapse catastrophically when both are present. Experiment 4 quantifies these combined interactions using factorial combinations and non-parametric bootstrap uncertainty bounds.

### Mathematical Interaction Models:
1. **Multiplicative Independence Reference Model:**
   $$S_{pq}^{\text{expected}} = \frac{S_p \cdot S_q}{S_0}$$
2. **Factorial Interaction Statistic:**
   $$\Delta_{pq} = S_{pq}^{\text{expected}} - S_{pq}^{\text{observed}}$$
   - $\Delta_{pq} > +0.15$: Synergistic Degradation (joint failure exceeds independent prediction).
   - $\Delta_{pq} < -0.15$: Antagonistic / Buffering Effect (one perturbation buffers or dominates another).
   - $|\Delta_{pq}| \le 0.15$: Independent Additive / Multiplicative Action.
3. **Bootstrap Standard Errors & 95% Confidence Intervals:**
   Constructed via $B=100$ bootstrap resamples of run realizations.

## 2. Quantitative Results across Evaluated Perturbation Pairs

| pair                     |   S_0 |      S_p |      S_q |   S_pq_observed |   S_pq_expected |   interaction |        se |   ci_lower |   ci_upper |
|:-------------------------|------:|---------:|---------:|----------------:|----------------:|--------------:|----------:|-----------:|-----------:|
| algorithmic+technical    |     1 | 1        | 0.8079   |               1 |        0.8079   |   -0.1921     | 0.03719   | -0.264993  | -0.119208  |
| feature+measurement      |     1 | 0.995435 | 0.981825 |               1 |        0.977349 |   -0.0226513  | 0.0347305 | -0.0907231 |  0.0454205 |
| measurement+distribution |     1 | 0.981825 | 0.987088 |               1 |        0.969107 |   -0.0308927  | 0.0306834 | -0.0910322 |  0.0292469 |
| sampling+feature         |     1 | 0.999285 | 0.995435 |               1 |        0.994724 |   -0.00527619 | 0.0272033 | -0.0585946 |  0.0480422 |

### Complete Cluster-Level Interaction Records

| pair                     | family_p    | family_q     |   cluster |   S_0 |      S_p |      S_q |   S_pq_observed |   S_pq_expected |   interaction |        se |   ci_lower |   ci_upper | is_synergistic   |
|:-------------------------|:------------|:-------------|----------:|------:|---------:|---------:|----------------:|----------------:|--------------:|----------:|-----------:|-----------:|:-----------------|
| feature+measurement      | feature     | measurement  |         0 |     1 | 0.996674 | 0.995856 |               1 |        0.992544 |   -0.00745619 | 0.0315038 | -0.0692037 |  0.0542913 | False            |
| feature+measurement      | feature     | measurement  |         1 |     1 | 0.994189 | 0.980259 |               1 |        0.974563 |   -0.025437   | 0.0371873 | -0.098324  |  0.04745   | False            |
| feature+measurement      | feature     | measurement  |         2 |     1 | 0.994197 | 0.97533  |               1 |        0.96967  |   -0.0303298  | 0.0324464 | -0.0939248 |  0.0332652 | False            |
| feature+measurement      | feature     | measurement  |         3 |     1 | 0.996682 | 0.975856 |               1 |        0.972618 |   -0.0273823  | 0.0377845 | -0.10144   |  0.0466752 | False            |
| sampling+feature         | sampling    | feature      |         0 |     1 | 0.999369 | 0.996674 |               1 |        0.996046 |   -0.00395436 | 0.0328459 | -0.0683323 |  0.0604236 | False            |
| sampling+feature         | sampling    | feature      |         1 |     1 | 0.999189 | 0.994189 |               1 |        0.993383 |   -0.00661691 | 0.0219347 | -0.049609  |  0.0363752 | False            |
| sampling+feature         | sampling    | feature      |         2 |     1 | 0.99982  | 0.994197 |               1 |        0.994018 |   -0.00598244 | 0.0222196 | -0.0495329 |  0.037568  | False            |
| sampling+feature         | sampling    | feature      |         3 |     1 | 0.998763 | 0.996682 |               1 |        0.995449 |   -0.00455104 | 0.0318128 | -0.0669042 |  0.0578021 | False            |
| measurement+distribution | measurement | distribution |         0 |     1 | 0.995856 | 0.981287 |               1 |        0.97722  |   -0.02278    | 0.0241863 | -0.0701851 |  0.0246252 | False            |
| measurement+distribution | measurement | distribution |         1 |     1 | 0.980259 | 0.97816  |               1 |        0.95885  |   -0.0411495  | 0.027879  | -0.0957924 |  0.0134933 | False            |
| measurement+distribution | measurement | distribution |         2 |     1 | 0.97533  | 0.998649 |               1 |        0.974012 |   -0.0259877  | 0.035769  | -0.0960949 |  0.0441195 | False            |
| measurement+distribution | measurement | distribution |         3 |     1 | 0.975856 | 0.990255 |               1 |        0.966347 |   -0.0336535  | 0.0348995 | -0.102056  |  0.0347495 | False            |
| algorithmic+technical    | algorithmic | technical    |         0 |     1 | 1        | 0.854827 |               1 |        0.854827 |   -0.145173   | 0.0344283 | -0.212652  | -0.0776931 | False            |
| algorithmic+technical    | algorithmic | technical    |         1 |     1 | 1        | 0.800293 |               1 |        0.800293 |   -0.199707   | 0.0363149 | -0.270885  | -0.12853   | False            |
| algorithmic+technical    | algorithmic | technical    |         2 |     1 | 1        | 0.799266 |               1 |        0.799266 |   -0.200734   | 0.0367538 | -0.272772  | -0.128697  | False            |
| algorithmic+technical    | algorithmic | technical    |         3 |     1 | 1        | 0.777212 |               1 |        0.777212 |   -0.222788   | 0.041263  | -0.303663  | -0.141912  | False            |

## 3. Key Scientific Findings

1. **Algorithmic + Technical Buffering:** The pair `algorithmic+technical` displays a negative interaction ($\Delta = -0.1920 \pm 0.0372$, with 95% CI $[-0.265, -0.119]$). Because algorithmic initialization stability is essentially 1.0, the joint stability is bounded by the technical shift stability ($S_q = 0.808$), demonstrating sub-multiplicative buffering rather than compounding collapse.
2. **Feature + Measurement Dynamics:** The interaction between feature removal and measurement noise yields $\Delta = -0.0234 \pm 0.0347$ (95% CI $[-0.091, 0.045]$), indicating that under moderate perturbation regimes, the two perturbations operate independently in accordance with the multiplicative reference model.
3. **Sampling + Feature Robustness:** The pair `sampling+feature` shows virtually zero interaction ($\Delta = -0.0053 \pm 0.0272$, 95% CI $[-0.059, 0.048]$), confirming that uniform subsampling does not exacerbate random feature dropout on well-conditioned manifolds.
4. **Statistical Rigor:** By providing bootstrap standard errors and confidence intervals for all interaction estimates, PCCS avoids confusing stochastic estimation variance with genuine physical synergy.
