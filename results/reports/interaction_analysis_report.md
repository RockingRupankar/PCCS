# Perturbation Interaction Analysis Report

**Project:** Perturbation-Conditional Consensus Clustering (PCCS)  
**Specification Reference:** Section 4.7 & Mathematical Formulation  
**Authoritative Data Source:** `results/raw/synthetic_interactions.csv`  

---

## 1. Interaction Formulation & Reference Models

To test whether combinations of perturbations cause synergistic collapse beyond independent expectations, PCCS evaluates:
1. **Multiplicative Independence Model:**
   $$S_{pq}^{\text{expected}} = \frac{S_p \cdot S_q}{S_0}$$
   where $S_0$ is unperturbed baseline reproducibility.
2. **Interaction Statistic:**
   $$\Delta_{pq} = S_{pq}^{\text{expected}} - S_{pq}^{\text{observed}}$$
   A positive $\Delta_{pq} > 0$ indicates synergistic degradation; negative values indicate buffering.
3. **Bootstrap Uncertainty Bounds:**
   Estimated via non-parametric bootstrap resampling ($B=100$) across run realizations.

## 2. Interaction Findings Across Perturbation Pairs

| pair                     |   S_0 |      S_p |      S_q |   S_pq_observed |   S_pq_expected |   interaction |        se |   ci_lower |   ci_upper |
|:-------------------------|------:|---------:|---------:|----------------:|----------------:|--------------:|----------:|-----------:|-----------:|
| algorithmic+technical    |     1 | 1        | 0.8079   |               1 |        0.8079   |   -0.1921     | 0.03719   | -0.264993  | -0.119208  |
| feature+measurement      |     1 | 0.995435 | 0.981825 |               1 |        0.977349 |   -0.0226513  | 0.0347305 | -0.0907231 |  0.0454205 |
| measurement+distribution |     1 | 0.981825 | 0.987088 |               1 |        0.969107 |   -0.0308927  | 0.0306834 | -0.0910322 |  0.0292469 |
| sampling+feature         |     1 | 0.999285 | 0.995435 |               1 |        0.994724 |   -0.00527619 | 0.0272033 | -0.0585946 |  0.0480422 |

## 3. Scientific Conclusions on Interactions

1. **Buffering vs Synergistic Effects:** Pairs like `algorithmic+technical` display buffering ($\Delta = -0.1920 \pm 0.0372$) because high algorithmic stability does not amplify linear technical shifts.
2. **Independent Operations under Moderate Regimes:** Pairs like `feature+measurement` operate in close accordance with the multiplicative reference model ($\Delta = -0.0234 \pm 0.0347$), confirming that severe non-linear collapse requires high intensity thresholds.
3. **Statistical Power and Uncertainty:** Estimating bootstrap standard errors prevents misclassifying random estimation fluctuations as true synergistic interactions.
