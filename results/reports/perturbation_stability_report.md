# Perturbation Stability Report

**Project:** Perturbation-Conditional Consensus Clustering (PCCS)  
**Specification Reference:** Section 4.5 & Problem Statement Section 1.1  
**Authoritative Data Source:** `results/raw/perturbation_sweeps_results.csv`  

---

## 1. Multi-Dimensional Stability by Perturbation Family

The foundational premise of PCCS is that a single aggregate stability score obscures the specific vulnerabilities of clusterings. Modeling stability as a multi-dimensional response tensor reveals stark differences in vulnerability across perturbation families:

| family       |   nominal_level |   ari_ref_mean |   ari_ref_std |   ami_ref_mean |   ami_ref_std |
|:-------------|----------------:|---------------:|--------------:|---------------:|--------------:|
| algorithmic  |            0.1  |      0.68318   |     0.419052  |      0.66648   |     0.417247  |
| algorithmic  |            0.25 |      0.683831  |     0.41957   |      0.668891  |     0.419177  |
| algorithmic  |            0.5  |      0.683041  |     0.418945  |      0.666881  |     0.417558  |
| distribution |            0.2  |      0.682021  |     0.407777  |      0.656221  |     0.398501  |
| distribution |            0.5  |      0.690269  |     0.411749  |      0.659315  |     0.398656  |
| distribution |            0.8  |      0.671043  |     0.415122  |      0.646513  |     0.402571  |
| feature      |            0.1  |      0.649668  |     0.406092  |      0.621096  |     0.394168  |
| feature      |            0.25 |      0.650077  |     0.402164  |      0.613587  |     0.385558  |
| feature      |            0.5  |      0.564537  |     0.366426  |      0.531642  |     0.33908   |
| measurement  |            0.1  |      0.696067  |     0.419614  |      0.67573   |     0.409375  |
| measurement  |            0.3  |      0.645997  |     0.395811  |      0.610908  |     0.376165  |
| measurement  |            0.6  |      0.582627  |     0.361905  |      0.541564  |     0.335084  |
| sampling     |            0.5  |      0.668997  |     0.407412  |      0.643501  |     0.395096  |
| sampling     |            0.75 |      0.698194  |     0.418382  |      0.675556  |     0.40714   |
| sampling     |            0.9  |      0.704998  |     0.420081  |      0.68286   |     0.411284  |
| technical    |            0.2  |      0.677544  |     0.419784  |      0.65008   |     0.404094  |
| technical    |            0.5  |      0.647576  |     0.392648  |      0.614768  |     0.37439   |
| technical    |            1    |      0.0281002 |     0.0586443 |      0.0229797 |     0.0479688 |

### Key Findings:
1. **Feature Masking and Technical Shifts Produce the Most Severe Degradation:** While algorithmic initializations remain stable ($ARI_{ref} \approx 0.683$), feature masking degrades stability to $0.5645$ and extreme technical shift collapses stability to $0.0281$.
2. **Algorithmic Stability Disguises Structural Vulnerabilities:** Clustering methods display high algorithmic reproducibility across seeds, falsely suggesting universal robustness if measured in isolation.
3. **Multi-Family Tensor Necessity:** Capturing degradation across all 6 families prevents overconfidence in reproducible but fragile clusters.
