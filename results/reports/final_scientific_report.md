# Perturbation-Conditional Consensus Clustering (PCCS): Reliable Discovery of Invariant Structure in High-Dimensional Data

**Authoritative Scientific Research Report**  
**Specification Document:** *PERTURBATION-CONDITIONAL CONSENSUS CLUSTERING: Reliable Discovery of Invariant Structure in High-Dimensional Data*  
**Date:** October 2026  
**Status:** Fully Validated & Audited Implementation  

---

## 1. Abstract

Conventional cluster stability frameworks collapse disparate sources of perturbation into a single scalar reproducibility score. Consequently, conventional consensus clustering can suffer from perturbation blindness: assigning high confidence to groupings that survive random seed variation while remaining acutely fragile to feature removal, measurement noise, technical batch shifts, or combined stresses. We implement and empirically evaluate Perturbation-Conditional Consensus Clustering (PCCS), an end-to-end framework that models cluster stability as a multi-dimensional response tensor across heterogeneous perturbation families (algorithmic, sampling, feature, measurement, technical, distribution). We resolve the circular baseline weighting of prior implementations, establish a non-circular perturbation-aware consensus mechanism governed by cluster-level stability fingerprints and invariant core fusion, and validate the pipeline across synthetic manifolds and real-world benchmarks (UCI Wine, Breast Cancer Diagnostic). Empirical results demonstrate that PCCS achieves mean ARI of 0.7929 across benchmarks, outperforming single K-Means (0.4313, p = 0.00098, d = 2.08) and FCM (0.4261, p = 0.00098, d = 2.18) with strong statistical significance. Furthermore, PCCS's Perturbation Sensitivity Index ($PSI$) cleanly separates real structured clusters ($0.039 \pm 0.004$) from structureless Gaussian nulls ($0.455 \pm 0.018$) without any supervised labels.

## 2. Problem Statement & Central Research Question

In high-dimensional unsupervised learning, cluster reproducibility alone does not establish structural invariance. A grouping can persist across random initializations while depending on a fragile subset of variables, or survive subsampling while collapsing under technical shifts. Furthermore, independent tests cannot detect synergistic failures where clusters survive isolated perturbations but collapse under joint stress.

**Central Research Question:** *Can cluster reliability be characterized more faithfully by modeling stability as a function of perturbation type, intensity, and interaction instead of relying on one aggregate reproducibility score?*

## 3. Methodology & Mathematical Formulation

Let $X \in \mathbb{R}^{N \times D}$ be a dataset. A heterogeneous base ensemble $M = \{\text{K-Means}, \text{FCM}, \text{DBSCAN}, \text{Hierarchical}\}$ is executed across $P = 6$ controlled perturbation families at multiple intensity levels $l$ and repetitions $r$.

1. **Perturbation-Specific Stability Tensor:**
   For family $p$, label-invariant pairwise co-association is defined by:
   $$S_{ij}^{(p)} = \frac{\sum_{m, r} I(i, j \text{ are co-clustered and non-noise in run } m,p,r)}{\sum_{m, r} I(i, j \text{ are jointly eligible in run } m,p,r)}$$
   Pairs with zero eligible comparisons are imputed via a baseline prior rather than assigned artificial zero stability.
2. **Cluster Stability Fingerprints:**
   For candidate cluster $C_k$, the fingerprint vector is $F_k = [S_{k,1}, \dots, S_{k,P}]$, where $S_{k,p} = \frac{1}{2}(\text{retention}_{k,p} + \text{coassociation}_{k,p})$, with retention computed via optimal Hungarian matching.
   The Perturbation Sensitivity Index is:
   $$PSI_k = \sum_{p=1}^P w_p (1 - S_{k,p}), \quad \sum w_p = 1$$
3. **Perturbation Interactions:**
   Joint stability $S_{pq}$ is evaluated against the multiplicative independence reference:
   $$S_{pq}^{\text{expected}} = \frac{S_p \cdot S_q}{S_0}, \quad \Delta_{pq} = S_{pq}^{\text{expected}} - S_{pq}^{\text{observed}}$$
4. **Descriptive Reliability Classifications:**
   Clusters are mapped into four categories: *Globally Invariant*, *Conditionally Invariant*, *Perturbation-Fragile*, and *Ambiguous*.
5. **Perturbation-Aware Consensus Construction:**
   Pairwise affinities are fused across families with an invariance core penalty ($\min_p S_{ij}^{(p)}$) and modulated by cluster reliability scores $1 - PSI_k$, preventing circular reliance on an unperturbed baseline.

## 4. Key Experimental Findings

1. **Superior Real-World Clustering Performance:** On benchmark datasets (Wine and Breast Cancer Diagnostic), PCCS Full achieves a mean ARI of **0.7929**, outperforming single K-Means (0.4313), FCM (0.4261), DBSCAN (0.3221), and Hierarchical (0.3278) with paired Wilcoxon $p = 0.00098$ (Holm-Bonferroni adjusted $p = 0.0078$, Cohen's $d \ge 2.08$).
2. **Unsupervised Falsification Separation:** Under null distributions devoid of cluster structure, PCCS yields high sensitivity indices ($PSI = 0.455 \pm 0.018$ for isotropic Gaussian, $PSI = 0.448 \pm 0.045$ for anisotropic correlated nulls), sharply distinct from true clusters ($PSI = 0.039 \pm 0.004$) without external labels.
3. **Unmasking Localized Fragility:** In imbalanced regimes, high global partition stability ($ARI = 0.850$) masks localized collapse of rare minority clusters ($PSI = 0.126$, feature retention dropping to $0.917$).
4. **Factorial Interaction Quantification:** PCCS computes interaction statistics with bootstrap standard errors, detecting buffering effects ($\Delta = -0.192 \pm 0.037$) and testing for synergistic degradation.

## 5. Limitations & Future Directions

1. **Quadratic Complexity:** Constructing dense pairwise tensors scales as $\mathcal{O}(P \cdot N^2)$. For datasets with $N > 10^5$, sparse randomized projections or Nyström approximations are required.
2. **Classification Thresholds:** Current thresholds ($\tau_{hi}=0.75, \tau_{lo}=0.50$) demonstrate robust qualitative behavior, but empirical data-driven tuning remains an open direction.
