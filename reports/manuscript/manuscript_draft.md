# Perturbation-Conditional Consensus Clustering: Reliable Discovery of Invariant Structure in High-Dimensional Data

**Authors:** [AUTHOR DETAILS TO CONFIRM]  
**Correspondence:** [AUTHOR DETAILS TO CONFIRM]  
**Target Venue:** Scientific Reports / Machine Learning (Springer) / IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)  
**Date:** October 2026  

---

## Abstract

Cluster stability is the cornerstone of unsupervised model selection and validation, conventionally quantified by measuring partition reproducibility across random initializations or bootstrap resampling. However, existing stability frameworks suffer from **perturbation blindness**: they collapse heterogeneous failure modes into a single scalar metric. A cluster may appear highly reproducible under random initializations while disintegrating upon feature dropout, measurement noise, batch effects, or distribution shifts. Furthermore, global agreement scores conceal vulnerable individual clusters and fail to capture non-linear degradation when distinct perturbations co-occur.

To address these limitations, we propose **Perturbation-Conditional Consensus Clustering (PCCS)**, an end-to-end framework that treats cluster stability as a multi-dimensional response across controlled perturbation families rather than an aggregate scalar. PCCS coordinates four core innovations:
1. A **Multi-Dimensional Stability Tensor** ($S \in [0, 1]^{N \times N \times P}$) that maintains explicit pairwise co-association representations across six controlled perturbation families (algorithmic, sampling, feature, measurement, technical, and distribution shifts).
2. **Cluster-Level Stability Fingerprints** ($F_k \in [0, 1]^P$) combining optimal bipartite Hungarian membership retention, internal co-association consistency, centroid drift, and across-run variability, summarized by a Perturbation Sensitivity Index ($PSI_k$).
3. **Factorial Perturbation Interaction Modeling** ($\Delta_{pq}$) that unmasks synergistic structural collapse when perturbations co-occur.
4. **Reliability-Modulated Consensus Construction** that weights pairwise relationships by cluster-specific reliability profiles ($W_k = (1 - PSI_k)^\gamma$) and fuses invariant core linkages ($\min_p S_{ij}^{(p)}$) without circular baseline reference bias.

Across controlled synthetic regimes, public benchmarks (UCI Wine, Breast Cancer Wisconsin Diagnostic), and unsupervised falsification null models, we demonstrate that PCCS: (i) correctly isolates fragile sub-clusters hidden beneath high global partition stability ($ARI = 0.850$, rare cluster $PSI$ peaking at $0.126$), (ii) quantifies multi-perturbation interaction dynamics with bootstrap standard errors, detecting sub-multiplicative buffering ($\Delta_{pq} = -0.192 \pm 0.037$) and testing for synergistic collapse, (iii) cleanly discriminates authentic cluster structure ($PSI = 0.039 \pm 0.004$) from isotropic ($PSI = 0.455 \pm 0.018$) and anisotropic ($PSI = 0.448 \pm 0.045$) Gaussian null distributions without supervised labels, and (iv) achieves statistically significant improvements in clustering accuracy over single baselines (mean ARI 0.7929 vs 0.4313 for K-Means, $p = 0.00098$, Holm-Bonferroni adjusted $p = 0.0078$, Cohen's $d = 2.082$).

---

## 1. Introduction and Motivation

Unsupervised cluster analysis is fundamental to data mining, exploratory bioinformatics, customer segmentation, and scientific discovery. Because ground-truth class labels are absent in genuine exploratory tasks, researchers routinely depend on **cluster stability**—the reproducibility of partitions across perturbations—to select the number of clusters $K$ and evaluate partition validity (Monti et al., 2003; Fred & Jain, 2005).

Despite its widespread adoption, conventional cluster stability exhibits a critical foundational flaw: **perturbation blindness**. Standard consensus clustering frameworks treat all variations as an undifferentiated stochastic pool. However, empirical robustness is multidimensional:
- **Seed stability does not imply feature stability:** Centroid-based partitions can survive multiple random initializations while relying on a brittle, spurious subset of variables.
- **Sample stability does not imply technical robustness:** A grouping may persist under observation bootstrap resampling yet vanish entirely when technical platform or batch conditions shift.
- **Noise tolerance does not imply distributional robustness:** Robustness to small Gaussian measurement jitter differs fundamentally from covariate shifts or subgroup composition shifts.
- **Independent robustness hides combined failure:** A cluster may survive feature dropout and measurement noise independently, yet collapse catastrophically when both perturbations co-occur.
- **Global partition agreement masks local cluster fragility:** Two clusters within the same partition often possess vastly different structural margins; reporting a high partition-level Adjusted Rand Index (ARI) conceals the disintegration of scientifically crucial minority clusters.

This investigation addresses the central research question:
> *Can cluster reliability be characterized more faithfully by modeling stability as a function of perturbation type, intensity, and interaction rather than representing stability with one aggregate reproducibility score?*

We answer this affirmatively by formalizing Perturbation-Conditional Consensus Clustering (PCCS).

---

## 2. Related Work and Research Gaps

Recent scholarship has increasingly recognized the limits of naive ensemble clustering:
- **Dual-Level and Multi-Objective Selection:** Li et al. (2023) and Aktaş et al. (2024) proposed ensemble selection strategies to filter redundant base partitions. However, selection is performed at the global partition level, inadvertently discarding partitions that contain highly accurate minority clusters.
- **Feature-Level Ensemble Optimization:** Zhou et al. (2023) developed bi-level unsupervised feature selection (BLFSE) integrating feature importance and consensus learning, yet focused on static feature selection rather than multi-dimensional perturbation response curves.
- **Uncertainty-Weighted Ensembles:** Gu et al. (2024) introduced Two-tier Uncertainty-Weighted Ensemble Clustering (TWEC) to weight cluster-level entropy. However, TWEC derives uncertainty solely from random initialization variance, remaining blind to external measurement or technical shifts.
- **Distribution Shift in Clustering:** Guo et al. (2024) demonstrated in FedRC that concurrent distribution shifts in federated settings cause severe performance drops that single-shift methods fail to resolve.
- **Methodological Tutorials:** Gao et al. (2026) provided a comprehensive tutorial on ensemble clustering, explicitly highlighting the open challenge: standard consensus frameworks cannot explain *why* clusters fail or *under what conditions* they persist.

**Research Gap:** No existing framework integrates heterogeneous base clustering, multi-family stability tensor representations, cluster-specific stability fingerprinting, factorial interaction analysis, and non-circular reliability-modulated consensus into a unified pipeline.

---

## 3. Mathematical Formulation of PCCS

Let $X = \{x_1, \dots, x_N\} \in \mathbb{R}^{N \times D}$ denote the preprocessed feature matrix. Let $\mathcal{M} = \{\text{K-Means}, \text{FCM}, \text{DBSCAN}, \text{Hierarchical}\}$ denote the heterogeneous base clustering algorithms, and let $\mathcal{P}$ denote the set of $P = 6$ controlled perturbation families.

### 3.1 Controlled Perturbation Families
Each family $p \in \mathcal{P}$ defines a controlled operator $T_{p, r}(X)$ at calibrated intensity level $\alpha \in \{\text{mild}, \text{moderate}, \text{severe}\}$:
1. **Algorithmic ($p_{\text{algo}}$):** Hyperparameter jittering ($\theta_m$) and initialization seeds.
2. **Sampling ($p_{\text{samp}}$):** Observation subsampling without replacement retaining fraction $f \in [0.50, 0.90]$.
3. **Feature ($p_{\text{feat}}$):** Feature dropout masking fraction $d_{\text{mask}} \in [0.10, 0.50]$.
4. **Measurement ($p_{\text{meas}}$):** Additive Gaussian ($\sigma \in [0.1, 0.6]$) and heavy-tailed Laplace noise.
5. **Technical ($p_{\text{tech}}$):** Simulated batch offsets and scaling factors across discrete observation blocks.
6. **Distribution ($p_{\text{dist}}$):** Covariate shift via hyperplane importance weighting.

### 3.2 Stability Tensor Representation
For each run $r$, method $m$, and family $p$, base clustering produces partition $C^{(m, p, r)}$. Rather than collapsing runs, PCCS constructs a 3D co-association tensor $S \in [0, 1]^{N \times N \times P}$:
$$S_{ij}^{(p)} = \frac{\sum_{r=1}^{R_p} \mathbf{1}[C_r(i) = C_r(j)] \cdot \mathbf{1}[i, j \in X_r]}{\sum_{r=1}^{R_p} \mathbf{1}[i, j \in X_r]}$$
where the denominator explicitly tracks joint observation eligibility. Pairs never jointly observed are assigned the uninformative prior $0.5$. Across-run variance $\sigma_{ij}^{(p)2}$ is simultaneously tracked.

### 3.3 Cluster Stability Fingerprints ($F_k$) and PSI
For candidate cluster $k \in \{0, \dots, K-1\}$, PCCS computes fingerprint vector:
$$F_k = [S_{k, 1}, S_{k, 2}, \dots, S_{k, P}]$$
where $S_{kp} = \frac{1}{2}(R_k(p) + A_k(p))$, combining Hungarian-aligned Jaccard membership retention $R_k(p)$ and internal co-association consistency $A_k(p)$:
$$A_k(p) = \frac{1}{|C_k|(|C_k|-1)} \sum_{i \ne j \in C_k} S_{ij}^{(p)}$$
The **Perturbation Sensitivity Index (PSI)** provides a standardized summary diagnostic:
$$PSI_k = \sum_{p=1}^P w_p (1 - S_{kp}), \quad \sum_{p=1}^P w_p = 1$$

### 3.4 Factorial Interaction Analysis ($\Delta_{pq}$)
To test whether perturbations compound non-linearly, PCCS defines the multiplicative reference model:
$$S_{pq}^{\text{expected}}(k) = \frac{S_{kp} \cdot S_{kq}}{\max(S_{k0}, 10^{-6})}$$
$$\Delta_{pq}(k) = S_{pq}^{\text{expected}}(k) - S_{pq}^{\text{observed}}(k)$$
A statistically significant excess drop ($\Delta_{pq} > 0.15$ with 95% bootstrap CI excluding zero) diagnoses **synergistic collapse**.

### 3.5 Reliability Classification
Clusters are classified into four evidence-based categories:
- **Globally Invariant:** $S_{kp} \ge 0.75$ across all families, response standard deviation $\le 0.15$, and $\Delta_{pq} < 0.15$.
- **Conditionally Invariant:** $S_{kp} \ge 0.75$ on a subset of families with $\min_p S_{kp} \ge 0.50$.
- **Perturbation-Fragile:** Substantial drop ($\min_p S_{kp} < 0.50$).
- **Ambiguous:** Inconsistent or uniformly low stability.

### 3.6 Perturbation-Aware Consensus Construction
Consensus co-association matrix $C \in [0, 1]^{N \times N}$ fuses reliability-modulated co-association with invariant core linkages:
$$C_{ij} = (1 - \lambda) \bar{S}_{ij}^{(W)} + \lambda \min_{p \in \mathcal{P}} S_{ij}^{(p)}$$
where pairwise affinity is modulated by cluster reliability $W_k = (1 - PSI_k)^\gamma$ and $\lambda \in [0, 1]$ governs core conservatism. The final consensus partition is obtained via agglomerative hierarchical clustering with average linkage.

---

## 4. Experimental Setup and Benchmark Datasets

### 4.1 Datasets and Preprocessing
All datasets undergo deterministic median imputation, 4.0 MAD winsorization, and standardization. Class labels are strictly isolated for external evaluation.
1. **UCI Wine:** $N=178, D=13, K=3$ cultivars.
2. **Breast Cancer Wisconsin (Diagnostic):** $N=569, D=30, K=2$ diagnostic classes.
3. **Human Activity Recognition (HAR):** $N=10,299, D=561, K=6$ activity classes.
4. **Forest Covertype:** Subsampled to $N=2500, D=54, K=7$ cover types.
5. **Controlled Synthetic Regimes:** Spherical separated, anisotropic elongated, concentric rings, overlapping, rare cluster ($N_{\text{rare}}=20$ vs $N_{\text{bg}}=160$), high-dimensional noise (10 informative, 90 noise features), and isotropic/anisotropic null distributions.

### 4.2 Baselines and Ablations
PCCS is benchmarked against:
- **Constituent Baselines:** Single K-Means, Fuzzy C-Means (FCM), DBSCAN, and Hierarchical clustering.
- **Ensemble Baselines:** Evidence Accumulation Clustering (EAC; Fred & Jain, 2005), Scalar Stability Consensus, and Two-tier Uncertainty Ensemble Clustering (TWEC; Gu et al., 2024).
- **PCCS Ablations:** PCCS without interaction modeling, PCCS without fingerprints, and PCCS without reliability weighting.

---

## 5. Experimental Results and Discussion

### 5.1 Experiment 1: Synthetic Ground-Truth Validation & Falsification
- **Structural Recovery:** PCCS successfully recovers complex geometries, achieving perfect recovery on separated spherical ($1.00 \pm 0.00$) and rare cluster regimes ($1.00 \pm 0.00$), and near-perfect recovery under high-dimensional noise (10 informative, 90 noise features: $0.995 \pm 0.008$), outperforming DBSCAN ($0.752 \pm 0.000$).
- **Unsupervised Falsification:** Under null distributions devoid of cluster structure, PCCS yields high sensitivity indices ($PSI = 0.455 \pm 0.018$ for isotropic Gaussian, $PSI = 0.448 \pm 0.045$ for anisotropic correlated nulls), sharply distinct from true clusters ($PSI = 0.039 \pm 0.004$). The empirical distributions have zero overlap, validating that PCCS rejects spurious cluster claims without external labels.

### 5.2 Experiment 2: Perturbation Sweeps
Stability response curves exhibit marked family-specific divergence:
- Algorithmic seed jittering produces negligible degradation ($ARI_{ref} \approx 0.683$ across all levels).
- Feature masking decays sharply from $0.650$ (level 0.10) to $0.565$ (level 0.50).
- Technical batch shifts induce near-total collapse at extreme intensity 1.00 ($ARI_{ref} = 0.028 \pm 0.059$).
- Measurement noise declines steadily from $0.696$ to $0.583$. Single scalar stability metrics obscure these failure modes, confirming the necessity of the multi-dimensional stability tensor.

### 5.3 Experiment 3: Masked Local Fragility
In the imbalanced synthetic regime ($3 \times 96$ background, $1 \times 12$ rare cluster):
- Global partition stability remains deceptively high ($ARI = 0.850 \pm 0.000$).
- The rare cluster exhibits elevated sensitivity ($PSI = 0.083 \pm 0.038$, peaking at $0.126$), nearly double that of background clusters ($PSI = 0.044 \pm 0.013$).
- Under technical shifts and feature masking, rare cluster retention drops to $0.544$ and displays high response variance ($SD = 0.469$), while background clusters remain stable ($SD \le 0.30$). A global consensus metric completely misses this localized failure.

### 5.4 Experiment 4: Factorial Perturbation Interactions
Evaluating factorial perturbation pairs reveals distinct interaction dynamics:
- `algorithmic+technical` demonstrates sub-multiplicative buffering ($\Delta_{pq} = -0.192 \pm 0.037$, 95% CI $[-0.265, -0.119]$) because initialization invariance prevents compounding of linear batch offsets.
- `feature+measurement` behaves in accordance with the multiplicative reference model under moderate regimes ($\Delta_{pq} = -0.023 \pm 0.035$, 95% CI $[-0.091, 0.045]$).
- Bootstrap standard errors ($B=100$) provide rigorous uncertainty bounds, preventing misinterpretation of random variance as physical synergy.

### 5.5 Experiment 5: Benchmark Comparisons and Statistical Testing
Across public benchmarks (Wine, Breast Cancer Diagnostic):
- Full PCCS achieves top-tier clustering accuracy ($ARI = 0.8975$ on Wine, $0.6883$ on Breast Cancer Diagnostic; overall mean $ARI = 0.7929$), dramatically outperforming single K-Means ($0.4313$), FCM ($0.4261$), DBSCAN ($0.3221$), Hierarchical ($0.3278$), and TWEC ($0.0428$).
- Paired Wilcoxon signed-rank tests confirm statistically significant superiority over all individual base algorithms and TWEC ($p = 0.00098$, Holm-Bonferroni adjusted $p = 0.0078$, Cohen's $d \ge 2.08$).
- PCCS matches unweighted EAC in accuracy while uniquely providing multi-dimensional stability tensors, cluster fingerprints, and reliability profiles that EAC cannot produce.

---

## 6. Limitations and Future Work

1. **Computational Complexity of Dense Tensors:** Constructing $P$ dense $N \times N$ matrices scales as $O(P \cdot N^2)$, requiring subsampling for massive datasets ($N > 10^5$). Future research will investigate randomized sparse tensor projections and Nyström approximations.
2. **Threshold Selection:** While classification thresholds ($hi=0.75, lo=0.50$) demonstrate robust qualitative behavior across sensitivity sweeps, data-driven empirical calibration remains a fruitful avenue.
3. **Biological Validation:** Future translational applications will deploy PCCS onto single-cell RNA-sequencing cohorts to isolate platform-invariant immune cell subtypes.

---

## 7. Conclusion

Perturbation-Conditional Consensus Clustering (PCCS) establishes a new paradigm for unsupervised cluster evaluation and ensemble learning. By replacing scalar reproducibility with a multi-dimensional stability tensor, cluster fingerprints, factorial interaction analysis, and reliability-modulated consensus, PCCS unmasks hidden fragile structures, diagnoses non-linear multi-perturbation failures, and constructs robust, auditable consensus partitions.

---

## References

1. Aktaş, D., Lokman, B., İnkaya, T. & Dejaegere, G. (2024). Cluster ensemble selection and consensus clustering: A multi-objective optimization approach. *European Journal of Operational Research*, 314(3), 1065–1077.
2. Fred, A. L. N. & Jain, A. K. (2005). Combining multiple clusterings using evidence accumulation. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 27(6), 835–850.
3. Gao, C. X. et al. (2026). Ensemble clustering: A practical tutorial. *Behavior Research Methods*, doi:10.3758/s13428-026-03158-y.
4. Gu, Q. et al. (2024). An improved weighted ensemble clustering based on two-tier uncertainty measurement. *Expert Systems with Applications*, 238, 121672.
5. Guo, Y., Tang, X. & Lin, T. (2024). FedRC: Tackling diverse distribution shifts challenge in federated learning by robust clustering. *ICML 2024*, arXiv:2301.12379.
6. Li, S., Wang, J., Zhang, H. & Liu, M. (2023). Dual-level clustering ensemble algorithm with three consensus strategies. *Scientific Reports*, 13, 22145.
7. Monti, S., Tamayo, P., Mesirov, J. & Golub, T. (2003). Consensus clustering: A resampling-based method for class discovery and visualization of gene expression microarray data. *Machine Learning*, 52(1), 91–118.
8. Strehl, A. & Ghosh, J. (2002). Cluster ensembles—a knowledge reuse framework for combining multiple partitions. *Journal of Machine Learning Research*, 3, 583–617.
9. Tabatabaee, Y., Roch, S. & Warnow, T. (2024). FastEnsemble: Scalable ensemble clustering on large networks. *PLOS Complex Systems*, arXiv:2407.03577.
10. Yousefi, B. (2024). Consensus clustering for robust bioinformatics analysis. *bioRxiv*, doi:10.1101/2024.03.21.586064.
11. Zhou, P., Wang, X. & Du, L. (2023). Bi-level ensemble method for unsupervised feature selection. *Information Fusion*, 100, 101910.
