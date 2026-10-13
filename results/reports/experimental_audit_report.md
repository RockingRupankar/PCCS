# Experimental Audit Report: Scientific Defect Analysis and Corrective Actions

**Project:** Perturbation-Conditional Consensus Clustering (PCCS)  
**Specification Reference:** Authoritative Research PDF  
**Auditor:** Autonomous Verification & Replication Agent  

---

## 1. Executive Summary

This audit documents the identification, diagnosis, and resolution of methodological defects discovered in prior experimental code:

1. **Consensus Circularity:** Prior code weighted partitions by `ari_ref` against an unperturbed baseline consensus, introducing severe confirmation bias. Corrected by implementing non-circular cluster-level reliability modulation ($W_k = (1 - PSI_k)^\gamma$) and invariant core fusion ($\min_p S_{ij}^{(p)}$).
2. **Supervised Testing Leakage:** Prior code evaluated stability fingerprints via a supervised logistic regression trained on class labels. Replaced with strictly unsupervised falsification tests using isotropic and anisotropic Gaussian null distributions.
3. **Missing Feedback of Cluster Reliability:** Prior implementations computed fingerprints purely as post-hoc diagnostics without influencing the consensus affinity matrix. Resolved by integrating reliability weights directly into pairwise co-association construction.
4. **DBSCAN Exclusion:** Prior code converted DBSCAN noise into singletons and discarded runs with low baseline agreement. Resolved with formal noise tracking (noise pairs assigned zero co-association, eligible denominators tracked).
5. **Unobserved Pair Imputation:** Prior code assigned artificial zero stability to unobserved pairs under subsampling. Resolved by tracking eligible comparisons and applying a baseline prior.
6. **Interaction Underpowering:** Prior code evaluated interactions at a single level without uncertainty. Resolved with factorial interaction sweeps, bootstrap standard errors ($B=100$), and 95% confidence intervals.
