"""
Main PCCS Estimator: Perturbation-Conditional Consensus Clustering.
Implements the full 16-step algorithmic pipeline and 10 required capabilities.
"""
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Sequence
import time
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClusterMixin

from pccs.preprocessing.pipeline import PreprocessingPipeline, validate_and_preprocess
from pccs.clustering.ensemble import fit_base_clusterer
from pccs.perturbations.engine import apply_perturbation, FAMILIES, DEFAULT_LEVELS
from pccs.stability.matching import hungarian_match, partition_metrics
from pccs.stability.tensor import StabilityTensor
from pccs.stability.fingerprints import compute_cluster_fingerprints, compute_psi
from pccs.stability.classification import classify_reliability
from pccs.interactions.factorial import compute_pairwise_interactions
from pccs.consensus.builder import build_perturbation_aware_consensus


@dataclass
class PCCSConfig:
    """Hyperparameter configuration for PCCS."""
    n_clusters: int = 4
    families: Sequence[str] = field(default_factory=lambda: list(FAMILIES))
    levels: Optional[Dict[str, Sequence[float]]] = None
    repetitions: int = 3
    methods: Sequence[str] = field(default_factory=lambda: ["kmeans", "fcm", "dbscan", "hierarchical"])
    baseline_seeds: int = 3
    fuzzifier: float = 1.3
    dbscan_min_samples: int = 5
    dbscan_eps_quantile: float = 0.6
    dbscan_eps_scale: float = 1.2
    n_batches: int = 3
    interaction: bool = True
    interaction_pairs: Optional[Sequence[Tuple[str, str]]] = None
    interaction_repetitions: int = 2
    use_reliability_weights: bool = True
    weight_power: float = 2.0
    family_weighting: str = "reliability"    # "reliability" | "equal"
    consistency_lambda: float = 0.5          # 0 = weighted mean over families, 1 = minimum over families
    linkage: str = "average"
    min_cluster_frac: float = 0.02
    hi: float = 0.75
    lo: float = 0.50
    max_response_sd: float = 0.15
    min_member_agreement: float = 0.2
    interaction_threshold: float = 0.15
    random_state: int = 0
    clip_mad: float = 4.0
    pca_variance: Optional[float] = None
    hierarchical_linkage: str = "ward"


class PCCS(BaseEstimator, ClusterMixin):
    """
    Perturbation-Conditional Consensus Clustering (PCCS).
    
    Attributes fitted:
    - labels_: np.ndarray, final consensus cluster assignments
    - fingerprint_: pd.DataFrame, cluster stability fingerprints F_k, PSI, and categories
    - interaction_: pd.DataFrame, pairwise factorial interaction degradations
    - tensor_: StabilityTensor, multi-dimensional co-association tensor
    - summary_: Dict[str, Any], global and consensus diagnostics
    - computational_stats_: Dict[str, Any], runtime breakdown and memory
    """
    def __init__(self, config: Optional[PCCSConfig] = None, **kwargs):
        if config is None:
            self.config = PCCSConfig(**kwargs)
        else:
            self.config = config
        self.levels_ = self.config.levels or DEFAULT_LEVELS

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None, batch: Optional[np.ndarray] = None) -> "PCCS":
        start_time = time.time()
        c = self.config
        rng = np.random.RandomState(c.random_state)
        timings: Dict[str, float] = {}

        # 1 & 2. Preprocess X and construct fixed working representation
        t0 = time.time()
        X_proc, self.pipeline_ = validate_and_preprocess(
            X, clip_mad=c.clip_mad, pca_variance=c.pca_variance
        )
        self.n_samples_, self.n_features_ = X_proc.shape
        N, D = self.n_samples_, self.n_features_
        timings["preprocessing"] = time.time() - t0

        # Method parameters mapping
        method_params = {
            "kmeans": {"n_init": 5},
            "fcm": {"fuzzifier": c.fuzzifier},
            "dbscan": {
                "min_samples": c.dbscan_min_samples,
                "eps_quantile": c.dbscan_eps_quantile,
                "eps_scale": c.dbscan_eps_scale,
            },
            "hierarchical": {"linkage": c.hierarchical_linkage},
        }

        # 3 & 4. Baseline ensemble on unperturbed dataset
        t0 = time.time()
        self.partitions_: List[Dict[str, Any]] = []
        baseline_partitions = []
        for s in range(c.baseline_seeds):
            b_seed = c.random_state * 100 + s
            for m in c.methods:
                lbls, info = fit_base_clusterer(
                    m, X_proc, c.n_clusters, random_state=b_seed, params=method_params.get(m)
                )
                part_entry = {
                    "kind": "baseline",
                    "family": "baseline",
                    "level": 0.0,
                    "level_index": 0,
                    "rep": s,
                    "method": m,
                    "seed": b_seed,
                    "labels": lbls,
                    "row_indices": np.arange(N),
                    "col_indices": np.arange(D),
                    "info": info,
                }
                self.partitions_.append(part_entry)
                baseline_partitions.append(part_entry)
        timings["baseline_ensemble"] = time.time() - t0

        # Construct reference partition from unperturbed K-Means/Agglomerative consensus
        ref_entry = next(p for p in baseline_partitions if p["method"] == "kmeans")
        self.reference_labels_ = ref_entry["labels"]

        # 5, 6 & 7. Perturbation engine & repeated clustering
        t0 = time.time()
        for f in c.families:
            lvls = self.levels_.get(f, (0.1, 0.25, 0.5))
            for l_idx, lvl in enumerate(lvls):
                for r in range(c.repetitions):
                    p_seed = int(rng.randint(0, 1000000))
                    X_pert, row_idx, col_idx, pert_metrics = apply_perturbation(
                        X_proc, f, lvl, p_seed, batch=batch
                    )
                    for m in c.methods:
                        m_seed = p_seed + 1
                        lbls, info = fit_base_clusterer(
                            m, X_pert, c.n_clusters, random_state=m_seed, params=method_params.get(m)
                        )
                        # Partition similarity to reference
                        sub_ref = self.reference_labels_[row_idx]
                        p_metrics = partition_metrics(sub_ref, lbls)

                        self.partitions_.append({
                            "kind": "single",
                            "family": f,
                            "level": float(lvl),
                            "level_index": l_idx,
                            "rep": r,
                            "method": m,
                            "seed": m_seed,
                            "labels": lbls,
                            "row_indices": row_idx,
                            "col_indices": col_idx,
                            "features": X_pert,
                            "ari_ref": p_metrics["ari"],
                            "ami_ref": p_metrics["ami"],
                            "info": info,
                            "pert_metrics": pert_metrics,
                        })
        timings["perturbation_clustering"] = time.time() - t0

        # 8 & 9. Construct stability tensor S in [0, 1]^(N x N x P)
        t0 = time.time()
        self.tensor_ = StabilityTensor(n_samples=N, families=list(c.families))
        for p in self.partitions_:
            if p["kind"] == "single":
                # Only include members with minimum agreement
                if p.get("ari_ref", 1.0) >= c.min_member_agreement or p["method"] == "dbscan":
                    self.tensor_.accumulate(p["family"], p["labels"], p["row_indices"])
        self.tensor_.finalize(unobserved_prior=0.5)
        timings["tensor_construction"] = time.time() - t0

        # 10. Cluster-level stability fingerprints & variability
        t0 = time.time()
        self.fingerprint_ = compute_cluster_fingerprints(
            self.reference_labels_,
            self.tensor_,
            self.partitions_,
            c.n_clusters,
            X=X_proc,
        )
        timings["fingerprinting"] = time.time() - t0

        # 11. Evaluate factorial interactions
        t0 = time.time()
        pairs = c.interaction_pairs or [
            ("feature", "measurement"),
            ("sampling", "feature"),
            ("measurement", "distribution"),
            ("algorithmic", "technical"),
        ]
        self.interaction_ = pd.DataFrame()
        if c.interaction:
            # Run combined pairwise perturbation runs
            pair_stabilities: Dict[Tuple[str, str], Dict[int, float]] = {}
            single_stabilities: Dict[str, Dict[int, float]] = {}
            baseline_stabilities: Dict[int, float] = {}

            # Baseline stability
            for k in range(c.n_clusters):
                baseline_stabilities[k] = float(self.fingerprint_.loc[self.fingerprint_["cluster"] == k, "algorithmic"].values[0])

            # Single stabilities
            for f in c.families:
                single_stabilities[f] = {}
                for k in range(c.n_clusters):
                    single_stabilities[f][k] = float(self.fingerprint_.loc[self.fingerprint_["cluster"] == k, f].values[0])

            # Pair stabilities
            for (p, q) in pairs:
                p_lvl = self.levels_.get(p, (0.25,))[len(self.levels_.get(p, (0.25,))) // 2]
                q_lvl = self.levels_.get(q, (0.25,))[len(self.levels_.get(q, (0.25,))) // 2]
                comb_rets = []

                for r in range(c.interaction_repetitions):
                    comb_seed = int(rng.randint(0, 1000000))
                    # Apply P then Q
                    X_p, r_p, c_p, _ = apply_perturbation(X_proc, p, p_lvl, comb_seed, batch=batch)
                    X_pq, r_q, c_q, _ = apply_perturbation(X_p, q, q_lvl, comb_seed + 1, batch=batch)
                    
                    shared_rows = r_p[r_q]
                    for m in c.methods[:2]:  # Use fast representative methods
                        lbls, _ = fit_base_clusterer(m, X_pq, c.n_clusters, random_state=comb_seed)
                        sub_ref = self.reference_labels_[shared_rows]
                        from pccs.stability.matching import cluster_retention
                        ret = cluster_retention(sub_ref, lbls, c.n_clusters)
                        comb_rets.append(ret)

                mean_comb_ret = np.mean(comb_rets, axis=0) if comb_rets else np.full(c.n_clusters, 0.5)
                pair_stabilities[(p, q)] = {k: float(mean_comb_ret[k]) for k in range(c.n_clusters)}

            self.interaction_ = compute_pairwise_interactions(
                baseline_stabilities,
                single_stabilities,
                pair_stabilities,
                c.n_clusters,
                bootstrap_samples=50,
                seed=c.random_state,
            )
        timings["interaction_analysis"] = time.time() - t0

        # 12. Reliability classification
        t0 = time.time()
        self.fingerprint_ = classify_reliability(
            self.fingerprint_,
            list(c.families),
            hi=c.hi,
            lo=c.lo,
            max_response_sd=c.max_response_sd,
            interaction_threshold=c.interaction_threshold,
            interactions_df=self.interaction_,
        )
        timings["classification"] = time.time() - t0

        # 13 & 14. Perturbation-aware consensus construction
        t0 = time.time()
        self.labels_, self.consensus_matrix_, consensus_diag = build_perturbation_aware_consensus(
            self.tensor_,
            self.fingerprint_,
            self.partitions_,
            c.n_clusters,
            consistency_lambda=c.consistency_lambda,
            weight_power=c.weight_power,
            linkage_method=c.linkage,
            min_cluster_frac=c.min_cluster_frac,
        )
        timings["consensus_construction"] = time.time() - t0

        # Compute consensus cluster centroids for inductive prediction
        self.centroids_ = np.zeros((c.n_clusters, X_proc.shape[1]), dtype=float)
        for k in range(c.n_clusters):
            mask = (self.labels_ == k)
            if np.sum(mask) > 0:
                self.centroids_[k] = np.mean(X_proc[mask], axis=0)

        # Diagnostics & computational statistics
        timings["total_execution"] = time.time() - start_time
        self.computational_stats_ = {
            "timings_seconds": timings,
            "n_partitions": len(self.partitions_),
            "tensor_memory_mb": float((len(c.families) * N * N * 8) / (1024 * 1024)),
        }
        self.summary_ = {
            "n_samples": N,
            "n_features": D,
            "n_clusters": c.n_clusters,
            "consensus_diagnostics": consensus_diag,
            "categories_summary": self.fingerprint_["reliability_category"].value_counts().to_dict(),
        }

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Assign new observations to nearest consensus cluster centroid."""
        if not hasattr(self, "centroids_"):
            raise ValueError("Model is not fitted yet.")
        from scipy.spatial.distance import cdist
        X_proc = self.pipeline_.transform(X)
        dists = cdist(X_proc, self.centroids_)
        return np.argmin(dists, axis=1)

    def fit_predict(self, X: np.ndarray, y: Optional[np.ndarray] = None, batch: Optional[np.ndarray] = None) -> np.ndarray:
        return self.fit(X, y=y, batch=batch).labels_
