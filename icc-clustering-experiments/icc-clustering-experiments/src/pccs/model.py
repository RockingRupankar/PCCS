"""
PCCS - Perturbation-Conditional Consensus Clustering.
Unified implementation coordinating the 16-step pipeline defined in the authoritative PDF.
"""
from dataclasses import dataclass, field
import itertools
import time
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, adjusted_rand_score

from .data import PreprocessingPipeline, validate_and_preprocess
from .ensemble import METHODS, cluster_single, BaseClustererResult
from .perturbations import FAMILIES, DEFAULT_LEVELS, make_fresh_view, apply_perturbation, PerturbedDataView
from .matching import hungarian_match, partition_metrics, cluster_retention
from .tensor import StabilityTensor
from .fingerprints import compute_cluster_fingerprints, compute_psi
from .interactions import compute_interactions
from .classification import classify_reliability
from .consensus import build_perturbation_aware_consensus, _cluster_precomputed


@dataclass
class PCCSConfig:
    n_clusters: int = 4
    methods: tuple = METHODS
    families: tuple = FAMILIES
    levels: dict = field(default_factory=lambda: dict(DEFAULT_LEVELS))
    repetitions: int = 2
    baseline_seeds: int = 3
    fuzzifier: float = 1.3
    dbscan_min_samples: int = 5
    dbscan_eps_quantile: float = 0.6
    dbscan_eps_scale: float = 1.2
    n_batches: int = 3
    interaction: bool = True
    interaction_pairs: Optional[tuple] = None
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


class PCCS:
    """
    Perturbation-Conditional Consensus Clustering (PCCS) estimator.
    """
    def __init__(self, cfg: Optional[PCCSConfig] = None, **kw):
        self.cfg = cfg or PCCSConfig()
        for k, v in kw.items():
            setattr(self.cfg, k, v)

        self.fitted_ = False
        self.pipeline_: Optional[PreprocessingPipeline] = None
        self.Z_: Optional[np.ndarray] = None
        self.n_: int = 0
        self.d_: int = 0
        self.partitions_: List[Dict[str, Any]] = []
        self.active_methods_: Tuple[str, ...] = ()
        self.ref_: Optional[np.ndarray] = None
        self.labels_: Optional[np.ndarray] = None
        self.tensor_: Optional[StabilityTensor] = None
        self.fallback_: bool = False
        self.fingerprint_: Optional[pd.DataFrame] = None
        self.interaction_: Optional[pd.DataFrame] = None
        self.summary_: Dict[str, Any] = {}
        self.timings_: Dict[str, float] = {}
        self.n_clusterings_: int = 0

    def fit(self, X: Any, batch: Optional[np.ndarray] = None) -> "PCCS":
        c = self.cfg
        t_start = time.time()
        timings: Dict[str, float] = {}
        rng = np.random.default_rng(c.random_state)

        # Step 1: Preprocessing and fixed representation
        t0 = time.time()
        self.Z_, self.pipeline_ = validate_and_preprocess(X, clip_mad=c.clip_mad, pca_variance=c.pca_variance)
        self.n_, self.d_ = self.Z_.shape
        n, d = self.n_, self.d_
        timings["preprocessing"] = time.time() - t0

        # Step 2: Baseline ensemble on unperturbed data
        t0 = time.time()
        parts: List[Dict[str, Any]] = []
        for m in c.methods:
            for s in range(c.baseline_seeds):
                view = make_fresh_view(self.Z_)
                res: BaseClustererResult = cluster_single(
                    method=m,
                    X=view.X,
                    n_clusters=c.n_clusters,
                    seed=int(rng.integers(1_000_000_000)),
                    jitter=0.0,
                    fuzzifier=c.fuzzifier,
                    dbscan_min_samples=c.dbscan_min_samples,
                    dbscan_eps_quantile=c.dbscan_eps_quantile,
                    dbscan_eps_scale=c.dbscan_eps_scale,
                    hierarchical_linkage=c.hierarchical_linkage,
                )
                parts.append(dict(
                    kind="baseline",
                    family="baseline",
                    level_index=0,
                    level=0.0,
                    realized_intensity=0.0,
                    rep=s,
                    method=m,
                    rows=view.rows,
                    cols=view.cols,
                    labels=res.labels,
                    n_clusters=res.n_clusters,
                    noise_count=res.noise_count,
                    runtime=res.runtime,
                ))

        # Provisional baseline reference partition
        b_num = np.zeros((n, n), dtype=np.float32)
        b_den = np.zeros((n, n), dtype=np.float32)
        for p in parts:
            valid = p["labels"] >= 0
            v_rows = p["rows"][valid]
            v_labs = p["labels"][valid]
            if len(v_rows) > 0:
                ix = np.ix_(v_rows, v_rows)
                b_num[ix] += (v_labs[:, None] == v_labs[None, :]).astype(np.float32)
                b_den[ix] += 1.0
        Cb = np.where(b_den > 0, b_num / np.maximum(b_den, 1e-9), 0.5)
        np.fill_diagonal(Cb, 1.0)
        self.ref_, _ = _cluster_precomputed(Cb, c.n_clusters, c.linkage, c.min_cluster_frac, c.random_state)
        timings["baseline"] = time.time() - t0

        # Step 3: Single-family controlled perturbations
        t0 = time.time()
        for fam in c.families:
            for li, lev in enumerate(c.levels.get(fam, (0.0,))):
                for r in range(c.repetitions):
                    view = apply_perturbation(
                        make_fresh_view(self.Z_),
                        family=fam,
                        level=lev,
                        rep=r,
                        rng=rng,
                        n_clusters=c.n_clusters,
                        batch=batch,
                        n_batches=c.n_batches,
                    )
                    for m in c.methods:
                        res = cluster_single(
                            method=m,
                            X=view.X,
                            n_clusters=c.n_clusters,
                            seed=int(rng.integers(1_000_000_000)),
                            jitter=view.jitter,
                            fuzzifier=c.fuzzifier,
                            dbscan_min_samples=c.dbscan_min_samples,
                            dbscan_eps_quantile=c.dbscan_eps_quantile,
                            dbscan_eps_scale=c.dbscan_eps_scale,
                            hierarchical_linkage=c.hierarchical_linkage,
                        )
                        parts.append(dict(
                            kind="single",
                            family=fam,
                            level_index=li,
                            level=lev,
                            realized_intensity=view.realized_intensity,
                            rep=r,
                            method=m,
                            rows=view.rows,
                            cols=view.cols,
                            labels=res.labels,
                            n_clusters=res.n_clusters,
                            noise_count=res.noise_count,
                            runtime=res.runtime,
                        ))
        timings["single_perturbations"] = time.time() - t0

        # Step 4: Pairwise interaction perturbations
        t0 = time.time()
        self.pairs_ = []
        if c.interaction:
            pair_combos = c.interaction_pairs or tuple(itertools.combinations(c.families, 2))
            for (p_, q_) in pair_combos:
                lp_list = c.levels.get(p_, (0.0,))
                lq_list = c.levels.get(q_, (0.0,))
                lp = lp_list[len(lp_list) // 2]
                lq = lq_list[len(lq_list) // 2]
                for r in range(c.interaction_repetitions):
                    v1 = apply_perturbation(
                        make_fresh_view(self.Z_),
                        family=p_,
                        level=lp,
                        rep=r,
                        rng=rng,
                        n_clusters=c.n_clusters,
                        batch=batch,
                        n_batches=c.n_batches,
                    )
                    v2 = apply_perturbation(
                        v1,
                        family=q_,
                        level=lq,
                        rep=r,
                        rng=rng,
                        n_clusters=c.n_clusters,
                        batch=batch,
                        n_batches=c.n_batches,
                    )
                    for m in c.methods:
                        res = cluster_single(
                            method=m,
                            X=v2.X,
                            n_clusters=c.n_clusters,
                            seed=int(rng.integers(1_000_000_000)),
                            jitter=v2.jitter,
                            fuzzifier=c.fuzzifier,
                            dbscan_min_samples=c.dbscan_min_samples,
                            dbscan_eps_quantile=c.dbscan_eps_quantile,
                            dbscan_eps_scale=c.dbscan_eps_scale,
                            hierarchical_linkage=c.hierarchical_linkage,
                        )
                        parts.append(dict(
                            kind="pair",
                            family=f"{p_}+{q_}",
                            level_index=len(lp_list) // 2,
                            level=(lp, lq),
                            realized_intensity=0.5 * (v1.realized_intensity + v2.realized_intensity),
                            rep=r,
                            method=m,
                            rows=v2.rows,
                            cols=v2.cols,
                            labels=res.labels,
                            n_clusters=res.n_clusters,
                            noise_count=res.noise_count,
                            runtime=res.runtime,
                        ))
                self.pairs_.append((p_, q_))
        timings["pair_perturbations"] = time.time() - t0

        # Step 5: Evaluate partition metrics against candidate reference
        for p in parts:
            pm = partition_metrics(self.ref_[p["rows"]], p["labels"], mask_noise=True)
            p["ari_ref"] = pm["ARI"]
            p["ami_ref"] = pm["AMI"]
            p["nmi_ref"] = pm["NMI"]

        self.partitions_ = parts
        self.n_clusterings_ = len(parts)
        self.active_methods_ = tuple(c.methods)

        # Step 6: Construct stability tensor
        t0 = time.time()
        self.tensor_ = StabilityTensor(n).fit_from_partitions(parts, c.families)
        timings["tensor"] = time.time() - t0

        # Step 7: Candidate cluster stability fingerprints & interactions
        t0 = time.time()
        cand_fp = compute_cluster_fingerprints(self.ref_, self.Z_, parts, self.tensor_, c.families)
        cand_inter = compute_interactions(self.ref_, self.Z_, parts, self.pairs_, random_state=c.random_state) if c.interaction else pd.DataFrame()
        timings["candidate_fingerprints"] = time.time() - t0

        # Step 8: Build perturbation-aware consensus
        t0 = time.time()
        cluster_rel = 1.0 - cand_fp["PSI"].to_numpy() if c.use_reliability_weights else None
        self.labels_, self.consensus_matrix_, self.fallback_ = build_perturbation_aware_consensus(
            tensor=self.tensor_,
            n_clusters=c.n_clusters,
            families=c.families,
            cluster_reliabilities=cluster_rel,
            candidate_labels=self.ref_,
            consistency_lambda=c.consistency_lambda,
            family_weighting=c.family_weighting,
            linkage=c.linkage,
            min_cluster_frac=c.min_cluster_frac,
            random_state=c.random_state,
        )
        timings["consensus"] = time.time() - t0

        # Step 9: Final cluster fingerprints, interactions, and reliability classification
        t0 = time.time()
        self.fingerprint_ = compute_cluster_fingerprints(self.labels_, self.Z_, parts, self.tensor_, c.families)
        self.interaction_ = compute_interactions(self.labels_, self.Z_, parts, self.pairs_, random_state=c.random_state) if c.interaction else pd.DataFrame()
        self.fingerprint_ = classify_reliability(
            self.fingerprint_,
            families=c.families,
            interaction_df=self.interaction_,
            hi_threshold=c.hi,
            lo_threshold=c.lo,
            max_sd_threshold=c.max_response_sd,
            interaction_threshold=c.interaction_threshold,
        )
        self.summary_ = self._compute_summary()
        timings["analysis"] = time.time() - t0

        timings["total"] = time.time() - t_start
        self.timings_ = timings
        self.fitted_ = True
        return self

    def build_consensus(
        self,
        methods: Optional[Tuple[str, ...]] = None,
        families: Optional[Tuple[str, ...]] = None,
        per_family: bool = True,
        reliability_weights: Optional[bool] = None,
        family_weighting: Optional[str] = None,
        lam: Optional[float] = None,
        return_tensor: bool = False,
    ) -> Any:
        """
        Rebuild consensus from stored partitions (fast variant exploration / ablation).
        """
        c = self.cfg
        fams = families or c.families
        meths = methods or c.methods
        rw = c.use_reliability_weights if reliability_weights is None else reliability_weights
        fw = family_weighting or c.family_weighting
        l_val = c.consistency_lambda if lam is None else lam

        filtered_parts = [p for p in self.partitions_ if p["method"] in meths]
        temp_tensor = StabilityTensor(self.n_).fit_from_partitions(filtered_parts, fams)

        if not per_family:
            # Pooled unweighted consensus
            n = self.n_
            num = np.zeros((n, n), dtype=np.float32)
            den = np.zeros((n, n), dtype=np.float32)
            for p in filtered_parts:
                if p["kind"] == "single":
                    valid = p["labels"] >= 0
                    v_rows = p["rows"][valid]
                    v_labs = p["labels"][valid]
                    if len(v_rows) > 0:
                        ix = np.ix_(v_rows, v_rows)
                        num[ix] += (v_labs[:, None] == v_labs[None, :]).astype(np.float32)
                        den[ix] += 1.0
            C_pooled = np.where(den > 0, num / np.maximum(den, 1e-9), 0.5)
            np.fill_diagonal(C_pooled, 1.0)
            labels, fb = _cluster_precomputed(C_pooled, c.n_clusters, c.linkage, c.min_cluster_frac, c.random_state)
            return (labels, temp_tensor.S_family, fb) if return_tensor else labels

        cand_fp = compute_cluster_fingerprints(self.ref_, self.Z_, filtered_parts, temp_tensor, fams)
        cluster_rel = 1.0 - cand_fp["PSI"].to_numpy() if rw else None

        labels, fused_C, fb = build_perturbation_aware_consensus(
            tensor=temp_tensor,
            n_clusters=c.n_clusters,
            families=fams,
            cluster_reliabilities=cluster_rel,
            candidate_labels=self.ref_,
            consistency_lambda=l_val,
            family_weighting=fw,
            linkage=c.linkage,
            min_cluster_frac=c.min_cluster_frac,
            random_state=c.random_state,
        )
        return (labels, temp_tensor.S_family, fb) if return_tensor else labels

    def _compute_summary(self) -> Dict[str, Any]:
        c = self.cfg
        singles = [p for p in self.partitions_ if p["kind"] == "single"]
        fam_stab = {f: float(np.mean([p["ari_ref"] for p in singles if p["family"] == f])) for f in self.tensor_.S_family}
        meth_stab = {m: float(np.mean([p["ari_ref"] for p in singles if p["method"] == m])) for m in c.methods}

        C = self.consensus_matrix_
        same = (self.labels_[:, None] == self.labels_[None, :])
        np.fill_diagonal(same, False)

        q = dict(
            within_cluster_coassociation=float(C[same].mean()) if same.any() else 1.0,
            between_cluster_coassociation=float(C[~same & ~np.eye(self.n_, dtype=bool)].mean()) if (~same).any() else 0.0,
        )
        if len(np.unique(self.labels_)) > 1:
            q["consensus_silhouette"] = float(silhouette_score(1.0 - C, self.labels_, metric="precomputed"))

        return dict(
            family_stability_ari=fam_stab,
            method_stability_ari=meth_stab,
            consensus_quality=q,
            reference_vs_consensus_ari=float(adjusted_rand_score(self.ref_, self.labels_)),
            active_methods=list(self.active_methods_),
            used_spectral_fallback=self.fallback_,
        )

    def run_table(self) -> pd.DataFrame:
        """Table of all executed runs with parameters, realized intensity, and partition metrics."""
        rows = []
        for p in self.partitions_:
            r = dict(
                kind=p["kind"],
                family=p["family"],
                level_index=p["level_index"],
                level=str(p["level"]) if isinstance(p["level"], (tuple, list)) else p["level"],
                realized_intensity=p.get("realized_intensity", 0.0),
                rep=p["rep"],
                method=p["method"],
                n_rows=len(p["rows"]),
                n_cols=len(p["cols"]),
                n_clusters=p.get("n_clusters", 0),
                noise_count=p.get("noise_count", 0),
                ari_ref=p.get("ari_ref", 0.0),
                ami_ref=p.get("ami_ref", 0.0),
                nmi_ref=p.get("nmi_ref", 0.0),
                runtime=p.get("runtime", 0.0),
            )
            rows.append(r)
        return pd.DataFrame(rows)

    def computational_stats(self) -> Dict[str, Any]:
        n = self.n_
        tensor_mb = len(self.tensor_.S_family) * n * n * 4 / 1e6 if self.tensor_ else 0.0
        return dict(
            n_clusterings=self.n_clusterings_,
            timings_seconds=self.timings_,
            tensor_memory_MB=tensor_mb,
        )
