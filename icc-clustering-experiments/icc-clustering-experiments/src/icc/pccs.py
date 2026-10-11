"""
PCCS - Perturbation-Conditional Consensus Clustering.

Implements the framework described in "Perturbation-Conditional Consensus Clustering: Problem Statement &
Approach" (sections cited below refer to that document).

Pipeline (document section 3 / algorithm steps 1-16)
  1. preprocessing with a fixed policy (median impute, winsorise, standardise)            [4.1, steps 1-2]
  2. heterogeneous base ensemble: k-means, fuzzy c-means, DBSCAN, hierarchical             [4.2, step 3]
  3. baseline ensemble on the unperturbed data -> reference (candidate) partition         [step 4]
  4. perturbation engine, six families with intensity levels and repetitions:
     algorithmic, sampling, feature, measurement, technical (batch-like), distribution    [4.3, steps 5-6]
  5. repeated clustering under every condition                                             [4.4, step 7]
  6. perturbation-specific stability tensor S[i, j, p] (one co-association matrix per
     family)                                                                               [4.5, step 9]
  7. cluster stability fingerprints F_k (membership retention, co-association
     consistency, centroid drift), variability, perturbation sensitivity index PSI        [4.6, step 10]
  8. perturbation interaction analysis (observed combined vs expected from single effects,
     expected = S_p * S_q / S_0 with S_0 the unperturbed reproducibility)                    [4.7, step 11]
  9. descriptive reliability categories                                                    [4.8, step 12]
 10. perturbation-aware weighted consensus (partition weights from agreement with the
     reference, family weights from family reliability, consistency across families)      [4.9, steps 13-14]
 11. final output: labels, fingerprints, PSI, interactions, categories, global summaries,
     consensus quality and computational statistics                                        [4.10, step 15]

Documented deviations from the document (see docs/pccs_traceability.md):
  * with sampling-type perturbations a pair of rows is not always observed together, so co-association
    is normalised by the (weighted) number of runs that observed the pair instead of 1/(M R);
  * the fingerprint entry S_kp is the mean of membership retention and co-association consistency
    (centroid drift is reported separately); family weights w_p in PSI are equal by default;
  * the technical family simulates batch-like effects (random batch assignment unless a batch vector is given);
  * the distribution family is a controlled covariate shift (importance resampling along a random direction);
  * reliability thresholds, the consistency blend and the interaction threshold are design choices.
"""
from dataclasses import dataclass, field
import itertools
import time
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans, SpectralClustering
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.neighbors import NearestNeighbors

from .model import preprocess_fit, _agglo_precomputed

FAMILIES = ("algorithmic", "sampling", "feature", "measurement", "technical", "distribution")
DEFAULT_LEVELS = {
    "algorithmic": (0.0, 1.0),          # 0 = seeds only, 1 = seeds + hyper-parameter jitter
    "sampling": (0.1, 0.3, 0.5),        # fraction of rows removed
    "feature": (0.2, 0.5, 0.8),         # fraction of features masked
    "measurement": (0.25, 0.5, 1.0),    # noise sd / outlier / missingness intensity (see _perturb)
    "technical": (0.1, 0.25, 0.5),      # batch offset sd (standardised units)
    "distribution": (0.5, 1.0, 2.0),    # covariate-shift strength (exponential tilting)
}
METHODS = ("kmeans", "fcm", "dbscan", "hierarchical")


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
    interaction_pairs: tuple = None          # None = all pairs of families
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
    min_member_agreement: float = 0.2        # methods whose unperturbed ARI to the reference is lower carry no perturbation information
    interaction_threshold: float = 0.15
    random_state: int = 0


# ----------------------------------------------------------------------------- helpers
def _singletonize(lab):
    """Noise label (-1) -> each noise point becomes its own cluster; relabel to 0..L-1."""
    lab = np.asarray(lab).copy()
    neg = lab < 0
    if neg.any():
        lab[neg] = lab.max() + 1 + np.arange(neg.sum())
    return np.unique(lab, return_inverse=True)[1]


def fcm(X, c, m, rng, iters=50, tol=1e-4):
    """Fuzzy c-means (log-domain memberships); returns hard labels and memberships."""
    cent = X[rng.choice(len(X), c, replace=False)].copy()
    p = 1.0 / (m - 1.0)
    U = None
    for _ in range(iters):
        d2 = ((X[:, None, :] - cent[None]) ** 2).sum(2) + 1e-12
        logit = -p * np.log(d2)
        logit -= logit.max(1, keepdims=True)
        U = np.exp(logit)
        U /= U.sum(1, keepdims=True)
        W = U ** m
        new = (W.T @ X) / W.sum(0)[:, None]
        done = np.abs(new - cent).max() < tol
        cent = new
        if done:
            break
    return U.argmax(1), U


def _responses(ref_full, Z, part, K):
    """Per reference cluster: membership retention (best-match Jaccard on observed rows) and centroid drift."""
    rows, lab, cols = part["rows"], part["labels"], part["cols"]
    ref = ref_full[rows]
    L = int(lab.max()) + 1
    key = ref * L + lab
    uk, cnt = np.unique(key, return_counts=True)
    kk, ll = uk // L, uk % L
    nk = np.bincount(ref, minlength=K)
    nl = np.bincount(lab, minlength=L)
    jac = cnt / (nk[kk] + nl[ll] - cnt)
    ret = np.full(K, np.nan)
    best = -np.ones(K, int)
    order = np.argsort(jac)
    ret[kk[order]] = jac[order]
    best[kk[order]] = ll[order]
    ret[nk == 0] = np.nan
    Zc = Z[:, cols]
    cref = np.vstack([Zc[rows[ref == k]].mean(0) if nk[k] else np.zeros(Zc.shape[1]) for k in range(K)])
    sep = np.mean([np.linalg.norm(cref[i] - cref[j]) for i in range(K) for j in range(i + 1, K)]) + 1e-9
    drift = np.full(K, np.nan)
    for k in range(K):
        if best[k] >= 0 and nk[k]:
            drift[k] = np.linalg.norm(cref[k] - Zc[rows[lab == best[k]]].mean(0)) / sep
    return ret, drift


# ----------------------------------------------------------------------------- model
class PCCS:
    def __init__(self, cfg=None, **kw):
        self.cfg = cfg or PCCSConfig()
        for k, v in kw.items():
            setattr(self.cfg, k, v)

    # ------------------------------------------------------------------ perturbation engine T_{p,r}
    def _perturb(self, st, family, level, rng, rep, batch):
        c = self.cfg
        rows, cols, V = st["rows"], st["cols"], st["V"]
        n, d = V.shape
        if family == "algorithmic":
            st["jitter"] = level
        elif family == "sampling":
            keep = max(int(round((1 - level) * n)), c.n_clusters * 5)
            pos = np.sort(rng.choice(n, keep, replace=False))
            st["rows"], st["V"] = rows[pos], V[pos]
        elif family == "feature":
            keep = max(int(round((1 - level) * d)), 2)
            pos = np.sort(rng.choice(d, keep, replace=False))
            st["cols"], st["V"] = cols[pos], V[:, pos]
        elif family == "measurement":
            kind = ("gaussian", "heavy_tailed", "outliers", "missing")[rep % 4]
            V = V.copy()
            if kind == "gaussian":
                V += rng.normal(0, level, V.shape)
            elif kind == "heavy_tailed":
                V += level * rng.standard_t(3, V.shape) / np.sqrt(3)
            elif kind == "outliers":
                for r in rng.choice(n, max(int(0.05 * level * n), 1), replace=False):
                    cc = rng.choice(d, min(3, d), replace=False)
                    V[r, cc] = rng.choice([-1, 1], len(cc)) * rng.uniform(6, 10, len(cc))
            else:
                M = rng.random(V.shape) < 0.2 * level
                mu = np.where(M, np.nan, V)
                mu = np.nanmean(mu, 0)
                V = np.where(M, np.where(np.isnan(mu), 0, mu), V)
            st["V"] = V
        elif family == "technical":
            b = (np.asarray(batch)[rows] if batch is not None else rng.integers(0, c.n_batches, n))
            b = np.unique(b, return_inverse=True)[1]
            B = int(b.max()) + 1
            off = rng.normal(0, level, (B, d))
            scl = np.clip(1 + 0.5 * level * rng.normal(size=(B, d)), 0.2, None)
            st["V"] = V * scl[b] + off[b]
        elif family == "distribution":
            u = V @ (lambda v: v / np.linalg.norm(v))(rng.normal(size=d))
            u = (u - u.mean()) / (u.std() + 1e-9)
            p = np.exp(np.clip(level * u, -30, 30))
            p /= p.sum()
            keep = max(int(0.7 * n), c.n_clusters * 5)
            pos = np.sort(rng.choice(n, keep, replace=False, p=p))
            st["rows"], st["V"] = rows[pos], V[pos]
        else:
            raise ValueError(family)
        return st

    # ------------------------------------------------------------------ base clusterers (theta_m jittered by `jitter`)
    def _cluster(self, method, V, jitter, rng):
        c, K = self.cfg, self.cfg.n_clusters
        seed = int(rng.integers(1_000_000_000))
        if method == "kmeans":
            n_init = int(rng.choice([1, 3])) if jitter > 0 else 1
            init = "random" if (jitter > 0 and rng.random() < 0.5) else "k-means++"
            return KMeans(K, n_init=n_init, init=init, random_state=seed).fit_predict(V)
        if method == "fcm":
            m = c.fuzzifier + (jitter * rng.uniform(-0.15, 0.25) if jitter > 0 else 0.0)
            return fcm(V, K, max(m, 1.05), np.random.default_rng(seed))[0]
        if method == "dbscan":
            ms = c.dbscan_min_samples + (int(rng.integers(-1, 3)) if jitter > 0 else 0)
            scale = c.dbscan_eps_scale * (1 + (jitter * rng.uniform(-0.2, 0.2) if jitter > 0 else 0))
            dist = NearestNeighbors(n_neighbors=ms + 1).fit(V).kneighbors(V)[0][:, -1]
            eps = float(np.quantile(dist, c.dbscan_eps_quantile) * scale)
            return _singletonize(DBSCAN(eps=eps, min_samples=ms).fit_predict(V))
        if method == "hierarchical":
            link = str(rng.choice(["ward", "average", "complete"])) if jitter > 0 else "ward"
            return AgglomerativeClustering(K, linkage=link).fit_predict(V)
        raise ValueError(method)

    def _fresh(self, Z):
        n, d = Z.shape
        return dict(rows=np.arange(n), cols=np.arange(d), V=Z, jitter=0.0)

    def _consensus_labels(self, C):
        c = self.cfg
        n = C.shape[0]
        lab = _agglo_precomputed(1.0 - C, c.n_clusters, c.linkage)
        fallback = False
        if np.bincount(lab, minlength=c.n_clusters).min() < c.min_cluster_frac * n:
            lab = SpectralClustering(c.n_clusters, affinity="precomputed", random_state=c.random_state).fit_predict(np.clip(C, 0, 1))
            fallback = True
        return lab, fallback

    # ------------------------------------------------------------------ fit
    def fit(self, X, batch=None):
        c = self.cfg
        t_all = time.time()
        rng = np.random.default_rng(c.random_state)
        timings = {}
        t = time.time()
        Z, self.prep_ = preprocess_fit(X)
        self.Z_, self.n_ = Z, Z.shape[0]
        K = c.n_clusters

        # baseline ensemble on unperturbed data -> reference partition (candidate clusters)
        parts = []
        for m in c.methods:
            for s in range(c.baseline_seeds):
                st = self._fresh(Z)
                lab = self._cluster(m, st["V"], 0.0, rng)
                parts.append(dict(kind="baseline", family="baseline", level_index=0, level=0.0, rep=s, method=m,
                                  rows=st["rows"], cols=st["cols"], labels=lab))
        n = self.n_
        Cb = np.zeros((n, n), np.float32)
        for p in parts:
            Cb += (p["labels"][:, None] == p["labels"][None, :])
        Cb /= len(parts)
        self.ref_, _ = self._consensus_labels(Cb)
        del Cb
        timings["baseline"] = time.time() - t

        # single-family perturbations
        t = time.time()
        for fam in c.families:
            for li, lev in enumerate(c.levels[fam]):
                for r in range(c.repetitions):
                    st = self._perturb(self._fresh(Z), fam, lev, rng, r, batch)
                    for m in c.methods:
                        parts.append(dict(kind="single", family=fam, level_index=li, level=lev, rep=r, method=m,
                                          rows=st["rows"], cols=st["cols"], labels=self._cluster(m, st["V"], st["jitter"], rng)))
        timings["single_perturbations"] = time.time() - t

        # pairwise combinations (interaction analysis)
        t = time.time()
        self.pairs_ = []
        if c.interaction:
            pairs = c.interaction_pairs or tuple(itertools.combinations(c.families, 2))
            for (p_, q_) in pairs:
                lp = c.levels[p_][len(c.levels[p_]) // 2]
                lq = c.levels[q_][len(c.levels[q_]) // 2]
                for r in range(c.interaction_repetitions):
                    st = self._perturb(self._fresh(Z), p_, lp, rng, r, batch)
                    st = self._perturb(st, q_, lq, rng, r, batch)
                    for m in c.methods:
                        parts.append(dict(kind="pair", family=f"{p_}+{q_}", level_index=len(c.levels[p_]) // 2, level=(lp, lq),
                                          rep=r, method=m, rows=st["rows"], cols=st["cols"],
                                          labels=self._cluster(m, st["V"], st["jitter"], rng)))
                self.pairs_.append((p_, q_))
        timings["pair_perturbations"] = time.time() - t

        for p in parts:
            p["ari_ref"] = float(adjusted_rand_score(self.ref_[p["rows"]], p["labels"]))
        self.partitions_ = parts
        self.n_clusterings_ = len(parts)
        base_agree = {m: float(np.mean([p["ari_ref"] for p in parts if p["kind"] == "baseline" and p["method"] == m])) for m in c.methods}
        self.baseline_agreement_ = base_agree
        self.active_methods_ = tuple(m for m in c.methods if base_agree[m] >= c.min_member_agreement) or tuple(c.methods)

        # perturbation-aware consensus + stability tensor
        t = time.time()
        self.labels_, self.tensor_, self.fallback_ = self.build_consensus(return_tensor=True)
        timings["consensus"] = time.time() - t

        # fingerprints, interactions, categories, summaries
        t = time.time()
        self.fingerprint_ = self.fingerprints(self.labels_)
        self.interaction_ = self.interactions(self.labels_) if c.interaction else pd.DataFrame()
        self._classify()
        self.summary_ = self._summary()
        timings["analysis"] = time.time() - t
        timings["total"] = time.time() - t_all
        self.timings_ = timings
        return self

    # ------------------------------------------------------------------ tensor + consensus (cheap; reuses stored partitions)
    def build_consensus(self, methods=None, families=None, per_family=True, reliability_weights=None,
                        family_weighting=None, lam=None, power=None, return_tensor=False):
        c = self.cfg
        methods = methods or c.methods
        families = families or c.families
        rw = c.use_reliability_weights if reliability_weights is None else reliability_weights
        fw = family_weighting or c.family_weighting
        lam = c.consistency_lambda if lam is None else lam
        power = c.weight_power if power is None else power
        n = self.n_
        S, alpha, num_tot, den_tot = {}, {}, np.zeros((n, n), np.float32), np.zeros((n, n), np.float32)
        for fam in families:
            num = np.zeros((n, n), np.float32)
            den = np.zeros((n, n), np.float32)
            ar = []
            for p in self.partitions_:
                if p["kind"] != "single" or p["family"] != fam or p["method"] not in methods:
                    continue
                w = max(p["ari_ref"], 1e-3) ** power if rw else 1.0
                ix = np.ix_(p["rows"], p["rows"])
                num[ix] += w * (p["labels"][:, None] == p["labels"][None, :])
                den[ix] += w
                ar.append(p["ari_ref"])
            if not ar:
                continue
            S[fam] = np.where(den > 0, num / np.maximum(den, 1e-9), 0).astype(np.float32)
            alpha[fam] = max(float(np.mean(ar)), 1e-3) if fw == "reliability" else 1.0
            num_tot += num
            den_tot += den
        fams = list(S)
        if per_family:
            a = np.array([alpha[f] for f in fams])
            a = a / a.sum()
            C = sum(ai * S[f] for ai, f in zip(a, fams))
            if lam > 0 and len(fams) > 1:
                C = (1 - lam) * C + lam * np.minimum.reduce([S[f] for f in fams])
        else:
            C = np.where(den_tot > 0, num_tot / np.maximum(den_tot, 1e-9), 0)
        C = ((C + C.T) / 2).astype(np.float32)
        np.fill_diagonal(C, 1.0)
        self._last_consensus = C
        lab, fb = self._consensus_labels(C)
        return (lab, S, fb) if return_tensor else lab

    # ------------------------------------------------------------------ fingerprints (4.6)
    def fingerprints(self, labels):
        c, K = self.cfg, self.cfg.n_clusters
        fams = list(self.tensor_)
        resp = {}
        for i, p in enumerate(self.partitions_):
            if p["kind"] == "single":
                resp[i] = _responses(labels, self.Z_, p, K)
        rows = []
        for k in range(K):
            idx = np.where(labels == k)[0]
            r = dict(cluster=k, size=int(len(idx)))
            ssd = []
            for f in fams:
                ids = [i for i, p in enumerate(self.partitions_) if p["kind"] == "single" and p["family"] == f and p["method"] in self.active_methods_]
                ret = np.array([resp[i][0][k] for i in ids])
                dr = np.array([resp[i][1][k] for i in ids])
                if len(idx) > 1:
                    sub = self.tensor_[f][np.ix_(idx, idx)]
                    coas = float((sub.sum() - len(idx)) / (len(idx) * (len(idx) - 1)))
                else:
                    coas = np.nan
                r[f] = float(np.nanmean([np.nanmean(ret), coas]))      # S_kp
                r[f"retention_{f}"] = float(np.nanmean(ret))
                r[f"coassoc_{f}"] = coas
                r[f"drift_{f}"] = float(np.nanmean(dr))
                r[f"sd_{f}"] = float(np.nanstd(ret))
                ssd.append(r[f"sd_{f}"])
            r["max_response_sd"] = float(np.max(ssd))
            vals = np.array([r[f] for f in fams])
            r["PSI"] = float(np.mean(1 - vals))                        # equal family weights w_p = 1/P
            rows.append(r)
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------ interactions (4.7)
    def interactions(self, labels):
        c, K = self.cfg, self.cfg.n_clusters
        mid = {f: len(c.levels[f]) // 2 for f in c.families}
        resp = {}
        for i, p in enumerate(self.partitions_):
            if p["kind"] in ("single", "pair", "baseline"):
                resp[i] = _responses(labels, self.Z_, p, K)[0]

        def mean_ret(kind, fam, li=None):
            ids = [i for i, p in enumerate(self.partitions_) if p["kind"] == kind and p["family"] == fam and p["method"] in self.active_methods_ and (li is None or p["level_index"] == li)]
            M = np.vstack([resp[i] for i in ids])
            ari = np.mean([self.partitions_[i]["ari_ref"] for i in ids])
            return np.nanmean(M, 0), ari

        s0, a0 = mean_ret("baseline", "baseline")          # reproducibility without any perturbation
        out = []
        for (p_, q_) in self.pairs_:
            sp, ap = mean_ret("single", p_, mid[p_])
            sq, aq = mean_ret("single", q_, mid[q_])
            spq, apq = mean_ret("pair", f"{p_}+{q_}")
            for k in range(K):
                exp_k = sp[k] * sq[k] / max(s0[k], 1e-6)
                out.append(dict(pair=f"{p_}+{q_}", cluster=k, S_0=s0[k], S_p=sp[k], S_q=sq[k], S_pq_observed=spq[k],
                                S_pq_expected=exp_k, interaction=exp_k - spq[k]))
            exp_g = ap * aq / max(a0, 1e-6)
            out.append(dict(pair=f"{p_}+{q_}", cluster="global", S_0=a0, S_p=ap, S_q=aq, S_pq_observed=apq,
                            S_pq_expected=exp_g, interaction=exp_g - apq))
        return pd.DataFrame(out)

    # ------------------------------------------------------------------ reliability categories (4.8)
    def _classify(self):
        c = self.cfg
        fams = list(self.tensor_)
        fp = self.fingerprint_
        inter_max = np.zeros(len(fp))
        if len(self.interaction_):
            ik = self.interaction_[self.interaction_.cluster != "global"].copy()
            ik["cluster"] = ik["cluster"].astype(int)
            m = ik.groupby("cluster").interaction.max()
            inter_max = np.array([m.get(k, 0.0) for k in fp.cluster])
        fp["max_interaction"] = inter_max
        cats = []
        for i, row in fp.iterrows():
            v = np.array([row[f] for f in fams])
            n_hi = int((v >= c.hi).sum())
            if n_hi == len(fams) and row["max_response_sd"] <= c.max_response_sd and inter_max[i] < c.interaction_threshold:
                cats.append("globally_invariant")
            elif n_hi == len(fams) or (n_hi >= 1 and np.nanmin(v) >= c.lo):
                cats.append("conditionally_invariant")
            elif n_hi >= 1:
                cats.append("perturbation_fragile")
            else:
                cats.append("ambiguous")
        fp["reliability"] = cats

    # ------------------------------------------------------------------ outputs (4.10)
    def run_table(self, labels=None):
        """One row per stored partition with partition-level agreement and mean membership retention."""
        labels = self.labels_ if labels is None else labels
        rows = []
        for p in self.partitions_:
            r = dict(kind=p["kind"], family=p["family"], level_index=p["level_index"],
                     level=str(p["level"]) if p["kind"] == "pair" else p["level"], rep=p["rep"], method=p["method"],
                     n_rows=len(p["rows"]), n_cols=len(p["cols"]), ari_ref=p["ari_ref"])
            if p["kind"] != "baseline":
                r["retention_mean"] = float(np.nanmean(_responses(labels, self.Z_, p, self.cfg.n_clusters)[0]))
            rows.append(r)
        return pd.DataFrame(rows)

    def _summary(self):
        c = self.cfg
        singles = [p for p in self.partitions_ if p["kind"] == "single"]
        fam_stab = {f: float(np.mean([p["ari_ref"] for p in singles if p["family"] == f])) for f in self.tensor_}
        meth_stab = {m: float(np.mean([p["ari_ref"] for p in singles if p["method"] == m])) for m in c.methods}
        C = self._last_consensus
        same = self.labels_[:, None] == self.labels_[None, :]
        np.fill_diagonal(same, False)
        q = dict(within_cluster_coassociation=float(C[same].mean()),
                 between_cluster_coassociation=float(C[~same & ~np.eye(self.n_, dtype=bool)].mean()))
        if len(np.unique(self.labels_)) > 1:
            q["consensus_silhouette"] = float(silhouette_score(1 - C, self.labels_, metric="precomputed"))
        return dict(family_stability_ari=fam_stab, method_stability_ari=meth_stab, consensus_quality=q,
                    reference_vs_consensus_ari=float(adjusted_rand_score(self.ref_, self.labels_)),
                    baseline_agreement_ari=self.baseline_agreement_, active_methods=list(self.active_methods_),
                    used_spectral_fallback=self.fallback_)

    def computational_stats(self):
        n = self.n_
        return dict(n_clusterings=self.n_clusterings_, timings_seconds=self.timings_,
                    tensor_memory_MB=len(self.tensor_) * n * n * 4 / 1e6)
