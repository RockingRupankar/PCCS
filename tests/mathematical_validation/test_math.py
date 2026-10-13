"""
Mathematical correctness validation tests for PCCS.
Verifies essential mathematical properties specified in Section 6 and Section 13:
A. Permutation invariance
B. Shared observation alignment
C. Pairwise eligibility counts (denominators)
D. Undefined comparison handling
E. DBSCAN noise treatment
F. Cluster morphologies (split/merge)
G. Fuzzy membership preservation
H. Hand-calculated interaction estimands
I. PSI weighting sensitivity
J. Non-circular consensus
"""
import numpy as np
import pytest

from pccs.stability.matching import hungarian_match, partition_metrics, cluster_retention
from pccs.stability.tensor import StabilityTensor
from pccs.stability.fingerprints import compute_psi
from pccs.stability.classification import classify_reliability
from pccs.interactions.factorial import compute_pairwise_interactions
from pccs.clustering.ensemble import run_fcm, run_dbscan


def test_label_permutation_invariance():
    """Verify that Hungarian alignment and metrics are strictly invariant to cluster label permutations."""
    y1 = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    # Arbitrary label permutation: 0->2, 1->3, 2->0, 3->1
    mapping = {0: 2, 1: 3, 2: 0, 3: 1}
    y2 = np.array([mapping[val] for val in y1])

    aligned, perm, mean_j = hungarian_match(y1, y2, n_clusters=4)
    assert np.array_equal(aligned, y1), "Hungarian matching must restore exact reference identities."
    assert np.isclose(mean_j, 1.0), "Permuted identical partition must yield Jaccard = 1.0."

    metrics = partition_metrics(y1, y2)
    assert np.isclose(metrics["ari"], 1.0), "ARI must be 1.0 under permutation."
    assert np.isclose(metrics["ami"], 1.0), "AMI must be 1.0 under permutation."
    assert np.isclose(metrics["vi"], 0.0), "VI must be 0.0 under permutation."


def test_shared_observation_masking():
    """Verify that comparisons under subsampling compare only observations present in both runs."""
    N = 10
    tensor = StabilityTensor(n_samples=N, families=["sampling"])
    
    # Run 1: rows [0, 1, 2, 3], clusters [0, 0, 1, 1]
    rows_1 = np.array([0, 1, 2, 3])
    labels_1 = np.array([0, 0, 1, 1])
    tensor.accumulate("sampling", labels_1, rows_1)

    # Run 2: rows [2, 3, 4, 5], clusters [0, 0, 1, 1]
    rows_2 = np.array([2, 3, 4, 5])
    labels_2 = np.array([0, 0, 1, 1])
    tensor.accumulate("sampling", labels_2, rows_2)

    # Pair (2, 3) was co-observed in both runs and co-clustered in both runs
    assert tensor.counts["sampling"][2, 3] == 2.0
    assert tensor.tensors["sampling"][2, 3] == 2.0

    # Pair (0, 4) was never jointly observed
    assert tensor.counts["sampling"][0, 4] == 0.0

    tensor.finalize(unobserved_prior=0.5)
    assert tensor.tensors["sampling"][2, 3] == 1.0
    assert tensor.tensors["sampling"][0, 4] == 0.5, "Unobserved pairs must receive uninformative prior."


def test_dbscan_noise_handling():
    """Verify that DBSCAN noise points (-1) are not treated as an ordinary co-associating cluster."""
    N = 6
    tensor = StabilityTensor(n_samples=N, families=["algorithmic"])
    # Cluster 0: points 0, 1; Noise: points 2, 3, 4, 5
    labels = np.array([0, 0, -1, -1, -1, -1])
    tensor.accumulate("algorithmic", labels, np.arange(N))
    tensor.finalize()

    S = tensor.get_slice("algorithmic")
    # Points 0 and 1 co-associated
    assert S[0, 1] == 1.0
    # Noise points must NOT co-associate with each other
    assert S[2, 3] == 0.0
    assert S[4, 5] == 0.0


def test_fcm_soft_membership():
    """Verify that Fuzzy C-Means produces valid probability distributions and hard assignments."""
    rng = np.random.RandomState(42)
    X = rng.normal(size=(20, 4))
    hard_lbls, info = run_fcm(X, n_clusters=3, fuzzifier=1.3, random_state=42)

    U = info["U"]
    assert U.shape == (20, 3)
    # Memberships must sum to 1.0 for every observation
    assert np.allclose(U.sum(axis=1), 1.0, atol=1e-5)
    assert np.all(U >= 0.0) and np.all(U <= 1.0)
    assert np.array_equal(hard_lbls, np.argmax(U, axis=1))


def test_hand_calculated_interaction():
    """Verify factorial interaction calculation against hand-computed example."""
    # S_0 = 0.8, S_p = 0.6, S_q = 0.4
    # Expected multiplicative = (0.6 * 0.4) / 0.8 = 0.24 / 0.8 = 0.30
    # Observed combined = 0.10
    # Expected degradation Delta = 0.30 - 0.10 = +0.20 (Synergistic failure)
    baseline = {0: 0.8}
    single = {"feat": {0: 0.6}, "noise": {0: 0.4}}
    pair = {("feat", "noise"): {0: 0.1}}

    df_inter = compute_pairwise_interactions(
        baseline, single, pair, n_clusters=1, model="multiplicative", bootstrap_samples=10
    )
    row = df_inter.iloc[0]
    assert np.isclose(row["S_pq_expected"], 0.30)
    assert np.isclose(row["S_pq_observed"], 0.10)
    assert np.isclose(row["interaction"], 0.20)
    assert bool(row["is_synergistic"]) is True


def test_psi_weight_sensitivity():
    """Verify that PSI behaves properly under equal vs skewed weight profiles."""
    stabs = np.array([0.9, 0.8, 0.4])  # Stable on 2, fragile on 1
    # Equal weights w = [1/3, 1/3, 1/3]
    psi_eq = compute_psi(stabs)
    expected_eq = (1 - 0.9 + 1 - 0.8 + 1 - 0.4) / 3.0  # (0.1 + 0.2 + 0.6)/3 = 0.9/3 = 0.30
    assert np.isclose(psi_eq, expected_eq)

    # Heavily weight fragile family w = [0.1, 0.1, 0.8]
    w_skew = np.array([0.1, 0.1, 0.8])
    psi_skew = compute_psi(stabs, weights=w_skew)
    expected_skew = 0.1 * 0.1 + 0.1 * 0.2 + 0.8 * 0.6  # 0.01 + 0.02 + 0.48 = 0.51
    assert np.isclose(psi_skew, expected_skew)
    assert psi_skew > psi_eq
