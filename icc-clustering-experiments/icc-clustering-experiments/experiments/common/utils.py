"""Shared metrics and small helpers."""
import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, mutual_info_score


def vi(a, b):
    def H(l):
        _, c = np.unique(l, return_counts=True)
        p = c / c.sum()
        return -(p * np.log(p)).sum()
    return float(H(a) + H(b) - 2 * mutual_info_score(a, b))


def scores(y, lab):
    return dict(ARI=float(adjusted_rand_score(y, lab)), NMI=float(normalized_mutual_info_score(y, lab)), VI=vi(y, lab))


def pairwise_ari(label_sets):
    """Mean pairwise ARI among partitions given as full-length vectors (-1 = not observed)."""
    L = [np.asarray(l) for l in label_sets]
    vals = []
    for i in range(len(L)):
        for j in range(i + 1, len(L)):
            m = (L[i] >= 0) & (L[j] >= 0)
            if m.sum() > 1:
                vals.append(adjusted_rand_score(L[i][m], L[j][m]))
    return float(np.mean(vals)) if vals else np.nan
