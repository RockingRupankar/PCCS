"""PCCS central question, tested as a falsification: is a perturbation-specific stability fingerprint a better
detector of real structure than one aggregate stability score?

For each dataset the PCCS perturbation engine (all six families, intensity levels, repetitions, k-means base method)
yields one stability value per family (mean ARI to the unperturbed reference partition).
Detectors of 'structured vs null' (higher stability = structured):
  scalar_mean   : mean over the six families (the single aggregate score criticised by the document)
  <family>      : one family alone
  min_family    : worst family
  fingerprint_logreg : logistic regression on the 6-vector (5-fold CV, repeated 5 times) - uses labels of the
                       *simulated* datasets only for this detector evaluation, never for clustering.
Output: data/exp0_pccs_fingerprint_features.csv and data/exp0_pccs_auc.csv (directed AUC; < 0.5 = null looks more stable).
"""
import sys
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from exp0_core import *

sys.path.insert(0, str(ROOT.parent / "src"))
from icc import PCCS, PCCSConfig, FAMILIES

DATA = ROOT / "exp0_falsification" / "data"
LEVELS, SEEDS = (0.5, 1.0, 1.5), range(20)


def main():
    rows = []
    for s in STRUCTURES:
        for lv in LEVELS:
            for sd in SEEDS:
                rng = np.random.default_rng(10_000 + sd + int(lv * 100) + 7 * STRUCTURES.index(s))
                X, y = make_structure(s, rng=rng)
                X = add_noise(X, "gaussian", lv, rng)
                m = PCCS(PCCSConfig(n_clusters=4, methods=("kmeans",), repetitions=2, baseline_seeds=2,
                                    interaction=False, random_state=sd)).fit(X)
                fam = m.summary_["family_stability_ari"]
                rows.append(dict(structure=s, level=lv, seed=sd, **fam, scalar_mean=float(np.mean(list(fam.values()))),
                                 min_family=float(np.min(list(fam.values())))))
        print(s, "done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(DATA / "exp0_pccs_fingerprint_features.csv", index=False)

    out = []
    for null in ("null_aniso", "null_iso"):
        for lv in LEVELS:
            d = df[(df.level == lv) & df.structure.isin(["structured", null])]
            yv = (d.structure == "structured").astype(int).to_numpy()
            for pred in ["scalar_mean", "min_family", *FAMILIES]:
                out.append(dict(task=f"structured_vs_{null}", level=lv, predictor=pred, auc=float(roc_auc_score(yv, d[pred]))))
            F = d[list(FAMILIES)].to_numpy()
            aucs = []
            for rep in range(5):
                cv = StratifiedKFold(5, shuffle=True, random_state=rep)
                p = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000)), F, yv, cv=cv, method="predict_proba")[:, 1]
                aucs.append(roc_auc_score(yv, p))
            out.append(dict(task=f"structured_vs_{null}", level=lv, predictor="fingerprint_logreg_cv", auc=float(np.mean(aucs))))
    res = pd.DataFrame(out)
    res.to_csv(DATA / "exp0_pccs_auc.csv", index=False)
    print(res.pivot_table(index="predictor", columns=["task", "level"], values="auc").round(2).to_string())


if __name__ == "__main__":
    main()
