"""Main sweep: Gaussian noise level x structure (-> exp0_agg.csv) and FCM fuzzifier curve (-> exp0_fcm_full_curve.csv)."""
import numpy as np, pandas as pd
from sklearn.metrics import adjusted_rand_score
from exp0_core import *

DATA = ROOT / "exp0_falsification" / "data"
LEVELS = [0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0]
SEEDS = range(20)


def main():
    rows = [run_cell(s, "gaussian", lv, sd) for s in STRUCTURES for lv in LEVELS for sd in SEEDS]
    raw = pd.DataFrame(rows)
    raw.to_csv(DATA / "exp0_raw.csv", index=False)
    agg = (raw.groupby(["structure", "level"])
           .agg(stab_seed=("stab_seed", "mean"), stab_seed_sd=("stab_seed", "std"),
                stab_boot=("stab_boot", "mean"), stab_boot_sd=("stab_boot", "std"),
                silhouette=("silhouette", "mean"), ari_truth=("ari_truth", "mean"),
                high_stab_rate=("stab_boot", lambda v: float((v >= 0.8).mean())), n_runs=("seed", "count"))
           .reset_index())
    agg.to_csv(DATA / "exp0_agg.csv", index=False)
    print(agg.round(3).to_string())

    # ---- fuzzy c-means: fuzzifier m sweep at several dimensions (phase-transition curve)
    out = []
    for d in (10, 50, 200):
        for sigma in (0.0, 1.0):
            for m in (1.05, 1.1, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 6.0, 10.0):
                for sd in range(8):
                    rng = np.random.default_rng(1000 + sd)
                    X, y = make_structure("structured", n=300, d=d, d_inf=5, rng=rng)
                    X = add_noise(X, "gaussian", sigma, rng)
                    U, C = fcm(X, 4, m, rng=rng)
                    ent = float(np.mean(-(U * np.log(U + 1e-12)).sum(1) / np.log(4)))
                    spread = float(np.mean([np.linalg.norm(C[i] - C[j]) for i in range(4) for j in range(i + 1, 4)]))
                    out.append(dict(d=d, sigma=sigma, m=m, seed=sd, ari=float(adjusted_rand_score(y, U.argmax(1))),
                                    membership_entropy=ent, center_spread=spread,
                                    mean_max_membership=float(U.max(1).mean())))
    fc = pd.DataFrame(out).groupby(["d", "sigma", "m"]).mean(numeric_only=True).reset_index().drop(columns="seed")
    fc.to_csv(DATA / "exp0_fcm_full_curve.csv", index=False)
    print(fc[fc.sigma == 0].round(3).to_string())


if __name__ == "__main__":
    main()
