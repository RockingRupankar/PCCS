"""PCCS on the real datasets (iris, wine, breast cancer; original and corrupted), 5 seeds.
digits (N=1797) is excluded: the dense per-family tensor and ~270 clusterings per fit exceed this study's compute budget.
Results are written separately; compare with exp3_summary.csv (same K, same data)."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "src"))
import numpy as np, pandas as pd
from common.utils import scores, pairwise_ari
from icc import PCCS, PCCSConfig

DATA = ROOT / "exp3_real_world_benchmarks" / "data"
rows, fps = [], []
for name in ("iris", "wine", "breast_cancer"):
    for variant in ("original", "corrupted"):
        df = pd.read_csv(DATA / (f"exp3_{name}.csv" if variant == "original" else f"exp3_{name}_corrupted.csv"))
        y = df.target.to_numpy()
        X = df.drop(columns="target").to_numpy(float)
        k = len(np.unique(y))
        labs = []
        for seed in range(5):
            m = PCCS(PCCSConfig(n_clusters=k, random_state=seed, interaction=(seed == 0))).fit(X)
            labs.append(m.labels_)
            rows.append(dict(dataset=name, variant=variant, method="pccs", seed=seed, n=len(y), d=X.shape[1], k=k,
                             **scores(y, m.labels_), reference_ARI=scores(y, m.ref_)["ARI"],
                             seconds=m.timings_["total"], active_methods="+".join(m.active_methods_)))
            if seed == 0:
                f = m.fingerprint_.copy(); f.insert(0, "dataset", name); f.insert(1, "variant", variant); fps.append(f)
        stab = pairwise_ari(labs)
        for r in rows[-5:]:
            r["seed_stability"] = stab
        print(name, variant, "ARI", round(np.mean([r["ARI"] for r in rows[-5:]]), 3), flush=True)
pd.DataFrame(rows).to_csv(DATA / "exp3_pccs_results.csv", index=False)
pd.concat(fps).to_csv(DATA / "exp3_pccs_fingerprints.csv", index=False)
