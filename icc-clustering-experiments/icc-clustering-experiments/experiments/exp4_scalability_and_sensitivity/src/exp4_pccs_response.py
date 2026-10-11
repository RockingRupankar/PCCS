"""Stability as a function of perturbation type, intensity and interaction (the central PCCS question).

One PCCS model per dataset (PCCS-style 'clean' and 'noisy', N=500, 3 generator seeds) with five intensity levels per
family (algorithmic: two), 4 repetitions, 4 base methods, and all 15 pairwise combinations at the middle level.
Outputs: data/exp4_pccs_response.csv (run-level agreement with the reference partition),
         data/exp4_pccs_interactions.csv (observed vs expected combined stability per pair and cluster).
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "src"))
import pandas as pd
from common.datasets import make_pccs
from icc import PCCS, PCCSConfig

DATA = ROOT / "exp4_scalability_and_sensitivity" / "data"
LEVELS = {
    "algorithmic": (0.0, 1.0),
    "sampling": (0.1, 0.2, 0.3, 0.4, 0.5),
    "feature": (0.1, 0.3, 0.5, 0.7, 0.9),
    "measurement": (0.1, 0.25, 0.5, 1.0, 2.0),
    "technical": (0.05, 0.1, 0.25, 0.5, 1.0),
    "distribution": (0.25, 0.5, 1.0, 2.0, 4.0),
}
resp, inter = [], []
for prof in ("clean", "noisy"):
    for seed in (42, 43, 44):
        X, y, _, _ = make_pccs(prof, 500, seed)
        m = PCCS(PCCSConfig(levels=LEVELS, repetitions=4, interaction_repetitions=4, random_state=seed)).fit(X)
        t = m.run_table()
        t["active_method"] = t.method.isin(m.active_methods_)
        t.insert(0, "profile", prof); t.insert(1, "data_seed", seed)
        resp.append(t)
        i = m.interaction_.copy(); i.insert(0, "profile", prof); i.insert(1, "data_seed", seed)
        inter.append(i)
        print(prof, seed, "done", "active:", m.active_methods_, f"{m.timings_['total']:.1f}s", flush=True)
pd.concat(resp).to_csv(DATA / "exp4_pccs_response.csv", index=False)
pd.concat(inter).to_csv(DATA / "exp4_pccs_interactions.csv", index=False)
