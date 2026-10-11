"""Fine noise grid to locate the sigma where structured stability falls to the null level."""
import numpy as np, pandas as pd
from exp0_core import *
levels = np.round(np.linspace(0, 3, 25), 3)
rows = [run_cell(s, "gaussian", float(lv), sd, B=8) for s in STRUCTURES for lv in levels for sd in range(10)]
df = pd.DataFrame(rows)
agg = df.groupby(["structure", "level"]).agg(stab_boot=("stab_boot", "mean"), stab_boot_sd=("stab_boot", "std"),
                                              stab_seed=("stab_seed", "mean")).reset_index()
agg.to_csv(ROOT / "exp0_falsification" / "data" / "exp0_finegrid.csv", index=False)
p = agg.pivot(index="level", columns="structure", values="stab_boot")
p["gap"] = p["structured"] - p[["null_iso", "null_aniso"]].max(axis=1)
print(p.round(3).to_string())
