"""'Scaled' regime: increasing dimension d (more pure-noise features), n=600, fixed informative dims."""
import pandas as pd
from exp0_core import *
DIMS = [20, 50, 100, 200, 400]
rows = [run_cell(s, "gaussian", lv, sd, n=600, d=d, d_inf=5, B=8)
        for s in STRUCTURES for d in DIMS for lv in (0.5, 1.0) for sd in range(8)]
df = pd.DataFrame(rows)
agg = (df.groupby(["structure", "d", "level"])
       .agg(stab_seed=("stab_seed", "mean"), stab_boot=("stab_boot", "mean"), stab_boot_sd=("stab_boot", "std"),
            ari_truth=("ari_truth", "mean"), silhouette=("silhouette", "mean")).reset_index())
agg.to_csv(ROOT / "exp0_falsification" / "data" / "exp0_scaled_agg.csv", index=False)
print(agg.round(3).to_string())
