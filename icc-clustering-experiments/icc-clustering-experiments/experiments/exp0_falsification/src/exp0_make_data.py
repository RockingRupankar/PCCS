"""Write one example dataset per structure (sigma=1 Gaussian noise) so the data can be inspected."""
import numpy as np, pandas as pd
from exp0_core import *
out = ROOT / "exp0_falsification" / "data"
for s in STRUCTURES:
    rng = np.random.default_rng(7)
    X, y = make_structure(s, rng=rng)
    X = add_noise(X, "gaussian", 1.0, rng)
    df = pd.DataFrame(X, columns=[f"x{i + 1:02d}" for i in range(X.shape[1])])
    df.insert(0, "true_cluster", y)
    df.to_csv(out / f"exp0_example_{s}_sigma1.csv", index=False, float_format="%.4f")
    print("saved", s, df.shape)
