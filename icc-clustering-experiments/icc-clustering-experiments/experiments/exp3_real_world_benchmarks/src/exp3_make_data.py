"""Real-world data that ships with scikit-learn (no download needed): iris, wine, breast_cancer, digits.
For each, also write a corrupted version: +100 pure-noise features, measurement noise, outlier rows, MCAR missing rows
(common/datasets.corrupt)."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits
from common.datasets import corrupt

OUT = ROOT / "exp3_real_world_benchmarks" / "data"
for name, loader in dict(iris=load_iris, wine=load_wine, breast_cancer=load_breast_cancer, digits=load_digits).items():
    b = loader()
    X, y = b.data.astype(float), b.target
    cols = [f"f{i + 1:03d}" for i in range(X.shape[1])]
    pd.DataFrame(X, columns=cols).assign(target=y).to_csv(OUT / f"exp3_{name}.csv", index=False, float_format="%.4f")
    Xc = corrupt(X, n_noise_features=100, seed=7)
    cc = [f"f{i + 1:03d}" for i in range(Xc.shape[1])]
    pd.DataFrame(Xc, columns=cc).assign(target=y).to_csv(OUT / f"exp3_{name}_corrupted.csv", index=False, float_format="%.3f")
    print(name, X.shape, "->", Xc.shape, "classes", len(np.unique(y)))
