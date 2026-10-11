# Data

## PCCS-style synthetic generator
`experiments/common/pccs_datagen.py` (wrapped by `common.datasets.make_pccs`).

- **Latent geometry.** Four clusters in 15 informative dimensions; mean pairwise centre distance 7.5; cluster sizes 35/30/21/14%; cluster-specific spread; three correlated blocks of five dimensions (rho 0.5). Clusters 1 and 2 form interlocking crescents in two dimensions.
- **Informative features (15).** Dimensions 0-5 linear Gaussian (strong), 6-8 quadratic or sinusoidal transforms (strong), 9-12 skewed or nonlinear (moderate), 13-14 weak linear (centre weight 0.25).
- **Redundant features (15).** Linear mixes of one or two standardised informative features plus independent noise.
- **Pure-noise features (70).** 30 Gaussian (two correlated blocks of 10, rho 0.4), 15 uniform, 15 Student-t(4), 10 log-normal; independent of the cluster.
- **Scales.** Every feature receives its own scale (0.1 to 500) and offset; column order is shuffled.
- **Corruption.** Measurement noise, outlier rows (5-20 cells set to +-(4-8) standardised units), MCAR missing cells in a fraction of rows, borderline rows (40-60% of the way to another centre), mislabel-like rows (85-100% of the way), and in the severe profile shuffled values in a subset of signal features.
- **Ground truth.** `true_cluster` is the generating cluster, also for borderline and mislabel-like rows.

## Real data
scikit-learn bundled datasets: iris (150 x 4, K=3), wine (178 x 13, K=3), breast cancer (569 x 30, K=2), digits (1797 x 64, K=10).
The corrupted variants standardise the features and add measurement noise (sd 0.3), 100 pure-noise features, 2% outlier rows, and MCAR missing cells in 3% of rows (seed 7).

## Limitations
Synthetic data follow a simple latent model; missingness is MCAR only; no batch or distribution-shift mechanism is simulated.
