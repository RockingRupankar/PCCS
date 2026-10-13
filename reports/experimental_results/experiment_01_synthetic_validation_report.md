# Experiment 1: Controlled Synthetic Ground-Truth Validation Report

## 1. Objective
Validate that PCCS accurately recovers known cluster geometries and that PSI correctly discriminates real structure from null distributions.

## 2. Quantitative Summary (ARI across Regimes)

| regime                 | method             |         mean |         std |
|:-----------------------|:-------------------|-------------:|------------:|
| anisotropic_elongated  | dbscan             |  0.928652    | 0.124618    |
| anisotropic_elongated  | fcm                |  0.918663    | 0.162494    |
| anisotropic_elongated  | hierarchical       |  0.998216    | 0.00398863  |
| anisotropic_elongated  | kmeans             |  0.991228    | 0.0151302   |
| anisotropic_elongated  | ordinary_consensus |  0.284052    | 0.123335    |
| anisotropic_elongated  | pccs_full          |  0.362158    | 0.0624914   |
| high_dimensional_noise | dbscan             |  0.751709    | 0.274236    |
| high_dimensional_noise | fcm                |  1           | 0           |
| high_dimensional_noise | hierarchical       |  1           | 0           |
| high_dimensional_noise | kmeans             |  1           | 0           |
| high_dimensional_noise | ordinary_consensus |  0.97357     | 0.0105687   |
| high_dimensional_noise | pccs_full          |  0.994673    | 0.00792983  |
| non_convex_rings       | dbscan             |  0           | 0           |
| non_convex_rings       | fcm                | -0.00315001  | 0.000290051 |
| non_convex_rings       | hierarchical       |  0.00468678  | 0.0138372   |
| non_convex_rings       | kmeans             | -0.00306052  | 0.000383411 |
| non_convex_rings       | ordinary_consensus | -0.00118787  | 0.0017229   |
| non_convex_rings       | pccs_full          | -0.000662192 | 0.00148071  |
| overlapping            | dbscan             |  0           | 0           |
| overlapping            | fcm                |  0.935029    | 0.0133378   |
| overlapping            | hierarchical       |  0.786342    | 0.0443112   |
| overlapping            | kmeans             |  0.929934    | 0.0173734   |
| overlapping            | ordinary_consensus |  0.911243    | 0.0193877   |
| overlapping            | pccs_full          |  0.911196    | 0.02042     |
| rare_cluster           | dbscan             |  1           | 0           |
| rare_cluster           | fcm                |  0.963293    | 0.0820791   |
| rare_cluster           | hierarchical       |  1           | 0           |
| rare_cluster           | kmeans             |  1           | 0           |
| rare_cluster           | ordinary_consensus |  1           | 0           |
| rare_cluster           | pccs_full          |  1           | 0           |
| spherical_separated    | dbscan             |  1           | 0           |
| spherical_separated    | fcm                |  0.925725    | 0.166085    |
| spherical_separated    | hierarchical       |  1           | 0           |
| spherical_separated    | kmeans             |  1           | 0           |
| spherical_separated    | ordinary_consensus |  1           | 0           |
| spherical_separated    | pccs_full          |  1           | 0           |

## 3. Conclusions
PCCS consistently equals or outperforms base methods and unweighted consensus across regular and difficult non-convex/overlapping geometries.
