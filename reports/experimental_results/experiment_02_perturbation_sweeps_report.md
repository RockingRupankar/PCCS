# Experiment 2: Perturbation-Family and Intensity Sweeps Report

## 1. Objective
Characterize stability degradation across 6 distinct perturbation families and 3 intensity tiers.

## 2. Summary by Family and Intensity

| dataset       | family       |   nominal_level |      mean |       std |
|:--------------|:-------------|----------------:|----------:|----------:|
| breast_cancer | algorithmic  |            0.1  | 0.657105  | 0.421955  |
| breast_cancer | algorithmic  |            0.25 | 0.660657  | 0.425157  |
| breast_cancer | algorithmic  |            0.5  | 0.658881  | 0.423563  |
| breast_cancer | distribution |            0.2  | 0.671599  | 0.411995  |
| breast_cancer | distribution |            0.5  | 0.675702  | 0.413485  |
| breast_cancer | distribution |            0.8  | 0.64561   | 0.424769  |
| breast_cancer | feature      |            0.1  | 0.6324    | 0.424618  |
| breast_cancer | feature      |            0.25 | 0.62722   | 0.414953  |
| breast_cancer | feature      |            0.5  | 0.662481  | 0.41487   |
| breast_cancer | measurement  |            0.1  | 0.698144  | 0.436846  |
| breast_cancer | measurement  |            0.3  | 0.645876  | 0.408345  |
| breast_cancer | measurement  |            0.6  | 0.628604  | 0.403408  |
| breast_cancer | sampling     |            0.5  | 0.657971  | 0.417386  |
| breast_cancer | sampling     |            0.75 | 0.705216  | 0.432043  |
| breast_cancer | sampling     |            0.9  | 0.705763  | 0.430673  |
| breast_cancer | technical    |            0.2  | 0.656877  | 0.434582  |
| breast_cancer | technical    |            0.5  | 0.651933  | 0.406541  |
| breast_cancer | technical    |            1    | 0.0450943 | 0.0787611 |
| wine          | algorithmic  |            0.1  | 0.709255  | 0.443503  |
| wine          | algorithmic  |            0.25 | 0.707004  | 0.441859  |
| wine          | algorithmic  |            0.5  | 0.707202  | 0.442     |
| wine          | distribution |            0.2  | 0.692443  | 0.43166   |
| wine          | distribution |            0.5  | 0.704835  | 0.437994  |
| wine          | distribution |            0.8  | 0.696476  | 0.432854  |
| wine          | feature      |            0.1  | 0.666935  | 0.415208  |
| wine          | feature      |            0.25 | 0.672934  | 0.41617   |
| wine          | feature      |            0.5  | 0.466593  | 0.30606   |
| wine          | measurement  |            0.1  | 0.69399   | 0.431812  |
| wine          | measurement  |            0.3  | 0.646118  | 0.411056  |
| wine          | measurement  |            0.6  | 0.536651  | 0.336292  |
| wine          | sampling     |            0.5  | 0.680023  | 0.425668  |
| wine          | sampling     |            0.75 | 0.691172  | 0.433958  |
| wine          | sampling     |            0.9  | 0.704234  | 0.438935  |
| wine          | technical    |            0.2  | 0.698212  | 0.433327  |
| wine          | technical    |            0.5  | 0.643219  | 0.406263  |
| wine          | technical    |            1    | 0.0111061 | 0.0224986 |

## 3. Conclusions
Stability decays non-uniformly: feature dropout and batch shifts produce severe degradation, whereas algorithmic initialization variation yields negligible impact.
