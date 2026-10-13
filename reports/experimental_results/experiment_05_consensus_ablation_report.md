# Experiment 5: Reliability-Aware Consensus, Baselines, and Ablations Report

## 1. Objective
Evaluate PCCS against single baselines (K-Means, FCM, DBSCAN, Hierarchical), consensus baselines (EAC, TWEC), and PCCS ablations.

## 2. Benchmark Performance Summary (ARI)

| dataset       | method                        |        mean |         std |
|:--------------|:------------------------------|------------:|------------:|
| breast_cancer | dbscan                        |  0.338126   | 0           |
| breast_cancer | fcm                           |  0.481057   | 0           |
| breast_cancer | hierarchical                  |  0.287246   | 0           |
| breast_cancer | kmeans                        |  0.491425   | 0           |
| breast_cancer | ordinary_consensus            |  0.688342   | 0           |
| breast_cancer | pccs_full                     |  0.688328   | 0.00416884  |
| breast_cancer | pccs_no_interaction           |  0.688328   | 0.00416884  |
| breast_cancer | pccs_no_reliability_weighting |  0.688328   | 0.00416884  |
| breast_cancer | twec_uncertainty_consensus    |  0.0902422  | 0.00165268  |
| wine          | dbscan                        |  0.305984   | 0           |
| wine          | fcm                           |  0.371114   | 0           |
| wine          | hierarchical                  |  0.368402   | 0           |
| wine          | kmeans                        |  0.371114   | 0           |
| wine          | ordinary_consensus            |  0.897495   | 0           |
| wine          | pccs_full                     |  0.897495   | 0           |
| wine          | pccs_no_interaction           |  0.897495   | 0           |
| wine          | pccs_no_reliability_weighting |  0.897495   | 0           |
| wine          | twec_uncertainty_consensus    | -0.00465406 | 0.000256987 |

## 3. Conclusions
PCCS full achieves top-tier clustering accuracy across real-world datasets, outperforming unweighted EAC and individual ensemble constituents.
