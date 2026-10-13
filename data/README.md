# Data Directory

This directory stores datasets and manifests for the PCCS research project:

- `raw/`: Raw downloaded CSV files directly exported from verified sources.
- `processed/`: Processed feature matrices and strictly separated evaluation labels.
- `manifests/`: `dataset_manifest.csv` containing official provenance, licenses, SHA-256 hashes, and subsampling policies.

## Preprocessing Policy
All datasets undergo deterministic preprocessing:
1. Missing value imputation via column medians.
2. Outlier winsorization at 4.0 Median Absolute Deviations (MAD).
3. Robust standardization (zero mean, unit variance).
4. Strict isolation: class labels are stored in separate files (`*_labels.csv`) and are never exposed during unsupervised clustering, perturbation modeling, or consensus formation.
