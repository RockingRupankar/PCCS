# Reproducibility

- **Environment used for the released results:** Python 3.12, numpy 2.4, pandas 3.0, scipy 1.17, scikit-learn 1.8, matplotlib 3.10; one CPU core, 3 GB RAM.
- **Seeds.** Every script fixes its seeds (generator seeds 42-44; method seeds 0-4; split seeds listed in `configs/`). Re-running a suite on the same library versions reproduces the CSV files; different scikit-learn versions can change k-means or spectral results slightly.
- **Runtimes (one core).** exp0 about 12 min, exp1 about 5 min, exp2 about 10 min, exp3 about 5 min, exp4 about 14 min (each includes its PCCS block).
- **Timing study.** exp4 wall-clock numbers depend on the machine; run it without other jobs and compare slopes, not absolute seconds.
- **Determinism notes.** Ward and spectral (kNN) clustering are deterministic given the data, so their "seed stability" in exp3 equals 1.0 by construction.
- **Verification.** `pytest` checks the library, generators, baselines, configs, and repository layout; `python scripts/summarize_results.py` rebuilds `docs/results_summary.md` from the CSVs.
