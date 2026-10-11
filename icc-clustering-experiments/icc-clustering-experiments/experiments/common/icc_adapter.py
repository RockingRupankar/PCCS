"""Single import point for the ICC implementation used by all experiments.

To benchmark a different ICC implementation (for example an ICC-v2 module), change only this file:
it must expose `ICC` (fit / fit_predict / predict / labels_ / fingerprint_), `ICCConfig`,
`preprocess_fit`, `preprocess_apply` and `_agglo_precomputed`.
"""
from icc.model import ICC, ICCConfig, preprocess_fit, preprocess_apply, _agglo_precomputed  # noqa: F401

ICCv1 = ICC  # name used inside the experiment scripts and result files ("icc_v1")
