"""Benchmark splits, evaluation metrics, and GPU linear probe for EcoVerse."""

from ecoverse.bench.metrics import (
    compute_counterfactual_metrics,
    compute_cross_obs_alignment,
    compute_flux_r2,
    compute_masked_nrmse,
)
from ecoverse.bench.probe import (
    compare_pretrained_vs_random_probe,
    fit_and_eval_linear_probe,
)
from ecoverse.bench.splits import SPLIT_NAMES, SplitManager, build_splits_json

__all__ = [
    "SPLIT_NAMES",
    "SplitManager",
    "build_splits_json",
    "compute_counterfactual_metrics",
    "compute_cross_obs_alignment",
    "compute_flux_r2",
    "compute_masked_nrmse",
    "compare_pretrained_vs_random_probe",
    "fit_and_eval_linear_probe",
]
