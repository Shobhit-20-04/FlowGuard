"""
FlowGuard Evaluation & Metrics Library
=======================================

Standard classification metrics, probability calibration (ECE),
and selective three-way decision metrics.
"""

from flowguard.evaluation.metrics import (
    compute_brier_score,
    compute_classification_metrics,
    compute_expected_calibration_error,
    compute_selective_metrics,
)

__all__ = [
    "compute_classification_metrics",
    "compute_expected_calibration_error",
    "compute_brier_score",
    "compute_selective_metrics",
]
