"""
Unit Test Suite: SAFE-3WD Decision Policy Engine & Calibrator
=============================================================

Tests:
1. SAFE-3WD regional decision partitioning (Benign, Attack, Investigate).
2. Boundary crossing and variance thresholding logic.
3. Cost-sensitive threshold optimization on calibration split.
4. Selective accuracy, coverage rate, and operational cost calculations.
"""

import numpy as np
import pandas as pd
import pytest

from flowguard.decision.safe_3wd import SafeThreeWayDecisionEngine, ThreeWayDecisionResult
from flowguard.decision.calibrator import CostSensitiveThresholdCalibrator
from flowguard.uncertainty.propagator import RiskProfile


@pytest.fixture
def mock_risk_profile():
    """Constructs a deterministic 4-flow risk profile representing all 3 regions."""
    # Flow 0: High confidence Benign (R_max < 0.20, low var)
    # Flow 1: High confidence Attack (R_min > 0.80, low var)
    # Flow 2: Ambiguous boundary-straddling flow (R_min < 0.20, R_max > 0.80)
    # Flow 3: Excessive variance flow (R_max < 0.20, but var > max_acceptable_variance)
    r_min = np.array([0.02, 0.85, 0.10, 0.05])
    r_max = np.array([0.15, 0.98, 0.90, 0.18])
    r_mean = np.array([0.08, 0.92, 0.50, 0.12])
    r_var = np.array([0.002, 0.003, 0.160, 0.060]) # Flow 3 has var > 0.04
    r_std = np.sqrt(r_var)
    r_spread = r_max - r_min
    r_median = r_mean
    raw = np.column_stack([r_min, r_mean, r_max])

    return RiskProfile(
        risk_min=r_min,
        risk_max=r_max,
        risk_mean=r_mean,
        risk_variance=r_var,
        risk_std=r_std,
        risk_spread=r_spread,
        risk_median=r_median,
        raw_risk_matrix=raw,
    )


def test_safe_3wd_decision_partitions(mock_risk_profile):
    engine = SafeThreeWayDecisionEngine(
        tau_safe=0.20,
        tau_attack=0.80,
        max_acceptable_variance=0.04,
    )
    result = engine.decide(mock_risk_profile)

    assert isinstance(result, ThreeWayDecisionResult)
    assert len(result.decisions) == 4

    # Flow 0 -> Benign
    assert result.decisions[0] == "BENIGN"
    assert result.numeric_decisions[0] == 0
    assert result.accepted_predictions[0] == 0

    # Flow 1 -> Attack
    assert result.decisions[1] == "ATTACK"
    assert result.numeric_decisions[1] == 1
    assert result.accepted_predictions[1] == 1

    # Flow 2 -> Investigate (interval straddles boundary)
    assert result.decisions[2] == "INVESTIGATE"
    assert result.numeric_decisions[2] == 2
    assert result.accepted_predictions[2] == -1

    # Flow 3 -> Investigate (variance 0.06 > 0.04)
    assert result.decisions[3] == "INVESTIGATE"
    assert result.numeric_decisions[3] == 2
    assert result.accepted_predictions[3] == -1


def test_safe_3wd_evaluation_metrics(mock_risk_profile):
    engine = SafeThreeWayDecisionEngine(tau_safe=0.20, tau_attack=0.80, max_acceptable_variance=0.04)
    y_true = np.array([0, 1, 1, 0]) # Flow 0: Benign, Flow 1: Attack, Flow 2: Attack, Flow 3: Benign

    metrics = engine.evaluate(
        risk_profile=mock_risk_profile,
        y_true=y_true,
        cost_fn=10.0,
        cost_fp=1.0,
        cost_inv=0.15,
    )

    # 2 flows accepted (Flow 0, Flow 1), 2 investigated (Flow 2, Flow 3)
    assert metrics["coverage"] == 0.50
    assert metrics["investigation_rate"] == 0.50
    assert metrics["investigated_count"] == 2

    # Both accepted flows were classified correctly (Flow 0 = Benign, Flow 1 = Attack)
    assert metrics["selective_accuracy"] == 1.00
    assert metrics["selective_risk"] == 0.00
    assert metrics["false_negatives_definite"] == 0
    assert metrics["false_positives_definite"] == 0

    # Operational Cost: 0 * 10 + 0 * 1 + 2 * 0.15 = 0.30
    assert pytest.approx(metrics["operational_cost"], 1e-4) == 0.30


def test_cost_sensitive_calibrator_fitting(mock_risk_profile):
    y_calib = np.array([0, 1, 1, 0])
    calibrator = CostSensitiveThresholdCalibrator(
        cost_fn=10.0,
        cost_fp=1.0,
        cost_inv=0.15,
        min_coverage=0.40,
        grid_steps=10,
    )
    assert not calibrator.is_fitted

    calibrator.fit(mock_risk_profile, y_calib)
    assert calibrator.is_fitted
    assert 0.0 < calibrator.optimal_tau_safe_ <= calibrator.optimal_tau_attack_ < 1.0

    calibrated_engine = calibrator.create_calibrated_engine()
    assert isinstance(calibrated_engine, SafeThreeWayDecisionEngine)
    assert calibrated_engine.tau_safe == calibrator.optimal_tau_safe_
    assert calibrated_engine.tau_attack == calibrator.optimal_tau_attack_
