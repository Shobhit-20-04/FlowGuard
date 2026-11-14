"""
FlowGuard Cost-Sensitive Threshold Calibrator
============================================

Optimizes SAFE-3WD decision boundaries (tau_safe, tau_attack, sigma_thresh)
strictly on the Calibration Set via cost-sensitive operational risk minimization:
  min Cost = C_FN * FN + C_FP * FP + C_INV * INV
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from flowguard.decision.safe_3wd import SafeThreeWayDecisionEngine
from flowguard.uncertainty.propagator import RiskProfile


class CostSensitiveThresholdCalibrator:
    """
    Calibrates optimal three-way decision boundaries using the held-out calibration split.
    """
    def __init__(
        self,
        cost_fn: float = 10.0,
        cost_fp: float = 1.0,
        cost_inv: float = 0.15,
        min_coverage: float = 0.40,
        grid_steps: int = 25,
    ):
        self.cost_fn = float(cost_fn)
        self.cost_fp = float(cost_fp)
        self.cost_inv = float(cost_inv)
        self.min_coverage = float(min_coverage)
        self.grid_steps = int(grid_steps)

        # Calibrated parameters
        self.is_fitted: bool = False
        self.optimal_tau_safe_: float = 0.20
        self.optimal_tau_attack_: float = 0.80
        self.optimal_sigma_thresh_: float = 0.05
        self.best_cost_: float = float("inf")
        self.calibration_history_: List[Dict[str, float]] = []

    def fit(
        self,
        calibration_risk_profile: RiskProfile,
        y_calibration: Union[pd.Series, np.ndarray],
    ) -> "CostSensitiveThresholdCalibrator":
        """
        Searches the risk space on the Calibration Split to find the optimal decision thresholds.

        Parameters
        ----------
        calibration_risk_profile : RiskProfile
            RiskProfile propagated from calibration flows.
        y_calibration : Union[pd.Series, np.ndarray]
            Ground truth labels (0: Benign, 1: Attack) for calibration flows.

        Returns
        -------
        CostSensitiveThresholdCalibrator
            Fitted calibrator ready to instantiate SafeThreeWayDecisionEngine.
        """
        y_cal = np.asarray(y_calibration).ravel()
        n_samples = len(y_cal)

        r_min = calibration_risk_profile.risk_min
        r_max = calibration_risk_profile.risk_max
        r_var = calibration_risk_profile.risk_variance

        # Search grid
        tau_safe_candidates = np.linspace(0.05, 0.45, self.grid_steps)
        tau_attack_candidates = np.linspace(0.55, 0.95, self.grid_steps)
        var_thresh_candidates = [0.02, 0.05, 0.10, 0.20]

        best_cost = float("inf")
        best_tau_safe = 0.20
        best_tau_attack = 0.80
        best_var = 0.05
        history = []

        is_attack_true = (y_cal == 1)
        is_benign_true = (y_cal == 0)

        for v_thresh in var_thresh_candidates:
            var_ok = (r_var <= v_thresh)

            for t_safe in tau_safe_candidates:
                benign_mask = (r_max < t_safe) & var_ok

                for t_attack in tau_attack_candidates:
                    if t_attack < t_safe:
                        continue

                    attack_mask = (r_min > t_attack) & var_ok
                    inv_mask = ~(benign_mask | attack_mask)

                    # Errors & Volumes
                    fn = int(np.sum(is_attack_true & benign_mask))
                    fp = int(np.sum(is_benign_true & attack_mask))
                    inv = int(np.sum(inv_mask))
                    accepted = int(np.sum(benign_mask | attack_mask))

                    coverage = accepted / max(1, n_samples)

                    # Operational cost
                    cost = (
                        (self.cost_fn * fn)
                        + (self.cost_fp * fp)
                        + (self.cost_inv * inv)
                    ) / max(1, n_samples)

                    # Penalize constraint violation (coverage < min_coverage)
                    if coverage < self.min_coverage:
                        cost += (self.min_coverage - coverage) * 10.0

                    history.append({
                        "tau_safe": float(t_safe),
                        "tau_attack": float(t_attack),
                        "sigma_thresh": float(v_thresh),
                        "cost": float(cost),
                        "coverage": float(coverage),
                        "false_negatives": fn,
                        "false_positives": fp,
                        "investigated": inv,
                    })

                    if cost < best_cost:
                        best_cost = cost
                        best_tau_safe = float(t_safe)
                        best_tau_attack = float(t_attack)
                        best_var = float(v_thresh)

        self.optimal_tau_safe_ = best_tau_safe
        self.optimal_tau_attack_ = best_tau_attack
        self.optimal_sigma_thresh_ = best_var
        self.best_cost_ = best_cost
        self.calibration_history_ = history
        self.is_fitted = True

        return self

    def create_calibrated_engine(self) -> SafeThreeWayDecisionEngine:
        """
        Returns a SafeThreeWayDecisionEngine instantiated with the calibrated optimal thresholds.
        """
        if not self.is_fitted:
            raise RuntimeError("Calibrator must be fitted before creating engine!")

        return SafeThreeWayDecisionEngine(
            tau_safe=self.optimal_tau_safe_,
            tau_attack=self.optimal_tau_attack_,
            max_acceptable_variance=self.optimal_sigma_thresh_,
        )
