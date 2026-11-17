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
        policy: str = "quantile",
        k_sigma: float = 1.0,
        alpha_confidence: Optional[float] = 0.90,
    ):
        self.cost_fn = float(cost_fn)
        self.cost_fp = float(cost_fp)
        self.cost_inv = float(cost_inv)
        self.min_coverage = float(min_coverage)
        self.grid_steps = int(grid_steps)
        self.policy = policy
        self.k_sigma = float(k_sigma)
        self.alpha_confidence = float(alpha_confidence) if alpha_confidence is not None else None

        # Calibrated parameters
        self.is_fitted: bool = False
        self.constraint_satisfied_: bool = False
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
        r_mean = calibration_risk_profile.risk_mean
        r_var = calibration_risk_profile.risk_variance
        sigma_r = np.sqrt(np.maximum(0.0, r_var))

        # Quantile bounds for quantile policy
        if self.policy == "quantile":
            if self.alpha_confidence is not None and self.alpha_confidence < 1.0:
                alpha = self.alpha_confidence
                q_low = 100.0 * (1.0 - alpha) / 2.0
                q_high = 100.0 * (1.0 + alpha) / 2.0
                if getattr(calibration_risk_profile, "risk_q_low", None) is not None and getattr(calibration_risk_profile, "alpha_confidence", None) == alpha:
                    r_lower = calibration_risk_profile.risk_q_low
                    r_upper = calibration_risk_profile.risk_q_high
                elif getattr(calibration_risk_profile, "raw_risk_matrix", None) is not None and calibration_risk_profile.raw_risk_matrix.shape[1] > 1:
                    r_lower = np.percentile(calibration_risk_profile.raw_risk_matrix, q_low, axis=1)
                    r_upper = np.percentile(calibration_risk_profile.raw_risk_matrix, q_high, axis=1)
                elif np.isclose(alpha, 0.90) and getattr(calibration_risk_profile, "risk_q05", None) is not None:
                    r_lower = calibration_risk_profile.risk_q05
                    r_upper = calibration_risk_profile.risk_q95
                else:
                    r_lower = r_min
                    r_upper = r_max
            else:
                r_lower = r_min
                r_upper = r_max

        # Search grid
        tau_safe_candidates = np.linspace(0.05, 0.45, self.grid_steps)
        tau_attack_candidates = np.linspace(0.55, 0.95, self.grid_steps)
        var_thresh_candidates = [0.02, 0.05, 0.10, 0.20]

        best_feasible_cost = float("inf")
        best_feasible_params = (0.20, 0.80, 0.05)
        feasible_found = False

        # Infeasible fallback: configuration with maximum coverage
        best_fallback_cost = float("inf")
        best_fallback_params = (0.20, 0.80, 0.05)
        max_fallback_coverage = -1.0

        history = []

        is_attack_true = (y_cal == 1)
        is_benign_true = (y_cal == 0)

        for v_thresh in var_thresh_candidates:
            var_ok = (r_var <= v_thresh)

            for t_safe in tau_safe_candidates:
                if self.policy == "quantile":
                    benign_mask = (r_upper < t_safe) & var_ok
                else:
                    benign_mask = (r_mean + self.k_sigma * sigma_r < t_safe) & var_ok

                for t_attack in tau_attack_candidates:
                    if t_attack < t_safe:
                        continue

                    if self.policy == "quantile":
                        attack_mask = (r_lower > t_attack) & var_ok
                    else:
                        attack_mask = (r_mean - self.k_sigma * sigma_r > t_attack) & var_ok

                    inv_mask = ~(benign_mask | attack_mask)

                    # Errors & Volumes
                    fn = int(np.sum(is_attack_true & benign_mask))
                    fp = int(np.sum(is_benign_true & attack_mask))
                    inv = int(np.sum(inv_mask))
                    accepted = int(np.sum(benign_mask | attack_mask))

                    coverage = accepted / max(1, n_samples)

                    # Pure operational cost
                    cost = (
                        (self.cost_fn * fn)
                        + (self.cost_fp * fp)
                        + (self.cost_inv * inv)
                    ) / max(1, n_samples)

                    is_feasible = (coverage >= self.min_coverage)

                    history.append({
                        "tau_safe": float(t_safe),
                        "tau_attack": float(t_attack),
                        "sigma_thresh": float(v_thresh),
                        "cost": float(cost),
                        "coverage": float(coverage),
                        "is_feasible": is_feasible,
                        "false_negatives": fn,
                        "false_positives": fp,
                        "investigated": inv,
                    })

                    if is_feasible:
                        if cost < best_feasible_cost:
                            best_feasible_cost = cost
                            best_feasible_params = (float(t_safe), float(t_attack), float(v_thresh))
                            feasible_found = True
                    else:
                        # Fallback heuristic: maximize coverage, break ties by operational cost
                        if coverage > max_fallback_coverage or (np.isclose(coverage, max_fallback_coverage) and cost < best_fallback_cost):
                            max_fallback_coverage = coverage
                            best_fallback_cost = cost
                            best_fallback_params = (float(t_safe), float(t_attack), float(v_thresh))

        if feasible_found:
            self.optimal_tau_safe_, self.optimal_tau_attack_, self.optimal_sigma_thresh_ = best_feasible_params
            self.best_cost_ = best_feasible_cost
            self.constraint_satisfied_ = True
        else:
            self.optimal_tau_safe_, self.optimal_tau_attack_, self.optimal_sigma_thresh_ = best_fallback_params
            self.best_cost_ = best_fallback_cost
            self.constraint_satisfied_ = False
            import logging
            logging.getLogger(__name__).warning(
                f"Calibrator: min_coverage constraint ({self.min_coverage:.2f}) could not be satisfied. "
                f"Selected maximum achievable coverage fallback ({max_fallback_coverage:.2f})."
            )

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
            policy=self.policy,
            k_sigma=self.k_sigma,
            alpha_confidence=self.alpha_confidence,
        )
