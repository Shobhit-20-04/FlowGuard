"""
FlowGuard SAFE-3WD Decision Policy Engine
========================================

Implements the Risk-Based Three-Way Decision (SAFE-3WD) rule.
Partitions network flows into three calibrated decision regions:
1. Region 1 (Benign / Fast Accept): R_max < tau_safe and sigma_R^2 <= sigma_thresh^2
2. Region 2 (Attack / Immediate Block): R_min > tau_attack and sigma_R^2 <= sigma_thresh^2
3. Region 3 (Investigate / Deep Inspection): Ambiguous flows (interval overlaps boundary or high variance)
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from flowguard.evaluation.metrics import compute_selective_metrics
from flowguard.uncertainty.propagator import RiskProfile


@dataclass
class ThreeWayDecisionResult:
    """
    Encapsulates the output of the SAFE-3WD decision rule for N network flows.
    """
    decisions: np.ndarray            # String array of shape (N,): 'BENIGN', 'ATTACK', 'INVESTIGATE'
    numeric_decisions: np.ndarray    # Integer array of shape (N,): 0 (Benign), 1 (Attack), 2 (Investigate)
    accepted_predictions: np.ndarray # Integer array: 0 (Benign), 1 (Attack), -1 (Investigate)
    confidences: np.ndarray          # Float array: confidence score for accepted flows
    risk_min: np.ndarray             # R_min
    risk_max: np.ndarray             # R_max
    risk_mean: np.ndarray            # mu_R
    risk_variance: np.ndarray        # sigma_R^2
    risk_spread: np.ndarray          # Delta_R = R_max - R_min

    def to_dataframe(self) -> pd.DataFrame:
        """
        Converts decision outputs into a structured pandas DataFrame.
        """
        return pd.DataFrame({
            "decision": self.decisions,
            "numeric_decision": self.numeric_decisions,
            "accepted_prediction": self.accepted_predictions,
            "confidence": self.confidences,
            "risk_min": self.risk_min,
            "risk_max": self.risk_max,
            "risk_mean": self.risk_mean,
            "risk_variance": self.risk_variance,
            "risk_spread": self.risk_spread,
        })

    def summary(self) -> Dict[str, Any]:
        """
        Computes summary statistics of the decision distribution.
        """
        n_total = len(self.decisions)
        n_benign = int(np.sum(self.numeric_decisions == 0))
        n_attack = int(np.sum(self.numeric_decisions == 1))
        n_inv = int(np.sum(self.numeric_decisions == 2))

        return {
            "total_flows": n_total,
            "benign_count": n_benign,
            "attack_count": n_attack,
            "investigate_count": n_inv,
            "coverage_rate": float((n_benign + n_attack) / max(1, n_total)),
            "investigation_rate": float(n_inv / max(1, n_total)),
            "benign_rate": float(n_benign / max(1, n_total)),
            "attack_rate": float(n_attack / max(1, n_total)),
        }


class SafeThreeWayDecisionEngine:
    """
    Risk-Based Three-Way Decision Policy (SAFE-3WD).

    Decisions are evaluated over the full propagated risk interval [R_min, R_max]
    and predictive variance sigma_R^2, avoiding overconfident mistakes on sampled traffic.
    """
    def __init__(
        self,
        tau_safe: float = 0.20,
        tau_attack: float = 0.80,
        max_acceptable_variance: float = 0.04,
        policy: str = "quantile",
        k_sigma: float = 1.0,
        alpha_confidence: Optional[float] = 0.90,
    ):
        assert 0.0 < tau_safe <= tau_attack < 1.0, (
            f"Invalid threshold bounds: tau_safe ({tau_safe}) must be <= tau_attack ({tau_attack}) in (0, 1)!"
        )
        if policy not in ["quantile", "moment"]:
            raise ValueError(f"Unknown decision policy '{policy}'. Must be 'quantile' or 'moment'.")

        if alpha_confidence is not None:
            assert 0.0 < alpha_confidence <= 1.0, (
                f"alpha_confidence must be in (0, 1], got {alpha_confidence}"
            )

        self.tau_safe = float(tau_safe)
        self.tau_attack = float(tau_attack)
        self.max_acceptable_variance = float(max_acceptable_variance)
        self.policy = policy
        self.k_sigma = float(k_sigma)
        self.alpha_confidence = float(alpha_confidence) if alpha_confidence is not None else None

    def decide(self, risk_profile: RiskProfile) -> ThreeWayDecisionResult:
        """
        Executes the SAFE-3WD rule over propagated risk distributions.

        Parameters
        ----------
        risk_profile : RiskProfile
            Propagated risk distribution from UncertaintyPropagator.

        Returns
        -------
        ThreeWayDecisionResult
            Vectorized decisions and confidence scores for all N flows.
        """
        r_min = risk_profile.risk_min
        r_max = risk_profile.risk_max
        r_mean = risk_profile.risk_mean
        r_var = risk_profile.risk_variance
        r_spread = risk_profile.risk_spread

        n_rows = len(r_min)
        var_ok = (r_var <= self.max_acceptable_variance)

        if self.policy == "quantile":
            # Quantile Credible Interval Policy:
            # For confidence level alpha in (0, 1), evaluates central credible interval:
            # Lower bound L = Q((1 - alpha) / 2)  (e.g., q05 for alpha=0.90)
            # Upper bound U = Q((1 + alpha) / 2)  (e.g., q95 for alpha=0.90)
            # If alpha_confidence is 1.0 or None, falls back to sample extrema [r_min, r_max].
            if self.alpha_confidence is not None and self.alpha_confidence < 1.0:
                alpha = self.alpha_confidence
                q_low = 100.0 * (1.0 - alpha) / 2.0
                q_high = 100.0 * (1.0 + alpha) / 2.0

                if getattr(risk_profile, "risk_q_low", None) is not None and getattr(risk_profile, "alpha_confidence", None) == alpha:
                    r_lower = risk_profile.risk_q_low
                    r_upper = risk_profile.risk_q_high
                elif getattr(risk_profile, "raw_risk_matrix", None) is not None and risk_profile.raw_risk_matrix.shape[1] > 1:
                    r_lower = np.percentile(risk_profile.raw_risk_matrix, q_low, axis=1)
                    r_upper = np.percentile(risk_profile.raw_risk_matrix, q_high, axis=1)
                elif np.isclose(alpha, 0.90) and getattr(risk_profile, "risk_q05", None) is not None:
                    r_lower = risk_profile.risk_q05
                    r_upper = risk_profile.risk_q95
                else:
                    r_lower = r_min
                    r_upper = r_max
            else:
                r_lower = r_min
                r_upper = r_max

            # Region 1 (Benign / Fast Accept): U < tau_safe and sigma_R^2 <= max_acceptable_variance
            # Region 2 (Attack / Immediate Block): L > tau_attack and sigma_R^2 <= max_acceptable_variance
            benign_mask = (r_upper < self.tau_safe) & var_ok
            attack_mask = (r_lower > self.tau_attack) & var_ok
        elif self.policy == "moment":
            # Moment-bounded policy:
            # Region 1 (Benign): mu_R + k * sigma_R < tau_safe and sigma_R^2 <= max_acceptable_variance
            # Region 2 (Attack): mu_R - k * sigma_R > tau_attack and sigma_R^2 <= max_acceptable_variance
            sigma_r = np.sqrt(np.maximum(0.0, r_var))
            benign_bound = r_mean + self.k_sigma * sigma_r
            attack_bound = r_mean - self.k_sigma * sigma_r
            benign_mask = (benign_bound < self.tau_safe) & var_ok
            attack_mask = (attack_bound > self.tau_attack) & var_ok

        # Region 3: Investigate (Defer / Deep Inspection)
        # Plausible flow reconstructions intersect boundary or variance is too high
        investigate_mask = ~(benign_mask | attack_mask)

        # Formulate output arrays
        decisions = np.full(n_rows, "INVESTIGATE", dtype=object)
        decisions[benign_mask] = "BENIGN"
        decisions[attack_mask] = "ATTACK"

        numeric_decisions = np.full(n_rows, 2, dtype=np.int32)
        numeric_decisions[benign_mask] = 0
        numeric_decisions[attack_mask] = 1

        accepted_predictions = np.full(n_rows, -1, dtype=np.int32)
        accepted_predictions[benign_mask] = 0
        accepted_predictions[attack_mask] = 1

        # Confidence scores
        confidences = np.zeros(n_rows, dtype=np.float64)
        if self.policy == "quantile":
            confidences[benign_mask] = np.maximum(0.0, 1.0 - r_upper[benign_mask])
            confidences[attack_mask] = np.maximum(0.0, r_lower[attack_mask])
        else:
            confidences[benign_mask] = np.maximum(0.0, 1.0 - (r_mean[benign_mask] + self.k_sigma * sigma_r[benign_mask]))
            confidences[attack_mask] = np.maximum(0.0, (r_mean[attack_mask] - self.k_sigma * sigma_r[attack_mask]))

        return ThreeWayDecisionResult(
            decisions=decisions,
            numeric_decisions=numeric_decisions,
            accepted_predictions=accepted_predictions,
            confidences=confidences,
            risk_min=r_min,
            risk_max=r_max,
            risk_mean=r_mean,
            risk_variance=r_var,
            risk_spread=r_spread,
        )

    def evaluate(
        self,
        risk_profile: RiskProfile,
        y_true: Union[pd.Series, np.ndarray],
        cost_fn: float = 10.0,
        cost_fp: float = 1.0,
        cost_inv: float = 0.15,
        secondary_triage_efficiency: Optional[float] = None,
        secondary_triage_tpr: Optional[float] = None,
        secondary_triage_fpr: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Rigorously evaluates SAFE-3WD performance, selective accuracy, coverage, and operational costs.

        Parameters
        ----------
        risk_profile : RiskProfile
            Propagated risk distribution from UncertaintyPropagator.
        y_true : Union[pd.Series, np.ndarray]
            True binary labels (0: Benign, 1: Attack).
        cost_fn : float
            Penalty per missed attack (False Negative).
        cost_fp : float
            Penalty per false alarm (False Positive).
        cost_inv : float
            Cost of deep manual/sandbox inspection per investigated flow.
        secondary_triage_efficiency : Optional[float]
            Legacy alias for secondary triage attack detection rate.
        secondary_triage_tpr : Optional[float]
            Secondary triage detection sensitivity P(detect | attack in triage).
        secondary_triage_fpr : Optional[float]
            Secondary triage false alarm probability P(flag | benign in triage).
        """
        y_t = np.asarray(y_true).ravel()
        res = self.decide(risk_profile)

        # Selective classification metrics
        sel_metrics = compute_selective_metrics(
            y_true=y_t,
            decisions=res.decisions,
            accepted_predictions=res.accepted_predictions,
        )

        # Regional performance breakdown
        is_accepted = res.accepted_predictions != -1
        is_benign = res.numeric_decisions == 0
        is_attack = res.numeric_decisions == 1
        is_inv = res.numeric_decisions == 2

        # Confusion counts on accepted flows
        false_negatives = int(np.sum((y_t == 1) & is_benign))  # Attack accepted as Benign (C_FN)
        false_positives = int(np.sum((y_t == 0) & is_attack))  # Benign blocked as Attack (C_FP)
        true_positives = int(np.sum((y_t == 1) & is_attack))
        true_negatives = int(np.sum((y_t == 0) & is_benign))

        # Investigation counts partitioned by true class
        total_attacks = int(np.sum(y_t == 1))
        total_benign = int(np.sum(y_t == 0))
        attack_inv_count = int(np.sum((y_t == 1) & is_inv))
        benign_inv_count = int(np.sum((y_t == 0) & is_inv))
        investigated_flows = int(np.sum(is_inv))

        n_accepted = int(np.sum(is_accepted))
        sel_acc = float((true_positives + true_negatives) / max(1, n_accepted))
        sel_dr = float(true_positives / max(1, (true_positives + false_negatives)))
        sel_fpr = float(false_positives / max(1, (false_positives + true_negatives)))

        # Direct (definite) global rates
        direct_detection_rate = float(true_positives / max(1, total_attacks))
        attack_investigation_rate = float(attack_inv_count / max(1, total_attacks))
        definite_fn_rate = float(false_negatives / max(1, total_attacks))

        direct_fpr = float(false_positives / max(1, total_benign))
        benign_investigation_rate = float(benign_inv_count / max(1, total_benign))
        direct_tnr = float(true_negatives / max(1, total_benign))

        # Selective F1 score on accepted flows
        prec_acc = float(true_positives / max(1, (true_positives + false_positives)))
        rec_acc = sel_dr
        sel_f1 = float(2.0 * prec_acc * rec_acc / max(1e-9, (prec_acc + rec_acc))) if (prec_acc + rec_acc) > 0 else 0.0

        # Operational Cost Evaluation
        total_cost = (
            (cost_fn * false_negatives)
            + (cost_fp * false_positives)
            + (cost_inv * investigated_flows)
        )
        normalized_cost = total_cost / max(1, len(y_t))

        result = {
            "coverage": sel_metrics["coverage"],
            "investigation_rate": sel_metrics["investigation_rate"],
            "selective_accuracy": sel_acc,
            "selective_risk": 1.0 - sel_acc,
            "selective_detection_rate": sel_dr,
            "selective_false_positive_rate": sel_fpr,
            "selective_f1_score": sel_f1,
            "direct_detection_rate": direct_detection_rate,
            "attack_investigation_rate": attack_investigation_rate,
            "definite_fn_rate": definite_fn_rate,
            "direct_fpr": direct_fpr,
            "benign_investigation_rate": benign_investigation_rate,
            "direct_tnr": direct_tnr,
            "true_positives_definite": true_positives,
            "true_negatives_definite": true_negatives,
            "false_negatives_definite": false_negatives,
            "false_positives_definite": false_positives,
            "attack_investigated_count": attack_inv_count,
            "benign_investigated_count": benign_inv_count,
            "investigated_count": investigated_flows,
            "total_attacks": total_attacks,
            "total_benign": total_benign,
            "total_samples": len(y_t),
            "operational_cost": float(total_cost),
            "normalized_cost_per_flow": float(normalized_cost),
            "tau_safe": self.tau_safe,
            "tau_attack": self.tau_attack,
            "max_acceptable_variance": self.max_acceptable_variance,
        }

        # Secondary Triage Modeling
        # Decouple secondary detection sensitivity (TPR) from secondary false positive rate (FPR)
        tpr = None
        fpr = None
        if secondary_triage_tpr is not None:
            tpr = float(np.clip(secondary_triage_tpr, 0.0, 1.0))
        elif secondary_triage_efficiency is not None:
            tpr = float(np.clip(secondary_triage_efficiency, 0.0, 1.0))

        if secondary_triage_fpr is not None:
            fpr = float(np.clip(secondary_triage_fpr, 0.0, 1.0))
        elif secondary_triage_efficiency is not None:
            # Backwards compatibility: if user only passes secondary_triage_efficiency
            fpr = float(np.clip(1.0 - secondary_triage_efficiency, 0.0, 1.0))

        if tpr is not None:
            if fpr is None:
                fpr = 0.02  # Standard nominal secondary inspection false alarm rate

            result["secondary_triage_tpr"] = tpr
            result["secondary_triage_fpr"] = fpr
            result["secondary_triage_efficiency"] = tpr

            post_dr = float(direct_detection_rate + tpr * attack_investigation_rate)
            post_fpr = float(direct_fpr + fpr * benign_investigation_rate)

            result["post_triage_detection_rate"] = post_dr
            result["post_triage_fpr"] = post_fpr

            # Post-triage residual false negatives and false positives
            # Exact continuous mathematical expectation:
            residual_fn_exp = float((1.0 - tpr) * attack_inv_count)
            secondary_fp_exp = float(fpr * benign_inv_count)

            post_triage_cost = (
                (cost_fn * (false_negatives + residual_fn_exp))
                + (cost_fp * (false_positives + secondary_fp_exp))
                + (cost_inv * investigated_flows)
            )
            result["residual_fn_count"] = int(round(residual_fn_exp))
            result["secondary_fp_count"] = int(round(secondary_fp_exp))
            result["post_triage_operational_cost"] = float(post_triage_cost)
            result["post_triage_cost_per_flow"] = float(post_triage_cost / max(1, len(y_t)))

        return result
