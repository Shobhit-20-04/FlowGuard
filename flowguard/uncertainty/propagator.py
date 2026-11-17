"""
FlowGuard Feature Uncertainty Propagation Engine
=================================================

Propagates the ensemble of S plausible true flow representations through the
trained base classifier f_theta to compute empirical risk distributions,
bounding intervals [R_min, R_max], epistemic spread, and predictive variance.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from flowguard.uncertainty.sdum import SamplingDistortionUncertaintyModel


@dataclass
class RiskProfile:
    """
    Encapsulates the propagated risk distribution across S Monte Carlo flow reconstructions.
    """
    risk_min: np.ndarray       # Shape (N,): Minimum sample risk score R_min = min_s r^(s)
    risk_max: np.ndarray       # Shape (N,): Maximum sample risk score R_max = max_s r^(s)
    risk_mean: np.ndarray      # Shape (N,): Expected risk mu_R = E_s[r^(s)]
    risk_variance: np.ndarray  # Shape (N,): Risk variance sigma_R^2 = Var_s[r^(s)]
    risk_std: np.ndarray       # Shape (N,): Risk standard deviation sigma_R
    risk_spread: np.ndarray    # Shape (N,): Sample interval width Delta_R = R_max - R_min
    risk_median: np.ndarray    # Shape (N,): Median risk score
    raw_risk_matrix: np.ndarray  # Shape (N, S): Full Monte Carlo risk evaluations
    risk_q05: Optional[np.ndarray] = None  # Shape (N,): 5th percentile risk (lower 90% credible bound)
    risk_q95: Optional[np.ndarray] = None  # Shape (N,): 95th percentile risk (upper 90% credible bound)
    alpha_confidence: float = 0.90          # Coverage level alpha in (0, 1)
    risk_q_low: Optional[np.ndarray] = None # Shape (N,): Lower credible bound Q((1-alpha)/2)
    risk_q_high: Optional[np.ndarray] = None # Shape (N,): Upper credible bound Q((1+alpha)/2)

    def to_dataframe(self, include_credible_interval: bool = False) -> pd.DataFrame:
        """
        Converts summary uncertainty statistics into a pandas DataFrame.
        """
        data = {
            "risk_min": self.risk_min,
            "risk_max": self.risk_max,
            "risk_mean": self.risk_mean,
            "risk_variance": self.risk_variance,
            "risk_std": self.risk_std,
            "risk_spread": self.risk_spread,
            "risk_median": self.risk_median,
        }
        if include_credible_interval:
            if self.risk_q_low is not None and self.risk_q_high is not None:
                data["risk_q_low"] = self.risk_q_low
                data["risk_q_high"] = self.risk_q_high
            if self.risk_q05 is not None and self.risk_q95 is not None:
                data["risk_q05"] = self.risk_q05
                data["risk_q95"] = self.risk_q95
        return pd.DataFrame(data)


class UncertaintyPropagator:
    """
    Propagates feature uncertainty through non-linear intrusion classifiers.
    """
    def __init__(
        self,
        base_classifier: Any,
        sdum: SamplingDistortionUncertaintyModel,
        n_mc_samples: int = 30,
        alpha_confidence: float = 0.90,
    ):
        self.base_classifier = base_classifier
        self.sdum = sdum
        self.n_mc_samples = n_mc_samples
        self.alpha_confidence = float(alpha_confidence)

    def propagate(
        self,
        X_sampled: pd.DataFrame,
        sampling_rate: float,
        n_mc_samples: Optional[int] = None,
        alpha_confidence: Optional[float] = None,
    ) -> RiskProfile:
        """
        Propagates degraded flow records through SDUM reconstruction and base classifier inference.

        Parameters
        ----------
        X_sampled : pd.DataFrame
            Observed degraded flows of shape (N, D).
        sampling_rate : float
            Packet sampling probability p in (0, 1].
        n_mc_samples : Optional[int]
            Number of Monte Carlo realizations S (defaults to self.n_mc_samples).

        Returns
        -------
        RiskProfile
            Empirical risk distribution and interval bounds for all N flows.
        """
        S = n_mc_samples if n_mc_samples is not None else self.n_mc_samples
        n_rows = len(X_sampled)

        # 1. Generate S plausible true flow realizations via SDUM
        reconstructed_dfs = self.sdum.reconstruct_flow_ensemble(
            X_sampled=X_sampled,
            sampling_rate=sampling_rate,
            n_mc_samples=S,
        )

        # 2. Vectorized Batch Inference:
        # Concatenate all S realizations into a single matrix of shape (N * S, D)
        # for high-throughput batch evaluation through f_theta
        stacked_df = pd.concat(reconstructed_dfs, axis=0, ignore_index=True)

        if hasattr(self.base_classifier, "predict_proba"):
            probs_flat = self.base_classifier.predict_proba(stacked_df)
            if probs_flat.ndim == 2 and probs_flat.shape[1] == 2:
                probs_flat = probs_flat[:, 1]
        else:
            raise ValueError("Base classifier must support predict_proba()!")

        # Reshape flat predictions (N * S,) into risk matrix (S, N) -> transpose to (N, S)
        # Each column s contains the predictions for realization s across all N flows
        risk_matrix = probs_flat.reshape((S, n_rows)).T  # Shape: (N, S)

        # 3. Compute Per-Flow Epistemic Risk Statistics & Credible Quantiles
        alpha = float(alpha_confidence if alpha_confidence is not None else self.alpha_confidence)
        assert 0.0 < alpha < 1.0, f"alpha_confidence must be in (0, 1), got {alpha}"

        # Central credible interval quantiles:
        # Lower: (1 - alpha) / 2  (e.g., 0.05 for alpha=0.90)
        # Upper: (1 + alpha) / 2  (e.g., 0.95 for alpha=0.90)
        q_low_pct = 100.0 * (1.0 - alpha) / 2.0
        q_high_pct = 100.0 * (1.0 + alpha) / 2.0

        r_min = np.min(risk_matrix, axis=1)
        r_max = np.max(risk_matrix, axis=1)
        r_mean = np.mean(risk_matrix, axis=1)
        r_var = np.var(risk_matrix, axis=1, ddof=1) if S > 1 else np.zeros(n_rows)
        r_std = np.sqrt(np.maximum(0.0, r_var))
        r_spread = r_max - r_min
        r_median = np.median(risk_matrix, axis=1)
        r_q_low = np.percentile(risk_matrix, q_low_pct, axis=1) if S > 1 else r_min
        r_q_high = np.percentile(risk_matrix, q_high_pct, axis=1) if S > 1 else r_max
        r_q05 = np.percentile(risk_matrix, 5.0, axis=1) if S > 1 else r_min
        r_q95 = np.percentile(risk_matrix, 95.0, axis=1) if S > 1 else r_max

        return RiskProfile(
            risk_min=r_min,
            risk_max=r_max,
            risk_mean=r_mean,
            risk_variance=r_var,
            risk_std=r_std,
            risk_spread=r_spread,
            risk_median=r_median,
            risk_q05=r_q05,
            risk_q95=r_q95,
            alpha_confidence=alpha,
            risk_q_low=r_q_low,
            risk_q_high=r_q_high,
            raw_risk_matrix=risk_matrix,
        )
