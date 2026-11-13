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
    risk_min: np.ndarray       # Shape (N,): Minimum risk score R_min = min_s r^(s)
    risk_max: np.ndarray       # Shape (N,): Maximum risk score R_max = max_s r^(s)
    risk_mean: np.ndarray      # Shape (N,): Expected risk mu_R = E_s[r^(s)]
    risk_variance: np.ndarray  # Shape (N,): Risk variance sigma_R^2 = Var_s[r^(s)]
    risk_std: np.ndarray       # Shape (N,): Risk standard deviation sigma_R
    risk_spread: np.ndarray    # Shape (N,): Interval width Delta_R = R_max - R_min
    risk_median: np.ndarray    # Shape (N,): Median risk score
    raw_risk_matrix: np.ndarray  # Shape (N, S): Full Monte Carlo risk evaluations

    def to_dataframe(self) -> pd.DataFrame:
        """
        Converts summary uncertainty statistics into a pandas DataFrame.
        """
        return pd.DataFrame({
            "risk_min": self.risk_min,
            "risk_max": self.risk_max,
            "risk_mean": self.risk_mean,
            "risk_variance": self.risk_variance,
            "risk_std": self.risk_std,
            "risk_spread": self.risk_spread,
            "risk_median": self.risk_median,
        })


class UncertaintyPropagator:
    """
    Propagates feature uncertainty through non-linear intrusion classifiers.
    """
    def __init__(
        self,
        base_classifier: Any,
        sdum: SamplingDistortionUncertaintyModel,
        n_mc_samples: int = 30,
    ):
        self.base_classifier = base_classifier
        self.sdum = sdum
        self.n_mc_samples = n_mc_samples

    def propagate(
        self,
        X_sampled: pd.DataFrame,
        sampling_rate: float,
        n_mc_samples: Optional[int] = None,
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

        # 3. Compute Per-Flow Epistemic Risk Statistics
        r_min = np.min(risk_matrix, axis=1)
        r_max = np.max(risk_matrix, axis=1)
        r_mean = np.mean(risk_matrix, axis=1)
        r_var = np.var(risk_matrix, axis=1, ddof=1) if S > 1 else np.zeros(n_rows)
        r_std = np.sqrt(np.maximum(0.0, r_var))
        r_spread = r_max - r_min
        r_median = np.median(risk_matrix, axis=1)

        return RiskProfile(
            risk_min=r_min,
            risk_max=r_max,
            risk_mean=r_mean,
            risk_variance=r_var,
            risk_std=r_std,
            risk_spread=r_spread,
            risk_median=r_median,
            raw_risk_matrix=risk_matrix,
        )
