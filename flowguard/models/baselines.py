"""
FlowGuard Comparative Baseline Paradigms
========================================

Implements the four competitive baseline paradigms against FlowGuard SAFE-3WD:
- Baseline 1 (B1): Oblivious Model (Standard NIDS blind to sampling)
- Baseline 2 (B2): Feature Imputation (Mean / MICE deterministic scaling: X_hat = X_tilde / p)
- Baseline 3 (B3): Sampling-Augmented Retraining (Multi-condition retrained robust classifier)
- Baseline 4 (B4): Deep Uncertainty / Ensemble (Parameter uncertainty without physical sampling model)
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from flowguard.evaluation.metrics import compute_classification_metrics, compute_selective_metrics
from flowguard.models.base_classifier import BaseIntrusionClassifier
from flowguard.sampling.simulator import PacketSamplingSimulator


class ObliviousBaseline:
    """
    Baseline 1 (B1): Oblivious Model
    Standard classifier trained on complete pristine flows, evaluated directly on raw sampled flows
    with fixed decision threshold tau = 0.5.
    """
    def __init__(self, base_model: BaseIntrusionClassifier, threshold: float = 0.5):
        self.base_model = base_model
        self.threshold = threshold

    def predict(self, X_degraded: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        probs = self.predict_proba(X_degraded)
        return (probs >= self.threshold).astype(np.int32)

    def predict_proba(self, X_degraded: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        return self.base_model.predict_proba(X_degraded)


class ImputationBaseline:
    """
    Baseline 2 (B2): Feature Imputation Model
    Scales observed packet counts, byte volumes, and temporal moments deterministically
    by 1/p (the classical point-estimate imputation in network flow literature):
      N_hat = k / p
      V_hat = v / p
      IAT_hat = IAT_tilde * p
    Feeds the imputed point-estimate vector into the pristine base classifier at threshold tau = 0.5.
    """
    def __init__(self, base_model: BaseIntrusionClassifier, threshold: float = 0.5):
        self.base_model = base_model
        self.threshold = threshold

    def impute(self, X_degraded: pd.DataFrame, sampling_rate: float) -> pd.DataFrame:
        """
        Applies deterministic 1/p feature rescaling to additive flow features.
        """
        p = float(np.clip(sampling_rate, 1e-4, 1.0))
        if p >= 1.0:
            return X_degraded.copy()

        df_imputed = X_degraded.copy()
        scale = 1.0 / p

        # 1. Rescale packet counts
        for col in df_imputed.columns:
            col_l = col.lower()
            if "packet" in col_l and ("total" in col_l or "subflow" in col_l):
                df_imputed[col] = df_imputed[col].values * scale
            elif "byte" in col_l or "header_len" in col_l:
                df_imputed[col] = df_imputed[col].values * scale
            elif "iat" in col_l and ("mean" in col_l or "total" in col_l or "min" in col_l or "max" in col_l):
                df_imputed[col] = df_imputed[col].values * p

        # Recalculate packet rates and byte rates
        if "flow_duration" in df_imputed.columns:
            dur_sec = np.maximum(1e-4, df_imputed["flow_duration"].values / 1e6)
            fwd_col = "total_fwd_packets" if "total_fwd_packets" in df_imputed.columns else None
            bwd_col = "total_bwd_packets" if "total_bwd_packets" in df_imputed.columns else None
            if fwd_col and "flow_packets_per_s" in df_imputed.columns:
                fwd_p = df_imputed[fwd_col].values
                bwd_p = df_imputed[bwd_col].values if bwd_col else 0.0
                df_imputed["flow_packets_per_s"] = (fwd_p + bwd_p) / dur_sec
            if "total_fwd_bytes" in df_imputed.columns and "flow_bytes_per_s" in df_imputed.columns:
                fwd_b = df_imputed["total_fwd_bytes"].values
                bwd_b = df_imputed["total_bwd_bytes"].values if "total_bwd_bytes" in df_imputed.columns else 0.0
                df_imputed["flow_bytes_per_s"] = (fwd_b + bwd_b) / dur_sec

        return df_imputed

    def predict(self, X_degraded: pd.DataFrame, sampling_rate: float) -> np.ndarray:
        probs = self.predict_proba(X_degraded, sampling_rate=sampling_rate)
        return (probs >= self.threshold).astype(np.int32)

    def predict_proba(self, X_degraded: pd.DataFrame, sampling_rate: float) -> np.ndarray:
        X_imp = self.impute(X_degraded, sampling_rate=sampling_rate)
        return self.base_model.predict_proba(X_imp)


class SamplingAugmentedBaseline:
    """
    Baseline 3 (B3): Sampling-Augmented Retraining
    Retrains the base classifier on a multi-condition mixture dataset containing both
    pristine flows and flows decimated across multiple sampling rates (p in [0.01, 1.0]).
    At test time, the model evaluates raw degraded flows directly at threshold tau = 0.5.
    """
    def __init__(
        self,
        model_type: str = "lightgbm",
        random_state: int = 42,
        multi_rate_sweep: Optional[List[float]] = None,
        threshold: float = 0.5,
    ):
        self.model_type = model_type
        self.random_state = random_state
        self.multi_rate_sweep = multi_rate_sweep or [1.0, 0.50, 0.20, 0.10, 0.05, 0.02, 0.01]
        self.threshold = threshold
        self.model: Optional[BaseIntrusionClassifier] = None
        self.is_fitted: bool = False

    def fit(self, X_pristine: pd.DataFrame, y_pristine: pd.Series) -> "SamplingAugmentedBaseline":
        """
        Creates an augmented multi-rate training set by decimating fractions of the pristine
        training dataset across the sampling rate sweep, then fits a robust classifier.
        """
        simulator = PacketSamplingSimulator(seed=self.random_state)
        n_samples = len(X_pristine)
        slice_size = n_samples // len(self.multi_rate_sweep)

        augmented_dfs = []
        augmented_ys = []

        rng = np.random.default_rng(self.random_state)
        perm = rng.permutation(n_samples)

        for i, p in enumerate(self.multi_rate_sweep):
            start_idx = i * slice_size
            end_idx = (i + 1) * slice_size if i < len(self.multi_rate_sweep) - 1 else n_samples
            idx_slice = perm[start_idx:end_idx]

            X_slice = X_pristine.iloc[idx_slice].copy()
            y_slice = y_pristine.iloc[idx_slice].copy()

            if p < 1.0:
                X_decimated = simulator.sample_random(X_slice, sampling_rate=p)
            else:
                X_decimated = X_slice

            augmented_dfs.append(X_decimated)
            augmented_ys.append(y_slice)

        X_aug = pd.concat(augmented_dfs, axis=0, ignore_index=True)
        y_aug = pd.concat(augmented_ys, axis=0, ignore_index=True)

        self.model = BaseIntrusionClassifier(
            model_type=self.model_type,
            random_state=self.random_state,
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=8,
            min_child_samples=40,
            reg_alpha=0.1,
            reg_lambda=1.0,
            colsample_bytree=0.8,
            subsample=0.8,
        )
        self.model.fit(X_aug, y_aug)
        self.is_fitted = True
        return self

    def predict(self, X_degraded: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        probs = self.predict_proba(X_degraded)
        return (probs >= self.threshold).astype(np.int32)

    def predict_proba(self, X_degraded: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        if not self.is_fitted or self.model is None:
            raise RuntimeError("SamplingAugmentedBaseline must be fitted before predict_proba()!")
        return self.model.predict_proba(X_degraded)


class DeepEnsembleUncertaintyBaseline:
    """
    Baseline 4 (B4): Deep Ensemble / Multi-Model Uncertainty
    Trains an ensemble of K independently seeded models on the training data.
    At inference time, computes predictive mean p_bar and epistemic variance sigma_ens^2.
    Applies selective rejection when model predictive variance exceeds a rejection threshold.
    """
    def __init__(
        self,
        n_ensemble: int = 5,
        model_type: str = "lightgbm",
        random_state: int = 42,
        variance_rejection_thresh: float = 0.05,
        threshold: float = 0.5,
    ):
        self.n_ensemble = n_ensemble
        self.model_type = model_type
        self.random_state = random_state
        self.variance_rejection_thresh = variance_rejection_thresh
        self.threshold = threshold
        self.models: List[BaseIntrusionClassifier] = []
        self.is_fitted: bool = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "DeepEnsembleUncertaintyBaseline":
        """
        Trains K diverse models using distinct random seeds and feature bagging.
        """
        self.models = []
        for k in range(self.n_ensemble):
            seed_k = self.random_state + k * 101
            m = BaseIntrusionClassifier(
                model_type=self.model_type,
                random_state=seed_k,
                n_estimators=100,
                learning_rate=0.05,
                num_leaves=25,
                max_depth=6,
                min_child_samples=30,
                colsample_bytree=0.7,
                subsample=0.8,
            )
            m.fit(X, y)
            self.models.append(m)

        self.is_fitted = True
        return self

    def predict_with_uncertainty(
        self, X_degraded: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes ensemble mean probability, predictive variance, binary predictions,
        and rejection mask (flows where variance > threshold).

        Returns
        -------
        Tuple[p_mean, p_var, decisions, rejected_mask]
        """
        if not self.is_fitted:
            raise RuntimeError("Ensemble must be fitted before inference!")

        preds_matrix = np.array([m.predict_proba(X_degraded) for m in self.models])  # (K, N)
        p_mean = np.mean(preds_matrix, axis=0)  # (N,)
        p_var = np.var(preds_matrix, axis=0, ddof=1)  # (N,)

        rejected_mask = p_var > self.variance_rejection_thresh

        # Standard binary predictions for non-rejected flows
        decisions = np.where(rejected_mask, "INVESTIGATE", np.where(p_mean >= self.threshold, "ATTACK", "BENIGN"))
        accepted_predictions = np.where(rejected_mask, -1, (p_mean >= self.threshold).astype(int))

        return p_mean, p_var, decisions, accepted_predictions
