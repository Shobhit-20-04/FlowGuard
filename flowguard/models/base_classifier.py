"""
FlowGuard Base Intrusion Classifier
====================================

Wraps high-performance gradient boosted decision trees (LightGBM, XGBoost)
and bagging ensembles (Random Forest) for network flow classification.
"""

import logging
import os
from typing import Any, Dict, Optional, Union
import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class BaseIntrusionClassifier:
    """
    Standardized wrapper for tree-based NIDS classifiers.
    """
    def __init__(
        self,
        model_type: str = "lightgbm",
        random_state: int = 42,
        **hyperparameters,
    ):
        self.model_type = model_type.lower()
        self.random_state = random_state
        self.hyperparameters = hyperparameters
        self.model = self._initialize_model()
        self.feature_names_: Optional[list] = None
        self.is_fitted: bool = False

    def _initialize_model(self) -> Any:
        if self.model_type in ["lightgbm", "lgb", "lgbm"]:
            import lightgbm as lgb
            default_params = {
                "n_estimators": 250,
                "learning_rate": 0.05,
                "num_leaves": 63,
                "max_depth": -1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": self.random_state,
                "verbose": -1,
                "n_jobs": -1,
            }
            default_params.update(self.hyperparameters)
            return lgb.LGBMClassifier(**default_params)

        elif self.model_type in ["random_forest", "rf"]:
            from sklearn.ensemble import RandomForestClassifier
            default_params = {
                "n_estimators": 200,
                "max_depth": 20,
                "min_samples_split": 5,
                "random_state": self.random_state,
                "n_jobs": -1,
            }
            default_params.update(self.hyperparameters)
            return RandomForestClassifier(**default_params)

        elif self.model_type in ["xgboost", "xgb"]:
            import xgboost as xgb
            default_params = {
                "n_estimators": 250,
                "learning_rate": 0.05,
                "max_depth": 6,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": self.random_state,
                "eval_metric": "logloss",
                "n_jobs": -1,
            }
            default_params.update(self.hyperparameters)
            return xgb.XGBClassifier(**default_params)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}. Choose lightgbm, random_forest, or xgboost.")

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> "BaseIntrusionClassifier":
        """
        Trains the classifier on pristine complete flow data.
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = X.columns.tolist()
        else:
            self.feature_names_ = [f"f_{i}" for i in range(X.shape[1])]

        logger.info(f"Training {self.model_type.upper()} model on {len(X)} samples with {X.shape[1]} features...")
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray], threshold: float = 0.5) -> np.ndarray:
        """
        Predicts binary intrusion label (0: Benign, 1: Attack) given decision threshold.
        """
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(np.int32)

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Returns continuous attack probability score P(Attack | X) in [0, 1].
        """
        if not self.is_fitted:
            raise RuntimeError("Classifier must be fitted before predict_proba()!")
        probs = self.model.predict_proba(X)
        if probs.shape[1] == 2:
            return probs[:, 1]
        return probs[:, 0]

    @property
    def feature_importances_(self) -> pd.Series:
        """
        Returns ranked feature importances.
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted!")
        importances = getattr(self.model, "feature_importances_", None)
        if importances is not None and self.feature_names_ is not None:
            return pd.Series(importances, index=self.feature_names_).sort_values(ascending=False)
        return pd.Series()

    def save(self, path: str) -> None:
        """
        Serializes trained model artifact to disk.
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        joblib.dump({
            "model_type": self.model_type,
            "random_state": self.random_state,
            "feature_names": self.feature_names_,
            "model": self.model,
        }, path)
        logger.info(f"Saved {self.model_type} model to {path}")

    @classmethod
    def load(cls, path: str) -> "BaseIntrusionClassifier":
        """
        Loads serialized model artifact from disk.
        """
        data = joblib.load(path)
        instance = cls(model_type=data["model_type"], random_state=data["random_state"])
        instance.model = data["model"]
        instance.feature_names_ = data["feature_names"]
        instance.is_fitted = True
        return instance
