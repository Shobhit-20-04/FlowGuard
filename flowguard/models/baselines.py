"""
FlowGuard Baseline Paradigms
============================

Implements the comparative baseline paradigms:
- Baseline 1: Oblivious Model (Standard NIDS oblivious to sampling)
- Baseline 2: Feature Imputation (Mean / MICE / Autoencoder scaling)
- Baseline 3: Sampling-Augmented Retraining
- Baseline 4: Traditional Deep Uncertainty (MC-Dropout / Deep Ensemble)
"""

from typing import Dict, Optional, Union
import numpy as np
import pandas as pd

from flowguard.models.base_classifier import BaseIntrusionClassifier

class ObliviousBaseline:
    """
    Baseline 1 (B1): Oblivious Model
    Standard classifier trained on complete flows, evaluated directly on raw sampled flows
    with fixed decision threshold tau = 0.5.
    """
    def __init__(self, base_model: BaseIntrusionClassifier, threshold: float = 0.5):
        self.base_model = base_model
        self.threshold = threshold

    def predict(self, X_degraded: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Direct point-estimate classification on sampled traffic without sampling awareness.
        """
        probs = self.predict_proba(X_degraded)
        return (probs >= self.threshold).astype(np.int32)

    def predict_proba(self, X_degraded: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Returns point-estimate risk scores P(Attack | X_tilde).
        """
        return self.base_model.predict_proba(X_degraded)
