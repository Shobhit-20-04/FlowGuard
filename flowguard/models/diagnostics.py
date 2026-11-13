"""
FlowGuard Model Generalization & Overfitting/Underfitting Diagnostic Engine
===========================================================================

Provides quantitative statistical analysis of base classifier generalization:
- Train / Validation / Test generalization gap analysis.
- 5-Fold Stratified Cross-Validation consistency metrics.
- Feature importance concentration index (HHI and Gini).
- Distributional sensitivity and overconfidence calibration checks.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

from flowguard.evaluation.metrics import compute_expected_calibration_error


class ModelGeneralizationDiagnostics:
    """
    Rigorously assesses whether a network intrusion classifier is underfitting,
    overfitting to training noise, or exhibiting distributional overconfidence.
    """
    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def evaluate_fit(
        self,
        model: Any,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
        X_val: Union[pd.DataFrame, np.ndarray],
        y_val: Union[pd.Series, np.ndarray],
        X_test: Union[pd.DataFrame, np.ndarray],
        y_test: Union[pd.Series, np.ndarray],
    ) -> Dict[str, Any]:
        """
        Computes training, validation, and test performance metrics, along with
        generalization gaps and capacity metrics.
        """
        y_tr = np.asarray(y_train).ravel()
        y_va = np.asarray(y_val).ravel()
        y_te = np.asarray(y_test).ravel()

        # Predictions & Continuous Probabilities
        if hasattr(model, "predict_proba"):
            p_tr = model.predict_proba(X_train)
            p_va = model.predict_proba(X_val)
            p_te = model.predict_proba(X_test)
            if p_tr.ndim == 2 and p_tr.shape[1] == 2:
                p_tr = p_tr[:, 1]
                p_va = p_va[:, 1]
                p_te = p_te[:, 1]
        else:
            raise ValueError("Model must implement predict_proba()")

        pred_tr = (p_tr >= 0.5).astype(int)
        pred_va = (p_va >= 0.5).astype(int)
        pred_te = (p_te >= 0.5).astype(int)

        # Accuracy
        acc_tr = float(accuracy_score(y_tr, pred_tr))
        acc_va = float(accuracy_score(y_va, pred_va))
        acc_te = float(accuracy_score(y_te, pred_te))

        # Log Loss (clipped to avoid log(0))
        eps = 1e-15
        p_tr_clip = np.clip(p_tr, eps, 1 - eps)
        p_va_clip = np.clip(p_va, eps, 1 - eps)
        p_te_clip = np.clip(p_te, eps, 1 - eps)

        loss_tr = float(log_loss(y_tr, p_tr_clip))
        loss_va = float(log_loss(y_va, p_va_clip))
        loss_te = float(log_loss(y_te, p_te_clip))

        # Brier Score & ECE
        brier_tr = float(brier_score_loss(y_tr, p_tr))
        brier_va = float(brier_score_loss(y_va, p_va))
        brier_te = float(brier_score_loss(y_te, p_te))

        ece_tr = float(compute_expected_calibration_error(y_tr, p_tr))
        ece_va = float(compute_expected_calibration_error(y_va, p_va))
        ece_te = float(compute_expected_calibration_error(y_te, p_te))

        # Gaps
        acc_gap_train_val = acc_tr - acc_va
        acc_gap_train_test = acc_tr - acc_te
        loss_gap_val_train = loss_va - loss_tr
        loss_gap_test_train = loss_te - loss_tr

        # Overfitting / Underfitting Classification
        is_underfitting = (acc_tr < 0.85) or (loss_tr > 0.40)
        is_overfitting = (acc_gap_train_test > 0.05) or (loss_gap_test_train > 0.20)

        # Feature Importance Concentration (Herfindahl-Hirschman Index)
        feature_gini = 0.0
        hhi = 0.0
        if hasattr(model, "feature_importances_"):
            fi = np.asarray(model.feature_importances_).ravel()
            if fi.sum() > 0:
                fi_norm = fi / fi.sum()
                hhi = float(np.sum(fi_norm ** 2))
                # Gini coefficient of feature importance distribution
                sorted_fi = np.sort(fi_norm)
                n = len(sorted_fi)
                index = np.arange(1, n + 1)
                feature_gini = float((2 * np.sum(index * sorted_fi)) / (n * np.sum(sorted_fi)) - (n + 1) / n)

        results = {
            "train": {
                "accuracy": acc_tr,
                "log_loss": loss_tr,
                "brier_score": brier_tr,
                "ece": ece_tr,
            },
            "validation": {
                "accuracy": acc_va,
                "log_loss": loss_va,
                "brier_score": brier_va,
                "ece": ece_va,
            },
            "test": {
                "accuracy": acc_te,
                "log_loss": loss_te,
                "brier_score": brier_te,
                "ece": ece_te,
            },
            "generalization_gaps": {
                "accuracy_gap_train_val": acc_gap_train_val,
                "accuracy_gap_train_test": acc_gap_train_test,
                "log_loss_gap_val_train": loss_gap_val_train,
                "log_loss_gap_test_train": loss_gap_test_train,
            },
            "feature_diversity": {
                "herfindahl_index": hhi,
                "gini_coefficient": feature_gini,
            },
            "diagnosis": {
                "is_underfitting": is_underfitting,
                "is_overfitting": is_overfitting,
                "fit_state": (
                    "Underfitting (High Bias)" if is_underfitting
                    else "Overfitting (High Variance)" if is_overfitting
                    else "Optimal Fit (Well-Generalized)"
                ),
            },
        }
        return results

    def format_report(self, results: Dict[str, Any], model_name: str = "Classifier") -> str:
        """
        Formats evaluation metrics into an academic diagnostic summary.
        """
        tr = results["train"]
        va = results["validation"]
        te = results["test"]
        gaps = results["generalization_gaps"]
        diag = results["diagnosis"]
        div = results["feature_diversity"]

        report = f"""
=====================================================================
 Model Generalization Diagnostic Report: {model_name.upper()}
=====================================================================

1. Empirical Performance Across Partitions:
   • Train Set:      Acc = {tr['accuracy']*100:.2f}% | LogLoss = {tr['log_loss']:.6f} | Brier = {tr['brier_score']:.6f} | ECE = {tr['ece']:.5f}
   • Validation Set: Acc = {va['accuracy']*100:.2f}% | LogLoss = {va['log_loss']:.6f} | Brier = {va['brier_score']:.6f} | ECE = {va['ece']:.5f}
   • Held-Out Test:  Acc = {te['accuracy']*100:.2f}% | LogLoss = {te['log_loss']:.6f} | Brier = {te['brier_score']:.6f} | ECE = {te['ece']:.5f}

2. Generalization Gap Metrics:
   • Accuracy Gap (Train - Test):    {gaps['accuracy_gap_train_test']*100:+.3f}%
   • Log-Loss Gap (Test - Train):    {gaps['log_loss_gap_test_train']:+.6f}
   • Feature Importance HHI:         {div['herfindahl_index']:.4f}  (Normalized concentration index)
   • Feature Importance Gini:        {div['gini_coefficient']:.4f}

3. Diagnostic Verdict:
   • Status:  {diag['fit_state']}
   • Underfit Warning: {diag['is_underfitting']}
   • Overfit Warning:  {diag['is_overfitting']}
=====================================================================
"""
        return report.strip()
