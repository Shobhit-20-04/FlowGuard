"""
FlowGuard Evaluation Metrics & Calibration Library
===================================================

Computes standard intrusion detection metrics (DR, FPR, FNR, F1),
calibration metrics (ECE, Brier Score), and three-way selective decision metrics.
"""

from typing import Dict, List, Optional, Tuple, Union
import logging
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)

def compute_expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 15,
) -> float:
    """
    Computes the Expected Calibration Error (ECE):
    Partitions probability range [0, 1] into n_bins equal-width bins,
    measuring the weighted average discrepancy between empirical accuracy and model confidence.
    """
    y_true = np.asarray(y_true).astype(np.int32)
    y_prob = np.asarray(y_prob).astype(np.float64)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n_total = len(y_true)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        # Handle inclusive right edge on last bin
        if i == n_bins - 1:
            in_bin = (y_prob >= bin_lower) & (y_prob <= bin_upper)
        else:
            in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper)

        bin_size = np.sum(in_bin)
        if bin_size > 0:
            bin_acc = np.mean(y_true[in_bin])
            bin_conf = np.mean(y_prob[in_bin])
            ece += (bin_size / n_total) * np.abs(bin_acc - bin_conf)

    return float(ece)

def compute_brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Mean squared error between predicted probability and true binary label.
    """
    y_true = np.asarray(y_true).astype(np.float64)
    y_prob = np.asarray(y_prob).astype(np.float64)
    return float(np.mean((y_prob - y_true) ** 2))

def compute_classification_metrics(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
    y_prob: Optional[Union[pd.Series, np.ndarray]] = None,
) -> Dict[str, float]:
    """
    Computes complete suite of standard and NIDS-critical evaluation metrics:
    - Detection Rate (Recall / Sensitivity)
    - False Negative Rate (FNR = 1 - DR)
    - False Positive Rate (FPR)
    - Precision, Accuracy, F1-Score
    - ROC-AUC, PR-AUC, ECE, Brier Score
    """
    y_t = np.asarray(y_true).astype(np.int32)
    y_p = np.asarray(y_pred).astype(np.int32)

    tn, fp, fn, tp = confusion_matrix(y_t, y_p, labels=[0, 1]).ravel()

    acc = accuracy_score(y_t, y_p)
    prec = precision_score(y_t, y_p, zero_division=0)
    rec = recall_score(y_t, y_p, zero_division=0) # Detection Rate
    f1 = f1_score(y_t, y_p, zero_division=0)

    # NIDS specific rates
    detection_rate = rec
    fnr = float(fn / max(1, (tp + fn)))
    fpr = float(fp / max(1, (fp + tn)))

    metrics = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "detection_rate": float(detection_rate),
        "f1_score": float(f1),
        "false_positive_rate": float(fpr),
        "false_negative_rate": float(fnr),
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "total_samples": int(len(y_t)),
    }

    if y_prob is not None:
        y_pr = np.asarray(y_prob).astype(np.float64)
        if len(np.unique(y_t)) > 1:
            try:
                metrics["roc_auc"] = float(roc_auc_score(y_t, y_pr))
            except Exception as e:
                logger.warning(f"ROC-AUC computation failed: {e}")
                metrics["roc_auc"] = float("nan")

            try:
                prec_c, rec_c, _ = precision_recall_curve(y_t, y_pr)
                metrics["pr_auc"] = float(auc(rec_c, prec_c))
            except Exception as e:
                logger.warning(f"PR-AUC computation failed: {e}")
                metrics["pr_auc"] = float("nan")
        else:
            logger.warning("Only one class present in y_true. ROC-AUC and PR-AUC are undefined.")
            metrics["roc_auc"] = float("nan")
            metrics["pr_auc"] = float("nan")

        metrics["brier_score"] = compute_brier_score(y_t, y_pr)
        metrics["expected_calibration_error"] = compute_expected_calibration_error(y_t, y_pr)

    return metrics

def compute_selective_metrics(
    y_true: np.ndarray,
    decisions: np.ndarray,
    accepted_predictions: np.ndarray,
) -> Dict[str, float]:
    """
    Computes metrics for a three-way decision system:
    - Coverage (C): Proportion of flows accepted without deep investigation.
    - Investigation Rate (I): Proportion of flows routed to Region 3 (Investigate).
    - Selective Accuracy: Accuracy on accepted decisions.
    - Selective Risk: 1 - Selective Accuracy.
    """
    y_true = np.asarray(y_true)
    decisions = np.asarray(decisions)
    accepted_predictions = np.asarray(accepted_predictions)

    # Validate decision labels
    valid_labels = {"BENIGN", "ATTACK", "INVESTIGATE", 0, 1, 2}
    unique_decs = set(np.unique(decisions))
    if not unique_decs.issubset(valid_labels):
        invalid = unique_decs - valid_labels
        raise ValueError(f"Invalid decision labels encountered in selective metrics: {invalid}")

    n_total = len(y_true)
    is_investigate = (decisions == "INVESTIGATE") | (decisions == 2)
    is_accepted = ~is_investigate

    n_accepted = np.sum(is_accepted)
    n_investigate = np.sum(is_investigate)

    coverage = float(n_accepted / max(1, n_total))
    investigation_rate = float(n_investigate / max(1, n_total))

    if n_accepted > 0:
        y_acc = y_true[is_accepted]
        pred_acc = accepted_predictions[is_accepted]
        sel_acc = float(np.mean(y_acc == pred_acc))
        sel_risk = 1.0 - sel_acc
        tp = int(np.sum((y_acc == 1) & (pred_acc == 1)))
        fp = int(np.sum((y_acc == 0) & (pred_acc == 1)))
        fn = int(np.sum((y_acc == 1) & (pred_acc == 0)))
        tn = int(np.sum((y_acc == 0) & (pred_acc == 0)))
        prec = float(tp / max(1, tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / max(1, tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2.0 * prec * rec / max(1e-9, prec + rec)) if (prec + rec) > 0 else 0.0
    else:
        sel_acc = 0.0
        sel_risk = 0.0
        prec = 0.0
        rec = 0.0
        f1 = 0.0

    return {
        "coverage": coverage,
        "investigation_rate": investigation_rate,
        "selective_accuracy": sel_acc,
        "selective_risk": sel_risk,
        "selective_precision": prec,
        "selective_recall": rec,
        "selective_f1_score": f1,
        "accepted_count": int(n_accepted),
        "investigated_count": int(n_investigate),
        "total_samples": int(n_total),
    }
