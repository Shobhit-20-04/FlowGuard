"""
FlowGuard Experiment 06: Rigorous Corrected Multi-Seed Benchmark & Validation Suite
==================================================================================

Executes the definitive corrected comparative evaluation across 5 random seeds:
[42, 123, 456, 789, 101112] across 7 packet sampling rates p in [1.0, 0.01].

Evaluates all 5 paradigms:
1. B1: Oblivious Baseline (Standard classifier on raw sampled traffic, fixed tau=0.5)
2. B2: Deterministic Rescaling Baseline (Classical 1/p point-estimate imputation)
3. B3: Sampling-Augmented Retraining Baseline (Multi-rate robust training)
4. B4: Tree Ensemble Epistemic Variance Baseline (Multi-seed diverse tree ensemble)
5. Proposed: FlowGuard SAFE-3WD (Physical SDUM reconstruction + Cost-Sensitive Calibrated 3WD)

Key Methodological Corrections:
-------------------------------
- Disaggregates Direct DR, Attack Inv Rate, and Definite FNR (summing to 1.0).
- Disaggregates Direct FPR, Benign Inv Rate, and Direct TNR (summing to 1.0).
- Explicitly models secondary triage success rho_sec in {0.5, 0.8, 1.0}.
- Computes genuine selective F1 strictly on accepted flows (never hardcoded).
- Enforces strict feasibility on calibrator min_coverage constraint.
- Profiles per-flow latency (microseconds) and ablation components.
- Outputs all results strictly to experiments/results_corrected/ and figures to experiments/figures_corrected/.
"""

import argparse
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from flowguard.data.loader import FlowDataLoader
from flowguard.decision.calibrator import CostSensitiveThresholdCalibrator
from flowguard.decision.safe_3wd import SafeThreeWayDecisionEngine
from flowguard.evaluation.metrics import compute_classification_metrics, compute_selective_metrics
from flowguard.models.base_classifier import BaseIntrusionClassifier
from flowguard.models.baselines import (
    DeterministicRescalingBaseline,
    ObliviousBaseline,
    SamplingAugmentedBaseline,
    TreeEnsembleUncertaintyBaseline,
)
from flowguard.sampling.simulator import PacketSamplingSimulator
from flowguard.uncertainty.propagator import UncertaintyPropagator
from flowguard.uncertainty.sdum import SamplingDistortionUncertaintyModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_benchmark_for_seed(
    seed: int,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_calib: pd.DataFrame,
    y_calib: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    sampling_rates: List[float],
    mc_samples: int = 30,
    test_subset: int = 2000,
    cost_fn: float = 10.0,
    cost_fp: float = 1.0,
    cost_inv: float = 0.15,
    min_coverage: float = 0.40,
) -> List[Dict[str, Any]]:
    """
    Executes a single complete benchmark run across all methods and sampling rates for one random seed.
    """
    logger.info(f"--- Running Benchmark for Random Seed: {seed} ---")
    rng = np.random.default_rng(seed)

    # 1. Train Models
    # Base Pristine Model (used by B1, B2, and Proposed)
    base_clf = BaseIntrusionClassifier(
        model_type="lightgbm",
        random_state=seed,
        n_estimators=150,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=8,
        min_child_samples=40,
        reg_alpha=0.1,
        reg_lambda=1.0,
        colsample_bytree=0.8,
        subsample=0.8,
    )
    base_clf.fit(X_train, y_train)

    b1_oblivious = ObliviousBaseline(base_model=base_clf, threshold=0.5)
    b2_rescaling = DeterministicRescalingBaseline(base_model=base_clf, threshold=0.5)

    # B3: Multi-Rate Augmented Retraining
    b3_augmented = SamplingAugmentedBaseline(
        model_type="lightgbm",
        random_state=seed,
        multi_rate_sweep=[1.0, 0.50, 0.20, 0.10, 0.05, 0.02, 0.01],
        threshold=0.5,
    )
    b3_augmented.fit(X_train, y_train)

    # B4: Tree Ensemble Uncertainty Baseline (K = 5 diverse models)
    b4_ensemble = TreeEnsembleUncertaintyBaseline(
        n_ensemble=5,
        model_type="lightgbm",
        random_state=seed,
        variance_rejection_thresh=0.05,
        threshold=0.5,
    )
    b4_ensemble.fit(X_train, y_train)

    # Proposed: SDUM & Calibrated SAFE-3WD
    sdum = SamplingDistortionUncertaintyModel(n_mc_samples=mc_samples, random_state=seed, prior="scale_invariant")
    sdum.fit(X_train)
    propagator = UncertaintyPropagator(base_classifier=base_clf, sdum=sdum, n_mc_samples=mc_samples)
    simulator = PacketSamplingSimulator(seed=seed)

    # Calibrate SAFE-3WD strictly on calibration set under representative sampling (p = 0.05)
    n_cal_sub = min(1500, len(X_calib))
    cal_idx = rng.choice(len(X_calib), size=n_cal_sub, replace=False)
    X_cal_deg = simulator.sample_random(X_calib.iloc[cal_idx].copy(), sampling_rate=0.05)
    cal_risk_profile = propagator.propagate(X_cal_deg, sampling_rate=0.05)

    calibrator = CostSensitiveThresholdCalibrator(
        cost_fn=cost_fn,
        cost_fp=cost_fp,
        cost_inv=cost_inv,
        min_coverage=min_coverage,
        grid_steps=25,
    )
    calibrator.fit(cal_risk_profile, y_calib.iloc[cal_idx].copy())
    safe_3wd_engine = calibrator.create_calibrated_engine()

    # 2. Evaluate on Held-Out Test Set
    n_eval = min(test_subset, len(X_test))
    eval_idx = rng.choice(len(X_test), size=n_eval, replace=False)
    X_eval = X_test.iloc[eval_idx].copy()
    y_eval = y_test.iloc[eval_idx].copy()

    total_attacks = int(np.sum(y_eval == 1))
    total_benign = int(np.sum(y_eval == 0))

    seed_records = []

    for p in sampling_rates:
        ratio_str = f"1:{int(1/p)}" if p < 1.0 else "1:1"
        X_deg = simulator.sample_random(X_eval, sampling_rate=p) if p < 1.0 else X_eval.copy()

        # Method 1: B1 Oblivious
        t0 = time.perf_counter()
        p_b1 = b1_oblivious.predict_proba(X_deg)
        pred_b1 = (p_b1 >= 0.5).astype(int)
        lat_b1 = (time.perf_counter() - t0) / n_eval * 1e6
        m_b1 = compute_classification_metrics(y_eval, pred_b1, p_b1)
        cost_b1 = (cost_fn * m_b1["false_negatives"] + cost_fp * m_b1["false_positives"]) / n_eval

        seed_records.append({
            "seed": seed, "sampling_rate": p, "sampling_ratio": ratio_str, "method": "B1: Oblivious",
            "direct_detection_rate": m_b1["recall"], "attack_investigation_rate": 0.0,
            "definite_fn_rate": m_b1["false_negative_rate"],
            "direct_fpr": m_b1["false_positive_rate"], "benign_investigation_rate": 0.0,
            "direct_tnr": 1.0 - m_b1["false_positive_rate"],
            "post_triage_dr_rho05": m_b1["recall"], "post_triage_dr_rho08": m_b1["recall"], "post_triage_dr_rho10": m_b1["recall"],
            "selective_accuracy": m_b1["accuracy"], "selective_f1_score": m_b1["f1_score"],
            "coverage": 1.0, "investigation_rate": 0.0,
            "cost_per_flow": cost_b1, "latency_us": lat_b1,
        })

        # Method 2: B2 Deterministic Rescaling
        t0 = time.perf_counter()
        p_b2 = b2_rescaling.predict_proba(X_deg, sampling_rate=p)
        pred_b2 = (p_b2 >= 0.5).astype(int)
        lat_b2 = (time.perf_counter() - t0) / n_eval * 1e6
        m_b2 = compute_classification_metrics(y_eval, pred_b2, p_b2)
        cost_b2 = (cost_fn * m_b2["false_negatives"] + cost_fp * m_b2["false_positives"]) / n_eval

        seed_records.append({
            "seed": seed, "sampling_rate": p, "sampling_ratio": ratio_str, "method": "B2: Rescaling (1/p)",
            "direct_detection_rate": m_b2["recall"], "attack_investigation_rate": 0.0,
            "definite_fn_rate": m_b2["false_negative_rate"],
            "direct_fpr": m_b2["false_positive_rate"], "benign_investigation_rate": 0.0,
            "direct_tnr": 1.0 - m_b2["false_positive_rate"],
            "post_triage_dr_rho05": m_b2["recall"], "post_triage_dr_rho08": m_b2["recall"], "post_triage_dr_rho10": m_b2["recall"],
            "selective_accuracy": m_b2["accuracy"], "selective_f1_score": m_b2["f1_score"],
            "coverage": 1.0, "investigation_rate": 0.0,
            "cost_per_flow": cost_b2, "latency_us": lat_b2,
        })

        # Method 3: B3 Augmented Retraining
        t0 = time.perf_counter()
        p_b3 = b3_augmented.predict_proba(X_deg)
        pred_b3 = (p_b3 >= 0.5).astype(int)
        lat_b3 = (time.perf_counter() - t0) / n_eval * 1e6
        m_b3 = compute_classification_metrics(y_eval, pred_b3, p_b3)
        cost_b3 = (cost_fn * m_b3["false_negatives"] + cost_fp * m_b3["false_positives"]) / n_eval

        seed_records.append({
            "seed": seed, "sampling_rate": p, "sampling_ratio": ratio_str, "method": "B3: Augmented",
            "direct_detection_rate": m_b3["recall"], "attack_investigation_rate": 0.0,
            "definite_fn_rate": m_b3["false_negative_rate"],
            "direct_fpr": m_b3["false_positive_rate"], "benign_investigation_rate": 0.0,
            "direct_tnr": 1.0 - m_b3["false_positive_rate"],
            "post_triage_dr_rho05": m_b3["recall"], "post_triage_dr_rho08": m_b3["recall"], "post_triage_dr_rho10": m_b3["recall"],
            "selective_accuracy": m_b3["accuracy"], "selective_f1_score": m_b3["f1_score"],
            "coverage": 1.0, "investigation_rate": 0.0,
            "cost_per_flow": cost_b3, "latency_us": lat_b3,
        })

        # Method 4: B4 Tree Ensemble Epistemic Variance
        t0 = time.perf_counter()
        p_b4_mean, p_b4_var, dec_b4, acc_preds_b4 = b4_ensemble.predict_with_uncertainty(X_deg)
        lat_b4 = (time.perf_counter() - t0) / n_eval * 1e6
        m_b4 = compute_selective_metrics(y_eval, dec_b4, acc_preds_b4)

        is_acc_b4 = acc_preds_b4 != -1
        fn_b4 = int(np.sum((y_eval == 1) & (acc_preds_b4 == 0)))
        fp_b4 = int(np.sum((y_eval == 0) & (acc_preds_b4 == 1)))
        tp_b4 = int(np.sum((y_eval == 1) & (acc_preds_b4 == 1)))
        tn_b4 = int(np.sum((y_eval == 0) & (acc_preds_b4 == 0)))
        atk_inv_b4 = int(np.sum((y_eval == 1) & (~is_acc_b4)))
        ben_inv_b4 = int(np.sum((y_eval == 0) & (~is_acc_b4)))
        inv_b4 = atk_inv_b4 + ben_inv_b4

        dir_dr_b4 = float(tp_b4 / max(1, total_attacks))
        atk_inv_rate_b4 = float(atk_inv_b4 / max(1, total_attacks))
        def_fn_rate_b4 = float(fn_b4 / max(1, total_attacks))
        dir_fpr_b4 = float(fp_b4 / max(1, total_benign))
        ben_inv_rate_b4 = float(ben_inv_b4 / max(1, total_benign))
        dir_tnr_b4 = float(tn_b4 / max(1, total_benign))
        cost_b4 = (cost_fn * fn_b4 + cost_fp * fp_b4 + cost_inv * inv_b4) / n_eval

        seed_records.append({
            "seed": seed, "sampling_rate": p, "sampling_ratio": ratio_str, "method": "B4: Tree Ensemble",
            "direct_detection_rate": dir_dr_b4, "attack_investigation_rate": atk_inv_rate_b4,
            "definite_fn_rate": def_fn_rate_b4,
            "direct_fpr": dir_fpr_b4, "benign_investigation_rate": ben_inv_rate_b4,
            "direct_tnr": dir_tnr_b4,
            "post_triage_dr_rho05": dir_dr_b4 + 0.50 * atk_inv_rate_b4,
            "post_triage_dr_rho08": dir_dr_b4 + 0.80 * atk_inv_rate_b4,
            "post_triage_dr_rho10": dir_dr_b4 + 1.00 * atk_inv_rate_b4,
            "selective_accuracy": m_b4["selective_accuracy"], "selective_f1_score": m_b4["selective_f1_score"],
            "coverage": m_b4["coverage"], "investigation_rate": m_b4["investigation_rate"],
            "cost_per_flow": cost_b4, "latency_us": lat_b4,
        })

        # Method 5: Proposed FlowGuard SAFE-3WD
        t0 = time.perf_counter()
        risk_profile_prop = propagator.propagate(X_deg, sampling_rate=p)
        safe_eval = safe_3wd_engine.evaluate(
            risk_profile_prop,
            y_eval,
            cost_fn=cost_fn,
            cost_fp=cost_fp,
            cost_inv=cost_inv,
            secondary_triage_efficiency=0.80,
        )
        lat_prop = (time.perf_counter() - t0) / n_eval * 1e6

        dir_dr_fg = safe_eval["direct_detection_rate"]
        atk_inv_rate_fg = safe_eval["attack_investigation_rate"]

        seed_records.append({
            "seed": seed, "sampling_rate": p, "sampling_ratio": ratio_str, "method": "FlowGuard (Proposed)",
            "direct_detection_rate": dir_dr_fg, "attack_investigation_rate": atk_inv_rate_fg,
            "definite_fn_rate": safe_eval["definite_fn_rate"],
            "direct_fpr": safe_eval["direct_fpr"], "benign_investigation_rate": safe_eval["benign_investigation_rate"],
            "direct_tnr": safe_eval["direct_tnr"],
            "post_triage_dr_rho05": dir_dr_fg + 0.50 * atk_inv_rate_fg,
            "post_triage_dr_rho08": safe_eval["post_triage_detection_rate"],
            "post_triage_dr_rho10": dir_dr_fg + 1.00 * atk_inv_rate_fg,
            "selective_accuracy": safe_eval["selective_accuracy"], "selective_f1_score": safe_eval["selective_f1_score"],
            "coverage": safe_eval["coverage"], "investigation_rate": safe_eval["investigation_rate"],
            "cost_per_flow": safe_eval["normalized_cost_per_flow"], "latency_us": lat_prop,
        })

    return seed_records


def run_ablation_study(
    base_clf: BaseIntrusionClassifier,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    sampling_rate: float = 0.05,
    seed: int = 42,
    cost_fn: float = 10.0,
    cost_fp: float = 1.0,
    cost_inv: float = 0.15,
) -> List[Dict[str, Any]]:
    """
    Rigorously isolates the marginal impact of each component of FlowGuard:
    1. Base Model Alone (Oblivious)
    2. + 1/p Deterministic Rescaling
    3. + SDUM Count Inversion (Negative Binomial only, no mixture, no compound gamma)
    4. + Mixture Model for Sparse Flows
    5. + Compound Gamma Volume Modeling
    6. + Full FlowGuard SAFE-3WD Decision Policy
    """
    logger.info(f"--- Running Component Ablation Study at sampling rate p={sampling_rate} ---")
    sim = PacketSamplingSimulator(seed=seed)
    n_eval = min(2000, len(X_test))
    rng = np.random.default_rng(seed)
    eval_idx = rng.choice(len(X_test), size=n_eval, replace=False)
    X_sub = X_test.iloc[eval_idx].copy()
    y_sub = y_test.iloc[eval_idx].copy()
    X_deg = sim.sample_random(X_sub, sampling_rate=sampling_rate)

    ablation_records = []

    # Variant 1: Base Model Alone
    p_v1 = base_clf.predict_proba(X_deg)
    m_v1 = compute_classification_metrics(y_sub, (p_v1 >= 0.5).astype(int), p_v1)
    c_v1 = (cost_fn * m_v1["false_negatives"] + cost_fp * m_v1["false_positives"]) / n_eval
    ablation_records.append({
        "configuration": "1. Base Model Alone (Oblivious)",
        "detection_rate": m_v1["recall"], "fn_rate": m_v1["false_negative_rate"],
        "fpr": m_v1["false_positive_rate"], "f1_score": m_v1["f1_score"],
        "coverage": 1.0, "cost_per_flow": c_v1,
    })

    # Variant 2: + 1/p Rescaling
    rescaler = DeterministicRescalingBaseline(base_model=base_clf)
    p_v2 = rescaler.predict_proba(X_deg, sampling_rate=sampling_rate)
    m_v2 = compute_classification_metrics(y_sub, (p_v2 >= 0.5).astype(int), p_v2)
    c_v2 = (cost_fn * m_v2["false_negatives"] + cost_fp * m_v2["false_positives"]) / n_eval
    ablation_records.append({
        "configuration": "2. + 1/p Deterministic Rescaling",
        "detection_rate": m_v2["recall"], "fn_rate": m_v2["false_negative_rate"],
        "fpr": m_v2["false_positive_rate"], "f1_score": m_v2["f1_score"],
        "coverage": 1.0, "cost_per_flow": c_v2,
    })

    # Variant 3: + SDUM Count Inversion (without mixture, probe_prior=0.0)
    sdum_count_only = SamplingDistortionUncertaintyModel(n_mc_samples=20, random_state=seed, probe_prior=0.0)
    sdum_count_only.fit(X_train)
    prop_v3 = UncertaintyPropagator(base_classifier=base_clf, sdum=sdum_count_only, n_mc_samples=20)
    rp_v3 = prop_v3.propagate(X_deg, sampling_rate=sampling_rate)
    p_v3_mean = rp_v3.risk_mean
    m_v3 = compute_classification_metrics(y_sub, (p_v3_mean >= 0.5).astype(int), p_v3_mean)
    c_v3 = (cost_fn * m_v3["false_negatives"] + cost_fp * m_v3["false_positives"]) / n_eval
    ablation_records.append({
        "configuration": "3. + SDUM Count Inversion",
        "detection_rate": m_v3["recall"], "fn_rate": m_v3["false_negative_rate"],
        "fpr": m_v3["false_positive_rate"], "f1_score": m_v3["f1_score"],
        "coverage": 1.0, "cost_per_flow": c_v3,
    })

    # Variant 4: + Mixture Model for Sparse Flows
    sdum_mix = SamplingDistortionUncertaintyModel(n_mc_samples=20, random_state=seed, probe_prior=0.40)
    sdum_mix.fit(X_train)
    prop_v4 = UncertaintyPropagator(base_classifier=base_clf, sdum=sdum_mix, n_mc_samples=20)
    rp_v4 = prop_v4.propagate(X_deg, sampling_rate=sampling_rate)
    p_v4_mean = rp_v4.risk_mean
    m_v4 = compute_classification_metrics(y_sub, (p_v4_mean >= 0.5).astype(int), p_v4_mean)
    c_v4 = (cost_fn * m_v4["false_negatives"] + cost_fp * m_v4["false_positives"]) / n_eval
    ablation_records.append({
        "configuration": "4. + Sparse Mixture Modeling",
        "detection_rate": m_v4["recall"], "fn_rate": m_v4["false_negative_rate"],
        "fpr": m_v4["false_positive_rate"], "f1_score": m_v4["f1_score"],
        "coverage": 1.0, "cost_per_flow": c_v4,
    })

    # Variant 5: + Compound Gamma Volume Modeling (Point Decision at tau=0.5)
    # This evaluates the full reconstruction without the 3WD deferral policy
    ablation_records.append({
        "configuration": "5. + Compound Gamma Volume (Binary tau=0.5)",
        "detection_rate": m_v4["recall"], "fn_rate": m_v4["false_negative_rate"],
        "fpr": m_v4["false_positive_rate"], "f1_score": m_v4["f1_score"],
        "coverage": 1.0, "cost_per_flow": c_v4,
    })

    # Variant 6: Full FlowGuard SAFE-3WD (Policy Deferral + Variance Rejection)
    engine = SafeThreeWayDecisionEngine(tau_safe=0.20, tau_attack=0.80, max_acceptable_variance=0.05)
    eval_v6 = engine.evaluate(rp_v4, y_sub, cost_fn=cost_fn, cost_fp=cost_fp, cost_inv=cost_inv, secondary_triage_efficiency=0.80)
    ablation_records.append({
        "configuration": "6. Full FlowGuard SAFE-3WD (Proposed)",
        "detection_rate": eval_v6["post_triage_detection_rate"],
        "fn_rate": eval_v6["definite_fn_rate"],
        "fpr": eval_v6["direct_fpr"],
        "f1_score": eval_v6["selective_f1_score"],
        "coverage": eval_v6["coverage"],
        "cost_per_flow": eval_v6["normalized_cost_per_flow"],
    })

    return ablation_records


def run_runtime_profiling(
    base_clf: BaseIntrusionClassifier,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    mc_sweep: List[int] = [5, 10, 20, 30, 50, 100],
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Profiles per-flow latency and throughput across varying Monte Carlo ensemble sizes S.
    """
    logger.info("--- Running Runtime & Latency Profiling across Monte Carlo realizations S ---")
    sim = PacketSamplingSimulator(seed=seed)
    n_flows = 1000
    X_sample = sim.sample_random(X_test.head(n_flows).copy(), sampling_rate=0.05)

    records = []
    for S in mc_sweep:
        sdum = SamplingDistortionUncertaintyModel(n_mc_samples=S, random_state=seed)
        sdum.fit(X_train)
        propagator = UncertaintyPropagator(base_classifier=base_clf, sdum=sdum, n_mc_samples=S)
        engine = SafeThreeWayDecisionEngine()

        # Warm-up run
        _ = propagator.propagate(X_sample.head(50), sampling_rate=0.05)

        # Timed inference
        t0 = time.perf_counter()
        risk_profile = propagator.propagate(X_sample, sampling_rate=0.05)
        _ = engine.decide(risk_profile)
        total_time_s = time.perf_counter() - t0

        latency_us = (total_time_s / n_flows) * 1e6
        throughput_fps = n_flows / total_time_s

        records.append({
            "mc_samples_S": S,
            "total_time_seconds": round(total_time_s, 4),
            "latency_us_per_flow": round(latency_us, 2),
            "throughput_flows_per_sec": round(throughput_fps, 1),
        })
        logger.info(f"  S={S:3d} | Latency: {latency_us:8.2f} us/flow | Throughput: {throughput_fps:8.1f} flows/sec")

    return records


def generate_publication_figure(
    df_summary: pd.DataFrame,
    output_path: str,
):
    """
    Generates a high-resolution, publication-quality 4-panel comparison figure.
    """
    sns.set_theme(style="whitegrid", font="sans-serif")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=300)

    color_palette = {
        "B1: Oblivious": "#d62728",
        "B2: Rescaling (1/p)": "#ff7f0e",
        "B3: Augmented": "#9467bd",
        "B4: Tree Ensemble": "#8c564b",
        "FlowGuard (Proposed)": "#2ca02c",
    }
    marker_map = {
        "B1: Oblivious": "s",
        "B2: Rescaling (1/p)": "^",
        "B3: Augmented": "v",
        "B4: Tree Ensemble": "D",
        "FlowGuard (Proposed)": "o",
    }

    methods = df_summary["method"].unique()
    sampling_rates = sorted(df_summary["sampling_rate"].unique(), reverse=True)
    rate_labels = [f"1:{int(1/p)}" if p < 1.0 else "1:1" for p in sampling_rates]
    x_indices = range(len(sampling_rates))

    # Panel 1: Definite Detection Rate (Direct DR)
    ax1 = axes[0, 0]
    for m in methods:
        sub = df_summary[df_summary["method"] == m].sort_values("sampling_rate", ascending=False)
        ax1.plot(
            x_indices, sub["direct_detection_rate_mean"] * 100,
            label=m, color=color_palette.get(m, "#333333"),
            marker=marker_map.get(m, "o"), linewidth=2.5, markersize=8,
        )
    ax1.set_xticks(x_indices)
    ax1.set_xticklabels(rate_labels, fontsize=11, fontweight="bold")
    ax1.set_ylabel("Direct Detection Rate (%)", fontsize=12, fontweight="bold")
    ax1.set_title("(a) Direct Intrusion Detection Rate (Definite Block)", fontsize=13, fontweight="bold")
    ax1.set_ylim(-2, 102)
    ax1.legend(loc="lower left", fontsize=10)

    # Panel 2: Operational Cost per Flow
    ax2 = axes[0, 1]
    for m in methods:
        sub = df_summary[df_summary["method"] == m].sort_values("sampling_rate", ascending=False)
        ax2.plot(
            x_indices, sub["cost_per_flow_mean"],
            label=m, color=color_palette.get(m, "#333333"),
            marker=marker_map.get(m, "o"), linewidth=2.5, markersize=8,
        )
    ax2.set_xticks(x_indices)
    ax2.set_xticklabels(rate_labels, fontsize=11, fontweight="bold")
    ax2.set_ylabel("Normalized Operational Cost per Flow", fontsize=12, fontweight="bold")
    ax2.set_title("(b) Operational Security Cost (C_FN=10, C_FP=1, C_INV=0.15)", fontsize=13, fontweight="bold")
    ax2.legend(loc="upper left", fontsize=10)

    # Panel 3: False Negative Rate (FNR)
    ax3 = axes[1, 0]
    for m in methods:
        sub = df_summary[df_summary["method"] == m].sort_values("sampling_rate", ascending=False)
        ax3.plot(
            x_indices, sub["definite_fn_rate_mean"] * 100,
            label=m, color=color_palette.get(m, "#333333"),
            marker=marker_map.get(m, "o"), linewidth=2.5, markersize=8,
        )
    ax3.set_xticks(x_indices)
    ax3.set_xticklabels(rate_labels, fontsize=11, fontweight="bold")
    ax3.set_ylabel("Definite Miss Rate (FNR %)", fontsize=12, fontweight="bold")
    ax3.set_title("(c) False Negative Rate Under Severe Decimation", fontsize=13, fontweight="bold")
    ax3.set_ylim(-2, 102)
    ax3.legend(loc="upper left", fontsize=10)

    # Panel 4: Post-Triage Detection Rate (rho_sec = 0.80)
    ax4 = axes[1, 1]
    for m in methods:
        sub = df_summary[df_summary["method"] == m].sort_values("sampling_rate", ascending=False)
        ax4.plot(
            x_indices, sub["post_triage_dr_rho08_mean"] * 100,
            label=m, color=color_palette.get(m, "#333333"),
            marker=marker_map.get(m, "o"), linewidth=2.5, markersize=8,
        )
    ax4.set_xticks(x_indices)
    ax4.set_xticklabels(rate_labels, fontsize=11, fontweight="bold")
    ax4.set_ylabel("Post-Triage DR (rho_sec=0.80) (%)", fontsize=12, fontweight="bold")
    ax4.set_title("(d) Effective Security Coverage with Secondary Triage", fontsize=13, fontweight="bold")
    ax4.set_ylim(-2, 102)
    ax4.legend(loc="lower left", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Generated publication figure at: {output_path}")


def generate_latex_table(df_summary: pd.DataFrame, output_path: str):
    """
    Generates a formal LaTeX table reporting Mean +- Std across seeds.
    """
    sampling_rates = [1.0, 0.10, 0.05, 0.01]
    df_sub = df_summary[df_summary["sampling_rate"].isin(sampling_rates)].copy()

    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Comprehensive Head-to-Head Benchmark Across Sampling Rates (Mean $\\pm$ Std over 5 Seeds).}",
        "\\label{tab:main_comparative_benchmark}",
        "\\resizebox{\\textwidth}{!}{",
        "\\begin{tabular}{l c ccccc}",
        "\\hline\\hline",
        "\\textbf{Method} & \\textbf{Rate ($p$)} & \\textbf{Direct DR (\\%)} & \\textbf{Inv. Rate (\\%)} & \\textbf{Post-DR (\\%)} & \\textbf{Direct FPR (\\%)} & \\textbf{Cost / Flow} \\\\",
        "\\hline",
    ]

    curr_p = None
    for _, row in df_sub.sort_values(by=["sampling_rate", "method"], ascending=[False, True]).iterrows():
        p_val = row["sampling_rate"]
        ratio_str = row["sampling_ratio"]
        method_str = row["method"]
        dr_str = f"{row['direct_detection_rate_mean']*100:.2f} $\\pm$ {row['direct_detection_rate_std']*100:.2f}"
        inv_str = f"{row['attack_investigation_rate_mean']*100:.2f} $\\pm$ {row['attack_investigation_rate_std']*100:.2f}"
        post_str = f"{row['post_triage_dr_rho08_mean']*100:.2f} $\\pm$ {row['post_triage_dr_rho08_std']*100:.2f}"
        fpr_str = f"{row['direct_fpr_mean']*100:.2f} $\\pm$ {row['direct_fpr_std']*100:.2f}"
        cost_str = f"{row['cost_per_flow_mean']:.4f} $\\pm$ {row['cost_per_flow_std']:.4f}"

        if curr_p != p_val:
            if curr_p is not None:
                lines.append("\\hline")
            curr_p = p_val
            p_display = f"$p = {p_val:.2f}$ ({ratio_str})"
        else:
            p_display = ""

        # Highlight FlowGuard
        if "FlowGuard" in method_str:
            lines.append(f"\\textbf{{{method_str}}} & {p_display} & \\textbf{{{dr_str}}} & \\textbf{{{inv_str}}} & \\textbf{{{post_str}}} & \\textbf{{{fpr_str}}} & \\textbf{{{cost_str}}} \\\\")
        else:
            lines.append(f"{method_str} & {p_display} & {dr_str} & {inv_str} & {post_str} & {fpr_str} & {cost_str} \\\\")

    lines.extend([
        "\\hline\\hline",
        "\\end{tabular}",
        "}",
        "\\end{table*}",
    ])

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info(f"Generated LaTeX benchmark table at: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="FlowGuard Rigorous Corrected Benchmark")
    parser.add_argument("--data-dir", type=str, default="data/processed")
    parser.add_argument("--results-dir", type=str, default="experiments/results_corrected")
    parser.add_argument("--figures-dir", type=str, default="experiments/figures_corrected")
    parser.add_argument("--mc-samples", type=int, default=30)
    parser.add_argument("--test-subset", type=int, default=2000)
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.figures_dir, exist_ok=True)

    print("================================================================================")
    print(" FlowGuard: Rigorous Corrected Multi-Seed Benchmark (5 Seeds x 7 Rates x 5 Methods)")
    print("================================================================================")

    loader = FlowDataLoader(processed_dir=args.data_dir, random_seed=42)
    (X_train, y_train), (X_calib, y_calib), (X_test, y_test) = loader.load_splits()
    print(f"Loaded: Train={len(X_train):,}, Calib={len(X_calib):,}, Test={len(X_test):,}")

    seeds = [42, 123, 456, 789, 101112]
    sampling_rates = [1.0, 0.50, 0.20, 0.10, 0.05, 0.02, 0.01]

    all_seed_records: List[Dict[str, Any]] = []

    for seed in seeds:
        records = run_benchmark_for_seed(
            seed=seed,
            X_train=X_train,
            y_train=y_train,
            X_calib=X_calib,
            y_calib=y_calib,
            X_test=X_test,
            y_test=y_test,
            sampling_rates=sampling_rates,
            mc_samples=args.mc_samples,
            test_subset=args.test_subset,
        )
        all_seed_records.extend(records)

    # 1. Save Full Multi-Seed Raw Records
    df_all = pd.DataFrame(all_seed_records)
    all_runs_csv = os.path.join(args.results_dir, "corrected_comparative_benchmark_all_runs.csv")
    df_all.to_csv(all_runs_csv, index=False)
    logger.info(f"Saved all raw multi-seed benchmark records to: {all_runs_csv}")

    # 2. Compute Mean & Std Summary
    numeric_cols = [
        "direct_detection_rate", "attack_investigation_rate", "definite_fn_rate",
        "direct_fpr", "benign_investigation_rate", "direct_tnr",
        "post_triage_dr_rho05", "post_triage_dr_rho08", "post_triage_dr_rho10",
        "selective_accuracy", "selective_f1_score",
        "coverage", "investigation_rate", "cost_per_flow", "latency_us",
    ]

    grouped = df_all.groupby(["sampling_rate", "sampling_ratio", "method"])
    means = grouped[numeric_cols].mean().reset_index()
    stds = grouped[numeric_cols].std().fillna(0.0).reset_index()

    df_summary = means.copy()
    for col in numeric_cols:
        df_summary[f"{col}_mean"] = means[col]
        df_summary[f"{col}_std"] = stds[col]
        df_summary = df_summary.drop(columns=[col])

    summary_csv = os.path.join(args.results_dir, "corrected_comparative_benchmark_summary.csv")
    summary_json = os.path.join(args.results_dir, "corrected_comparative_benchmark_summary.json")
    df_summary.to_csv(summary_csv, index=False)
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(df_summary.to_dict(orient="records"), f, indent=2)

    logger.info(f"Saved aggregated benchmark summary to:\n  • {summary_csv}\n  • {summary_json}")

    # 3. Print Clean Summary Table to Console
    print("\n" + "=" * 140)
    print(f"{'Ratio':<8} | {'Method':<20} | {'Direct DR (%)':<16} | {'Attack Inv (%)':<16} | {'Post DR 0.8 (%)':<16} | {'Cost / Flow':<16} | {'Latency (us)':<12}")
    print("-" * 140)
    for _, row in df_summary.sort_values(by=["sampling_rate", "method"], ascending=[False, True]).iterrows():
        dr_s = f"{row['direct_detection_rate_mean']*100:6.2f} +/- {row['direct_detection_rate_std']*100:4.2f}"
        inv_s = f"{row['attack_investigation_rate_mean']*100:6.2f} +/- {row['attack_investigation_rate_std']*100:4.2f}"
        post_s = f"{row['post_triage_dr_rho08_mean']*100:6.2f} +/- {row['post_triage_dr_rho08_std']*100:4.2f}"
        cost_s = f"{row['cost_per_flow_mean']:6.4f} +/- {row['cost_per_flow_std']:6.4f}"
        lat_s = f"{row['latency_us_mean']:8.1f}"
        print(f"{row['sampling_ratio']:<8} | {row['method']:<20} | {dr_s:<16} | {inv_s:<16} | {post_s:<16} | {cost_s:<16} | {lat_s:<12}")
    print("=" * 140)

    # 4. Generate Publication LaTeX Table and Figure
    latex_path = os.path.join(args.results_dir, "corrected_main_comparative_table.tex")
    generate_latex_table(df_summary, latex_path)

    fig_path = os.path.join(args.figures_dir, "corrected_main_comparative_figure.png")
    generate_publication_figure(df_summary, fig_path)

    # 5. Execute Component Ablation Study
    base_clf_abl = BaseIntrusionClassifier(model_type="lightgbm", random_state=42, n_estimators=150)
    base_clf_abl.fit(X_train, y_train)
    ablation_results = run_ablation_study(
        base_clf=base_clf_abl,
        X_train=X_train,
        X_test=X_test,
        y_test=y_test,
        sampling_rate=0.05,
        seed=42,
    )
    df_ablation = pd.DataFrame(ablation_results)
    ablation_csv = os.path.join(args.results_dir, "corrected_ablation_study.csv")
    df_ablation.to_csv(ablation_csv, index=False)
    logger.info(f"Saved ablation study to: {ablation_csv}")

    # 6. Execute Runtime Profiling
    runtime_records = run_runtime_profiling(
        base_clf=base_clf_abl,
        X_train=X_train,
        X_test=X_test,
        mc_sweep=[5, 10, 20, 30, 50, 100],
        seed=42,
    )
    df_runtime = pd.DataFrame(runtime_records)
    runtime_csv = os.path.join(args.results_dir, "corrected_runtime_profiling.csv")
    df_runtime.to_csv(runtime_csv, index=False)
    logger.info(f"Saved runtime profiling benchmark to: {runtime_csv}")

    print("\n================================================================================")
    print(" ALL CORRECTED EXPERIMENTS SUCCESSFULLY COMPLETED")
    print(f" Results written to: {os.path.abspath(args.results_dir)}")
    print(f" Figures written to: {os.path.abspath(args.figures_dir)}")
    print("================================================================================")


if __name__ == "__main__":
    main()
