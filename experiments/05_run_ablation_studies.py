"""
FlowGuard Experiment 05: Comprehensive Ablation Studies (Milestone 4)
=====================================================================

Executes three rigorous ablation studies:
1. Ablation 1: Impact of Monte Carlo Sample Size S in {5, 10, 20, 30, 50}.
   (Quantifies variance convergence vs. inference latency tradeoff).
2. Ablation 2: Decision Rule Architecture:
   - Point Threshold on mu_R
   - Traditional Point-Based 3WD
   - FlowGuard SAFE-3WD Interval Bounding [R_min, R_max].
3. Ablation 3: Robustness to Mismatched Sampling Rate Estimation:
   - Drifts p_est / p_true from -30% to +30%.
   - Validates operational stability against telemetry estimation errors.

Outputs publication-ready figures to experiments/figures/ and JSON/CSV records.
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List

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
from flowguard.models.base_classifier import BaseIntrusionClassifier
from flowguard.sampling.simulator import PacketSamplingSimulator
from flowguard.uncertainty.propagator import UncertaintyPropagator
from flowguard.uncertainty.sdum import SamplingDistortionUncertaintyModel


def main():
    parser = argparse.ArgumentParser(description="FlowGuard Milestone 4: Comprehensive Ablation Studies")
    parser.add_argument("--data-dir", type=str, default="data/processed", help="Path to processed parquet datasets")
    parser.add_argument("--model", type=str, default="lightgbm", choices=["lightgbm", "random_forest", "xgboost"], help="Base classifier")
    parser.add_argument("--results-dir", type=str, default="experiments/results", help="Directory for benchmark outputs")
    parser.add_argument("--figures-dir", type=str, default="experiments/figures", help="Directory for generated plots")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic random seed")
    parser.add_argument("--test-subset", type=int, default=1500, help="Subset of test flows for ablations")
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.figures_dir, exist_ok=True)

    print("================================================================================")
    print(" FlowGuard Step 6: Comprehensive Ablation Studies (Milestone 4)")
    print("================================================================================")

    # 1. Load Data Splits
    loader = FlowDataLoader(processed_dir=args.data_dir, random_seed=args.seed)
    (X_train, y_train), (X_calib, y_calib), (X_test, y_test) = loader.load_splits()

    n_eval = min(args.test_subset, len(X_test))
    eval_idx = np.random.default_rng(args.seed).choice(len(X_test), size=n_eval, replace=False)
    X_eval = X_test.iloc[eval_idx].copy()
    y_eval = y_test.iloc[eval_idx].copy()

    # 2. Train Base Model & Calibrate SAFE-3WD
    base_clf = BaseIntrusionClassifier(model_type=args.model, random_state=args.seed, n_estimators=150)
    base_clf.fit(X_train, y_train)

    sdum_base = SamplingDistortionUncertaintyModel(n_mc_samples=30, random_state=args.seed)
    sdum_base.fit(X_train)
    propagator_base = UncertaintyPropagator(base_classifier=base_clf, sdum=sdum_base, n_mc_samples=30)
    simulator = PacketSamplingSimulator(seed=args.seed)

    # Calibrate on Calibration Split
    p_cal = 0.05
    n_cal_sub = min(1000, len(X_calib))
    cal_idx = np.random.default_rng(args.seed).choice(len(X_calib), size=n_cal_sub, replace=False)
    X_cal_deg = simulator.sample_random(X_calib.iloc[cal_idx].copy(), sampling_rate=p_cal)
    cal_profile = propagator_base.propagate(X_cal_deg, sampling_rate=p_cal)

    calibrator = CostSensitiveThresholdCalibrator(cost_fn=10.0, cost_fp=1.0, cost_inv=0.15, min_coverage=0.40)
    calibrator.fit(cal_profile, y_calib.iloc[cal_idx].copy())
    calibrated_engine = calibrator.create_calibrated_engine()
    tau_safe = calibrated_engine.tau_safe
    tau_attack = calibrated_engine.tau_attack

    # -------------------------------------------------------------------------
    # ABLATION 1: Impact of Monte Carlo Sample Size S in {5, 10, 20, 30, 50}
    # -------------------------------------------------------------------------
    print("\n[1/3] Running Ablation 1: Impact of Monte Carlo Sample Size S...")
    s_candidates = [5, 10, 20, 30, 50]
    ablation_1_records = []
    p_fixed = 0.05
    X_deg_fixed = simulator.sample_random(X_eval, sampling_rate=p_fixed)

    for S in s_candidates:
        sdum_s = SamplingDistortionUncertaintyModel(n_mc_samples=S, random_state=args.seed)
        sdum_s.fit(X_train)
        prop_s = UncertaintyPropagator(base_classifier=base_clf, sdum=sdum_s, n_mc_samples=S)

        t0 = time.perf_counter()
        profile_s = prop_s.propagate(X_deg_fixed, sampling_rate=p_fixed, n_mc_samples=S)
        lat = (time.perf_counter() - t0) / n_eval * 1e6

        eval_s = calibrated_engine.evaluate(profile_s, y_eval)

        rec = {
            "sample_size_S": S,
            "mean_risk_spread": float(np.mean(profile_s.risk_spread)),
            "mean_risk_variance": float(np.mean(profile_s.risk_variance)),
            "selective_accuracy": eval_s["selective_accuracy"],
            "coverage": eval_s["coverage"],
            "investigation_rate": eval_s["investigation_rate"],
            "latency_us_per_flow": lat,
        }
        ablation_1_records.append(rec)
        print(f"      S = {S:2d} | Spread: {rec['mean_risk_spread']:.4f} | Variance: {rec['mean_risk_variance']:.5f} | Sel Acc: {rec['selective_accuracy']*100:.2f}% | Latency: {lat:6.1f} μs")

    # -------------------------------------------------------------------------
    # ABLATION 2: Decision Policy Architecture (Point vs. Interval)
    # -------------------------------------------------------------------------
    print("\n[2/3] Running Ablation 2: Decision Policy Architecture Comparison...")
    ablation_2_records = []
    rates_for_policy = [1.0, 0.20, 0.05, 0.01]

    for p in rates_for_policy:
        ratio_str = f"1:{int(1/p)}" if p < 1.0 else "1:1"
        X_deg_p = simulator.sample_random(X_eval, sampling_rate=p) if p < 1.0 else X_eval.copy()
        profile_p = propagator_base.propagate(X_deg_p, sampling_rate=p)

        # Policy A: Binary Point Threshold (tau = 0.5 on mu_R)
        pred_a = (profile_p.risk_mean >= 0.5).astype(int)
        acc_a = float(np.mean(pred_a == y_eval))
        fp_a = int(np.sum((y_eval == 0) & (pred_a == 1)))
        fpr_a = float(fp_a / max(1, np.sum(y_eval == 0)))

        # Policy B: Point-based 3WD (mu_R < tau_safe -> Benign, mu_R > tau_attack -> Attack, else Investigate)
        dec_b = np.where(profile_p.risk_mean < tau_safe, "BENIGN", np.where(profile_p.risk_mean > tau_attack, "ATTACK", "INVESTIGATE"))
        acc_preds_b = np.where(dec_b == "INVESTIGATE", -1, np.where(dec_b == "ATTACK", 1, 0))
        is_acc_b = acc_preds_b != -1
        sel_acc_b = float(np.mean(y_eval[is_acc_b] == acc_preds_b[is_acc_b])) if is_acc_b.any() else 0.0
        fp_b = int(np.sum((y_eval == 0) & (acc_preds_b == 1)))
        fpr_b = float(fp_b / max(1, np.sum(y_eval[is_acc_b] == 0))) if is_acc_b.any() else 0.0
        cov_b = float(np.mean(is_acc_b))

        # Policy C: FlowGuard SAFE-3WD (Interval Bounding: R_max < tau_safe, R_min > tau_attack)
        eval_c = calibrated_engine.evaluate(profile_p, y_eval)

        ablation_2_records.extend([
            {"sampling_ratio": ratio_str, "policy": "Policy A: Binary Mean (mu_R >= 0.5)", "selective_accuracy": acc_a, "fpr": fpr_a, "coverage": 1.0},
            {"sampling_ratio": ratio_str, "policy": "Policy B: Point-Based 3WD (mu_R thresholds)", "selective_accuracy": sel_acc_b, "fpr": fpr_b, "coverage": cov_b},
            {"sampling_ratio": ratio_str, "policy": "Policy C: FlowGuard SAFE-3WD (Interval [R_min, R_max])", "selective_accuracy": eval_c["selective_accuracy"], "fpr": eval_c["selective_false_positive_rate"], "coverage": eval_c["coverage"]},
        ])
        print(f"      {ratio_str:<5} | Policy A Acc: {acc_a*100:6.2f}% (FPR: {fpr_a*100:5.2f}%) | Policy B Acc: {sel_acc_b*100:6.2f}% (FPR: {fpr_b*100:5.2f}%) | SAFE-3WD Acc: {eval_c['selective_accuracy']*100:6.2f}% (FPR: {eval_c['selective_false_positive_rate']*100:5.2f}%)")

    # -------------------------------------------------------------------------
    # ABLATION 3: Robustness to Mismatched Sampling Rate Estimation
    # -------------------------------------------------------------------------
    print("\n[3/3] Running Ablation 3: Robustness to Mismatched Sampling Rate Drift...")
    p_true = 0.05
    X_deg_true = simulator.sample_random(X_eval, sampling_rate=p_true)
    drift_factors = [0.70, 0.80, 0.90, 1.00, 1.10, 1.20, 1.30]
    ablation_3_records = []

    for alpha in drift_factors:
        p_assumed = p_true * alpha
        profile_drift = propagator_base.propagate(X_deg_true, sampling_rate=p_assumed)
        eval_drift = calibrated_engine.evaluate(profile_drift, y_eval)

        rec = {
            "drift_factor": alpha,
            "drift_percentage": f"{(alpha - 1.0)*100:+.0f}%",
            "p_true": p_true,
            "p_assumed": p_assumed,
            "selective_accuracy": eval_drift["selective_accuracy"],
            "selective_fpr": eval_drift["selective_false_positive_rate"],
            "coverage": eval_drift["coverage"],
            "investigation_rate": eval_drift["investigation_rate"],
        }
        ablation_3_records.append(rec)
        print(f"      Drift: {rec['drift_percentage']:>5} (p_assumed={p_assumed:.4f}) | Sel Acc: {rec['selective_accuracy']*100:6.2f}% | FPR: {rec['selective_fpr']*100:5.2f}% | Coverage: {rec['coverage']*100:5.2f}%")

    # Save Records
    with open(os.path.join(args.results_dir, "ablation_mc_size.json"), "w") as f:
        json.dump(ablation_1_records, f, indent=2)
    with open(os.path.join(args.results_dir, "ablation_policy_comparison.json"), "w") as f:
        json.dump(ablation_2_records, f, indent=2)
    with open(os.path.join(args.results_dir, "ablation_mismatched_rate.json"), "w") as f:
        json.dump(ablation_3_records, f, indent=2)

    # Multi-panel visualization
    print("\nGenerating publication figure for Ablation Studies...")
    sns.set_theme(style="whitegrid", font="sans-serif")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # Panel 1: Convergence vs Latency
    ax = axes[0]
    ax2 = ax.twinx()
    df1 = pd.DataFrame(ablation_1_records)
    line1 = ax.plot(df1["sample_size_S"], df1["mean_risk_variance"], marker="o", color="#1f77b4", linewidth=2.5, label="Risk Variance (σ_R²)")
    line2 = ax2.plot(df1["sample_size_S"], df1["latency_us_per_flow"], marker="s", color="#d62728", linewidth=2.5, linestyle="--", label="Latency (μs/flow)")
    ax.set_title("(a) Convergence vs. Latency Tradeoff (S)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Monte Carlo Sample Size S", fontsize=11)
    ax.set_ylabel("Predictive Variance σ_R²", fontsize=11, color="#1f77b4")
    ax2.set_ylabel("Inference Latency (μs/flow)", fontsize=11, color="#d62728")
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc="center right", frameon=True)
    ax.grid(True, alpha=0.3)

    # Panel 2: Policy Comparison (FPR Suppression)
    ax = axes[1]
    df2 = pd.DataFrame(ablation_2_records)
    sns.barplot(data=df2, x="sampling_ratio", y="fpr", hue="policy", ax=ax, palette=["#d62728", "#ff7f0e", "#2ca02c"])
    ax.set_title("(b) False Positive Rate: Binary vs. SAFE-3WD", fontsize=12, fontweight="bold")
    ax.set_xlabel("Sampling Ratio", fontsize=11)
    ax.set_ylabel("False Positive Rate", fontsize=11)
    ax.legend(title="Decision Policy", loc="upper left", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    # Panel 3: Mismatch Robustness
    ax = axes[2]
    df3 = pd.DataFrame(ablation_3_records)
    ax.plot(df3["drift_factor"], df3["selective_accuracy"] * 100, marker="o", color="#2ca02c", linewidth=2.5, label="Selective Accuracy (%)")
    ax.plot(df3["drift_factor"], df3["coverage"] * 100, marker="^", color="#1f77b4", linewidth=2.5, linestyle="-.", label="Coverage Rate (%)")
    ax.plot(df3["drift_factor"], df3["selective_fpr"] * 100, marker="x", color="#d62728", linewidth=2.0, label="False Positive Rate (%)")
    ax.set_title("(c) Robustness to Sampling Rate Drift (±30%)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Drift Ratio (p_assumed / p_true)", fontsize=11)
    ax.set_ylabel("Percentage (%)", fontsize=11)
    ax.axvline(1.0, color="gray", linestyle=":", label="Exact Rate (Nominal)")
    ax.legend(loc="center left", frameon=True, fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(args.figures_dir, "ablation_studies_analysis.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\nSaved ablation figure to: {fig_path}")
    print("================================================================================\n")


if __name__ == "__main__":
    main()
