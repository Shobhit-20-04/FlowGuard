"""
FlowGuard Step 5: SAFE-3WD Calibration & Decision Policy Evaluation (Milestone 3)
=================================================================================

Executes:
1. Loads pristine Train, Calibration, and Held-Out Test splits.
2. Trains regularized base classifier and fits SDUM on Train.
3. Calibrates optimal decision boundaries (tau_safe, tau_attack, sigma_thresh)
   strictly on the held-out Calibration Split using cost-sensitive risk minimization.
4. Evaluates SAFE-3WD on the Held-Out Test Set across sampling sweeps p in [1.0, 0.01].
5. Compares SAFE-3WD vs. The Oblivious Baseline (Selective Acc vs. Oblivious Acc, FPR reduction, Cost).
6. Outputs publication-quality CSV/JSON benchmarks and high-resolution figures.
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
from flowguard.evaluation.metrics import compute_classification_metrics
from flowguard.models.base_classifier import BaseIntrusionClassifier
from flowguard.sampling.simulator import PacketSamplingSimulator
from flowguard.uncertainty.propagator import UncertaintyPropagator, RiskProfile
from flowguard.uncertainty.sdum import SamplingDistortionUncertaintyModel


def main():
    parser = argparse.ArgumentParser(description="FlowGuard Milestone 3: SAFE-3WD Calibration & Evaluation")
    parser.add_argument("--data-dir", type=str, default="data/processed", help="Path to processed parquet datasets")
    parser.add_argument("--model", type=str, default="lightgbm", choices=["lightgbm", "random_forest", "xgboost"], help="Base classifier")
    parser.add_argument("--mc-samples", type=int, default=30, help="Number of Monte Carlo realizations S")
    parser.add_argument("--cost-fn", type=float, default=10.0, help="Penalty for False Negative (missed attack)")
    parser.add_argument("--cost-fp", type=float, default=1.0, help="Penalty for False Positive (false alarm)")
    parser.add_argument("--cost-inv", type=float, default=0.15, help="Penalty for deep investigation")
    parser.add_argument("--min-coverage", type=float, default=0.40, help="Minimum acceptable automated decision coverage")
    parser.add_argument("--results-dir", type=str, default="experiments/results", help="Directory for benchmark outputs")
    parser.add_argument("--figures-dir", type=str, default="experiments/figures", help="Directory for generated plots")
    parser.add_argument("--models-dir", type=str, default="experiments/models", help="Directory for model checkpoints")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic random seed")
    parser.add_argument("--test-subset", type=int, default=2000, help="Subset of test flows for MC propagation benchmark")
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.figures_dir, exist_ok=True)
    os.makedirs(args.models_dir, exist_ok=True)

    print("=====================================================================")
    print(" FlowGuard Step 5: SAFE-3WD Calibration & Evaluation (Milestone 3)")
    print("=====================================================================")

    # 1. Load Dataset Partitions
    print(f"\n[1/5] Loading flow datasets from: {os.path.abspath(args.data_dir)}")
    loader = FlowDataLoader(processed_dir=args.data_dir, random_seed=args.seed)
    (X_train, y_train), (X_calib, y_calib), (X_test, y_test) = loader.load_splits()
    print(f"      Train Set:       {len(X_train):,} flows")
    print(f"      Calibration Set: {len(X_calib):,} flows (STRICTLY RESERVED FOR CALIBRATION)")
    print(f"      Test Set:        {len(X_test):,} flows (HELD-OUT BENCHMARK)")

    # 2. Train Base Classifier & Fit SDUM on Training Set
    print(f"\n[2/5] Training regularized {args.model.upper()} classifier and fitting SDUM...")
    base_clf = BaseIntrusionClassifier(
        model_type=args.model,
        random_state=args.seed,
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
    base_clf.fit(X_train, y_train)

    sdum = SamplingDistortionUncertaintyModel(n_mc_samples=args.mc_samples, random_state=args.seed)
    sdum.fit(X_train)

    propagator = UncertaintyPropagator(base_classifier=base_clf, sdum=sdum, n_mc_samples=args.mc_samples)
    simulator = PacketSamplingSimulator(seed=args.seed)

    # 3. Cost-Sensitive Threshold Calibration on the Calibration Split
    print(f"\n[3/5] Calibrating decision boundaries on the Calibration Split (C_FN={args.cost_fn}, C_FP={args.cost_fp}, C_INV={args.cost_inv})...")
    # Decimate calibration split under representative sampling (p = 0.05) to expose uncertainty
    p_calib = 0.05
    n_cal_sub = min(1500, len(X_calib))
    cal_idx = np.random.default_rng(args.seed).choice(len(X_calib), size=n_cal_sub, replace=False)
    X_cal_eval = X_calib.iloc[cal_idx].copy()
    y_cal_eval = y_calib.iloc[cal_idx].copy()

    X_cal_deg = simulator.sample_random(X_cal_eval, sampling_rate=p_calib)
    calib_risk_profile = propagator.propagate(X_cal_deg, sampling_rate=p_calib)

    calibrator = CostSensitiveThresholdCalibrator(
        cost_fn=args.cost_fn,
        cost_fp=args.cost_fp,
        cost_inv=args.cost_inv,
        min_coverage=args.min_coverage,
        grid_steps=25,
    )
    calibrator.fit(calib_risk_profile, y_cal_eval)
    calibrated_engine = calibrator.create_calibrated_engine()

    print(f"      Calibration Complete:")
    print(f"      • Optimal tau_safe:           {calibrator.optimal_tau_safe_:.4f}")
    print(f"      • Optimal tau_attack:         {calibrator.optimal_tau_attack_:.4f}")
    print(f"      • Optimal max variance:       {calibrator.optimal_sigma_thresh_:.4f}")
    print(f"      • Calibration Empirical Cost: {calibrator.best_cost_:.4f}")

    # Save calibrated parameters
    params_path = os.path.join(args.models_dir, "calibrated_safe_3wd_params.json")
    with open(params_path, "w") as f:
        json.dump({
            "optimal_tau_safe": calibrator.optimal_tau_safe_,
            "optimal_tau_attack": calibrator.optimal_tau_attack_,
            "optimal_sigma_thresh": calibrator.optimal_sigma_thresh_,
            "best_calibration_cost": calibrator.best_cost_,
            "cost_weights": {
                "cost_fn": args.cost_fn,
                "cost_fp": args.cost_fp,
                "cost_inv": args.cost_inv,
            },
            "min_coverage_constraint": args.min_coverage,
        }, f, indent=2)

    # 4. Evaluation Across Sampling Sweeps on the Held-Out Test Set
    print(f"\n[4/5] Benchmarking SAFE-3WD vs. Oblivious Baseline on Held-Out Test Set...")
    n_eval = min(args.test_subset, len(X_test))
    eval_idx = np.random.default_rng(args.seed).choice(len(X_test), size=n_eval, replace=False)
    X_eval = X_test.iloc[eval_idx].copy()
    y_eval = y_test.iloc[eval_idx].copy()

    sampling_rates = [1.0, 0.50, 0.20, 0.10, 0.05, 0.02, 0.01]
    benchmark_records: List[Dict[str, Any]] = []

    print("\n" + "=" * 115)
    print(f"{'Sampling':<14} | {'Oblivious Acc':<14} | {'Oblivious FPR':<14} | {'FlowGuard Sel Acc':<18} | {'FlowGuard FPR':<14} | {'Coverage':<10} | {'Inv. Rate':<10}")
    print("-" * 115)

    for p in sampling_rates:
        if p >= 1.0:
            X_deg = X_eval.copy()
        else:
            X_deg = simulator.sample_random(X_eval, sampling_rate=p)

        # Baseline: Oblivious Model
        p_oblivious = base_clf.predict_proba(X_deg)
        pred_oblivious = (p_oblivious >= 0.5).astype(int)
        obliv_metrics = compute_classification_metrics(y_eval, pred_oblivious, p_oblivious)

        # Oblivious Operational Cost
        obliv_cost = (
            (args.cost_fn * obliv_metrics["false_negatives"])
            + (args.cost_fp * obliv_metrics["false_positives"])
        ) / max(1, len(y_eval))

        # FlowGuard: SAFE-3WD
        test_risk_profile = propagator.propagate(X_deg, sampling_rate=p)
        safe_eval = calibrated_engine.evaluate(
            risk_profile=test_risk_profile,
            y_true=y_eval,
            cost_fn=args.cost_fn,
            cost_fp=args.cost_fp,
            cost_inv=args.cost_inv,
        )

        ratio_str = f"1:{int(1/p)}" if p < 1.0 else "1:1 (Pristine)"

        record = {
            "sampling_rate": p,
            "sampling_ratio": ratio_str,
            # Oblivious Metrics
            "oblivious_accuracy": obliv_metrics["accuracy"],
            "oblivious_detection_rate": obliv_metrics["recall"],
            "oblivious_fpr": obliv_metrics["false_positive_rate"],
            "oblivious_fnr": obliv_metrics["false_negative_rate"],
            "oblivious_f1": obliv_metrics["f1_score"],
            "oblivious_cost_per_flow": float(obliv_cost),
            # SAFE-3WD Metrics
            "flowguard_coverage": safe_eval["coverage"],
            "flowguard_investigation_rate": safe_eval["investigation_rate"],
            "flowguard_selective_accuracy": safe_eval["selective_accuracy"],
            "flowguard_selective_detection_rate": safe_eval["selective_detection_rate"],
            "flowguard_selective_fpr": safe_eval["selective_false_positive_rate"],
            "flowguard_cost_per_flow": safe_eval["normalized_cost_per_flow"],
            "flowguard_cost_reduction_ratio": float(obliv_cost / max(1e-6, safe_eval["normalized_cost_per_flow"])),
            # Counts
            "true_positives_definite": safe_eval["true_positives_definite"],
            "true_negatives_definite": safe_eval["true_negatives_definite"],
            "false_positives_definite": safe_eval["false_positives_definite"],
            "false_negatives_definite": safe_eval["false_negatives_definite"],
            "investigated_count": safe_eval["investigated_count"],
            "total_samples": safe_eval["total_samples"],
        }
        benchmark_records.append(record)

        print(
            f"{ratio_str:<14} | {obliv_metrics['accuracy']*100:13.2f}% | "
            f"{obliv_metrics['false_positive_rate']*100:13.2f}% | "
            f"{safe_eval['selective_accuracy']*100:17.2f}% | "
            f"{safe_eval['selective_false_positive_rate']*100:13.2f}% | "
            f"{safe_eval['coverage']*100:9.2f}% | "
            f"{safe_eval['investigation_rate']*100:9.2f}%"
        )

    print("=" * 115)

    # 5. Save Benchmark Data & Generate Publication Figures
    print("\n[5/5] Saving benchmark data and generating publication figures...")
    df_results = pd.DataFrame(benchmark_records)
    csv_path = os.path.join(args.results_dir, "safe_3wd_benchmark.csv")
    json_path = os.path.join(args.results_dir, "safe_3wd_benchmark.json")
    df_results.to_csv(csv_path, index=False)
    with open(json_path, "w") as f:
        json.dump(benchmark_records, f, indent=2)

    print(f"      Benchmark saved to:")
    print(f"      • {csv_path}")
    print(f"      • {json_path}")

    # Generate 4-panel publication visualization
    sns.set_theme(style="whitegrid", font="sans-serif")
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    x_labels = [r["sampling_ratio"] for r in benchmark_records]
    x_positions = range(len(benchmark_records))

    # Panel A: Selective Accuracy vs Oblivious Accuracy
    ax = axes[0, 0]
    obliv_accs = [r["oblivious_accuracy"] * 100 for r in benchmark_records]
    safe_accs = [r["flowguard_selective_accuracy"] * 100 for r in benchmark_records]
    ax.plot(x_positions, safe_accs, marker="o", color="#2ca02c", linewidth=2.8, markersize=8, label="FlowGuard (SAFE-3WD)")
    ax.plot(x_positions, obliv_accs, marker="s", color="#d62728", linewidth=2.5, linestyle="--", markersize=8, label="Oblivious Baseline (B1)")
    ax.set_title("(a) Selective Accuracy vs. Oblivious Accuracy Across Decimation", fontsize=12, fontweight="bold")
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, rotation=30, ha="right")
    ax.set_ylim(20, 105)
    ax.legend(loc="lower left", frameon=True)
    ax.grid(True, alpha=0.3)

    # Panel B: False Positive Rate (False Alarm Suppression)
    ax = axes[0, 1]
    obliv_fprs = [r["oblivious_fpr"] * 100 for r in benchmark_records]
    safe_fprs = [r["flowguard_selective_fpr"] * 100 for r in benchmark_records]
    ax.plot(x_positions, obliv_fprs, marker="s", color="#d62728", linewidth=2.5, linestyle="--", markersize=8, label="Oblivious Baseline (False Alarms)")
    ax.plot(x_positions, safe_fprs, marker="o", color="#1f77b4", linewidth=2.8, markersize=8, label="FlowGuard SAFE-3WD (Definite Decisions)")
    ax.set_title("(b) False Positive Rate: Catastrophic Surge vs. FlowGuard Suppression", fontsize=12, fontweight="bold")
    ax.set_ylabel("False Positive Rate (%)", fontsize=11)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, rotation=30, ha="right")
    ax.legend(loc="upper left", frameon=True)
    ax.grid(True, alpha=0.3)

    # Panel C: Three-Way Decision Region Distribution (Stacked Bar)
    ax = axes[1, 0]
    benign_pcts = np.array([
        (r["true_negatives_definite"] + r["false_negatives_definite"]) / r["total_samples"] * 100
        for r in benchmark_records
    ])
    attack_pcts = np.array([
        (r["true_positives_definite"] + r["false_positives_definite"]) / r["total_samples"] * 100
        for r in benchmark_records
    ])
    inv_pcts = np.array([
        r["investigated_count"] / r["total_samples"] * 100
        for r in benchmark_records
    ])

    p1 = ax.bar(x_positions, benign_pcts, color="#1f77b4", alpha=0.85, label="Region 1: Benign (Fast Accept)")
    p2 = ax.bar(x_positions, attack_pcts, bottom=benign_pcts, color="#d62728", alpha=0.85, label="Region 2: Attack (Immediate Block)")
    p3 = ax.bar(x_positions, inv_pcts, bottom=benign_pcts + attack_pcts, color="#ff7f0e", alpha=0.85, label="Region 3: Investigate (Defer)")

    ax.set_title("(c) Decision Region Partitioning Across Sampling Rates", fontsize=12, fontweight="bold")
    ax.set_ylabel("Flow Distribution (%)", fontsize=11)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, rotation=30, ha="right")
    ax.set_ylim(0, 105)
    ax.legend(loc="lower left", frameon=True)
    ax.grid(True, axis="y", alpha=0.3)

    # Panel D: Operational Risk Cost Reduction
    ax = axes[1, 1]
    obliv_costs = [r["oblivious_cost_per_flow"] for r in benchmark_records]
    safe_costs = [r["flowguard_cost_per_flow"] for r in benchmark_records]
    width = 0.35
    ax.bar([p - width/2 for p in x_positions], obliv_costs, width=width, color="#d62728", alpha=0.8, label="Oblivious Operational Cost")
    ax.bar([p + width/2 for p in x_positions], safe_costs, width=width, color="#2ca02c", alpha=0.8, label="FlowGuard SAFE-3WD Cost")
    ax.set_title("(d) Operational Cost Comparison per Flow (Cost-Sensitive Penalty)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Normalized Cost / Flow", fontsize=11)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, rotation=30, ha="right")
    ax.legend(loc="upper left", frameon=True)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(args.figures_dir, "safe_3wd_evaluation.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\nSaved publication-ready figure to:")
    print(f"• {fig_path}")
    print("=====================================================================\n")


if __name__ == "__main__":
    main()
