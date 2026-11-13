"""
FlowGuard Experiment 01: Base Classifier & The Oblivious Baseline Evaluation
=============================================================================

Demonstrates Milestone 1:
1. Trains base classifier (LightGBM) on pristine complete flow traffic.
2. Evaluates performance on complete pristine test data (~99% baseline).
3. Evaluates the uncorrected "Oblivious" model under multiple packet sampling rates
   (p in {1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01}) and burst packet loss regimes.
4. Generates publication-ready figures showing the catastrophic surge in False Negatives (FNR).

Usage:
    python experiments/01_evaluate_oblivious.py [--model lightgbm] [--save-model]
"""

import argparse
import json
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from flowguard.data import FlowDataLoader
from flowguard.evaluation import compute_classification_metrics
from flowguard.models import BaseIntrusionClassifier, ObliviousBaseline
from flowguard.sampling import PacketSamplingSimulator

def main():
    parser = argparse.ArgumentParser(description="FlowGuard Milestone 1: Oblivious Baseline Evaluation")
    parser.add_argument("--model", type=str, default="lightgbm", choices=["lightgbm", "random_forest", "xgboost"], help="Base model type")
    parser.add_argument("--data-dir", type=str, default="data/processed", help="Processed parquet directory")
    parser.add_argument("--results-dir", type=str, default="experiments/results", help="Results directory")
    parser.add_argument("--figures-dir", type=str, default="experiments/figures", help="Figures directory")
    parser.add_argument("--models-dir", type=str, default="experiments/models", help="Model checkpoints directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.figures_dir, exist_ok=True)
    os.makedirs(args.models_dir, exist_ok=True)

    print("=====================================================================")
    print(" FlowGuard Step 3: Base Classifier & Oblivious Baseline (Milestone 1)")
    print("=====================================================================")

    # 1. Load pristine splits
    print("\n[1/4] Loading pristine flow datasets from:", os.path.abspath(args.data_dir))
    loader = FlowDataLoader(processed_dir=args.data_dir, random_seed=args.seed)
    (X_train, y_train), (X_calib, y_calib), (X_test, y_test) = loader.load_splits()
    print(f"      Train Set: {len(X_train):,} flows | Calibration Set: {len(X_calib):,} | Test Set: {len(X_test):,}")

    # 2. Train pristine base classifier
    print(f"\n[2/4] Training pristine {args.model.upper()} classifier on complete traffic (p = 1.0)...")
    start_train = time.time()
    base_model = BaseIntrusionClassifier(model_type=args.model, random_state=args.seed)
    base_model.fit(X_train, y_train)
    train_time = time.time() - start_train
    print(f"      Model trained in {train_time:.2f} seconds.")

    # Save trained baseline model
    model_save_path = os.path.join(args.models_dir, f"pristine_{args.model}.joblib")
    base_model.save(model_save_path)

    # 3. Establish Pristine Baseline Ceiling
    print("\n[3/4] Establishing pristine test set baseline ceiling (complete flows)...")
    y_test_pred_pristine = base_model.predict(X_test)
    y_test_prob_pristine = base_model.predict_proba(X_test)
    pristine_metrics = compute_classification_metrics(y_test, y_test_pred_pristine, y_test_prob_pristine)

    print("---------------------------------------------------------------------")
    print(f"  Pristine Complete Traffic Performance (p = 1.0):")
    print(f"  • Detection Rate (Recall): {pristine_metrics['recall']*100:.2f}%")
    print(f"  • False Negative Rate (FNR): {pristine_metrics['false_negative_rate']*100:.2f}%  (Missed Attacks: {pristine_metrics['false_negatives']})")
    print(f"  • False Positive Rate (FPR): {pristine_metrics['false_positive_rate']*100:.2f}%  (False Alarms: {pristine_metrics['false_positives']})")
    print(f"  • F1-Score: {pristine_metrics['f1_score']*100:.2f}%")
    print(f"  • ROC-AUC: {pristine_metrics.get('roc_auc', 0)*100:.2f}%")
    print(f"  • Expected Calibration Error (ECE): {pristine_metrics.get('expected_calibration_error', 0):.4f}")
    print("---------------------------------------------------------------------")

    # 4. Evaluate Oblivious Model across Sampling Sweeps
    print("\n[4/4] Exposing the Oblivious Model under packet sampling decimation...")
    sampling_rates = [1.0, 0.50, 0.20, 0.10, 0.05, 0.02, 0.01]
    simulator = PacketSamplingSimulator(seed=args.seed)
    oblivious_model = ObliviousBaseline(base_model=base_model, threshold=0.5)

    results_table = []

    print(f"\n{'Sampling Rate (p)':<18} | {'Recall / DR (%)':<16} | {'FNR (%)':<10} | {'FPR (%)':<10} | {'F1-Score (%)':<12} | {'ECE':<8}")
    print("-" * 85)

    for p in sampling_rates:
        if p == 1.0:
            X_degraded = X_test
        else:
            X_degraded = simulator.sample_random(X_test, sampling_rate=p)

        y_pred = oblivious_model.predict(X_degraded)
        y_prob = oblivious_model.predict_proba(X_degraded)
        metrics = compute_classification_metrics(y_test, y_pred, y_prob)

        metrics["sampling_rate"] = p
        metrics["sampling_ratio"] = f"1:{int(round(1.0/p))}" if p < 1.0 else "1:1 (Complete)"
        metrics["mechanism"] = "uniform_random_sampling"
        results_table.append(metrics)

        print(
            f"{metrics['sampling_ratio']:<18} | "
            f"{metrics['recall']*100:>14.2f}% | "
            f"{metrics['false_negative_rate']*100:>8.2f}% | "
            f"{metrics['false_positive_rate']*100:>8.2f}% | "
            f"{metrics['f1_score']*100:>10.2f}% | "
            f"{metrics['expected_calibration_error']:>8.4f}"
        )

    # Also evaluate under Burst Packet Loss (Gilbert-Elliott)
    print("\nEvaluating under Burst Packet Loss (Gilbert-Elliott switch buffer drops):")
    burst_losses = [0.02, 0.05, 0.10, 0.15]
    for loss in burst_losses:
        X_burst = simulator.sample_burst_loss(X_test, loss_rate=loss, burst_length=5.0)
        y_pred_burst = oblivious_model.predict(X_burst)
        y_prob_burst = oblivious_model.predict_proba(X_burst)
        b_metrics = compute_classification_metrics(y_test, y_pred_burst, y_prob_burst)
        b_metrics["sampling_rate"] = 1.0 - loss
        b_metrics["sampling_ratio"] = f"Loss {loss*100:.0f}%"
        b_metrics["mechanism"] = "gilbert_elliott_burst_loss"
        results_table.append(b_metrics)

        print(
            f"{b_metrics['sampling_ratio']:<18} | "
            f"{b_metrics['recall']*100:>14.2f}% | "
            f"{b_metrics['false_negative_rate']*100:>8.2f}% | "
            f"{b_metrics['false_positive_rate']*100:>8.2f}% | "
            f"{b_metrics['f1_score']*100:>10.2f}% | "
            f"{b_metrics['expected_calibration_error']:>8.4f}"
        )

    # Save benchmark records to CSV & JSON
    df_results = pd.DataFrame(results_table)
    csv_path = os.path.join(args.results_dir, "oblivious_benchmark.csv")
    json_path = os.path.join(args.results_dir, "oblivious_benchmark.json")
    df_results.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_table, f, indent=2)

    print(f"\nSaved benchmark results to:")
    print(f"  • {os.path.abspath(csv_path)}")
    print(f"  • {os.path.abspath(json_path)}")

    # 5. Generate Publication-Quality Figures
    print("\nGenerating publication figures for faculty review...")
    df_sampling = df_results[df_results["mechanism"] == "uniform_random_sampling"].copy()
    
    # Invert order for plotting from p=1.0 down to p=0.01
    rates = df_sampling["sampling_rate"].values
    fnrs = df_sampling["false_negative_rate"].values * 100
    recalls = df_sampling["recall"].values * 100
    f1s = df_sampling["f1_score"].values * 100
    eces = df_sampling["expected_calibration_error"].values

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

    # Plot 1: False Negative Surge & Recall Collapse
    ax1.plot(rates, fnrs, "o-", color="#D32F2F", linewidth=2.5, markersize=7, label="False Negative Rate (FNR %)")
    ax1.plot(rates, recalls, "s-", color="#1976D2", linewidth=2.5, markersize=7, label="Detection Rate / Recall (%)")
    ax1.plot(rates, f1s, "^--", color="#388E3C", linewidth=2.0, markersize=6, label="F1-Score (%)")
    ax1.set_xscale("log")
    ax1.set_xlabel("Packet Sampling Rate $p$ (Log Scale)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Metric Value (%)", fontsize=12, fontweight="bold")
    ax1.set_title("Catastrophic Intrusion Detection Failure\nUnder Oblivious Packet Sampling (Milestone 1)", fontsize=13, fontweight="bold")
    ax1.legend(loc="best", frameon=True, fontsize=10)
    ax1.set_ylim(-2, 105)
    ax1.grid(True, which="both", ls="--", alpha=0.6)

    # Annotate key failure points
    ax1.annotate(
        f"Pristine Baseline\nFNR: {fnrs[0]:.1f}%\nDR: {recalls[0]:.1f}%",
        xy=(1.0, fnrs[0]),
        xytext=(0.3, 18),
        arrowprops=dict(facecolor="black", shrink=0.08, width=1, headwidth=6),
        fontsize=9,
        fontweight="bold"
    )
    ax1.annotate(
        f"Catastrophic Collapse (1:100)\nFNR Surges to {fnrs[-1]:.1f}%\nDR drops to {recalls[-1]:.1f}%",
        xy=(0.01, fnrs[-1]),
        xytext=(0.015, 65),
        arrowprops=dict(facecolor="#D32F2F", shrink=0.08, width=1.5, headwidth=6),
        fontsize=9,
        fontweight="bold",
        color="#D32F2F"
    )

    # Plot 2: Expected Calibration Error (ECE) Drift
    ax2.plot(rates, eces, "d-", color="#E65100", linewidth=2.5, markersize=7, label="Expected Calibration Error (ECE)")
    ax2.set_xscale("log")
    ax2.set_xlabel("Packet Sampling Rate $p$ (Log Scale)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Expected Calibration Error (ECE)", fontsize=12, fontweight="bold")
    ax2.set_title("Severe Overconfidence Calibration Drift\nUnder Oblivious Model Deployment", fontsize=13, fontweight="bold")
    ax2.legend(loc="best", frameon=True, fontsize=10)
    ax2.grid(True, which="both", ls="--", alpha=0.6)

    plt.tight_layout()
    fig_path = os.path.join(args.figures_dir, "oblivious_detection_collapse.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()

    print(f"\nSaved publication-ready figure to:")
    print(f"  • {os.path.abspath(fig_path)}")
    print("=====================================================================\n")

if __name__ == "__main__":
    main()
