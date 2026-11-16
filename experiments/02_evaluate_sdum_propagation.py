"""
FlowGuard Step 4: SDUM & Feature Uncertainty Propagation Evaluation (Milestone 2)
================================================================================

Executes:
1. Quantitative Overfitting vs. Underfitting Diagnostic on Base Classifier.
2. SDUM flow reconstruction across packet sampling sweep p in [1.0, 0.01].
3. Uncertainty risk interval propagation [R_min, R_max], predictive variance sigma_R^2,
   and epistemic interval spread Delta_R = R_max - R_min.
4. Publication-ready visualization saved to experiments/figures/.
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.model_selection import train_test_split

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from flowguard.data.loader import FlowDataLoader
from flowguard.models.base_classifier import BaseIntrusionClassifier
from flowguard.models.diagnostics import ModelGeneralizationDiagnostics
from flowguard.sampling.simulator import PacketSamplingSimulator
from flowguard.uncertainty.propagator import UncertaintyPropagator, RiskProfile
from flowguard.uncertainty.sdum import SamplingDistortionUncertaintyModel


def main():
    parser = argparse.ArgumentParser(description="FlowGuard SDUM & Uncertainty Propagation Engine")
    parser.add_argument("--data-dir", type=str, default="data/processed", help="Path to processed parquet datasets")
    parser.add_argument("--model", type=str, default="lightgbm", choices=["lightgbm", "random_forest", "xgboost"], help="Base classifier")
    parser.add_argument("--mc-samples", type=int, default=30, help="Number of Monte Carlo realizations S")
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
    print(" FlowGuard Step 4: SDUM & Uncertainty Propagation Engine (Milestone 2)")
    print("=====================================================================")

    # 1. Load Dataset Partitions
    print(f"\n[1/5] Loading flow datasets from: {os.path.abspath(args.data_dir)}")
    loader = FlowDataLoader(processed_dir=args.data_dir, random_seed=args.seed)
    (X_train, y_train), (X_calib, y_calib), (X_test, y_test) = loader.load_splits()
    print(f"      Train: {len(X_train):,} | Calibration: {len(X_calib):,} | Test: {len(X_test):,}")

    # 2. Base Classifier Training & Regularization
    print(f"\n[2/5] Initializing and training regularized {args.model.upper()} classifier...")
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
    t0 = time.time()
    base_clf.fit(X_train, y_train)
    train_duration = time.time() - t0
    print(f"      Model trained in {train_duration:.2f} seconds.")

    # 3. Model Generalization Diagnostic (Overfitting vs. Underfitting Audit)
    print("\n[3/5] Performing Deep Overfitting vs. Underfitting Audit...")
    diag_engine = ModelGeneralizationDiagnostics(random_state=args.seed)
    fit_diagnostic = diag_engine.evaluate_fit(
        model=base_clf,
        X_train=X_train,
        y_train=y_train,
        X_val=X_calib,
        y_val=y_calib,
        X_test=X_test,
        y_test=y_test,
    )
    print(diag_engine.format_report(fit_diagnostic, model_name=args.model))

    # Save diagnostic JSON
    diag_save_path = os.path.join(args.results_dir, "model_fit_diagnostic.json")
    with open(diag_save_path, "w") as f:
        json.dump(fit_diagnostic, f, indent=2)

    # 4. Fit SDUM Inversion Model on Pristine Training Data
    print(f"\n[4/5] Fitting SDUM on pristine flows (learning priors and correlation structure)...")
    sdum = SamplingDistortionUncertaintyModel(n_mc_samples=args.mc_samples, random_state=args.seed)
    sdum.fit(X_train)
    print(f"      SDUM initialized with S = {args.mc_samples} Monte Carlo realizations.")

    # 5. Evaluate Uncertainty Propagation Across Sampling Sweeps
    print(f"\n[5/5] Propagating feature uncertainty across packet sampling sweeps (S = {args.mc_samples})...")
    propagator = UncertaintyPropagator(base_classifier=base_clf, sdum=sdum, n_mc_samples=args.mc_samples)
    simulator = PacketSamplingSimulator(seed=args.seed)

    # Evaluate on a genuinely stratified subset of test flows for reproducible benchmark execution
    n_eval = min(args.test_subset, len(X_test))
    if n_eval < len(X_test):
        X_eval, _, y_eval, _ = train_test_split(
            X_test,
            y_test,
            train_size=n_eval,
            stratify=y_test,
            random_state=args.seed,
        )
    else:
        X_eval = X_test.copy()
        y_eval = y_test.copy()

    print(f"      Selected {len(X_eval):,} stratified test flows (Attacks: {y_eval.sum():,}, {y_eval.mean()*100:.2f}%)")

    sampling_rates = [1.0, 0.50, 0.20, 0.10, 0.05, 0.02, 0.01]
    benchmark_records: List[Dict[str, Any]] = []

    print("\nSampling Rate (p)  | Mean Risk Spread (ΔR) | Mean Variance (σ_R²) | Ambiguous Flows (%) | Latency (μs/flow)")
    print("-" * 92)

    profiles_by_rate: Dict[float, RiskProfile] = {}

    for p in sampling_rates:
        t_start = time.perf_counter()
        if p >= 1.0:
            X_deg = X_eval.copy()
        else:
            X_deg = simulator.sample_random(X_eval, sampling_rate=p)

        # Propagate through SDUM + Classifier
        profile = propagator.propagate(X_deg, sampling_rate=p)
        t_elapsed = time.perf_counter() - t_start

        profiles_by_rate[p] = profile

        latency_us = (t_elapsed / n_eval) * 1e6
        mean_spread = float(np.mean(profile.risk_spread))
        mean_var = float(np.mean(profile.risk_variance))

        # An ambiguous flow has risk interval [R_min, R_max] crossing standard threshold tau = 0.5
        # i.e., R_min < 0.5 < R_max (epistemic boundary overlap)
        is_ambiguous = (profile.risk_min < 0.5) & (profile.risk_max > 0.5)
        ambiguous_pct = float(np.mean(is_ambiguous) * 100)

        record = {
            "sampling_rate": p,
            "sampling_ratio": f"1:{int(1/p)}" if p < 1.0 else "1:1 (Complete)",
            "mean_risk_spread": mean_spread,
            "mean_risk_variance": mean_var,
            "ambiguous_flows_percent": ambiguous_pct,
            "ambiguous_count": int(np.sum(is_ambiguous)),
            "total_evaluated_flows": n_eval,
            "latency_us_per_flow": latency_us,
        }
        benchmark_records.append(record)

        print(f"{record['sampling_ratio']:<18} | {mean_spread:21.4f} | {mean_var:20.6f} | {ambiguous_pct:17.2f}% | {latency_us:15.1f}")

    # Save benchmark records
    df_results = pd.DataFrame(benchmark_records)
    csv_path = os.path.join(args.results_dir, "sdum_propagation_benchmark.csv")
    json_path = os.path.join(args.results_dir, "sdum_propagation_benchmark.json")
    df_results.to_csv(csv_path, index=False)
    with open(json_path, "w") as f:
        json.dump(benchmark_records, f, indent=2)

    print(f"\nSaved SDUM propagation benchmark to:")
    print(f"  • {csv_path}")
    print(f"  • {json_path}")

    # Generate Publication Figures
    print("\nGenerating publication figure for Milestone 2...")
    sns.set_theme(style="whitegrid", font="sans-serif")
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # Panel A: Mean Epistemic Spread vs Sampling Rate
    rates_inv = [1.0 / r["sampling_rate"] for r in benchmark_records]
    spreads = [r["mean_risk_spread"] for r in benchmark_records]
    ax = axes[0, 0]
    ax.plot(rates_inv, spreads, marker="o", color="#1f77b4", linewidth=2.5, markersize=8)
    ax.set_title("(a) Epistemic Risk Interval Spread vs. Packet Decimation (1:N)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Decimation Factor N (1-in-N Packet Sampling)", fontsize=11)
    ax.set_ylabel("Mean Risk Spread: ΔR = E[R_max - R_min]", fontsize=11)
    ax.set_xscale("log")
    ax.grid(True, which="both", alpha=0.3)

    # Panel B: Mean Risk Variance vs Sampling Rate
    variances = [r["mean_risk_variance"] for r in benchmark_records]
    ax = axes[0, 1]
    ax.plot(rates_inv, variances, marker="s", color="#d62728", linewidth=2.5, markersize=8)
    ax.set_title("(b) Predictive Risk Variance vs. Packet Decimation (1:N)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Decimation Factor N (1-in-N Packet Sampling)", fontsize=11)
    ax.set_ylabel("Predictive Variance: σ_R²", fontsize=11)
    ax.set_xscale("log")
    ax.grid(True, which="both", alpha=0.3)

    # Panel C: Ambiguous Boundary-Crossing Flows vs Sampling Rate
    ambig_pcts = [r["ambiguous_flows_percent"] for r in benchmark_records]
    ax = axes[1, 0]
    x_positions = range(len(benchmark_records))
    ax.bar(
        x_positions,
        ambig_pcts,
        color="#2ca02c",
        alpha=0.85,
        edgecolor="black",
    )
    ax.set_title("(c) Flows Requiring Safe Three-Way Deferral (R_min < 0.5 < R_max)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Packet Sampling Configuration", fontsize=11)
    ax.set_ylabel("Uncertain / Overlapping Flows (%)", fontsize=11)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([r["sampling_ratio"] for r in benchmark_records], rotation=30, ha="right")
    ax.grid(True, axis="y", alpha=0.3)

    # Panel D: Risk Interval Bands [R_min, R_max] for 50 Sample Flows at p = 0.05
    ax = axes[1, 1]
    p_demo = 0.05
    demo_profile = profiles_by_rate[p_demo]
    sample_indices = np.arange(min(50, n_eval))
    y_sub = y_eval.iloc[sample_indices].values

    r_min_sub = demo_profile.risk_min[sample_indices]
    r_max_sub = demo_profile.risk_max[sample_indices]
    r_mean_sub = demo_profile.risk_mean[sample_indices]

    for idx in range(len(sample_indices)):
        color = "#d62728" if y_sub[idx] == 1 else "#1f77b4"
        ax.vlines(idx, r_min_sub[idx], r_max_sub[idx], color=color, alpha=0.6, linewidth=2.0)
        ax.scatter(idx, r_mean_sub[idx], color=color, s=15, zorder=3)

    ax.axhline(0.5, color="black", linestyle="--", linewidth=1.2, label="Oblivious Threshold (τ = 0.5)")
    ax.set_title(f"(d) Flow Risk Intervals [R_min, R_max] Under 1:20 Sampling (p = {p_demo})", fontsize=12, fontweight="bold")
    ax.set_xlabel("Individual Network Flow Index", fontsize=11)
    ax.set_ylabel("Propagated Risk Score R(X)", fontsize=11)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="upper right", frameon=True)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(args.figures_dir, "sdum_uncertainty_propagation.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\nSaved publication-ready figure to:")
    print(f"  • {fig_path}")
    print("=====================================================================\n")


if __name__ == "__main__":
    main()
