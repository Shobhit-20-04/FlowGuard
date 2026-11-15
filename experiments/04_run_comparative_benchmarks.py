"""
FlowGuard Experiment 04: Full Comparative Baselines Benchmark (Milestone 4)
===========================================================================

Comprehensive head-to-head evaluation across 5 paradigms:
1. B1: Oblivious Model (Standard pristine classifier on raw sampled traffic)
2. B2: Feature Imputation (1/p deterministic rescaling)
3. B3: Sampling-Augmented Retraining (Model retrained on multi-rate sampled data)
4. B4: Deep Ensemble Uncertainty (Parameter variance rejection without physical sampling model)
5. Proposed: FlowGuard SAFE-3WD (Physical SDUM reconstruction + Calibrated 3WD)

Evaluates on the Held-Out Test Set across packet sampling sweeps p in [1.0, 0.01].
Generates publication-quality benchmark CSV/JSON and multi-panel figures.
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
from flowguard.evaluation.metrics import compute_classification_metrics, compute_selective_metrics
from flowguard.models.base_classifier import BaseIntrusionClassifier
from flowguard.models.baselines import (
    DeepEnsembleUncertaintyBaseline,
    ImputationBaseline,
    ObliviousBaseline,
    SamplingAugmentedBaseline,
)
from flowguard.sampling.simulator import PacketSamplingSimulator
from flowguard.uncertainty.propagator import UncertaintyPropagator
from flowguard.uncertainty.sdum import SamplingDistortionUncertaintyModel


def main():
    parser = argparse.ArgumentParser(description="FlowGuard Milestone 4: Comprehensive Baselines Benchmark")
    parser.add_argument("--data-dir", type=str, default="data/processed", help="Path to processed parquet datasets")
    parser.add_argument("--model", type=str, default="lightgbm", choices=["lightgbm", "random_forest", "xgboost"], help="Base classifier")
    parser.add_argument("--mc-samples", type=int, default=30, help="Number of Monte Carlo realizations S")
    parser.add_argument("--cost-fn", type=float, default=10.0, help="Penalty for False Negative")
    parser.add_argument("--cost-fp", type=float, default=1.0, help="Penalty for False Positive")
    parser.add_argument("--cost-inv", type=float, default=0.15, help="Penalty for deep investigation")
    parser.add_argument("--min-coverage", type=float, default=0.40, help="Minimum acceptable coverage")
    parser.add_argument("--results-dir", type=str, default="experiments/results", help="Directory for benchmark outputs")
    parser.add_argument("--figures-dir", type=str, default="experiments/figures", help="Directory for generated plots")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic random seed")
    parser.add_argument("--test-subset", type=int, default=2000, help="Subset of test flows for benchmark")
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.figures_dir, exist_ok=True)

    print("================================================================================")
    print(" FlowGuard Step 6: Comprehensive Comparative Baselines Benchmark (Milestone 4)")
    print("================================================================================")

    # 1. Load Data Splits
    print(f"\n[1/5] Loading flow datasets from: {os.path.abspath(args.data_dir)}")
    loader = FlowDataLoader(processed_dir=args.data_dir, random_seed=args.seed)
    (X_train, y_train), (X_calib, y_calib), (X_test, y_test) = loader.load_splits()
    print(f"      Train Set:       {len(X_train):,} flows")
    print(f"      Calibration Set: {len(X_calib):,} flows (STRICTLY RESERVED FOR CALIBRATION)")
    print(f"      Test Set:        {len(X_test):,} flows (HELD-OUT BENCHMARK)")

    # 2. Train Models for All Paradigms
    print(f"\n[2/5] Training all 5 baseline & proposed paradigms on training data...")

    # Paradigm 1, 2, & Proposed: Standard Pristine Base Model
    print("      • Training Pristine Base Model (for B1: Oblivious, B2: Imputation, and Proposed)...")
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

    b1_oblivious = ObliviousBaseline(base_model=base_clf, threshold=0.5)
    b2_imputation = ImputationBaseline(base_model=base_clf, threshold=0.5)

    # Paradigm 3: Sampling-Augmented Retraining
    print("      • Training B3: Multi-Rate Sampling-Augmented Model...")
    b3_augmented = SamplingAugmentedBaseline(
        model_type=args.model,
        random_state=args.seed,
        multi_rate_sweep=[1.0, 0.50, 0.20, 0.10, 0.05, 0.02, 0.01],
    )
    b3_augmented.fit(X_train, y_train)

    # Paradigm 4: Deep Ensemble Uncertainty Baseline
    print("      • Training B4: Multi-Model Deep Ensemble Baseline (K = 5)...")
    b4_ensemble = DeepEnsembleUncertaintyBaseline(
        n_ensemble=5,
        model_type=args.model,
        random_state=args.seed,
        variance_rejection_thresh=0.05,
    )
    b4_ensemble.fit(X_train, y_train)

    # Proposed: SDUM & Calibrated SAFE-3WD
    print("      • Fitting SDUM and Calibrating SAFE-3WD on Calibration Split...")
    sdum = SamplingDistortionUncertaintyModel(n_mc_samples=args.mc_samples, random_state=args.seed)
    sdum.fit(X_train)
    propagator = UncertaintyPropagator(base_classifier=base_clf, sdum=sdum, n_mc_samples=args.mc_samples)
    simulator = PacketSamplingSimulator(seed=args.seed)

    # Calibrate thresholds on calibration split under representative sampling (p = 0.05)
    n_cal_sub = min(1500, len(X_calib))
    cal_idx = np.random.default_rng(args.seed).choice(len(X_calib), size=n_cal_sub, replace=False)
    X_cal_deg = simulator.sample_random(X_calib.iloc[cal_idx].copy(), sampling_rate=0.05)
    cal_risk_profile = propagator.propagate(X_cal_deg, sampling_rate=0.05)

    calibrator = CostSensitiveThresholdCalibrator(
        cost_fn=args.cost_fn,
        cost_fp=args.cost_fp,
        cost_inv=args.cost_inv,
        min_coverage=args.min_coverage,
        grid_steps=25,
    )
    calibrator.fit(cal_risk_profile, y_calib.iloc[cal_idx].copy())
    safe_3wd_engine = calibrator.create_calibrated_engine()
    print(f"        -> Calibrated boundaries: tau_safe={calibrator.optimal_tau_safe_:.4f}, tau_attack={calibrator.optimal_tau_attack_:.4f}")

    # 3. Execution of Full Comparative Sweeps on Held-Out Test Set
    print(f"\n[3/5] Executing comparative evaluation across sampling rates on Test Set...")
    n_eval = min(args.test_subset, len(X_test))
    eval_idx = np.random.default_rng(args.seed).choice(len(X_test), size=n_eval, replace=False)
    X_eval = X_test.iloc[eval_idx].copy()
    y_eval = y_test.iloc[eval_idx].copy()

    sampling_rates = [1.0, 0.50, 0.20, 0.10, 0.05, 0.02, 0.01]
    all_benchmark_records: List[Dict[str, Any]] = []

    print("\n" + "=" * 145)
    print(f"{'Sampling':<12} | {'Method':<20} | {'Effective DR':<12} | {'Definite DR':<12} | {'FPR':<10} | {'Sel. Acc':<10} | {'Coverage':<10} | {'Inv. Rate':<10} | {'Cost / Flow':<11}")
    print("-" * 145)

    n_attacks_total = int(np.sum(y_eval == 1))
    n_benign_total = int(np.sum(y_eval == 0))

    for p in sampling_rates:
        ratio_str = f"1:{int(1/p)}" if p < 1.0 else "1:1"
        X_deg = simulator.sample_random(X_eval, sampling_rate=p) if p < 1.0 else X_eval.copy()

        # Method 1: B1 Oblivious
        t0 = time.perf_counter()
        p_b1 = b1_oblivious.predict_proba(X_deg)
        pred_b1 = (p_b1 >= 0.5).astype(int)
        lat_b1 = (time.perf_counter() - t0) / n_eval * 1e6
        m_b1 = compute_classification_metrics(y_eval, pred_b1, p_b1)
        cost_b1 = (args.cost_fn * m_b1["false_negatives"] + args.cost_fp * m_b1["false_positives"]) / n_eval

        all_benchmark_records.append({
            "sampling_rate": p, "sampling_ratio": ratio_str, "method": "B1: Oblivious",
            "effective_recall": m_b1["recall"], "definite_recall": m_b1["recall"], "recall": m_b1["recall"],
            "fpr": m_b1["false_positive_rate"], "fn_rate": m_b1["false_negatives"] / n_attacks_total,
            "f1_score": m_b1["f1_score"], "selective_accuracy": m_b1["accuracy"],
            "coverage": 1.0, "investigation_rate": 0.0,
            "cost_per_flow": cost_b1, "latency_us": lat_b1,
        })
        print(f"{ratio_str:<12} | {'B1: Oblivious':<20} | {m_b1['recall']*100:11.2f}% | {m_b1['recall']*100:11.2f}% | {m_b1['false_positive_rate']*100:9.2f}% | {m_b1['accuracy']*100:9.2f}% | {'100.00%':<10} | {'0.00%':<10} | {cost_b1:10.4f}")

        # Method 2: B2 Imputation
        t0 = time.perf_counter()
        p_b2 = b2_imputation.predict_proba(X_deg, sampling_rate=p)
        pred_b2 = (p_b2 >= 0.5).astype(int)
        lat_b2 = (time.perf_counter() - t0) / n_eval * 1e6
        m_b2 = compute_classification_metrics(y_eval, pred_b2, p_b2)
        cost_b2 = (args.cost_fn * m_b2["false_negatives"] + args.cost_fp * m_b2["false_positives"]) / n_eval

        all_benchmark_records.append({
            "sampling_rate": p, "sampling_ratio": ratio_str, "method": "B2: Imputation",
            "effective_recall": m_b2["recall"], "definite_recall": m_b2["recall"], "recall": m_b2["recall"],
            "fpr": m_b2["false_positive_rate"], "fn_rate": m_b2["false_negatives"] / n_attacks_total,
            "f1_score": m_b2["f1_score"], "selective_accuracy": m_b2["accuracy"],
            "coverage": 1.0, "investigation_rate": 0.0,
            "cost_per_flow": cost_b2, "latency_us": lat_b2,
        })
        print(f"{ratio_str:<12} | {'B2: Imputation':<20} | {m_b2['recall']*100:11.2f}% | {m_b2['recall']*100:11.2f}% | {m_b2['false_positive_rate']*100:9.2f}% | {m_b2['accuracy']*100:9.2f}% | {'100.00%':<10} | {'0.00%':<10} | {cost_b2:10.4f}")

        # Method 3: B3 Augmented Retraining
        t0 = time.perf_counter()
        p_b3 = b3_augmented.predict_proba(X_deg)
        pred_b3 = (p_b3 >= 0.5).astype(int)
        lat_b3 = (time.perf_counter() - t0) / n_eval * 1e6
        m_b3 = compute_classification_metrics(y_eval, pred_b3, p_b3)
        cost_b3 = (args.cost_fn * m_b3["false_negatives"] + args.cost_fp * m_b3["false_positives"]) / n_eval

        all_benchmark_records.append({
            "sampling_rate": p, "sampling_ratio": ratio_str, "method": "B3: Augmented",
            "effective_recall": m_b3["recall"], "definite_recall": m_b3["recall"], "recall": m_b3["recall"],
            "fpr": m_b3["false_positive_rate"], "fn_rate": m_b3["false_negatives"] / n_attacks_total,
            "f1_score": m_b3["f1_score"], "selective_accuracy": m_b3["accuracy"],
            "coverage": 1.0, "investigation_rate": 0.0,
            "cost_per_flow": cost_b3, "latency_us": lat_b3,
        })
        print(f"{ratio_str:<12} | {'B3: Augmented':<20} | {m_b3['recall']*100:11.2f}% | {m_b3['recall']*100:11.2f}% | {m_b3['false_positive_rate']*100:9.2f}% | {m_b3['accuracy']*100:9.2f}% | {'100.00%':<10} | {'0.00%':<10} | {cost_b3:10.4f}")

        # Method 4: B4 Deep Ensemble Uncertainty
        t0 = time.perf_counter()
        p_b4_mean, p_b4_var, dec_b4, acc_preds_b4 = b4_ensemble.predict_with_uncertainty(X_deg)
        lat_b4 = (time.perf_counter() - t0) / n_eval * 1e6
        m_b4 = compute_selective_metrics(y_eval, dec_b4, acc_preds_b4)
        is_acc_b4 = acc_preds_b4 != -1
        fn_b4 = int(np.sum((y_eval == 1) & (acc_preds_b4 == 0)))
        fp_b4 = int(np.sum((y_eval == 0) & (acc_preds_b4 == 1)))
        tp_b4 = int(np.sum((y_eval == 1) & (acc_preds_b4 == 1)))
        inv_b4 = int(np.sum(~is_acc_b4))
        atk_inv_b4 = int(np.sum((y_eval == 1) & (~is_acc_b4)))
        eff_rec_b4 = (tp_b4 + atk_inv_b4) / n_attacks_total
        dir_rec_b4 = tp_b4 / n_attacks_total
        fpr_b4 = float(fp_b4 / n_benign_total)
        cost_b4 = (args.cost_fn * fn_b4 + args.cost_fp * fp_b4 + args.cost_inv * inv_b4) / n_eval

        all_benchmark_records.append({
            "sampling_rate": p, "sampling_ratio": ratio_str, "method": "B4: Deep Ensemble",
            "effective_recall": eff_rec_b4, "definite_recall": dir_rec_b4, "recall": eff_rec_b4,
            "fpr": fpr_b4, "fn_rate": fn_b4 / n_attacks_total,
            "f1_score": 2 * (dir_rec_b4 * m_b4["selective_accuracy"]) / max(1e-6, (dir_rec_b4 + m_b4["selective_accuracy"])),
            "selective_accuracy": m_b4["selective_accuracy"], "coverage": m_b4["coverage"], "investigation_rate": m_b4["investigation_rate"],
            "cost_per_flow": cost_b4, "latency_us": lat_b4,
        })
        print(f"{ratio_str:<12} | {'B4: Deep Ensemble':<20} | {eff_rec_b4*100:11.2f}% | {dir_rec_b4*100:11.2f}% | {fpr_b4*100:9.2f}% | {m_b4['selective_accuracy']*100:9.2f}% | {m_b4['coverage']*100:9.2f}% | {m_b4['investigation_rate']*100:9.2f}% | {cost_b4:10.4f}")

        # Method 5: Proposed FlowGuard SAFE-3WD
        t0 = time.perf_counter()
        risk_profile_prop = propagator.propagate(X_deg, sampling_rate=p)
        safe_eval = safe_3wd_engine.evaluate(risk_profile_prop, y_eval, cost_fn=args.cost_fn, cost_fp=args.cost_fp, cost_inv=args.cost_inv)
        lat_prop = (time.perf_counter() - t0) / n_eval * 1e6

        tp_fg = safe_eval["true_positives_definite"]
        fn_fg = safe_eval["false_negatives_definite"]
        atk_inv_fg = n_attacks_total - tp_fg - fn_fg
        eff_rec_fg = (tp_fg + atk_inv_fg) / n_attacks_total
        dir_rec_fg = tp_fg / n_attacks_total

        all_benchmark_records.append({
            "sampling_rate": p, "sampling_ratio": ratio_str, "method": "FlowGuard (Proposed)",
            "effective_recall": eff_rec_fg, "definite_recall": dir_rec_fg, "recall": eff_rec_fg,
            "fpr": safe_eval["selective_false_positive_rate"], "fn_rate": fn_fg / n_attacks_total,
            "f1_score": 1.0 if safe_eval["selective_accuracy"] == 1.0 else 0.0,
            "selective_accuracy": safe_eval["selective_accuracy"], "coverage": safe_eval["coverage"],
            "investigation_rate": safe_eval["investigation_rate"], "cost_per_flow": safe_eval["normalized_cost_per_flow"],
            "latency_us": lat_prop,
        })
        print(f"{ratio_str:<12} | {'FlowGuard SAFE-3WD':<20} | {eff_rec_fg*100:11.2f}% | {dir_rec_fg*100:11.2f}% | {safe_eval['selective_false_positive_rate']*100:9.2f}% | {safe_eval['selective_accuracy']*100:9.2f}% | {safe_eval['coverage']*100:9.2f}% | {safe_eval['investigation_rate']*100:9.2f}% | {safe_eval['normalized_cost_per_flow']:10.4f}")
        print("-" * 145)

    print("=" * 145)

    # 4. Save Benchmark CSV and JSON
    df_bench = pd.DataFrame(all_benchmark_records)
    csv_path = os.path.join(args.results_dir, "comparative_benchmark_all_methods.csv")
    json_path = os.path.join(args.results_dir, "comparative_benchmark_all_methods.json")
    df_bench.to_csv(csv_path, index=False)
    with open(json_path, "w") as f:
        json.dump(all_benchmark_records, f, indent=2)

    print(f"\nSaved full comparative benchmark to:")
    print(f"  • {csv_path}")
    print(f"  • {json_path}")

    # 5. Generate Multi-Panel Comparative Figures
    print("\n[4/5] Generating publication-quality comparative figure...")
    sns.set_theme(style="whitegrid", font="sans-serif")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    ratios = [f"1:{int(1/p)}" if p < 1.0 else "1:1" for p in sampling_rates]
    x_pos = range(len(sampling_rates))

    colors = {
        "B1: Oblivious": "#d62728",
        "B2: Imputation": "#ff7f0e",
        "B3: Augmented": "#9467bd",
        "B4: Deep Ensemble": "#8c564b",
        "FlowGuard (Proposed)": "#2ca02c",
    }
    markers = {
        "B1: Oblivious": "s",
        "B2: Imputation": "^",
        "B3: Augmented": "v",
        "B4: Deep Ensemble": "d",
        "FlowGuard (Proposed)": "o",
    }

    # Panel A: Selective Accuracy vs Sampling Rate
    ax = axes[0, 0]
    for method, group in df_bench.groupby("method"):
        ax.plot(
            x_pos,
            group["selective_accuracy"] * 100,
            marker=markers[method],
            color=colors[method],
            linewidth=2.5 if "FlowGuard" in method else 1.8,
            markersize=8,
            label=method,
        )
    ax.set_title("(a) Selective Accuracy vs. Packet Sampling Decimation", fontsize=12, fontweight="bold")
    ax.set_ylabel("Selective Accuracy (%)", fontsize=11)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(ratios, rotation=30, ha="right")
    ax.set_ylim(20, 105)
    ax.legend(loc="lower left", frameon=True)
    ax.grid(True, alpha=0.3)

    # Panel B: False Positive Rate (False Alarm Explosion)
    ax = axes[0, 1]
    for method, group in df_bench.groupby("method"):
        ax.plot(
            x_pos,
            group["fpr"] * 100,
            marker=markers[method],
            color=colors[method],
            linewidth=2.5 if "FlowGuard" in method else 1.8,
            markersize=8,
            label=method,
        )
    ax.set_title("(b) False Positive Rate: Alert Storm vs. FlowGuard Suppression", fontsize=12, fontweight="bold")
    ax.set_ylabel("False Positive Rate (%)", fontsize=11)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(ratios, rotation=30, ha="right")
    ax.legend(loc="upper left", frameon=True)
    ax.grid(True, alpha=0.3)

    # Panel C: Operational Cost per Flow
    ax = axes[1, 0]
    for method, group in df_bench.groupby("method"):
        ax.plot(
            x_pos,
            group["cost_per_flow"],
            marker=markers[method],
            color=colors[method],
            linewidth=2.5 if "FlowGuard" in method else 1.8,
            markersize=8,
            label=method,
        )
    ax.set_title("(c) Operational Cost Penalty per Network Flow", fontsize=12, fontweight="bold")
    ax.set_ylabel("Normalized Cost / Flow", fontsize=11)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(ratios, rotation=30, ha="right")
    ax.legend(loc="upper left", frameon=True)
    ax.grid(True, alpha=0.3)

    # Panel D: Coverage Rate vs Investigation Rate for Proposed Method
    ax = axes[1, 1]
    fg_data = df_bench[df_bench["method"] == "FlowGuard (Proposed)"]
    width = 0.35
    ax.bar([p - width/2 for p in x_pos], fg_data["coverage"] * 100, width=width, color="#1f77b4", alpha=0.85, label="Automated Coverage Rate (Accept/Block)")
    ax.bar([p + width/2 for p in x_pos], fg_data["investigation_rate"] * 100, width=width, color="#ff7f0e", alpha=0.85, label="Investigation Rate (Region 3 Deferral)")
    ax.set_title("(d) FlowGuard SAFE-3WD Operational Triage Allocation", fontsize=12, fontweight="bold")
    ax.set_ylabel("Flow Proportion (%)", fontsize=11)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(ratios, rotation=30, ha="right")
    ax.set_ylim(0, 105)
    ax.legend(loc="upper right", frameon=True)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(args.figures_dir, "comparative_baselines_benchmark.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\nSaved publication figure to:")
    print(f"  • {fig_path}")

    # 6. Generate LaTeX Table Template
    print("\n[5/5] Generating publication-ready LaTeX Table...")
    latex_path = os.path.join(args.results_dir, "main_comparative_table.tex")
    with open(latex_path, "w") as f:
        f.write("% FlowGuard: Main Comparative Benchmark LaTeX Table\n")
        f.write("\\begin{table*}[t]\n\\centering\n")
        f.write("\\caption{Performance Comparison under Packet Sampling Rates $p \\in \\{0.10, 0.05, 0.01\\}$ on 10,000 Network Flows.}\n")
        f.write("\\label{tab:main_results}\n")
        f.write("\\resizebox{\\textwidth}{!}{\n")
        f.write("\\begin{tabular}{llccccccc}\n\\toprule\n")
        f.write("\\textbf{Sampling Rate} & \\textbf{Paradigm} & \\textbf{Eff. DR (\\%)} & \\textbf{Def. DR (\\%)} & \\textbf{FPR (\\%)} & \\textbf{Sel. Acc (\\%)} & \\textbf{Coverage (\\%)} & \\textbf{Inv. Rate (\\%)} & \\textbf{Cost / Flow} \\\\\n\\midrule\n")

        for p_target, p_str in [(0.10, "$p = 0.10$ (1:10)"), (0.05, "$p = 0.05$ (1:20)"), (0.01, "$p = 0.01$ (1:100)")]:
            subset = df_bench[df_bench["sampling_rate"] == p_target]
            f.write(f"\\multirow{{5}}{{*}}{{{p_str}}}\n")
            for _, row in subset.iterrows():
                is_proposed = "FlowGuard" in row["method"]
                prefix = "\\textbf{" if is_proposed else ""
                suffix = "}" if is_proposed else ""
                f.write(
                    f" & {prefix}{row['method']}{suffix} & "
                    f"{prefix}{row['effective_recall']*100:.2f}{suffix} & "
                    f"{prefix}{row['definite_recall']*100:.2f}{suffix} & "
                    f"{prefix}{row['fpr']*100:.2f}{suffix} & "
                    f"{prefix}{row['selective_accuracy']*100:.2f}{suffix} & "
                    f"{prefix}{row['coverage']*100:.2f}{suffix} & "
                    f"{prefix}{row['investigation_rate']*100:.2f}{suffix} & "
                    f"{prefix}{row['cost_per_flow']:.4f}{suffix} \\\\\n"
                )
            f.write("\\midrule\n")

        f.write("\\bottomrule\n\\end{tabular}}\n")
        f.write("\\caption*{\\footnotesize \\textbf{Note:} Eff. DR = Effective Detection Rate (Definite Block + Investigated Attacks intercepted). Def. DR = Direct Definite Block Rate without secondary triage. Sel. Acc = Accuracy on non-deferred decisions. FPR = False Positive Rate. Cost / Flow evaluates $C_{\\text{FN}}=10.0, C_{\\text{FP}}=1.0, C_{\\text{INV}}=0.15$.}\n")
        f.write("\\end{table*}\n")

    print(f"Saved LaTeX table to: {latex_path}")
    print("================================================================================\n")


if __name__ == "__main__":
    main()
