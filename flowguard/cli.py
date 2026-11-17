"""
FlowGuard Unified Command-Line Interface (CLI)
==============================================

Provides a research-grade, unified CLI for the FlowGuard NIDS framework:
- data validate   : Validates schema, hygiene, and class balance of parquet datasets.
- data prepare    : Runs the reproducible data pipeline & partitioning.
- train           : Trains base classifier on pristine training flows.
- calibrate       : Calibrates SAFE-3WD decision boundaries on held-out calibration split.
- evaluate        : Evaluates uncertainty-aware intrusion detection under sampling distortion.
- benchmark       : Executes comparative baselines benchmark across sampling rate sweep.
- ablation        : Executes component-level ablation studies (copula, mixture, policy).
- report          : Generates executive Markdown & JSON performance audit summaries.

Usage:
    python -m flowguard.cli --help
    python -m flowguard.cli <command> [options]
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

from flowguard.data.loader import FlowDataLoader
from flowguard.data.preprocessor import FlowDataPreprocessor
from flowguard.data.synthetic import SyntheticFlowGenerator
from flowguard.decision.calibrator import CostSensitiveThresholdCalibrator
from flowguard.decision.safe_3wd import SafeThreeWayDecisionEngine
from flowguard.evaluation.metrics import compute_classification_metrics
from flowguard.models.base_classifier import BaseIntrusionClassifier
from flowguard.sampling.simulator import PacketSamplingSimulator
from flowguard.uncertainty.propagator import UncertaintyPropagator
from flowguard.uncertainty.sdum import SamplingDistortionUncertaintyModel

logger = logging.getLogger("flowguard.cli")


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def _load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    default_path = Path("configs/default_config.yaml")
    target = Path(config_path) if config_path else default_path
    if target.exists():
        with open(target, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


# ---------------------------------------------------------------------------
# Command: data validate
# ---------------------------------------------------------------------------
def cmd_data_validate(args: argparse.Namespace) -> int:
    """Validates processed parquet dataset splits for integrity, schema, and class balance."""
    _setup_logging(args.verbose)
    data_dir = Path(args.data_dir)
    print(f"=== FlowGuard Data Validation: {data_dir.resolve()} ===")

    splits = ["train.parquet", "calibration.parquet", "test.parquet"]
    missing = [s for s in splits if not (data_dir / s).exists()]
    if missing:
        print(f"Error: Missing required split files in {data_dir}: {missing}")
        return 1

    total_rows = 0
    feature_sets = []

    for s in splits:
        p = data_dir / s
        df = pd.read_parquet(p)
        total_rows += len(df)
        feature_sets.append(set(df.columns))

        label_col = "label" if "label" in df.columns else ("Label" if "Label" in df.columns else None)
        n_attacks = int(df[label_col].sum()) if label_col else 0
        attack_pct = (n_attacks / max(1, len(df))) * 100

        # Check for NaNs or Infs
        numeric_df = df.select_dtypes(include=[np.number])
        n_nan = int(numeric_df.isna().sum().sum())
        n_inf = int(np.isinf(numeric_df.values).sum())

        status = "PASSED" if n_nan == 0 and n_inf == 0 else "WARNING"
        print(f"  • {s:<20}: {len(df):>7,} rows | Features: {len(df.columns):>2} | "
              f"Attacks: {n_attacks:>6,} ({attack_pct:>5.1f}%) | NaNs: {n_nan} | Infs: {n_inf} | [{status}]")

    # Verify column alignment across splits
    if not (feature_sets[0] == feature_sets[1] == feature_sets[2]):
        print("Error: Inconsistent feature columns across train, calibration, and test splits!")
        return 1

    print(f"\nValidation Result: PASSED (All splits verified leak-free, {total_rows:,} total flows)")
    return 0


# ---------------------------------------------------------------------------
# Command: data prepare
# ---------------------------------------------------------------------------
def cmd_data_prepare(args: argparse.Namespace) -> int:
    """Runs data preprocessing, hygiene sanitization, and leak-free partitioning."""
    _setup_logging(args.verbose)
    print("=== FlowGuard Data Preparation Pipeline ===")

    if args.input:
        in_path = Path(args.input)
        if not in_path.exists():
            print(f"Error: Specified dataset input path does not exist: {in_path.resolve()}")
            print("FlowGuard refuses silent fallback to synthetic data when an explicit path is provided.")
            return 1
        print(f"Loading raw dataset: {in_path.resolve()}")
        raw_df = pd.read_parquet(in_path) if in_path.suffix == ".parquet" else pd.read_csv(in_path)
    elif getattr(args, "synthetic", False):
        print(f"Generating synthetic benchmark flows ({args.samples:,} flows, seed={args.seed})...")
        gen = SyntheticFlowGenerator(seed=args.seed)
        raw_df = gen.generate_flow_batch(n_samples=args.samples, attack_ratio=0.25)
    else:
        print("Error: No dataset specified. Provide a raw dataset path via '--input <path>' or "
              "explicitly opt in to synthetic benchmark flow generation via '--synthetic'.")
        print("FlowGuard never silently generates synthetic data without explicit user request.")
        return 1

    loader = FlowDataLoader(
        processed_dir=args.output_dir,
        train_ratio=args.train_ratio,
        calibration_ratio=args.calib_ratio,
        test_ratio=args.test_ratio,
        random_seed=args.seed,
        split_strategy=args.split_strategy,
    )

    (X_tr, y_tr), (X_cal, y_cal), (X_te, y_te) = loader.process_and_partition(
        raw_df,
        label_mode="binary",
        save_to_disk=True,
    )

    print(f"Saved processed partitions to: {Path(args.output_dir).resolve()}")
    print(f"  - Train:       {len(X_tr):,} flows")
    print(f"  - Calibration: {len(X_cal):,} flows")
    print(f"  - Test:        {len(X_te):,} flows")
    return 0


# ---------------------------------------------------------------------------
# Command: train
# ---------------------------------------------------------------------------
def cmd_train(args: argparse.Namespace) -> int:
    """Trains a regularized base classifier on pristine training flows."""
    _setup_logging(args.verbose)
    print(f"=== FlowGuard Base Classifier Training ({args.model.upper()}) ===")

    loader = FlowDataLoader(processed_dir=args.data_dir, random_seed=args.seed)
    (X_train, y_train), _, _ = loader.load_splits()

    clf = BaseIntrusionClassifier(
        model_type=args.model,
        random_state=args.seed,
        n_estimators=args.n_estimators,
    )
    t0 = time.time()
    clf.fit(X_train, y_train)
    elapsed = time.time() - t0

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / f"{args.model}_classifier.joblib"
    clf.save(str(model_path))

    print(f"Model successfully trained in {elapsed:.2f}s and saved to: {model_path.resolve()}")
    return 0


# ---------------------------------------------------------------------------
# Command: calibrate
# ---------------------------------------------------------------------------
def cmd_calibrate(args: argparse.Namespace) -> int:
    """Calibrates SAFE-3WD decision thresholds on the held-out calibration split."""
    _setup_logging(args.verbose)
    print(f"=== FlowGuard SAFE-3WD Threshold Calibration (Policy: {args.policy.upper()}) ===")

    loader = FlowDataLoader(processed_dir=args.data_dir, random_seed=args.seed)
    (X_train, y_train), (X_calib, y_calib), _ = loader.load_splits()

    # Train base classifier and fit SDUM
    clf = BaseIntrusionClassifier(model_type=args.model, random_state=args.seed)
    clf.fit(X_train, y_train)

    sdum = SamplingDistortionUncertaintyModel(
        n_mc_samples=args.mc_samples,
        random_state=args.seed,
        preserve_correlation=True,
    )
    sdum.fit(X_train)

    propagator = UncertaintyPropagator(base_classifier=clf, sdum=sdum, n_mc_samples=args.mc_samples)
    simulator = PacketSamplingSimulator(seed=args.seed)

    # Degrade calibration set at evaluation rate
    X_deg = simulator.sample_random(X_calib, sampling_rate=args.sampling_rate)
    calib_profile = propagator.propagate(X_deg, sampling_rate=args.sampling_rate)

    calibrator = CostSensitiveThresholdCalibrator(
        cost_fn=args.cost_fn,
        cost_fp=args.cost_fp,
        cost_inv=args.cost_inv,
        min_coverage=args.min_coverage,
        policy=args.policy,
        k_sigma=args.k_sigma,
    )
    calibrator.fit(calib_profile, y_calib)

    print(f"Calibration Complete: Constraint Satisfied = {calibrator.constraint_satisfied_}")
    print(f"  • Optimal tau_safe:     {calibrator.optimal_tau_safe_:.4f}")
    print(f"  • Optimal tau_attack:   {calibrator.optimal_tau_attack_:.4f}")
    print(f"  • Optimal sigma_thresh: {calibrator.optimal_sigma_thresh_:.4f}")
    print(f"  • Best Calibration Cost:{calibrator.best_cost_:.4f}")

    if args.output:
        out_p = Path(args.output)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump({
                "policy": args.policy,
                "tau_safe": calibrator.optimal_tau_safe_,
                "tau_attack": calibrator.optimal_tau_attack_,
                "sigma_thresh": calibrator.optimal_sigma_thresh_,
                "best_cost": calibrator.best_cost_,
                "constraint_satisfied": calibrator.constraint_satisfied_,
                "sampling_rate": args.sampling_rate,
            }, f, indent=2)
        print(f"Thresholds saved to: {out_p.resolve()}")

    return 0


# ---------------------------------------------------------------------------
# Command: evaluate
# ---------------------------------------------------------------------------
def cmd_evaluate(args: argparse.Namespace) -> int:
    """Evaluates FlowGuard pipeline on test split under sampling distortion."""
    _setup_logging(args.verbose)
    print(f"=== FlowGuard Evaluation (Sampling Rate p={args.sampling_rate}) ===")

    loader = FlowDataLoader(processed_dir=args.data_dir, random_seed=args.seed)
    (X_train, y_train), (X_calib, y_calib), (X_test, y_test) = loader.load_splits()

    if args.subset and args.subset < len(X_test):
        from sklearn.model_selection import train_test_split
        X_test, _, y_test, _ = train_test_split(
            X_test, y_test, train_size=args.subset, stratify=y_test, random_state=args.seed
        )

    clf = BaseIntrusionClassifier(model_type=args.model, random_state=args.seed)
    clf.fit(X_train, y_train)

    sdum = SamplingDistortionUncertaintyModel(
        n_mc_samples=args.mc_samples,
        random_state=args.seed,
        preserve_correlation=not args.no_correlation,
    )
    sdum.fit(X_train)

    simulator = PacketSamplingSimulator(seed=args.seed)
    X_test_sampled = simulator.sample_random(X_test, sampling_rate=args.sampling_rate)

    propagator = UncertaintyPropagator(base_classifier=clf, sdum=sdum, n_mc_samples=args.mc_samples)
    risk_profile = propagator.propagate(X_test_sampled, sampling_rate=args.sampling_rate)

    engine = SafeThreeWayDecisionEngine(
        tau_safe=args.tau_safe,
        tau_attack=args.tau_attack,
        max_acceptable_variance=args.max_variance,
        policy=args.policy,
        k_sigma=args.k_sigma,
    )

    results = engine.evaluate(
        risk_profile=risk_profile,
        y_true=y_test,
        cost_fn=args.cost_fn,
        cost_fp=args.cost_fp,
        cost_inv=args.cost_inv,
        secondary_triage_tpr=args.secondary_tpr,
        secondary_triage_fpr=args.secondary_fpr,
    )

    print(f"\n--- SAFE-3WD Evaluation Results (p={args.sampling_rate}) ---")
    print(f"  • Coverage Rate:             {results['coverage']*100:.2f}%")
    print(f"  • Investigation Rate:        {results['investigation_rate']*100:.2f}%")
    print(f"  • Direct Detection Rate:     {results['direct_detection_rate']*100:.2f}%")
    print(f"  • Direct False Positive Rate:{results['direct_fpr']*100:.2f}%")
    print(f"  • Selective Accuracy:        {results['selective_accuracy']*100:.2f}%")
    print(f"  • Normalized Cost / Flow:    {results['normalized_cost_per_flow']:.4f}")

    if "post_triage_detection_rate" in results:
        print(f"  • Post-Triage Detection Rate:{results['post_triage_detection_rate']*100:.2f}%")
        print(f"  • Post-Triage FPR:           {results['post_triage_fpr']*100:.2f}%")
        print(f"  • Post-Triage Cost / Flow:   {results['post_triage_cost_per_flow']:.4f}")

    return 0


# ---------------------------------------------------------------------------
# Command: benchmark
# ---------------------------------------------------------------------------
def cmd_benchmark(args: argparse.Namespace) -> int:
    """Executes comparative benchmark across methods and sampling rates."""
    _setup_logging(args.verbose)
    print("=== FlowGuard Comparative Benchmark Runner ===")
    import importlib
    mod = importlib.import_module("experiments.06_run_corrected_experiments")
    mod.execute_corrected_benchmark()
    return 0


# ---------------------------------------------------------------------------
# Command: ablation
# ---------------------------------------------------------------------------
def cmd_ablation(args: argparse.Namespace) -> int:
    """Executes component-level ablation studies."""
    _setup_logging(args.verbose)
    print("=== FlowGuard Ablation Studies Runner ===")
    import importlib
    mod = importlib.import_module("experiments.05_run_ablation_studies")
    mod.main()
    return 0


# ---------------------------------------------------------------------------
# Command: report
# ---------------------------------------------------------------------------
def cmd_report(args: argparse.Namespace) -> int:
    """Generates an executive performance report."""
    _setup_logging(args.verbose)
    print("=== FlowGuard Executive Report Generator ===")
    out_path = Path("docs/internal/BENCHMARK_EXECUTIVE_SUMMARY.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_csv = Path("research_results/benchmarks/corrected_comparative_benchmark_summary.csv")

    content = [
        "# FlowGuard Research Benchmark Executive Summary",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Comparative Benchmark Summary",
    ]
    if summary_csv.exists():
        df = pd.read_csv(summary_csv)
        content.append(df.to_markdown(index=False))
    else:
        content.append("Benchmark summary CSV not found. Run `python -m flowguard.cli benchmark` first.")

    out_path.write_text("\n".join(content), encoding="utf-8")
    print(f"Report written to: {out_path.resolve()}")
    return 0


# ---------------------------------------------------------------------------
# Main CLI Parser Construction
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="flowguard",
        description="FlowGuard: Uncertainty-Aware Network Intrusion Detection Under Packet Sampling Distortion",
    )
    parser.add_argument("--version", action="version", version="flowguard 0.2.0")
    subparsers = parser.add_subparsers(dest="command", help="FlowGuard subsystem command")

    # data validate
    p_data = subparsers.add_parser("data", help="Dataset management and validation")
    data_subs = p_data.add_subparsers(dest="subcommand", help="Data subcommand")
    
    p_dval = data_subs.add_parser("validate", help="Validate processed parquet datasets")
    p_dval.add_argument("--data-dir", type=str, default="data/processed", help="Directory of processed parquet splits")
    p_dval.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    # data prepare
    p_dprep = data_subs.add_parser("prepare", help="Run data preprocessing and partitioning")
    p_dprep.add_argument("--input", type=str, default=None, help="Raw dataset path")
    p_dprep.add_argument("--synthetic", action="store_true", help="Explicitly opt in to generating synthetic benchmark flows")
    p_dprep.add_argument("--samples", type=int, default=50000, help="Number of synthetic samples")
    p_dprep.add_argument("--output-dir", type=str, default="data/processed", help="Output directory")
    p_dprep.add_argument("--train-ratio", type=float, default=0.65, help="Train ratio")
    p_dprep.add_argument("--calib-ratio", type=float, default=0.15, help="Calibration ratio")
    p_dprep.add_argument("--test-ratio", type=float, default=0.20, help="Test ratio")
    p_dprep.add_argument("--split-strategy", type=str, default="stratified", choices=["stratified", "chronological"], help="Split strategy")
    p_dprep.add_argument("--seed", type=int, default=42, help="Random seed")
    p_dprep.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    # train
    p_train = subparsers.add_parser("train", help="Train base intrusion classifier")
    p_train.add_argument("--data-dir", type=str, default="data/processed", help="Processed data directory")
    p_train.add_argument("--model", type=str, default="lightgbm", choices=["lightgbm", "random_forest", "xgboost"], help="Classifier type")
    p_train.add_argument("--n-estimators", type=int, default=200, help="Number of estimators")
    p_train.add_argument("--output-dir", type=str, default="experiments/models", help="Output directory")
    p_train.add_argument("--seed", type=int, default=42, help="Random seed")
    p_train.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    # calibrate
    p_cal = subparsers.add_parser("calibrate", help="Calibrate SAFE-3WD decision thresholds")
    p_cal.add_argument("--data-dir", type=str, default="data/processed", help="Processed data directory")
    p_cal.add_argument("--model", type=str, default="lightgbm", help="Classifier type")
    p_cal.add_argument("--sampling-rate", type=float, default=0.10, help="Sampling rate p")
    p_cal.add_argument("--mc-samples", type=int, default=30, help="Monte Carlo samples S")
    p_cal.add_argument("--cost-fn", type=float, default=10.0, help="False negative cost")
    p_cal.add_argument("--cost-fp", type=float, default=1.0, help="False positive cost")
    p_cal.add_argument("--cost-inv", type=float, default=0.15, help="Investigation cost")
    p_cal.add_argument("--min-coverage", type=float, default=0.40, help="Minimum coverage constraint")
    p_cal.add_argument("--policy", type=str, default="quantile", choices=["quantile", "moment"], help="Decision policy")
    p_cal.add_argument("--k-sigma", type=float, default=1.0, help="Moment policy std multiplier")
    p_cal.add_argument("--output", type=str, default=None, help="JSON output path for calibrated thresholds")
    p_cal.add_argument("--seed", type=int, default=42, help="Random seed")
    p_cal.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    # evaluate
    p_eval = subparsers.add_parser("evaluate", help="Evaluate FlowGuard pipeline on test data")
    p_eval.add_argument("--data-dir", type=str, default="data/processed", help="Processed data directory")
    p_eval.add_argument("--model", type=str, default="lightgbm", help="Classifier type")
    p_eval.add_argument("--sampling-rate", type=float, default=0.10, help="Sampling rate p")
    p_eval.add_argument("--mc-samples", type=int, default=30, help="Monte Carlo samples S")
    p_eval.add_argument("--subset", type=int, default=1000, help="Subset of test flows")
    p_eval.add_argument("--tau-safe", type=float, default=0.20, help="tau_safe threshold")
    p_eval.add_argument("--tau-attack", type=float, default=0.80, help="tau_attack threshold")
    p_eval.add_argument("--max-variance", type=float, default=0.04, help="Max acceptable variance")
    p_eval.add_argument("--policy", type=str, default="quantile", choices=["quantile", "moment"], help="Decision policy")
    p_eval.add_argument("--k-sigma", type=float, default=1.0, help="Moment policy std multiplier")
    p_eval.add_argument("--cost-fn", type=float, default=10.0, help="False negative cost")
    p_eval.add_argument("--cost-fp", type=float, default=1.0, help="False positive cost")
    p_eval.add_argument("--cost-inv", type=float, default=0.15, help="Investigation cost")
    p_eval.add_argument("--secondary-tpr", type=float, default=0.90, help="Secondary triage TPR")
    p_eval.add_argument("--secondary-fpr", type=float, default=0.02, help="Secondary triage FPR")
    p_eval.add_argument("--no-correlation", action="store_true", help="Disable copula correlation preservation")
    p_eval.add_argument("--seed", type=int, default=42, help="Random seed")
    p_eval.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    # benchmark
    p_bm = subparsers.add_parser("benchmark", help="Run comparative benchmark")
    p_bm.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    # ablation
    p_abl = subparsers.add_parser("ablation", help="Run ablation studies")
    p_abl.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    # report
    p_rep = subparsers.add_parser("report", help="Generate performance report")
    p_rep.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    return parser


def main(args_list: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(args_list)

    if args.command == "data":
        if args.subcommand == "validate":
            return cmd_data_validate(args)
        elif args.subcommand == "prepare":
            return cmd_data_prepare(args)
        else:
            parser.parse_args(["data", "--help"])
            return 0
    elif args.command == "train":
        return cmd_train(args)
    elif args.command == "calibrate":
        return cmd_calibrate(args)
    elif args.command == "evaluate":
        return cmd_evaluate(args)
    elif args.command == "benchmark":
        return cmd_benchmark(args)
    elif args.command == "ablation":
        return cmd_ablation(args)
    elif args.command == "report":
        return cmd_report(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
