"""
FlowGuard Experiment 00: Data Pipeline & Partitioning
======================================================

Preprocesses raw network flow records (or synthesizes benchmark flows),
applies strict hygiene, performs a stratified 65/15/20 split, and saves
high-speed Apache Parquet datasets to data/processed/.

Usage:
    python experiments/00_prepare_data.py [--input path/to/raw.csv] [--samples 50000]
"""

import argparse
import os
import sys
import time

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from flowguard.data import FlowDataLoader, FlowDataPreprocessor, SyntheticFlowGenerator

def main():
    parser = argparse.ArgumentParser(description="FlowGuard Data Preprocessing & Partitioning")
    parser.add_argument("--input", type=str, default=None, help="Path to raw CSV dataset")
    parser.add_argument("--samples", type=int, default=50000, help="Number of samples if generating benchmark data")
    parser.add_argument("--output-dir", type=str, default="data/processed", help="Output directory for processed parquet")
    parser.add_argument("--label-mode", type=str, default="binary", choices=["binary", "multiclass"], help="Label mode")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    print("=================================================================")
    print(" FlowGuard Step 2: High-Performance Data Pipeline & Partitioning")
    print("=================================================================")

    start_time = time.time()

    if args.input and os.path.exists(args.input):
        print(f"\n[1/3] Loading raw dataset from: {args.input}")
        if args.input.endswith(".parquet"):
            raw_df = pd.read_parquet(args.input)
        else:
            raw_df = pd.read_csv(args.input)
        print(f"      Loaded {len(raw_df):,} rows, {len(raw_df.columns)} columns.")
    else:
        print(f"\n[1/3] Generating realistic benchmark flow dataset ({args.samples:,} flows)...")
        gen = SyntheticFlowGenerator(seed=args.seed)
        raw_df = gen.generate_flow_batch(n_samples=args.samples, attack_ratio=0.25)
        print(f"      Generated {len(raw_df):,} synthetic flow records across 4 attack categories.")

    # Show raw class distribution
    label_col = "Label" if "Label" in raw_df.columns else "label"
    print("\n      Raw Class Distribution:")
    for lbl, count in raw_df[label_col].value_counts().items():
        print(f"        - {lbl}: {count:,} ({count/len(raw_df)*100:.2f}%)")

    # Step 2: Stratified Partitioning & Preprocessing
    print(f"\n[2/3] Executing leak-free preprocessing and stratified 65/15/20 split...")
    loader = FlowDataLoader(
        processed_dir=args.output_dir,
        train_ratio=0.65,
        calibration_ratio=0.15,
        test_ratio=0.20,
        random_seed=args.seed,
    )

    (X_train, y_train), (X_calib, y_calib), (X_test, y_test) = loader.process_and_partition(
        raw_df,
        label_mode=args.label_mode,
        save_to_disk=True,
    )

    elapsed = time.time() - start_time

    # Step 3: Summary & Verification
    print(f"\n[3/3] Data Pipeline Complete in {elapsed:.2f} seconds!")
    print("-----------------------------------------------------------------")
    print(f"  • Train Set:       {len(X_train):>6,} rows | Features: {X_train.shape[1]} | Attacks: {y_train.sum():>5,} ({y_train.mean()*100:.1f}%)")
    print(f"  • Calibration Set: {len(X_calib):>6,} rows | Features: {X_calib.shape[1]} | Attacks: {y_calib.sum():>5,} ({y_calib.mean()*100:.1f}%)")
    print(f"  • Test Set:        {len(X_test):>6,} rows | Features: {X_test.shape[1]} | Attacks: {y_test.sum():>5,} ({y_test.mean()*100:.1f}%)")
    print("-----------------------------------------------------------------")
    print(f"  Saved Parquet splits to: {os.path.abspath(args.output_dir)}")
    print("  - train.parquet")
    print("  - calibration.parquet")
    print("  - test.parquet")
    print("  - preprocessor_meta.json")
    print("=================================================================\n")

if __name__ == "__main__":
    main()
