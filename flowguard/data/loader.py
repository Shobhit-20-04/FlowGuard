"""
FlowGuard Dataset Loader & Stratified Partitioner
=================================================

Handles loading, stratified partitioning (65% Train / 15% Calibration / 20% Test),
and high-speed Apache Parquet serialization.
"""

import json
import logging
import os
from typing import Dict, Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from flowguard.data.preprocessor import FlowDataPreprocessor

logger = logging.getLogger(__name__)

class FlowDataLoader:
    """
    Manages loading, splitting, and saving sanitized network flow datasets.
    """
    def __init__(
        self,
        processed_dir: str = "data/processed",
        train_ratio: float = 0.65,
        calibration_ratio: float = 0.15,
        test_ratio: float = 0.20,
        random_seed: int = 42,
        split_strategy: str = "stratified",
        include_quarantined_in_train: bool = False,
    ):
        assert abs((train_ratio + calibration_ratio + test_ratio) - 1.0) < 1e-5, (
            "Split ratios must sum to 1.0!"
        )
        if split_strategy not in ["stratified", "chronological"]:
            raise ValueError(f"Unknown split_strategy '{split_strategy}'. Must be 'stratified' or 'chronological'.")

        self.processed_dir = processed_dir
        self.train_ratio = train_ratio
        self.calibration_ratio = calibration_ratio
        self.test_ratio = test_ratio
        self.random_seed = random_seed
        self.split_strategy = split_strategy
        self.include_quarantined_in_train = bool(include_quarantined_in_train)
        self.preprocessor = FlowDataPreprocessor()

        # Quarantined records tracking for chronological splitting
        self.quarantined_records_: pd.DataFrame = pd.DataFrame()
        self.quarantined_count_: int = 0

    def process_and_partition(
        self,
        df: pd.DataFrame,
        label_mode: str = "binary",
        save_to_disk: bool = True,
    ) -> Tuple[
        Tuple[pd.DataFrame, pd.Series],
        Tuple[pd.DataFrame, pd.Series],
        Tuple[pd.DataFrame, pd.Series],
    ]:
        """
        Executes a leak-free train/calibration/test pipeline:
        1. Split according to selected strategy ('stratified' or 'chronological').
        2. Fit preprocessor strictly on Train, then transform Train, Calibration, and Test.
        3. Optionally saves to Parquet in processed_dir.
        """
        n_total = len(df)

        if self.split_strategy == "chronological":
            # Identify timestamp column if available and sort chronologically using stable sort
            df_sorted = df.copy()
            ts_col = None
            for col in df_sorted.columns:
                c_clean = str(col).strip().lower()
                if c_clean in ["timestamp", "time", "date", "flow_start_time"]:
                    ts_col = col
                    break

            if ts_col:
                # Convert timestamps safely with datetime parsing
                ts_series = pd.to_datetime(df_sorted[ts_col], errors="coerce")
                df_sorted["_parsed_ts"] = ts_series

                valid_mask = df_sorted["_parsed_ts"].notna()
                df_invalid = df_sorted[~valid_mask].copy() if (~valid_mask).any() else pd.DataFrame(columns=df_sorted.columns)
                df_valid = df_sorted[valid_mask].copy()

                if len(df_valid) == 0:
                    logger.warning("All timestamps are invalid or unparseable. Falling back to sequential row splitting.")
                    n_train = int(n_total * self.train_ratio)
                    n_calib = int(n_total * self.calibration_ratio)
                    df_train = df_sorted.iloc[:n_train].copy()
                    df_calib = df_sorted.iloc[n_train:n_train + n_calib].copy()
                    df_test = df_sorted.iloc[n_train + n_calib:].copy()
                else:
                    # Sort valid records stably by timestamp
                    df_valid = df_valid.sort_values(by="_parsed_ts", kind="mergesort").reset_index(drop=True)
                    n_valid = len(df_valid)
                    unique_ts = df_valid["_parsed_ts"].unique()

                    if len(unique_ts) >= 3:
                        # Deterministic timestamp group-boundary policy:
                        # Ensures T_train < T_calib < T_test strictly, preventing timestamp leakage.
                        n_train_target = max(1, int(n_valid * self.train_ratio))
                        n_calib_target = max(1, int(n_valid * self.calibration_ratio))

                        # Cut 1: Train boundary timestamp T1
                        t1_idx = min(n_train_target - 1, n_valid - 3)
                        t1 = df_valid["_parsed_ts"].iloc[t1_idx]
                        if t1 >= unique_ts[-2]:
                            # Ensure at least two unique timestamps remain for calib and test
                            t1 = unique_ts[max(0, len(unique_ts) - 3)]

                        df_train_valid = df_valid[df_valid["_parsed_ts"] <= t1].copy()
                        df_rem = df_valid[df_valid["_parsed_ts"] > t1].copy().reset_index(drop=True)

                        # Cut 2: Calib boundary timestamp T2
                        n_rem = len(df_rem)
                        unique_rem_ts = df_rem["_parsed_ts"].unique()
                        if len(unique_rem_ts) >= 2:
                            t2_idx = min(max(0, n_calib_target - 1), n_rem - 2)
                            t2 = df_rem["_parsed_ts"].iloc[t2_idx]
                            if t2 >= unique_rem_ts[-1]:
                                t2 = unique_rem_ts[-2]

                            df_calib = df_rem[df_rem["_parsed_ts"] <= t2].copy()
                            df_test = df_rem[df_rem["_parsed_ts"] > t2].copy()
                        else:
                            # Fallback if remaining unique timestamps cannot be split strictly
                            mid = max(1, n_rem // 2)
                            df_calib = df_rem.iloc[:mid].copy()
                            df_test = df_rem.iloc[mid:].copy()

                        # Quarantined records with invalid/missing timestamps handling:
                        # Excluded from evaluation splits by default; included in train only if explicitly configured.
                        self.quarantined_records_ = df_invalid
                        self.quarantined_count_ = len(df_invalid)

                        if len(df_invalid) > 0:
                            if self.include_quarantined_in_train:
                                logger.warning(
                                    f"Chronological split: Merging {self.quarantined_count_} quarantined records "
                                    f"(missing/invalid timestamps) into the training partition by explicit configuration."
                                )
                                df_train = pd.concat([df_train_valid, df_invalid], ignore_index=True)
                            else:
                                logger.info(
                                    f"Chronological split: Quarantined {self.quarantined_count_} records with "
                                    f"missing/invalid timestamps (excluded from training/evaluation splits)."
                                )
                                df_train = df_train_valid
                        else:
                            df_train = df_train_valid
                    else:
                        logger.warning(
                            f"Chronological split: Only {len(unique_ts)} unique timestamps found. "
                            f"Falling back to sequential row splitting across valid records."
                        )
                        self.quarantined_records_ = df_invalid
                        self.quarantined_count_ = len(df_invalid)
                        n_train = max(1, int(n_valid * self.train_ratio))
                        n_calib = max(1, int(n_valid * self.calibration_ratio))
                        if len(df_invalid) > 0 and self.include_quarantined_in_train:
                            df_train = pd.concat([df_valid.iloc[:n_train], df_invalid], ignore_index=True)
                        else:
                            df_train = df_valid.iloc[:n_train].copy()
                        df_calib = df_valid.iloc[n_train:n_train + n_calib].copy()
                        df_test = df_valid.iloc[n_train + n_calib:].copy()

                df_sorted = df_sorted.drop(columns=["_parsed_ts"], errors="ignore")
                df_train = df_train.drop(columns=["_parsed_ts"], errors="ignore")
                df_calib = df_calib.drop(columns=["_parsed_ts"], errors="ignore")
                df_test = df_test.drop(columns=["_parsed_ts"], errors="ignore")
            else:
                logger.warning(
                    "Chronological split requested, but no timestamp column ('timestamp', 'time', 'date') found. "
                    "Proceeding with sequential row partitioning."
                )
                n_train = int(n_total * self.train_ratio)
                n_calib = int(n_total * self.calibration_ratio)

                df_train = df_sorted.iloc[:n_train].copy()
                df_calib = df_sorted.iloc[n_train:n_train + n_calib].copy()
                df_test = df_sorted.iloc[n_train + n_calib:].copy()
        else:
            # Step 1: Initial split to isolate held-out Test set via stratified sampling
            test_fraction = self.test_ratio
            df_train_calib, df_test = train_test_split(
                df,
                test_size=test_fraction,
                random_state=self.random_seed,
                stratify=df["Label"] if "Label" in df.columns else (df["label"] if "label" in df.columns else None),
            )

            # Step 2: Split remaining data into Train and Calibration
            calib_fraction_of_remainder = self.calibration_ratio / (self.train_ratio + self.calibration_ratio)
            df_train, df_calib = train_test_split(
                df_train_calib,
                test_size=calib_fraction_of_remainder,
                random_state=self.random_seed,
                stratify=df_train_calib["Label"] if "Label" in df_train_calib.columns else (
                    df_train_calib["label"] if "label" in df_train_calib.columns else None
                ),
            )

        # Step 3: Fit preprocessor strictly on Train
        self.preprocessor.fit(df_train)

        X_train, y_train = self.preprocessor.transform(df_train, label_mode=label_mode)
        X_calib, y_calib = self.preprocessor.transform(df_calib, label_mode=label_mode)
        X_test, y_test = self.preprocessor.transform(df_test, label_mode=label_mode)

        logger.info(
            f"Partitioned data: Train={len(X_train)} ({len(X_train)/len(df)*100:.1f}%), "
            f"Calibration={len(X_calib)} ({len(X_calib)/len(df)*100:.1f}%), "
            f"Test={len(X_test)} ({len(X_test)/len(df)*100:.1f}%)"
        )

        if save_to_disk:
            self.save_splits(X_train, y_train, X_calib, y_calib, X_test, y_test)

        return (X_train, y_train), (X_calib, y_calib), (X_test, y_test)

    def save_splits(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_calib: pd.DataFrame,
        y_calib: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> None:
        """
        Saves feature matrices and labels to Apache Parquet files in processed_dir.
        """
        os.makedirs(self.processed_dir, exist_ok=True)

        train_df = X_train.copy()
        train_df["target_label"] = y_train.values
        train_path = os.path.join(self.processed_dir, "train.parquet")
        train_df.to_parquet(train_path, index=False, compression="snappy")

        calib_df = X_calib.copy()
        calib_df["target_label"] = y_calib.values
        calib_path = os.path.join(self.processed_dir, "calibration.parquet")
        calib_df.to_parquet(calib_path, index=False, compression="snappy")

        test_df = X_test.copy()
        test_df["target_label"] = y_test.values
        test_path = os.path.join(self.processed_dir, "test.parquet")
        test_df.to_parquet(test_path, index=False, compression="snappy")

        meta = {
            "feature_columns": self.preprocessor.feature_columns_,
            "zero_variance_columns": self.preprocessor.zero_variance_columns_,
            "median_values": self.preprocessor.median_values_,
            "upper_clipping_bounds": self.preprocessor.upper_clipping_bounds_,
            "train_size": len(X_train),
            "calib_size": len(X_calib),
            "test_size": len(X_test),
        }
        meta_path = os.path.join(self.processed_dir, "preprocessor_meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        logger.info(f"Saved processed Parquet datasets to {self.processed_dir}")

    def load_splits(self) -> Tuple[
        Tuple[pd.DataFrame, pd.Series],
        Tuple[pd.DataFrame, pd.Series],
        Tuple[pd.DataFrame, pd.Series],
    ]:
        """
        Fast loader to read the pre-computed Parquet splits directly from disk.
        """
        train_path = os.path.join(self.processed_dir, "train.parquet")
        calib_path = os.path.join(self.processed_dir, "calibration.parquet")
        test_path = os.path.join(self.processed_dir, "test.parquet")

        if not (os.path.exists(train_path) and os.path.exists(calib_path) and os.path.exists(test_path)):
            raise FileNotFoundError(
                f"Parquet splits not found in {self.processed_dir}. Run process_and_partition() first!"
            )

        train_df = pd.read_parquet(train_path)
        calib_df = pd.read_parquet(calib_path)
        test_df = pd.read_parquet(test_path)

        X_train = train_df.drop(columns=["target_label"])
        y_train = train_df["target_label"]

        X_calib = calib_df.drop(columns=["target_label"])
        y_calib = calib_df["target_label"]

        X_test = test_df.drop(columns=["target_label"])
        y_test = test_df["target_label"]

        return (X_train, y_train), (X_calib, y_calib), (X_test, y_test)
