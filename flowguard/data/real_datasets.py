"""
FlowGuard Real-World Dataset Ingestion & Schema Mapping Library
==============================================================

Provides schema mappers, automated column normalization, and validation loaders
for benchmark intrusion detection datasets:
1. CIC-IDS2017 (Canadian Institute for Cybersecurity, UNB)
2. CSE-CIC-IDS2018 (Communications Security Establishment & CIC)

Real Dataset Ingestion Architecture:
------------------------------------
In academic network intrusion detection research, raw PCAP and CSV files from
CIC-IDS2017/2018 feature column variations, leading/trailing whitespace, non-UTF8 encodings,
and inconsistent label naming (e.g. 'Web Attack  Brute Force').
This module maps all raw variants to the canonical FlowGuard 80-feature schema,
strips socket identifiers to prevent shortcut learning, and ensures leak-free processing.
"""

import glob
import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from flowguard.data.loader import FlowDataLoader
from flowguard.data.schema import IDENTIFIER_COLUMNS, RAW_COLUMN_MAP

logger = logging.getLogger(__name__)

# Expected raw CSV filenames for CIC-IDS2017
CIC_IDS_2017_FILES = [
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
]

# Expected raw CSV filenames for CSE-CIC-IDS2018
CSE_CIC_IDS_2018_FILES = [
    "02-14-2018.csv",
    "02-15-2018.csv",
    "02-16-2018.csv",
    "02-20-2018.csv",
    "02-21-2018.csv",
    "02-22-2018.csv",
    "02-23-2018.csv",
    "02-28-2018.csv",
    "03-01-2018.csv",
    "03-02-2018.csv",
]

# Canonical label normalization mapping
LABEL_NORMALIZATION: Dict[str, str] = {
    "benign": "BENIGN",
    "dos": "ATTACK",
    "ddos": "ATTACK",
    "portscan": "ATTACK",
    "bot": "ATTACK",
    "infilteration": "ATTACK",
    "web attack": "ATTACK",
    "brute force": "ATTACK",
    "ftp-patator": "ATTACK",
    "ssh-patator": "ATTACK",
    "heartbleed": "ATTACK",
}


def normalize_raw_column_name(col: str) -> str:
    """
    Cleans leading/trailing whitespace, punctuation, and converts to canonical mapping.
    """
    clean_col = col.strip()
    # Check direct canonical mapping
    if clean_col in RAW_COLUMN_MAP:
        return RAW_COLUMN_MAP[clean_col]

    # Handle case-insensitive match
    for k, v in RAW_COLUMN_MAP.items():
        if clean_col.lower() == k.lower():
            return v

    # Fallback to snake_case
    snake = clean_col.lower().replace(" ", "_").replace("/", "_per_").replace("-", "_")
    return snake


def normalize_attack_label(raw_label: str) -> str:
    """
    Maps heterogeneous attack string labels into standardized 'BENIGN' or 'ATTACK'.
    """
    lbl_clean = str(raw_label).strip().lower()
    if "benign" in lbl_clean:
        return "BENIGN"
    return "ATTACK"


class RealDatasetPipeline:
    """
    Ingestion, validation, and preprocessing pipeline for real benchmark datasets.
    """
    def __init__(self, raw_data_dir: str = "data/raw", processed_dir: str = "data/processed"):
        self.raw_data_dir = raw_data_dir
        self.processed_dir = processed_dir

    def verify_dataset_availability(self, dataset_name: str = "cic-ids-2017") -> Dict[str, Any]:
        """
        Inspects raw_data_dir for the presence of real dataset CSVs.
        """
        if not os.path.exists(self.raw_data_dir):
            return {
                "available": False,
                "found_files": [],
                "missing_files": CIC_IDS_2017_FILES if dataset_name == "cic-ids-2017" else CSE_CIC_IDS_2018_FILES,
                "total_rows": 0,
                "dataset_name": dataset_name,
                "message": f"Raw data directory '{self.raw_data_dir}' does not exist.",
            }

        expected = CIC_IDS_2017_FILES if dataset_name.lower() in ["cic-ids-2017", "cicids2017"] else CSE_CIC_IDS_2018_FILES
        found = []
        missing = []

        for fname in expected:
            full_p = os.path.join(self.raw_data_dir, fname)
            if os.path.exists(full_p):
                found.append(full_p)
            else:
                missing.append(fname)

        # Also search for any matching CSVs
        any_csvs = glob.glob(os.path.join(self.raw_data_dir, "**", "*.csv"), recursive=True)

        return {
            "available": len(found) > 0 or len(any_csvs) > 0,
            "found_files": found if len(found) > 0 else any_csvs,
            "missing_files": missing,
            "total_files_found": len(found) if len(found) > 0 else len(any_csvs),
            "dataset_name": dataset_name,
            "instructions": (
                "Download official CSVs from the University of New Brunswick (UNB) dataset repository: "
                "https://www.unb.ca/cic/datasets/ids-2017.html or https://www.unb.ca/cic/datasets/ids-2018.html "
                f"and extract them into '{os.path.abspath(self.raw_data_dir)}'."
            ),
        }

    def load_and_standardize_csv(
        self,
        file_path: str,
        n_rows: Optional[int] = None,
        drop_identifiers: bool = True,
    ) -> pd.DataFrame:
        """
        Loads a single raw CSV file, normalizes header names, strips socket identifiers,
        and harmonizes binary labels.
        """
        # Try UTF-8 first, fallback to latin-1 / cp1252
        try:
            df = pd.read_csv(file_path, nrows=n_rows, encoding="utf-8", low_memory=False)
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, nrows=n_rows, encoding="latin-1", low_memory=False)

        # Normalize column names
        rename_map = {col: normalize_raw_column_name(col) for col in df.columns}
        df = df.rename(columns=rename_map)

        # Harmonize label column
        label_col = None
        for col in ["label", "Label"]:
            if col in df.columns:
                label_col = col
                break
        if label_col:
            df["label"] = df[label_col].apply(normalize_attack_label)
            if label_col != "label":
                df = df.drop(columns=[label_col])

        # Drop socket identifiers to prevent shortcut learning
        if drop_identifiers:
            cols_to_drop = [c for c in df.columns if c in IDENTIFIER_COLUMNS]
            if cols_to_drop:
                df = df.drop(columns=cols_to_drop)

        # Convert object columns that should be numeric
        for col in df.columns:
            if col != "label" and df[col].dtype == object:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

        # Replace infinite values with NaN then median impute
        numeric_cols = [c for c in df.columns if c != "label"]
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

        return df

    def create_processed_benchmark_splits(
        self,
        sample_size_per_file: int = 5000,
        random_seed: int = 42,
    ) -> Tuple[
        Tuple[pd.DataFrame, pd.Series],
        Tuple[pd.DataFrame, pd.Series],
        Tuple[pd.DataFrame, pd.Series],
    ]:
        """
        Creates standardized train/calibration/test parquet files from available raw CSVs.
        If no raw CSVs are found, raises FileNotFoundError with download instructions.
        """
        status = self.verify_dataset_availability()
        if not status["available"]:
            raise FileNotFoundError(
                f"No real dataset CSV files found in '{self.raw_data_dir}'.\n{status['instructions']}"
            )

        dfs = []
        for fpath in status["found_files"]:
            logger.info(f"Loading and standardizing: {fpath}")
            df_slice = self.load_and_standardize_csv(fpath, n_rows=sample_size_per_file * 2)
            if len(df_slice) > sample_size_per_file:
                df_slice = df_slice.sample(n=sample_size_per_file, random_state=random_seed)
            dfs.append(df_slice)

        combined_df = pd.concat(dfs, axis=0, ignore_index=True)
        loader = FlowDataLoader(processed_dir=self.processed_dir, random_seed=random_seed)
        return loader.process_and_partition(combined_df, label_mode="binary", save_to_disk=True)
