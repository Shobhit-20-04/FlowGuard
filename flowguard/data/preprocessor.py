"""
FlowGuard Data Preprocessor
===========================

Handles data hygiene, outlier clipping, inf/nan sanitization,
zero-variance feature pruning, and label taxonomy mapping.
Follows the Scikit-Learn Transformer paradigm to prevent data leakage.
"""

import logging
import re
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd

from flowguard.data.schema import (
    GROUP_A_FEATURES,
    GROUP_B_FEATURES,
    GROUP_C_FEATURES,
    IDENTIFIER_COLUMNS,
    LABEL_TAXONOMY,
    RAW_COLUMN_MAP,
)

logger = logging.getLogger(__name__)

class FlowDataPreprocessor:
    """
    Production-grade preprocessor for raw network flow data (CIC-IDS2017/2018, NetFlow).
    """
    def __init__(
        self,
        target_col: str = "label",
        drop_identifiers: bool = True,
        clip_outliers: bool = True,
        outlier_quantile: float = 0.9999,
        remove_zero_variance: bool = True,
    ):
        self.target_col = target_col
        self.drop_identifiers = drop_identifiers
        self.clip_outliers = clip_outliers
        self.outlier_quantile = outlier_quantile
        self.remove_zero_variance = remove_zero_variance

        # Fitted attributes (learnt strictly on Training data)
        self.is_fitted: bool = False
        self.feature_columns_: List[str] = []
        self.zero_variance_columns_: List[str] = []
        self.upper_clipping_bounds_: Dict[str, float] = {}
        self.median_values_: Dict[str, float] = {}

    def _normalize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Strips whitespace and maps raw dataset column names to clean snake_case.
        """
        df = df.copy()
        # Clean whitespaces
        cleaned_cols = [re.sub(r'\s+', ' ', col.strip()) for col in df.columns]
        df.columns = cleaned_cols

        # Map known variations to canonical snake_case
        renamed = {}
        for col in df.columns:
            if col in RAW_COLUMN_MAP:
                renamed[col] = RAW_COLUMN_MAP[col]
            else:
                # Fallback: lowercase and replace spaces/symbols with underscore
                clean_name = re.sub(r'[^a-zA-Z0-9_]', '', col.lower().replace(' ', '_').replace('/', '_per_'))
                renamed[col] = clean_name
        df = df.rename(columns=renamed)
        return df

    def standardize_labels(self, label_series: pd.Series, mode: str = "binary") -> pd.Series:
        """
        Maps diverse string labels (e.g. 'DDoS', 'DoS Hulk', 'FTP-Patator')
        to standardized integers based on binary or multiclass taxonomy.
        """
        raw_labels = label_series.astype(str).str.strip()

        if mode == "binary":
            # 0 for Benign, 1 for any Attack
            binary_labels = pd.Series(0, index=label_series.index, dtype=np.int32)
            is_attack = raw_labels.str.upper() != "BENIGN"
            binary_labels[is_attack] = 1
            return binary_labels

        elif mode == "multiclass":
            mapping = LABEL_TAXONOMY["multiclass"]
            mc_labels = pd.Series(mapping["OTHER"], index=label_series.index, dtype=np.int32)
            
            upper = raw_labels.str.upper()
            mc_labels[upper == "BENIGN"] = mapping["BENIGN"]
            mc_labels[upper.str.contains("DDOS")] = mapping["DDOS"]
            mc_labels[upper.str.contains("DOS") & ~upper.str.contains("DDOS")] = mapping["DOS"]
            mc_labels[upper.str.contains("PORT.*SCAN")] = mapping["PORTSCAN"]
            mc_labels[upper.str.contains("PATATOR|BRUTE")] = mapping["BRUTE_FORCE"]
            mc_labels[upper.str.contains("BOT")] = mapping["BOTNET"]
            mc_labels[upper.str.contains("INFILTR")] = mapping["INFILTRATION"]
            mc_labels[upper.str.contains("WEB|XSS|SQL")] = mapping["WEB_ATTACK"]
            return mc_labels
        else:
            raise ValueError(f"Unknown label mode: {mode}. Choose 'binary' or 'multiclass'.")

    def sanitize_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Replaces infinite values, fixes physically impossible negative quantities,
        and sanitizes numeric features.
        """
        df = df.copy()

        # Replace Inf and -Inf with NaN
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Enforce non-negativity for physical flow features (counts, bytes, durations, IATs)
        strictly_non_negative = set(GROUP_A_FEATURES + GROUP_B_FEATURES + GROUP_C_FEATURES)
        for col in df.select_dtypes(include=[np.number]).columns:
            if col in strictly_non_negative:
                df[col] = df[col].clip(lower=0.0)

        return df

    def fit(self, df: pd.DataFrame) -> "FlowDataPreprocessor":
        """
        Fits sanitization statistics, clipping thresholds, and active feature masks
        strictly on the training dataset.
        """
        df_clean = self._normalize_column_names(df)

        # Drop identifier columns
        if self.drop_identifiers:
            cols_to_drop = [c for c in df_clean.columns if c in IDENTIFIER_COLUMNS]
            df_clean = df_clean.drop(columns=cols_to_drop, errors="ignore")

        df_clean = self.sanitize_values(df_clean)

        numeric_df = df_clean.select_dtypes(include=[np.number])
        if self.target_col in numeric_df.columns:
            numeric_df = numeric_df.drop(columns=[self.target_col])

        # Compute medians for imputation
        self.median_values_ = numeric_df.median().to_dict()

        # Identify constant / zero-variance columns
        if self.remove_zero_variance:
            stds = numeric_df.std()
            self.zero_variance_columns_ = stds[stds == 0.0].index.tolist()
        else:
            self.zero_variance_columns_ = []

        active_numeric = numeric_df.drop(columns=self.zero_variance_columns_, errors="ignore")

        # Compute upper clipping bounds (99.99th percentile) to protect downstream estimators
        if self.clip_outliers:
            self.upper_clipping_bounds_ = active_numeric.quantile(self.outlier_quantile).to_dict()
        else:
            self.upper_clipping_bounds_ = {}

        self.feature_columns_ = active_numeric.columns.tolist()
        self.is_fitted = True
        logger.info(
            f"Fitted preprocessor: {len(self.feature_columns_)} active features, "
            f"{len(self.zero_variance_columns_)} zero-variance pruned."
        )
        return self

    def transform(self, df: pd.DataFrame, label_mode: str = "binary") -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Applies fitted transformations to training, validation, or test data.
        Returns:
            X (pd.DataFrame): Processed feature matrix with standardized column names.
            y (Optional[pd.Series]): Encoded labels (0/1 or multiclass), or None if not present.
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted on training data before calling transform()!")

        df_clean = self._normalize_column_names(df)

        # Extract and standardize target if present
        y = None
        target_name = None
        for cand in [self.target_col, "label", "Label"]:
            if cand in df_clean.columns:
                target_name = cand
                break

        if target_name is not None:
            y = self.standardize_labels(df_clean[target_name], mode=label_mode)
            df_clean = df_clean.drop(columns=[target_name])

        # Drop identifiers
        if self.drop_identifiers:
            cols_to_drop = [c for c in df_clean.columns if c in IDENTIFIER_COLUMNS]
            df_clean = df_clean.drop(columns=cols_to_drop, errors="ignore")

        # Sanitize Inf / negatives
        df_clean = self.sanitize_values(df_clean)

        # Drop zero-variance columns discovered during fit
        if self.zero_variance_columns_:
            df_clean = df_clean.drop(columns=self.zero_variance_columns_, errors="ignore")

        # Align columns to fitted feature set (fill missing features with medians)
        for col in self.feature_columns_:
            if col not in df_clean.columns:
                df_clean[col] = self.median_values_.get(col, 0.0)

        # Retain strictly the fitted feature columns in identical order
        X = df_clean[self.feature_columns_].copy()

        # Impute any remaining NaNs with fitted medians
        X.fillna(self.median_values_, inplace=True)

        # Apply upper bound outlier clipping
        if self.clip_outliers and self.upper_clipping_bounds_:
            for col, upper_val in self.upper_clipping_bounds_.items():
                if col in X.columns and pd.notna(upper_val):
                    X[col] = X[col].clip(upper=upper_val)

        return X, y

    def fit_transform(
        self, df: pd.DataFrame, label_mode: str = "binary"
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Fits preprocessor on data and transforms it in a single pass.
        """
        return self.fit(df).transform(df, label_mode=label_mode)
