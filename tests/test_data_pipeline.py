"""
Unit & Integration Tests for FlowGuard Data Pipeline
====================================================
"""

import os
import tempfile
import numpy as np
import pandas as pd
import pytest

from flowguard.data import (
    GROUP_A_FEATURES,
    GROUP_B_FEATURES,
    GROUP_C_FEATURES,
    IDENTIFIER_COLUMNS,
    FeatureGroup,
    FlowDataLoader,
    FlowDataPreprocessor,
    SyntheticFlowGenerator,
)

def test_feature_group_disjointness():
    """Verify that feature groups have distinct non-overlapping members."""
    set_a = set(GROUP_A_FEATURES)
    set_b = set(GROUP_B_FEATURES)
    set_c = set(GROUP_C_FEATURES)

    assert len(set_a.intersection(set_b)) == 0, "Group A and Group B must not overlap!"
    assert len(set_a.intersection(set_c)) == 0, "Group A and Group C must not overlap!"
    assert len(set_b.intersection(set_c)) == 0, "Group B and Group C must not overlap!"

def test_synthetic_flow_generation():
    """Verify synthetic generator creates valid, non-empty flows with expected labels."""
    gen = SyntheticFlowGenerator(seed=123)
    df = gen.generate_flow_batch(n_samples=500, attack_ratio=0.30)

    assert len(df) == 500
    assert "Label" in df.columns
    assert "Total Fwd Packets" in df.columns
    assert "Flow Duration" in df.columns

    # Class distribution test
    benign_count = (df["Label"] == "BENIGN").sum()
    attack_count = (df["Label"] != "BENIGN").sum()
    assert benign_count == 350
    assert attack_count == 150

def test_preprocessor_hygiene_and_inf_handling():
    """Test that preprocessor sanitizes inf, negative durations, and strips headers."""
    dirty_data = pd.DataFrame({
        " Source IP ": ["192.168.1.1", "10.0.0.1"],
        " Flow Duration ": [-500.0, np.inf],
        " Total Fwd Packets ": [10, 20],
        " Total Length of Fwd Packets ": [1500.0, -100.0],
        " Label ": ["BENIGN", "DoS Hulk"],
    })

    preprocessor = FlowDataPreprocessor(drop_identifiers=True)
    X, y = preprocessor.fit_transform(dirty_data, label_mode="binary")

    # Source IP should be dropped
    assert "source_ip" not in X.columns
    # Negatives and infs should be sanitized
    assert (X["flow_duration"] >= 0).all()
    assert not np.isinf(X.values).any()
    assert not np.isnan(X.values).any()
    # Labels should be 0 and 1
    assert y.tolist() == [0, 1]

def test_stratified_split_and_no_leakage():
    """Verify stratified split preserves label ratio and prevents index leakage."""
    gen = SyntheticFlowGenerator(seed=42)
    df = gen.generate_flow_batch(n_samples=1000, attack_ratio=0.20)

    with tempfile.TemporaryDirectory() as tmpdir:
        loader = FlowDataLoader(
            processed_dir=tmpdir,
            train_ratio=0.65,
            calibration_ratio=0.15,
            test_ratio=0.20,
            random_seed=42,
        )

        (X_train, y_train), (X_calib, y_calib), (X_test, y_test) = loader.process_and_partition(
            df, label_mode="binary", save_to_disk=True
        )

        # Check split sizes
        assert len(X_train) == 650
        assert len(X_calib) == 150
        assert len(X_test) == 200

        # Check label balance preservation (stratification)
        assert abs(y_train.mean() - 0.20) < 0.02
        assert abs(y_calib.mean() - 0.20) < 0.02
        assert abs(y_test.mean() - 0.20) < 0.02

        # Verify Parquet round-trip
        (r_X_tr, r_y_tr), (r_X_ca, r_y_ca), (r_X_te, r_y_te) = loader.load_splits()
        assert len(r_X_tr) == 650
        assert len(r_X_ca) == 150
        assert len(r_X_te) == 200
        assert (r_y_tr.values == y_train.values).all()
        assert (r_y_ca.values == y_calib.values).all()
        assert (r_y_te.values == y_test.values).all()
