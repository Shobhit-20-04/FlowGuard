"""
Unit & Integration Tests for Step 3: Sampling Simulator & Base Classifier
========================================================================
"""

import os
import tempfile
import numpy as np
import pandas as pd
import pytest

from flowguard.evaluation import (
    compute_brier_score,
    compute_classification_metrics,
    compute_expected_calibration_error,
)
from flowguard.models import BaseIntrusionClassifier, ObliviousBaseline
from flowguard.sampling import (
    PacketSamplingSimulator,
    binomial_count_moments,
    gilbert_elliott_parameters,
    inverted_count_posterior_moments,
    sampled_iat_moments,
)

def test_distortion_math_formulas():
    """Verify theoretical moment formulas match mathematical definitions."""
    # Binomial moments
    exp_k, var_k = binomial_count_moments(true_count=100.0, sampling_rate=0.10)
    assert abs(exp_k - 10.0) < 1e-5
    assert abs(var_k - 9.0) < 1e-5

    # Inverted count posterior moments
    exp_n, var_n = inverted_count_posterior_moments(observed_count=10.0, sampling_rate=0.10)
    assert abs(exp_n - 100.0) < 1e-5
    assert abs(var_n - 900.0) < 1e-5

    # Sampled IAT moments
    exp_iat, var_iat = sampled_iat_moments(mean_true_iat=100.0, std_true_iat=20.0, sampling_rate=0.10)
    assert abs(exp_iat - 1000.0) < 1e-5
    assert var_iat > (20.0 ** 2)

    # Gilbert-Elliott parameters
    params = gilbert_elliott_parameters(loss_rate=0.05, burst_length=5.0)
    assert abs(params["p_BG"] - 0.20) < 1e-5
    assert abs(params["effective_loss_rate"] - 0.05) < 0.01

def test_sampling_simulator_random():
    """Test that packet sampling decimates packet counts and scales volumes proportionally."""
    df = pd.DataFrame({
        "total_fwd_packets": [1000, 500, 200],
        "total_bwd_packets": [500, 200, 100],
        "total_fwd_bytes": [100_000.0, 50_000.0, 20_000.0],
        "total_bwd_bytes": [50_000.0, 20_000.0, 10_000.0],
        "flow_iat_mean": [50.0, 100.0, 200.0],
        "flow_iat_std": [20.0, 40.0, 80.0],
        "syn_flag_count": [10, 5, 2],
        "flow_duration": [1_000_000.0, 1_000_000.0, 1_000_000.0],
    })

    sim = PacketSamplingSimulator(seed=42)
    sampled_df = sim.sample_random(df, sampling_rate=0.10)

    # Counts must be decimated
    assert (sampled_df["total_fwd_packets"] < df["total_fwd_packets"]).all()
    # Volumes must be decimated
    assert (sampled_df["total_fwd_bytes"] < df["total_fwd_bytes"]).all()
    # IAT mean must be inflated
    assert (sampled_df["flow_iat_mean"] > df["flow_iat_mean"]).all()
    # Non-negativity
    assert (sampled_df.values >= 0).all()

def test_base_classifier_lifecycle():
    """Test training, probability prediction, feature importances, and serialization."""
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.normal(size=(200, 10)), columns=[f"feat_{i}" for i in range(10)])
    y = (X["feat_0"] + X["feat_1"] > 0).astype(int)

    clf = BaseIntrusionClassifier(model_type="lightgbm", random_state=42, n_estimators=20)
    clf.fit(X, y)

    preds = clf.predict(X)
    probs = clf.predict_proba(X)

    assert len(preds) == 200
    assert len(probs) == 200
    assert set(np.unique(preds)).issubset({0, 1})
    assert (probs >= 0.0).all() and (probs <= 1.0).all()
    assert len(clf.feature_importances_) == 10

    # Serialization test
    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        clf.save(tmp_path)
        loaded_clf = BaseIntrusionClassifier.load(tmp_path)
        loaded_probs = loaded_clf.predict_proba(X)
        assert np.allclose(probs, loaded_probs)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def test_oblivious_baseline_and_metrics():
    """Verify ObliviousBaseline wrapper and metric calculation suite."""
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.normal(size=(100, 5)), columns=[f"f_{i}" for i in range(5)])
    y = (X["f_0"] > 0).astype(int)

    base = BaseIntrusionClassifier(model_type="lightgbm", random_state=42, n_estimators=10)
    base.fit(X, y)

    oblivious = ObliviousBaseline(base_model=base, threshold=0.5)
    preds = oblivious.predict(X)
    probs = oblivious.predict_proba(X)

    metrics = compute_classification_metrics(y, preds, probs)
    assert "accuracy" in metrics
    assert "recall" in metrics
    assert "false_negative_rate" in metrics
    assert "expected_calibration_error" in metrics
    assert abs((metrics["recall"] + metrics["false_negative_rate"]) - 1.0) < 1e-5
