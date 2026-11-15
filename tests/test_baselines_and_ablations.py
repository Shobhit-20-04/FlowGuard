"""
Unit Test Suite: Comparative Baselines (B1, B2, B3, B4) & Ablation Mechanics
=============================================================================

Tests:
1. Baseline 2 (ImputationBaseline): 1/p feature rescaling and inference.
2. Baseline 3 (SamplingAugmentedBaseline): Multi-condition training and inference.
3. Baseline 4 (DeepEnsembleUncertaintyBaseline): Ensemble variance and rejection.
4. Ablation study routines: Monte Carlo size S stability and sampling mismatch robustness.
"""

import numpy as np
import pandas as pd
import pytest

from flowguard.data.synthetic import SyntheticFlowGenerator
from flowguard.data.loader import FlowDataLoader
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


@pytest.fixture(scope="module")
def small_benchmark_data():
    """Generates a small dataset for fast deterministic baseline testing."""
    gen = SyntheticFlowGenerator(seed=42)
    raw_df = gen.generate_flow_batch(n_samples=400, attack_ratio=0.25)
    loader = FlowDataLoader(processed_dir="data/processed", random_seed=42)
    (X_tr, y_tr), (X_cal, y_cal), (X_te, y_te) = loader.process_and_partition(raw_df, save_to_disk=False)
    return (X_tr, y_tr), (X_cal, y_cal), (X_te, y_te)


def test_imputation_baseline_rescaling(small_benchmark_data):
    (X_train, y_train), _, (X_test, _) = small_benchmark_data
    base_clf = BaseIntrusionClassifier(model_type="lightgbm", random_state=42, n_estimators=30)
    base_clf.fit(X_train, y_train)

    imputer = ImputationBaseline(base_model=base_clf)
    sim = PacketSamplingSimulator(seed=42)
    p = 0.10
    X_deg = sim.sample_random(X_test.head(20), sampling_rate=p)

    X_imp = imputer.impute(X_deg, sampling_rate=p)

    # Imputed forward packets should be exactly observed_packets * (1/p)
    assert np.allclose(X_imp["total_fwd_packets"].values, X_deg["total_fwd_packets"].values * (1.0 / p))

    probs = imputer.predict_proba(X_deg, sampling_rate=p)
    preds = imputer.predict(X_deg, sampling_rate=p)

    assert len(probs) == len(X_deg)
    assert np.all((probs >= 0.0) & (probs <= 1.0))
    assert np.all((preds == 0) | (preds == 1))


def test_sampling_augmented_baseline(small_benchmark_data):
    (X_train, y_train), _, (X_test, _) = small_benchmark_data
    aug_model = SamplingAugmentedBaseline(
        model_type="lightgbm",
        random_state=42,
        multi_rate_sweep=[1.0, 0.20, 0.05],
    )
    assert not aug_model.is_fitted

    aug_model.fit(X_train, y_train)
    assert aug_model.is_fitted

    sim = PacketSamplingSimulator(seed=42)
    X_deg = sim.sample_random(X_test.head(25), sampling_rate=0.05)

    probs = aug_model.predict_proba(X_deg)
    preds = aug_model.predict(X_deg)

    assert len(preds) == len(X_deg)
    assert np.all((probs >= 0.0) & (probs <= 1.0))
    assert np.all((preds == 0) | (preds == 1))


def test_deep_ensemble_uncertainty_baseline(small_benchmark_data):
    (X_train, y_train), _, (X_test, _) = small_benchmark_data
    ensemble = DeepEnsembleUncertaintyBaseline(
        n_ensemble=3,
        model_type="lightgbm",
        random_state=42,
        variance_rejection_thresh=0.03,
    )
    assert not ensemble.is_fitted

    ensemble.fit(X_train, y_train)
    assert ensemble.is_fitted
    assert len(ensemble.models) == 3

    sim = PacketSamplingSimulator(seed=42)
    X_deg = sim.sample_random(X_test.head(30), sampling_rate=0.05)

    p_mean, p_var, decisions, accepted_preds = ensemble.predict_with_uncertainty(X_deg)

    assert len(p_mean) == len(X_deg)
    assert np.all(p_var >= 0.0)
    assert set(np.unique(decisions)).issubset({"BENIGN", "ATTACK", "INVESTIGATE"})
    assert set(np.unique(accepted_preds)).issubset({-1, 0, 1})


def test_mismatched_sampling_rate_stability(small_benchmark_data):
    """Verifies SDUM stability when nominal router rate differs from true sampling rate."""
    (X_train, y_train), _, (X_test, _) = small_benchmark_data
    base_clf = BaseIntrusionClassifier(model_type="lightgbm", random_state=42, n_estimators=30)
    base_clf.fit(X_train, y_train)

    sdum = SamplingDistortionUncertaintyModel(n_mc_samples=10, random_state=42)
    sdum.fit(X_train)
    prop = UncertaintyPropagator(base_classifier=base_clf, sdum=sdum, n_mc_samples=10)

    sim = PacketSamplingSimulator(seed=42)
    p_true = 0.05
    X_deg = sim.sample_random(X_test.head(20), sampling_rate=p_true)

    # Mismatched rates (+20% and -20% drift)
    p_under = 0.04
    p_over = 0.06

    profile_under = prop.propagate(X_deg, sampling_rate=p_under)
    profile_over = prop.propagate(X_deg, sampling_rate=p_over)

    # Invariants should hold without NaNs or infinities
    assert not np.isnan(profile_under.risk_mean).any()
    assert not np.isnan(profile_over.risk_mean).any()
    assert np.all((profile_under.risk_min >= 0.0) & (profile_under.risk_max <= 1.0))
    assert np.all((profile_over.risk_min >= 0.0) & (profile_over.risk_max <= 1.0))
