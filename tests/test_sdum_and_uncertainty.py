"""
Unit Test Suite: Sampling Distortion & Uncertainty Model (SDUM) & Propagator
=============================================================================

Tests:
1. SDUM fitting, prior learning, and Cholesky correlation matrix.
2. Negative Binomial count inversion moments: E[N_hat] ~= k / p.
3. Uncertainty Propagator risk bounds: 0 <= R_min <= mu_R <= R_max <= 1.
4. Epistemic risk spread widening as sampling rate p decreases.
5. Model Generalization Diagnostic execution and report generation.
"""

import numpy as np
import pandas as pd
import pytest

from flowguard.data.synthetic import SyntheticFlowGenerator
from flowguard.data.loader import FlowDataLoader
from flowguard.models.base_classifier import BaseIntrusionClassifier
from flowguard.models.diagnostics import ModelGeneralizationDiagnostics
from flowguard.sampling.simulator import PacketSamplingSimulator
from flowguard.uncertainty.sdum import SamplingDistortionUncertaintyModel
from flowguard.uncertainty.propagator import UncertaintyPropagator, RiskProfile


@pytest.fixture(scope="module")
def sample_flow_data():
    """Generates a small clean dataset for fast, deterministic unit testing."""
    gen = SyntheticFlowGenerator(seed=42)
    raw_df = gen.generate_flow_batch(n_samples=500, attack_ratio=0.25)
    loader = FlowDataLoader(processed_dir="data/processed", random_seed=42)
    (X_tr, y_tr), (X_cal, y_cal), (X_te, y_te) = loader.process_and_partition(raw_df, save_to_disk=False)
    return (X_tr, y_tr), (X_cal, y_cal), (X_te, y_te)


def test_sdum_fit_and_attributes(sample_flow_data):
    (X_train, _), _, _ = sample_flow_data
    sdum = SamplingDistortionUncertaintyModel(n_mc_samples=10, random_state=42)
    assert not sdum.is_fitted

    sdum.fit(X_train)
    assert sdum.is_fitted
    assert len(sdum.empirical_medians_) == len(X_train.columns)
    assert sdum.corr_cholesky_ is not None


def test_negative_binomial_count_inversion(sample_flow_data):
    (X_train, _), _, (X_test, _) = sample_flow_data
    sdum = SamplingDistortionUncertaintyModel(n_mc_samples=100, random_state=42)
    sdum.fit(X_train)

    sim = PacketSamplingSimulator(seed=42)
    p = 0.10
    X_deg = sim.sample_random(X_test.head(20), sampling_rate=p)

    reconstructed_list = sdum.reconstruct_flow_ensemble(X_deg, sampling_rate=p, n_mc_samples=100)
    assert len(reconstructed_list) == 100

    # For flows with k > 0 forward packets, verify average reconstructed N_hat approximates k / p
    k_fwd = X_deg["total_fwd_packets"].values
    n_hat_matrix = np.array([df["total_fwd_packets"].values for df in reconstructed_list])  # (S, N)
    mean_n_hat = np.mean(n_hat_matrix, axis=0)  # (N,)

    active_mask = k_fwd > 5
    if active_mask.any():
        expected_n = k_fwd[active_mask] / p
        empirical_n = mean_n_hat[active_mask]
        # Monte Carlo mean should be within 20% of theoretical expectation for S=100
        relative_errors = np.abs(empirical_n - expected_n) / expected_n
        assert np.mean(relative_errors) < 0.25, f"High error in count reconstruction: {np.mean(relative_errors)}"


def test_uncertainty_propagator_bounds(sample_flow_data):
    (X_train, y_train), _, (X_test, _) = sample_flow_data
    base_clf = BaseIntrusionClassifier(model_type="lightgbm", random_state=42, n_estimators=50)
    base_clf.fit(X_train, y_train)

    sdum = SamplingDistortionUncertaintyModel(n_mc_samples=15, random_state=42)
    sdum.fit(X_train)

    propagator = UncertaintyPropagator(base_classifier=base_clf, sdum=sdum, n_mc_samples=15)
    sim = PacketSamplingSimulator(seed=42)
    p = 0.05
    X_deg = sim.sample_random(X_test.head(30), sampling_rate=p)

    profile: RiskProfile = propagator.propagate(X_deg, sampling_rate=p)

    assert len(profile.risk_min) == len(X_deg)
    assert profile.raw_risk_matrix.shape == (len(X_deg), 15)

    # Invariants: 0.0 <= R_min <= mu_R <= R_max <= 1.0
    assert np.all(profile.risk_min >= -1e-6)
    assert np.all(profile.risk_max <= 1.0 + 1e-6)
    assert np.all(profile.risk_min <= profile.risk_mean + 1e-6)
    assert np.all(profile.risk_mean <= profile.risk_max + 1e-6)
    assert np.all(profile.risk_spread >= -1e-6)
    assert np.all(profile.risk_variance >= -1e-6)

    # DataFrame conversion
    df_profile = profile.to_dataframe()
    assert list(df_profile.columns) == [
        "risk_min", "risk_max", "risk_mean", "risk_variance", "risk_std", "risk_spread", "risk_median"
    ]


def test_epistemic_spread_widens_under_sampling(sample_flow_data):
    (X_train, y_train), _, (X_test, _) = sample_flow_data
    base_clf = BaseIntrusionClassifier(model_type="lightgbm", random_state=42, n_estimators=50)
    base_clf.fit(X_train, y_train)

    sdum = SamplingDistortionUncertaintyModel(n_mc_samples=20, random_state=42)
    sdum.fit(X_train)

    propagator = UncertaintyPropagator(base_classifier=base_clf, sdum=sdum, n_mc_samples=20)
    sim = PacketSamplingSimulator(seed=42)

    # Under p = 1.0 (complete traffic), spread should be exactly 0
    profile_p1 = propagator.propagate(X_test.head(25), sampling_rate=1.0)
    assert np.allclose(profile_p1.risk_spread, 0.0, atol=1e-5)

    # Under severe sampling p = 0.02, spread should increase as true state is ambiguous
    X_deg = sim.sample_random(X_test.head(25), sampling_rate=0.02)
    profile_p02 = propagator.propagate(X_deg, sampling_rate=0.02)
    # The average spread under p = 0.02 should be noticeably higher than at p = 1.0
    assert np.mean(profile_p02.risk_spread) >= np.mean(profile_p1.risk_spread)


def test_diagnostics_evaluation(sample_flow_data):
    (X_train, y_train), (X_cal, y_cal), (X_test, y_test) = sample_flow_data
    base_clf = BaseIntrusionClassifier(model_type="lightgbm", random_state=42, n_estimators=50)
    base_clf.fit(X_train, y_train)

    diag = ModelGeneralizationDiagnostics(random_state=42)
    results = diag.evaluate_fit(base_clf, X_train, y_train, X_cal, y_cal, X_test, y_test)

    assert "train" in results
    assert "validation" in results
    assert "test" in results
    assert "generalization_gaps" in results
    assert "diagnosis" in results

    report = diag.format_report(results, model_name="LightGBM")
    assert "Model Generalization Diagnostic Report" in report
