"""
Unit Test Suite: Advanced Mathematical, Copula, Triage & Metric Corrections
==========================================================================

Validates:
1. Gaussian Copula correlation preservation across reconstructed flow features.
2. Decoupled secondary triage parameters (TPR vs FPR) and residual cost accounting.
3. SAFE-3WD moment-bounded vs empirical quantile-bounded decision policies.
4. Exact geometric-sum temporal IAT moment inversion.
5. Selective metric validation on unrecognized decision labels.
6. PR-AUC and ROC-AUC single-class safety.
7. Chronological dataset partitioning without temporal leakage.
"""

import numpy as np
import pandas as pd
import pytest

from flowguard.data.loader import FlowDataLoader
from flowguard.decision.safe_3wd import SafeThreeWayDecisionEngine, ThreeWayDecisionResult
from flowguard.evaluation.metrics import compute_classification_metrics, compute_selective_metrics
from flowguard.sampling.distortion_math import invert_sampled_iat_moments, sampled_iat_moments
from flowguard.uncertainty.propagator import RiskProfile
from flowguard.uncertainty.sdum import SamplingDistortionUncertaintyModel


def test_copula_correlation_preservation():
    """
    Verifies that SDUM with preserve_correlation=True uses the Gaussian Copula
    to generate realizations that preserve cross-feature correlations.
    """
    rng = np.random.default_rng(42)
    n_samples = 200

    # Create strongly correlated pristine data: bytes scale strongly with packets
    fwd_pkts = rng.integers(10, 100, size=n_samples)
    bwd_pkts = (fwd_pkts * 0.7 + rng.normal(0, 2, size=n_samples)).astype(np.int64)
    bwd_pkts = np.maximum(1, bwd_pkts)
    fwd_bytes = fwd_pkts * 120.0 + rng.normal(0, 50, size=n_samples)
    bwd_bytes = bwd_pkts * 200.0 + rng.normal(0, 50, size=n_samples)
    duration = fwd_pkts * 1000.0 + rng.normal(0, 100, size=n_samples)

    pristine_df = pd.DataFrame({
        "total_fwd_packets": fwd_pkts,
        "total_bwd_packets": bwd_pkts,
        "total_fwd_bytes": fwd_bytes,
        "total_bwd_bytes": bwd_bytes,
        "flow_duration": duration,
        "flow_bytes_per_s": (fwd_bytes + bwd_bytes) / (duration / 1e6),
        "flow_packets_per_s": (fwd_pkts + bwd_pkts) / (duration / 1e6),
        "flow_iat_mean": duration / fwd_pkts,
    })

    # Fit SDUM with correlation preservation
    sdum = SamplingDistortionUncertaintyModel(
        n_mc_samples=30,
        random_state=42,
        preserve_correlation=True,
    )
    sdum.fit(pristine_df)
    assert sdum.corr_cholesky_ is not None

    # Sample data at 1-in-10 sampling
    sampled_df = pristine_df.copy()
    sampled_df["total_fwd_packets"] = rng.binomial(fwd_pkts, 0.10)
    sampled_df["total_bwd_packets"] = rng.binomial(bwd_pkts, 0.10)
    sampled_df["total_fwd_bytes"] = sampled_df["total_fwd_packets"] * 120.0
    sampled_df["total_bwd_bytes"] = sampled_df["total_bwd_packets"] * 200.0

    reconstructed_ensemble = sdum.reconstruct_flow_ensemble(sampled_df, sampling_rate=0.10, n_mc_samples=10)
    assert len(reconstructed_ensemble) == 10

    # Measure average correlation across realizations between reconstructed packets and bytes
    corrs = []
    for df_s in reconstructed_ensemble:
        r = np.corrcoef(df_s["total_fwd_packets"], df_s["total_fwd_bytes"])[0, 1]
        if not np.isnan(r):
            corrs.append(r)

    mean_corr = np.mean(corrs)
    # Copula draws should maintain strong positive correlation (typically > 0.6)
    assert mean_corr > 0.50, f"Expected strong correlation (>0.50) preserved by Copula, got {mean_corr:.3f}"


def test_decoupled_secondary_triage_metrics():
    """
    Tests that secondary triage parameters (TPR vs FPR) are decoupled
    and residual unblocked attacks are appropriately penalized.
    """
    engine = SafeThreeWayDecisionEngine(tau_safe=0.20, tau_attack=0.80, max_acceptable_variance=0.04)

    # 4 flows: 2 accepted (1 TN, 1 TP), 2 investigated (1 Attack, 1 Benign)
    r_min = np.array([0.02, 0.85, 0.10, 0.05])
    r_max = np.array([0.15, 0.98, 0.90, 0.18])
    r_mean = np.array([0.08, 0.92, 0.50, 0.12])
    r_var = np.array([0.002, 0.003, 0.160, 0.060])
    raw = np.column_stack([r_min, r_mean, r_max])

    profile = RiskProfile(
        risk_min=r_min,
        risk_max=r_max,
        risk_mean=r_mean,
        risk_variance=r_var,
        risk_std=np.sqrt(r_var),
        risk_spread=r_max - r_min,
        risk_median=r_mean,
        raw_risk_matrix=raw,
    )
    y_true = np.array([0, 1, 1, 0]) # 2 benign, 2 attacks

    metrics = engine.evaluate(
        risk_profile=profile,
        y_true=y_true,
        cost_fn=10.0,
        cost_fp=1.0,
        cost_inv=0.15,
        secondary_triage_tpr=0.90,
        secondary_triage_fpr=0.02,
    )

    # Direct rates:
    # 1 TP (flow 1), 1 TN (flow 0), 1 attack investigated (flow 2), 1 benign investigated (flow 3)
    assert metrics["direct_detection_rate"] == 0.50  # 1 / 2 attacks
    assert metrics["attack_investigation_rate"] == 0.50 # 1 / 2 attacks
    assert metrics["direct_fpr"] == 0.0 # 0 / 2 benign
    assert metrics["benign_investigation_rate"] == 0.50 # 1 / 2 benign

    # Decoupled post-triage rates:
    # PostDR = DirectDR (0.50) + 0.90 * AttackInv (0.50) = 0.95
    # PostFPR = DirectFPR (0.00) + 0.02 * BenignInv (0.50) = 0.01
    assert np.isclose(metrics["post_triage_detection_rate"], 0.95)
    assert np.isclose(metrics["post_triage_fpr"], 0.01)

    # Post triage cost should include residual missed attack (1 - 0.90 = 0.10 of flow 2)
    assert "post_triage_operational_cost" in metrics
    assert metrics["post_triage_operational_cost"] > 0


def test_safe_3wd_moment_policy():
    """
    Tests the moment-bounded policy (mu_R +/- k * sigma_R) in SAFE-3WD.
    """
    engine = SafeThreeWayDecisionEngine(
        tau_safe=0.20,
        tau_attack=0.80,
        max_acceptable_variance=0.04,
        policy="moment",
        k_sigma=1.0,
    )

    r_min = np.array([0.02, 0.85, 0.10, 0.05])
    r_max = np.array([0.15, 0.98, 0.90, 0.18])
    r_mean = np.array([0.08, 0.92, 0.50, 0.12])
    r_var = np.array([0.002, 0.003, 0.160, 0.060])
    raw = np.column_stack([r_min, r_mean, r_max])

    profile = RiskProfile(
        risk_min=r_min,
        risk_max=r_max,
        risk_mean=r_mean,
        risk_variance=r_var,
        risk_std=np.sqrt(r_var),
        risk_spread=r_max - r_min,
        risk_median=r_mean,
        raw_risk_matrix=raw,
    )

    res = engine.decide(profile)
    assert res.decisions[0] == "BENIGN"
    assert res.decisions[1] == "ATTACK"
    assert res.decisions[2] == "INVESTIGATE"
    assert res.decisions[3] == "INVESTIGATE" # Fails variance <= 0.04


def test_geometric_sum_iat_moment_inversion():
    """
    Validates exact inversion of sampled inter-arrival times:
    Var(tau_tilde) = Var(tau) / p + ((1 - p) / p^2) * (E[tau])^2
    Inversion: Var(tau) = p * Var(tau_tilde) - p * (1 - p) * (E[tau_tilde])^2
    """
    true_mean = 50.0
    true_std = 20.0
    p = 0.10

    sampled_mean, sampled_var = sampled_iat_moments(
        mean_true_iat=true_mean,
        std_true_iat=true_std,
        sampling_rate=p,
    )
    sampled_std = np.sqrt(sampled_var)

    # Invert moments
    rec_mean, rec_std = invert_sampled_iat_moments(
        mean_sampled_iat=sampled_mean,
        std_sampled_iat=sampled_std,
        sampling_rate=p,
    )

    assert np.isclose(rec_mean, true_mean, rtol=1e-4)
    assert np.isclose(rec_std, true_std, rtol=1e-4)


def test_pr_auc_and_metric_edge_cases():
    """
    Verifies PR-AUC computation and handling of single-class edge cases without silent 0.5.
    """
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9])

    metrics = compute_classification_metrics(y_true, y_pred, y_prob)
    assert "pr_auc" in metrics
    assert np.isclose(metrics["pr_auc"], 1.0)
    assert np.isclose(metrics["roc_auc"], 1.0)

    # Single-class partition edge case
    y_single = np.array([0, 0, 0, 0])
    metrics_single = compute_classification_metrics(y_single, y_single, y_prob)
    assert np.isnan(metrics_single["roc_auc"])
    assert np.isnan(metrics_single["pr_auc"])


def test_selective_metrics_label_validation():
    """
    Verifies that compute_selective_metrics raises a ValueError on unrecognized decision labels.
    """
    y_true = np.array([0, 1])
    invalid_decisions = np.array(["BENIGN", "UNKNOWN_LABEL"])
    accepted_preds = np.array([0, 1])

    with pytest.raises(ValueError, match="Invalid decision labels"):
        compute_selective_metrics(y_true, invalid_decisions, accepted_preds)


def test_chronological_splitting():
    """
    Verifies that FlowDataLoader splits sequentially when split_strategy='chronological'.
    """
    timestamps = pd.date_range("2026-01-01", periods=100, freq="1min")
    df = pd.DataFrame({
        "timestamp": timestamps,
        "feature_1": np.arange(100),
        "total_fwd_packets": np.ones(100),
        "total_bwd_packets": np.ones(100),
        "total_fwd_bytes": np.full(100, 100.0),
        "total_bwd_bytes": np.full(100, 100.0),
        "label": np.zeros(100),
    })

    loader = FlowDataLoader(
        train_ratio=0.60,
        calibration_ratio=0.20,
        test_ratio=0.20,
        split_strategy="chronological",
    )
    (X_tr, y_tr), (X_cal, y_cal), (X_te, y_te) = loader.process_and_partition(df, save_to_disk=False)

    assert len(X_tr) == 60
    assert len(X_cal) == 20
    assert len(X_te) == 20
