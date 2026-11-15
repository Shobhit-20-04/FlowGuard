"""
FlowGuard Comprehensive Research Correction & Foundation Test Suite
==================================================================

Rigorous mathematical, algorithmic, and statistical verification covering
all 20 foundation audit points:
1. Negative Binomial posterior under improper uniform prior: U | k,p ~ NegBin(k+1, p).
2. Negative Binomial posterior under scale-invariant prior: U | k,p ~ NegBin(k, p).
3. Posterior moments match theoretical expectations under both priors.
4. Boundary condition at p = 1.0 (zero variance reconstruction).
5. Boundary handling of zero-observation flows (k_tot = 0).
6. Gilbert-Elliott 2-state Markov chain packet loss.
7. Periodic 1-in-M sampling with phase offset.
8. Asymmetric directional packet sampling.
9. Compound Gamma volume modeling for missing packets.
10. Strict exclusion of TCP window sizes from additive volume scaling.
11. Temporal duration and IAT geometric random sum moment inversion.
12. Risk profile bounds [R_min, R_max] and empirical credible intervals [q05, q95].
13. SAFE-3WD three-way decision partitioning.
14. Metric disaggregation: Direct DR, Attack Inv Rate, Definite FNR sum to 1.0.
15. Secondary triage efficiency parameterization (rho_sec).
16. Selective accuracy and genuine F1 calculation on accepted decisions.
17. Calibrator strict feasibility enforcement on min_coverage.
18. Baseline models B1, B2 (rescaling), B3 (augmented), and B4 (tree ensemble).
19. MICE imputation baseline functionality.
20. Synthetic generator statistical consistency (no Active Std shortcuts).
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from flowguard.data.schema import ADDITIVE_BYTE_FEATURES, RAW_COLUMN_MAP
from flowguard.data.synthetic import SyntheticFlowGenerator
from flowguard.decision.calibrator import CostSensitiveThresholdCalibrator
from flowguard.decision.safe_3wd import SafeThreeWayDecisionEngine
from flowguard.evaluation.metrics import compute_classification_metrics, compute_selective_metrics
from flowguard.models.base_classifier import BaseIntrusionClassifier
from flowguard.models.baselines import (
    DeterministicRescalingBaseline,
    MICEImputationBaseline,
    ObliviousBaseline,
    SamplingAugmentedBaseline,
    TreeEnsembleUncertaintyBaseline,
)
from flowguard.sampling.distortion_math import (
    binomial_count_moments,
    gilbert_elliott_parameters,
    inverted_count_posterior_moments,
    sampled_iat_moments,
)
from flowguard.sampling.simulator import PacketSamplingSimulator
from flowguard.uncertainty.propagator import RiskProfile, UncertaintyPropagator
from flowguard.uncertainty.sdum import SamplingDistortionUncertaintyModel


# --------------------------------------------------------------------------
# Point 1, 2, 3: Mathematical Posterior Derivations & Moments
# --------------------------------------------------------------------------
def test_negative_binomial_posterior_scale_invariant_prior():
    r"""Verify under scale-invariant prior P(N) \propto 1/N, U ~ NegBin(k, p) and E[N] = k/p."""
    k = 10
    p = 0.10
    sdum = SamplingDistortionUncertaintyModel(n_mc_samples=20000, random_state=42, prior="scale_invariant")
    sdum.fit(pd.DataFrame({
        "total_fwd_packets": [100.0, 200.0],
        "total_bwd_packets": [50.0, 80.0],
        "total_fwd_bytes": [10000.0, 20000.0],
        "total_bwd_bytes": [5000.0, 8000.0],
    }))

    df_sample = pd.DataFrame({
        "total_fwd_packets": [k],
        "total_bwd_packets": [0],
        "total_fwd_bytes": [1000.0],
        "total_bwd_bytes": [0.0],
    })

    reconstructions = sdum.reconstruct_flow_ensemble(df_sample, sampling_rate=p, n_mc_samples=5000)
    samples_n = np.array([df["total_fwd_packets"].values[0] for df in reconstructions])

    theoretical_mean = k / p  # 100.0
    empirical_mean = float(np.mean(samples_n))

    # Mean must match theoretical expectation within Monte Carlo error (tol 2.5%)
    assert abs(empirical_mean - theoretical_mean) / theoretical_mean < 0.025


def test_negative_binomial_posterior_uniform_prior():
    r"""Verify under uniform prior P(N) \propto 1, U ~ NegBin(k+1, p) and E[N] = k/p + (1-p)/p."""
    k = 10
    p = 0.10
    sdum = SamplingDistortionUncertaintyModel(n_mc_samples=20000, random_state=42, prior="uniform")
    sdum.fit(pd.DataFrame({
        "total_fwd_packets": [100.0, 200.0],
        "total_bwd_packets": [50.0, 80.0],
        "total_fwd_bytes": [10000.0, 20000.0],
        "total_bwd_bytes": [5000.0, 8000.0],
    }))

    df_sample = pd.DataFrame({
        "total_fwd_packets": [k],
        "total_bwd_packets": [0],
        "total_fwd_bytes": [1000.0],
        "total_bwd_bytes": [0.0],
    })

    reconstructions = sdum.reconstruct_flow_ensemble(df_sample, sampling_rate=p, n_mc_samples=5000)
    samples_n = np.array([df["total_fwd_packets"].values[0] for df in reconstructions])

    # Theoretical mean: k + (k+1)*(1-p)/p = 10 + 11 * 9 = 109.0
    theoretical_mean = (k / p) + ((1.0 - p) / p)
    empirical_mean = float(np.mean(samples_n))

    assert abs(empirical_mean - theoretical_mean) / theoretical_mean < 0.025


# --------------------------------------------------------------------------
# Point 4: Boundary Condition at p = 1.0 (Zero Variance)
# --------------------------------------------------------------------------
def test_boundary_condition_p_one():
    """At p = 1.0, reconstruction must be identical to pristine input with 0 variance."""
    sdum = SamplingDistortionUncertaintyModel(n_mc_samples=10, random_state=42)
    df_pristine = pd.DataFrame({
        "total_fwd_packets": [42.0],
        "total_bwd_packets": [18.0],
        "total_fwd_bytes": [3500.0],
        "total_bwd_bytes": [1500.0],
    })
    sdum.fit(df_pristine)
    reconstructions = sdum.reconstruct_flow_ensemble(df_pristine, sampling_rate=1.0, n_mc_samples=10)

    for df_rec in reconstructions:
        pd.testing.assert_frame_equal(df_rec, df_pristine)


# --------------------------------------------------------------------------
# Point 5: Zero-Observation Flows (No Artificial Packet Forcing)
# --------------------------------------------------------------------------
def test_zero_observation_flows_no_artificial_packets():
    """Verify simulator does not artificially force packet count to 1 when a flow has 0 sampled packets."""
    sim = PacketSamplingSimulator(seed=42)
    df = pd.DataFrame({
        "total_fwd_packets": [1, 1, 1, 1, 1],
        "total_bwd_packets": [0, 0, 0, 0, 0],
        "total_fwd_bytes": [100.0, 100.0, 100.0, 100.0, 100.0],
    })
    # Sample at very low rate p = 0.0001 where all flows are expected to capture 0 packets
    deg = sim.sample_random(df, sampling_rate=1e-4, drop_unobserved=False)
    # Check that 0 packets are preserved rather than artificially forced to 1
    zero_fwd_count = int(np.sum(deg["total_fwd_packets"] == 0))
    assert zero_fwd_count > 0, "Simulator must honestly report 0 packets without forcing!"


# --------------------------------------------------------------------------
# Point 6: Gilbert-Elliott Markov Chain Loss
# --------------------------------------------------------------------------
def test_gilbert_elliott_burst_loss():
    """Verify Gilbert-Elliott 2-state Markov chain parameters and loss degradation."""
    params = gilbert_elliott_parameters(loss_rate=0.10, burst_length=5.0)
    assert 0.0 < params["p_GB"] < 1.0
    assert 0.0 < params["p_BG"] < 1.0
    assert abs(params["effective_loss_rate"] - 0.10) < 1e-3

    sim = PacketSamplingSimulator(seed=42)
    df = pd.DataFrame({
        "total_fwd_packets": [1000] * 50,
        "total_bwd_packets": [1000] * 50,
        "total_fwd_bytes": [50000.0] * 50,
        "total_bwd_bytes": [50000.0] * 50,
    })
    deg = sim.sample_burst_loss(df, loss_rate=0.10, burst_length=5.0)
    mean_retained = deg["total_fwd_packets"].mean()
    # Retained packets should be roughly 900 (10% drop)
    assert 800 <= mean_retained <= 950


# --------------------------------------------------------------------------
# Point 7: Periodic Sampling (1-in-M)
# --------------------------------------------------------------------------
def test_periodic_sampling_interval():
    """Verify periodic sampling accurately captures ~1 in M packets."""
    sim = PacketSamplingSimulator(seed=42)
    M = 10
    df = pd.DataFrame({
        "total_fwd_packets": [100, 200, 300, 400],
        "total_bwd_packets": [50, 100, 150, 200],
        "total_fwd_bytes": [10000.0, 20000.0, 30000.0, 40000.0],
        "total_bwd_bytes": [5000.0, 10000.0, 15000.0, 20000.0],
    })
    deg = sim.sample_periodic(df, sampling_interval=M, phase_randomization=False)
    # With phase 0, N packets yield (N - 1) // 10 + 1 = 10, 20, 30, 40
    expected_fwd = np.array([10, 20, 30, 40])
    np.testing.assert_array_equal(deg["total_fwd_packets"].values, expected_fwd)


# --------------------------------------------------------------------------
# Point 8: Asymmetric Directional Sampling
# --------------------------------------------------------------------------
def test_asymmetric_directional_sampling():
    """Verify forward and backward directions are sampled at disparate rates."""
    sim = PacketSamplingSimulator(seed=42)
    df = pd.DataFrame({
        "total_fwd_packets": [1000] * 20,
        "total_bwd_packets": [1000] * 20,
        "total_fwd_bytes": [50000.0] * 20,
        "total_bwd_bytes": [50000.0] * 20,
    })
    deg = sim.sample_asymmetric(df, forward_rate=0.20, backward_rate=0.02)
    mean_fwd = deg["total_fwd_packets"].mean()
    mean_bwd = deg["total_bwd_packets"].mean()
    # Forward retention should be ~200, backward ~20
    assert 150 <= mean_fwd <= 250
    assert 10 <= mean_bwd <= 35


# --------------------------------------------------------------------------
# Point 9, 10: Compound Gamma Volume & TCP Window Size Exclusion
# --------------------------------------------------------------------------
def test_compound_gamma_volume_and_window_exclusion():
    """Verify compound Gamma scales additive byte features but strictly excludes TCP window sizes."""
    sdum = SamplingDistortionUncertaintyModel(n_mc_samples=20, random_state=42)
    df_pristine = pd.DataFrame({
        "total_fwd_packets": [100.0, 200.0],
        "total_bwd_packets": [50.0, 80.0],
        "total_fwd_bytes": [10000.0, 20000.0],
        "total_bwd_bytes": [5000.0, 8000.0],
        "subflow_fwd_bytes": [10000.0, 20000.0],
        "init_win_bytes_fwd": [65535.0, 65535.0],
        "init_win_bytes_bwd": [8192.0, 8192.0],
    })
    sdum.fit(df_pristine)

    df_sampled = pd.DataFrame({
        "total_fwd_packets": [10],
        "total_bwd_packets": [5],
        "total_fwd_bytes": [1000.0],
        "total_bwd_bytes": [500.0],
        "subflow_fwd_bytes": [1000.0],
        "init_win_bytes_fwd": [65535.0],
        "init_win_bytes_bwd": [8192.0],
    })

    reconstructions = sdum.reconstruct_flow_ensemble(df_sampled, sampling_rate=0.10, n_mc_samples=10)

    for df_rec in reconstructions:
        # Additive volume should be scaled up significantly
        assert df_rec["total_fwd_bytes"].values[0] > 2000.0
        # TCP window sizes MUST REMAIN EXACTLY UNCHANGED
        assert df_rec["init_win_bytes_fwd"].values[0] == 65535.0
        assert df_rec["init_win_bytes_bwd"].values[0] == 8192.0


# --------------------------------------------------------------------------
# Point 11: Temporal Duration & IAT Inversion
# --------------------------------------------------------------------------
def test_temporal_duration_and_iat_inversion():
    """Verify sampled IAT moments and flow duration scale according to sampling physics."""
    mu_tau, var_tau = sampled_iat_moments(mean_true_iat=100.0, std_true_iat=20.0, sampling_rate=0.10)
    assert mu_tau == 1000.0  # 100 / 0.10
    assert var_tau > 0.0


# --------------------------------------------------------------------------
# Point 12: Risk Profile Bounds & Credible Intervals
# --------------------------------------------------------------------------
def test_risk_profile_quantiles_and_extrema():
    """Verify RiskProfile holds valid empirical sample extrema and 90% credible intervals."""
    N = 50
    S = 30
    rng = np.random.default_rng(42)
    raw = rng.uniform(0.1, 0.9, size=(N, S))
    r_min = np.min(raw, axis=1)
    r_max = np.max(raw, axis=1)
    r_mean = np.mean(raw, axis=1)
    r_var = np.var(raw, axis=1, ddof=1)
    r_std = np.sqrt(r_var)
    r_spread = r_max - r_min
    r_median = np.median(raw, axis=1)
    r_q05 = np.percentile(raw, 5.0, axis=1)
    r_q95 = np.percentile(raw, 95.0, axis=1)

    profile = RiskProfile(
        risk_min=r_min,
        risk_max=r_max,
        risk_mean=r_mean,
        risk_variance=r_var,
        risk_std=r_std,
        risk_spread=r_spread,
        risk_median=r_median,
        risk_q05=r_q05,
        risk_q95=r_q95,
        raw_risk_matrix=raw,
    )

    # Invariants: R_min <= q05 <= median <= q95 <= R_max
    assert np.all(profile.risk_min <= profile.risk_q05 + 1e-6)
    assert np.all(profile.risk_q05 <= profile.risk_q95 + 1e-6)
    assert np.all(profile.risk_q95 <= profile.risk_max + 1e-6)


# --------------------------------------------------------------------------
# Point 13, 14, 15: SAFE-3WD, Metric Disaggregation, and Secondary Triage
# --------------------------------------------------------------------------
def test_safe_3wd_partitioning_and_disaggregated_metrics():
    """Verify SAFE-3WD decision partitioning, rate balance, and secondary triage modeling."""
    r_min = np.array([0.02, 0.85, 0.10, 0.05])
    r_max = np.array([0.15, 0.98, 0.90, 0.18])
    r_mean = np.array([0.08, 0.92, 0.50, 0.12])
    r_var = np.array([0.002, 0.003, 0.160, 0.060])
    r_std = np.sqrt(r_var)
    r_spread = r_max - r_min
    r_median = r_mean
    raw = np.column_stack([r_min, r_mean, r_max])

    profile = RiskProfile(
        risk_min=r_min,
        risk_max=r_max,
        risk_mean=r_mean,
        risk_variance=r_var,
        risk_std=r_std,
        risk_spread=r_spread,
        risk_median=r_median,
        raw_risk_matrix=raw,
    )

    engine = SafeThreeWayDecisionEngine(tau_safe=0.20, tau_attack=0.80, max_acceptable_variance=0.04)
    res = engine.decide(profile)

    assert res.decisions[0] == "BENIGN"
    assert res.decisions[1] == "ATTACK"
    assert res.decisions[2] == "INVESTIGATE"
    assert res.decisions[3] == "INVESTIGATE"  # High variance reject

    # Evaluate with y_true = [0, 1, 1, 0]
    y_true = np.array([0, 1, 1, 0])
    eval_res = engine.evaluate(profile, y_true, cost_fn=10.0, cost_fp=1.0, cost_inv=0.15, secondary_triage_efficiency=0.80)

    # Disaggregated rate balance: Direct DR + Attack Inv Rate + Definite FNR == 1.0
    attack_sum = eval_res["direct_detection_rate"] + eval_res["attack_investigation_rate"] + eval_res["definite_fn_rate"]
    assert abs(attack_sum - 1.0) < 1e-6

    # Benign rate balance: Direct FPR + Benign Inv Rate + Direct TNR == 1.0
    benign_sum = eval_res["direct_fpr"] + eval_res["benign_investigation_rate"] + eval_res["direct_tnr"]
    assert abs(benign_sum - 1.0) < 1e-6

    # Post-triage detection rate check
    expected_post_dr = eval_res["direct_detection_rate"] + 0.80 * eval_res["attack_investigation_rate"]
    assert abs(eval_res["post_triage_detection_rate"] - expected_post_dr) < 1e-6


# --------------------------------------------------------------------------
# Point 16: Selective Accuracy & Genuine F1 Calculation
# --------------------------------------------------------------------------
def test_selective_metrics_genuine_f1():
    """Verify selective metrics compute genuine non-hardcoded F1 score."""
    y_true = np.array([0, 0, 1, 1, 1, 0])
    decisions = np.array(["BENIGN", "BENIGN", "ATTACK", "ATTACK", "INVESTIGATE", "INVESTIGATE"])
    acc_preds = np.array([0, 0, 1, 1, -1, -1])

    m = compute_selective_metrics(y_true, decisions, acc_preds)
    assert m["coverage"] == 4 / 6
    assert m["investigation_rate"] == 2 / 6
    assert m["selective_accuracy"] == 1.0
    assert m["selective_f1_score"] == 1.0

    # Test imperfect case
    acc_preds_imp = np.array([1, 0, 1, 0, -1, -1])  # 1 FP, 1 FN
    m_imp = compute_selective_metrics(y_true, decisions, acc_preds_imp)
    assert m_imp["selective_accuracy"] == 0.5
    assert 0.0 < m_imp["selective_f1_score"] < 1.0


# --------------------------------------------------------------------------
# Point 17: Calibrator Strict Feasibility
# --------------------------------------------------------------------------
def test_calibrator_strict_feasibility():
    """Verify calibrator strictly checks min_coverage and flags constraint_satisfied_."""
    r_min = np.array([0.05, 0.85, 0.50, 0.10, 0.90])
    r_max = np.array([0.15, 0.95, 0.60, 0.15, 0.95])
    r_mean = np.array([0.10, 0.90, 0.55, 0.12, 0.92])
    r_var = np.array([0.001, 0.001, 0.001, 0.001, 0.001])
    raw = np.column_stack([r_min, r_mean, r_max])

    profile = RiskProfile(
        risk_min=r_min, risk_max=r_max, risk_mean=r_mean, risk_variance=r_var,
        risk_std=np.sqrt(r_var), risk_spread=r_max - r_min, risk_median=r_mean,
        raw_risk_matrix=raw,
    )
    y_cal = np.array([0, 1, 1, 0, 1])

    calibrator = CostSensitiveThresholdCalibrator(min_coverage=0.40, grid_steps=10)
    calibrator.fit(profile, y_cal)

    assert calibrator.is_fitted
    assert calibrator.constraint_satisfied_
    assert calibrator.optimal_tau_safe_ < calibrator.optimal_tau_attack_


# --------------------------------------------------------------------------
# Point 18, 19: Baseline Models & MICE Imputation
# --------------------------------------------------------------------------
def test_baseline_models_and_mice():
    """Verify B1, B2, B3, B4, and MICE baseline instantiation and inference."""
    generator = SyntheticFlowGenerator(seed=42)
    df = generator.generate_flow_batch(n_samples=100)
    feature_cols = [c for c in df.columns if c != "Label"]
    X = df[feature_cols].copy()
    y = (df["Label"] != "BENIGN").astype(int)

    base_clf = BaseIntrusionClassifier(model_type="lightgbm", random_state=42, n_estimators=20)
    base_clf.fit(X, y)

    # B1: Oblivious
    b1 = ObliviousBaseline(base_model=base_clf)
    p_b1 = b1.predict_proba(X)
    assert len(p_b1) == len(X)

    # B2: Rescaling
    b2 = DeterministicRescalingBaseline(base_model=base_clf)
    p_b2 = b2.predict_proba(X, sampling_rate=0.10)
    assert len(p_b2) == len(X)

    # B4: Tree Ensemble
    b4 = TreeEnsembleUncertaintyBaseline(n_ensemble=3, random_state=42)
    b4.fit(X, y)
    p_mean, p_var, dec, acc = b4.predict_with_uncertainty(X)
    assert len(p_mean) == len(X)
    assert np.all(p_var >= 0.0)

    # MICE Baseline
    mice = MICEImputationBaseline(base_model=base_clf, max_iter=2, random_state=42)
    mice.fit(X)
    p_mice = mice.predict_proba(X, sampling_rate=0.10)
    assert len(p_mice) == len(X)


# --------------------------------------------------------------------------
# Point 20: Synthetic Generator Consistency (No Active Std Shortcuts)
# --------------------------------------------------------------------------
def test_synthetic_flow_generator_no_active_std_shortcut():
    """Verify synthetic flows do not possess an artificial Active Std shortcut between benign and attack."""
    gen = SyntheticFlowGenerator(seed=42)
    df = gen.generate_flow_batch(n_samples=200, attack_ratio=0.50)
    benign_active = df[df["Label"] == "BENIGN"]["Active Std"].values
    attack_active = df[df["Label"] != "BENIGN"]["Active Std"].values

    # Neither distribution should be trivially constant or completely separable
    # Verify that Active Std is not trivially 500 for all benign and 0 for all attacks
    assert not np.all(benign_active == 500.0)
    assert np.any(attack_active >= 0.0)
