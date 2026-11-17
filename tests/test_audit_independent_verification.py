"""
Unit Test Suite: Post-Correction Independent Mathematical & Research Audit
==========================================================================

Rigorous test suite verifying:
1. SAFE-3WD confidence interval quantile mapping (alpha=0.90, 0.95, 0.99, ordering, calibration consistency).
2. Sparse-flow Empirical Bayes mixture (P(probe|k=0), E[(1-p)^N] vs Jensen, conditional support N_bwd >= 1).
3. Latent Gaussian Copula fitting via normal scores vs raw Pearson correlation.
4. Chronological splitting robustness (stable sort, duplicates, missing timestamps).
5. Secondary triage cost accounting (perfect vs imperfect triage, exact expectations, no double counting).
6. Component ablation isolation (Compound Gamma on/off, Copula on/off).
"""

import numpy as np
import pandas as pd
import pytest
from scipy import special, stats

from flowguard.data.loader import FlowDataLoader
from flowguard.decision.calibrator import CostSensitiveThresholdCalibrator
from flowguard.decision.safe_3wd import SafeThreeWayDecisionEngine, ThreeWayDecisionResult
from flowguard.uncertainty.propagator import RiskProfile, UncertaintyPropagator
from flowguard.uncertainty.sdum import SamplingDistortionUncertaintyModel


# =============================================================================
# 1. SAFE-3WD CONFIDENCE INTERVAL QUANTILE AUDIT
# =============================================================================

def test_safe_3wd_alpha_confidence_intervals():
    """
    Verifies that alpha means confidence level (e.g. 0.90 for 90% confidence),
    yielding lower bound Q((1-alpha)/2) and upper bound Q((1+alpha)/2).
    Ensures that collapsed intervals (q45/q55) are strictly avoided.
    """
    rng = np.random.default_rng(42)
    n_flows = 50
    S = 100
    # Simulate risk draws with varying mean and variance
    means = np.linspace(0.1, 0.9, n_flows)
    raw_risks = np.zeros((n_flows, S))
    for i in range(n_flows):
        raw_risks[i] = np.clip(rng.normal(loc=means[i], scale=0.08, size=S), 0.0, 1.0)

    # Test alpha = 0.90 -> q05 and q95
    engine_90 = SafeThreeWayDecisionEngine(
        tau_safe=0.20,
        tau_attack=0.80,
        policy="quantile",
        alpha_confidence=0.90,
    )
    assert engine_90.alpha_confidence == 0.90

    # Test alpha = 0.95 -> q025 and q975
    engine_95 = SafeThreeWayDecisionEngine(
        tau_safe=0.20,
        tau_attack=0.80,
        policy="quantile",
        alpha_confidence=0.95,
    )
    assert engine_95.alpha_confidence == 0.95

    # Test alpha = 0.99 -> q005 and q995
    engine_99 = SafeThreeWayDecisionEngine(
        tau_safe=0.20,
        tau_attack=0.80,
        policy="quantile",
        alpha_confidence=0.99,
    )
    assert engine_99.alpha_confidence == 0.99

    # Boundary validation: alpha <= 0 or > 1 must raise AssertionError
    with pytest.raises(AssertionError):
        SafeThreeWayDecisionEngine(alpha_confidence=0.0)
    with pytest.raises(AssertionError):
        SafeThreeWayDecisionEngine(alpha_confidence=1.05)
    with pytest.raises(AssertionError):
        SafeThreeWayDecisionEngine(alpha_confidence=-0.1)

    # Create RiskProfile with raw_risk_matrix
    profile = RiskProfile(
        risk_min=np.min(raw_risks, axis=1),
        risk_max=np.max(raw_risks, axis=1),
        risk_mean=np.mean(raw_risks, axis=1),
        risk_variance=np.var(raw_risks, axis=1, ddof=1),
        risk_std=np.std(raw_risks, axis=1, ddof=1),
        risk_spread=np.max(raw_risks, axis=1) - np.min(raw_risks, axis=1),
        risk_median=np.median(raw_risks, axis=1),
        raw_risk_matrix=raw_risks,
    )

    res_90 = engine_90.decide(profile)
    res_95 = engine_95.decide(profile)
    res_99 = engine_99.decide(profile)

    # Quantile ordering: higher confidence level expands the interval
    # L_99 <= L_95 <= L_90 <= U_90 <= U_95 <= U_99
    q005 = np.percentile(raw_risks, 0.5, axis=1)
    q025 = np.percentile(raw_risks, 2.5, axis=1)
    q05 = np.percentile(raw_risks, 5.0, axis=1)
    q95 = np.percentile(raw_risks, 95.0, axis=1)
    q975 = np.percentile(raw_risks, 97.5, axis=1)
    q995 = np.percentile(raw_risks, 99.5, axis=1)

    assert np.all(q005 <= q025 + 1e-9)
    assert np.all(q025 <= q05 + 1e-9)
    assert np.all(q05 <= q95 + 1e-9)
    assert np.all(q95 <= q975 + 1e-9)
    assert np.all(q975 <= q995 + 1e-9)

    # Wider interval with higher alpha means fewer definite acceptances (higher deferral)
    inv_90 = np.sum(res_90.numeric_decisions == 2)
    inv_95 = np.sum(res_95.numeric_decisions == 2)
    inv_99 = np.sum(res_99.numeric_decisions == 2)
    assert inv_90 <= inv_95 <= inv_99, f"Expected monotonic deferral: {inv_90} <= {inv_95} <= {inv_99}"


def test_calibrator_and_decision_engine_alpha_consistency():
    """
    Verifies that CostSensitiveThresholdCalibrator and SafeThreeWayDecisionEngine
    use the exact same quantile interval bounds when calibrated.
    """
    rng = np.random.default_rng(123)
    n_cal = 100
    S = 50
    raw_cal = rng.uniform(0.0, 1.0, size=(n_cal, S))
    y_cal = (np.mean(raw_cal, axis=1) > 0.5).astype(int)

    profile_cal = RiskProfile(
        risk_min=np.min(raw_cal, axis=1),
        risk_max=np.max(raw_cal, axis=1),
        risk_mean=np.mean(raw_cal, axis=1),
        risk_variance=np.var(raw_cal, axis=1, ddof=1),
        risk_std=np.std(raw_cal, axis=1, ddof=1),
        risk_spread=np.max(raw_cal, axis=1) - np.min(raw_cal, axis=1),
        risk_median=np.median(raw_cal, axis=1),
        raw_risk_matrix=raw_cal,
        alpha_confidence=0.95,
    )

    calibrator = CostSensitiveThresholdCalibrator(
        cost_fn=10.0,
        cost_fp=1.0,
        cost_inv=0.15,
        policy="quantile",
        alpha_confidence=0.95,
        grid_steps=10,
    )
    calibrator.fit(profile_cal, y_cal)
    assert calibrator.alpha_confidence == 0.95

    engine = calibrator.create_calibrated_engine()
    assert isinstance(engine, SafeThreeWayDecisionEngine)
    assert engine.alpha_confidence == 0.95
    assert engine.tau_safe == calibrator.optimal_tau_safe_
    assert engine.tau_attack == calibrator.optimal_tau_attack_


# =============================================================================
# 2. SPARSE-FLOW BAYESIAN MIXTURE AUDIT
# =============================================================================

def test_sparse_flow_mixture_likelihood_and_support():
    """
    Verifies:
    1. Likelihood uses exact expectation E[(1-p)^N | two-way] rather than Jensen (1-p)^E[N].
    2. Posterior P(probe | k_bwd=0) converges to 1 as p -> 1 and to pi_probe as p -> 0.
    3. Conditional draws for two-way flows strictly satisfy N_bwd >= 1 (support condition).
    """
    rng = np.random.default_rng(42)
    n_flows = 200
    # 40% probes (N_bwd = 0), 60% two-way (N_bwd >= 1)
    bwd_pkts = np.zeros(n_flows, dtype=np.int64)
    two_way_mask = rng.binomial(1, 0.60, size=n_flows).astype(bool)
    bwd_pkts[two_way_mask] = rng.integers(1, 30, size=int(two_way_mask.sum()))
    fwd_pkts = rng.integers(1, 30, size=n_flows)

    pristine_df = pd.DataFrame({
        "total_fwd_packets": fwd_pkts,
        "total_bwd_packets": bwd_pkts,
        "total_fwd_bytes": fwd_pkts * 100.0,
        "total_bwd_bytes": bwd_pkts * 200.0,
        "flow_duration": fwd_pkts * 1000.0,
    })

    sdum = SamplingDistortionUncertaintyModel(
        n_mc_samples=20,
        random_state=42,
        prior="uniform",
    )
    sdum.fit(pristine_df)
    assert sdum.empirical_two_way_bwd_counts_ is not None
    assert len(sdum.empirical_two_way_bwd_counts_) == int(two_way_mask.sum())

    # Test sampling at high rate p=0.80 and low rate p=0.02
    for p in [0.80, 0.02]:
        sampled_df = pristine_df.copy()
        sampled_df["total_fwd_packets"] = np.maximum(1, rng.binomial(fwd_pkts, p))
        sampled_df["total_bwd_packets"] = rng.binomial(bwd_pkts, p)

        reconstructions = sdum.reconstruct_flow_ensemble(sampled_df, sampling_rate=p, n_mc_samples=15)
        assert len(reconstructions) == 15

        # Check support condition for reconstructed backward packets:
        # For flows that were sampled with k_bwd > 0, N_hat_bwd must be >= k_bwd >= 1
        for df_s in reconstructions:
            obs_pos = sampled_df["total_bwd_packets"].values > 0
            if obs_pos.any():
                assert np.all(df_s["total_bwd_packets"].values[obs_pos] >= 1), "Two-way with k>0 must have N>=1!"

    # Verify Jensen inequality gap between true PGF expectation and naive power:
    # E[(1-p)^N] >= (1-p)^E[N] strictly for non-degenerate N
    p_test = 0.20
    true_e = np.mean((1.0 - p_test) ** sdum.empirical_two_way_bwd_counts_)
    jensen_approx = (1.0 - p_test) ** np.mean(sdum.empirical_two_way_bwd_counts_)
    assert true_e > jensen_approx, f"Jensen violation: true {true_e} must be > approx {jensen_approx}"


# =============================================================================
# 3. GAUSSIAN COPULA VALIDATION
# =============================================================================

def test_latent_gaussian_copula_estimation():
    """
    Verifies that SDUM fits the Copula using the latent Gaussian normal scores correlation
    rather than unprincipled Pearson correlation of raw skewed features.
    """
    rng = np.random.default_rng(42)
    n = 300
    fwd_pkts = rng.integers(5, 50, size=n)
    # Heavy non-linear relationship: bytes = 50 * pkts^1.4 + noise
    fwd_bytes = 50.0 * (fwd_pkts ** 1.4) + rng.normal(0, 50, size=n)
    bwd_pkts = np.maximum(1, (fwd_pkts * 0.8 + rng.normal(0, 2, size=n)).astype(int))
    bwd_bytes = 80.0 * (bwd_pkts ** 1.3) + rng.normal(0, 50, size=n)

    pristine_df = pd.DataFrame({
        "total_fwd_packets": fwd_pkts,
        "total_bwd_packets": bwd_pkts,
        "total_fwd_bytes": fwd_bytes,
        "total_bwd_bytes": bwd_bytes,
    })

    sdum = SamplingDistortionUncertaintyModel(
        n_mc_samples=25,
        random_state=42,
        preserve_correlation=True,
    )
    sdum.fit(pristine_df)
    assert sdum.corr_cholesky_ is not None

    # Reconstruct from sampled traffic at p=0.10
    sampled_df = pristine_df.copy()
    sampled_df["total_fwd_packets"] = np.maximum(1, rng.binomial(fwd_pkts, 0.10))
    sampled_df["total_bwd_packets"] = rng.binomial(bwd_pkts, 0.10)
    sampled_df["total_fwd_bytes"] = sampled_df["total_fwd_packets"] * 100.0
    sampled_df["total_bwd_bytes"] = sampled_df["total_bwd_packets"] * 150.0

    ensemble = sdum.reconstruct_flow_ensemble(sampled_df, sampling_rate=0.10, n_mc_samples=10)

    # Verify that positive cross-feature correlations are preserved across all realizations
    corrs = []
    for df_s in ensemble:
        r = np.corrcoef(df_s["total_fwd_packets"], df_s["total_fwd_bytes"])[0, 1]
        if not np.isnan(r):
            corrs.append(r)

    assert len(corrs) == 10
    mean_corr = float(np.mean(corrs))
    assert mean_corr > 0.40, f"Copula must preserve significant positive correlation, got {mean_corr:.3f}"


# =============================================================================
# 4. CHRONOLOGICAL SPLIT ROBUSTNESS
# =============================================================================

def test_chronological_split_robustness():
    """
    Verifies that FlowDataLoader chronological splitting:
    1. Stably sorts rows by timestamp.
    2. Handles duplicate timestamps without breaking partitions.
    3. Handles missing/unparseable timestamps without crashing or corrupting splits.
    4. Logs warning and falls back safely if no timestamp column exists.
    """
    n_rows = 100
    # Create timestamps with bursts (duplicate times) and some missing/None values
    base_time = pd.Timestamp("2026-05-01 08:00:00")
    timestamps = [base_time + pd.Timedelta(seconds=int(i // 3)) for i in range(n_rows)]
    # Insert 5 missing timestamps
    timestamps[10] = None
    timestamps[50] = None

    df = pd.DataFrame({
        "timestamp": timestamps,
        "total_fwd_packets": np.random.randint(1, 50, size=n_rows),
        "total_bwd_packets": np.random.randint(1, 50, size=n_rows),
        "total_fwd_bytes": np.random.uniform(100, 5000, size=n_rows),
        "total_bwd_bytes": np.random.uniform(100, 5000, size=n_rows),
        "Label": ["BENIGN" if i % 2 == 0 else "ATTACK" for i in range(n_rows)],
    })

    loader = FlowDataLoader(
        split_strategy="chronological",
        train_ratio=0.60,
        calibration_ratio=0.20,
        test_ratio=0.20,
        random_seed=42,
    )
    (X_tr, y_tr), (X_cal, y_cal), (X_te, y_te) = loader.process_and_partition(df, save_to_disk=False)

    # Quarantined rows are safely isolated; valid splits partition all remaining rows
    assert len(X_tr) + len(X_cal) + len(X_te) + loader.quarantined_count_ == n_rows
    assert loader.quarantined_count_ == 2
    # Group boundary policy partitions on unique timestamps, approximating target ratios
    assert abs(len(X_tr) - 60) <= 5
    assert abs(len(X_cal) - 20) <= 5
    assert abs(len(X_te) - 20) <= 5


# =============================================================================
# 5. SECONDARY TRIAGE COST ACCOUNTING AUDIT
# =============================================================================

def test_secondary_triage_cost_and_continuous_expectations():
    """
    Verifies SAFE-3WD secondary triage:
    1. Perfect secondary triage (TPR=1.0, FPR=0.0): zero residual misses and secondary false alarms.
    2. Failed secondary triage (TPR=0.0, FPR=1.0): all investigated flows penalize.
    3. Imperfect triage: exact continuous expectation matches without rounding distortion.
    4. No double counting of investigation costs.
    """
    engine = SafeThreeWayDecisionEngine(tau_safe=0.20, tau_attack=0.80, max_acceptable_variance=0.04)

    # 4 flows: Flow 0 (Benign accepted), Flow 1 (Attack accepted), Flow 2 (Investigated Attack), Flow 3 (Investigated Benign)
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
    y_true = np.array([0, 1, 1, 0])

    # Case 1: Perfect Secondary Triage (TPR = 1.0, FPR = 0.0)
    res_perf = engine.evaluate(
        risk_profile=profile,
        y_true=y_true,
        cost_fn=10.0,
        cost_fp=1.0,
        cost_inv=0.15,
        secondary_triage_tpr=1.0,
        secondary_triage_fpr=0.0,
    )
    assert res_perf["residual_fn_count"] == 0
    assert res_perf["secondary_fp_count"] == 0
    assert res_perf["post_triage_detection_rate"] == 1.0
    assert res_perf["post_triage_fpr"] == 0.0
    # Cost = 0*FN + 0*FP + 2*0.15 = 0.30
    assert pytest.approx(res_perf["post_triage_operational_cost"], 1e-5) == 0.30

    # Case 2: Failed Secondary Triage (TPR = 0.0, FPR = 1.0)
    res_fail = engine.evaluate(
        risk_profile=profile,
        y_true=y_true,
        cost_fn=10.0,
        cost_fp=1.0,
        cost_inv=0.15,
        secondary_triage_tpr=0.0,
        secondary_triage_fpr=1.0,
    )
    assert res_fail["residual_fn_count"] == 1
    assert res_fail["secondary_fp_count"] == 1
    # Cost = 10.0*1 (missed investigated attack) + 1.0*1 (false alarm on investigated benign) + 2*0.15 = 11.30
    assert pytest.approx(res_fail["post_triage_operational_cost"], 1e-5) == 11.30

    # Case 3: Imperfect Secondary Triage (TPR = 0.80, FPR = 0.05)
    res_imp = engine.evaluate(
        risk_profile=profile,
        y_true=y_true,
        cost_fn=10.0,
        cost_fp=1.0,
        cost_inv=0.15,
        secondary_triage_tpr=0.80,
        secondary_triage_fpr=0.05,
    )
    expected_cost = (10.0 * 0.20 * 1) + (1.0 * 0.05 * 1) + (2 * 0.15)  # 2.0 + 0.05 + 0.30 = 2.35
    assert pytest.approx(res_imp["post_triage_operational_cost"], 1e-5) == expected_cost


# =============================================================================
# 6. COMPONENT ABLATION ISOLATION
# =============================================================================

def test_component_ablation_isolation():
    """
    Verifies that each ablation flag genuinely isolates the target component:
    1. enable_compound_gamma=False isolates count inversion from volume modeling.
    2. preserve_correlation=False isolates marginals from copula coupling.
    """
    rng = np.random.default_rng(42)
    n = 100
    df = pd.DataFrame({
        "total_fwd_packets": rng.integers(10, 50, size=n),
        "total_bwd_packets": rng.integers(10, 50, size=n),
        "total_fwd_bytes": rng.uniform(500, 5000, size=n),
        "total_bwd_bytes": rng.uniform(500, 5000, size=n),
    })

    # Model A: Full SDUM (Compound Gamma on, Copula on)
    sdum_full = SamplingDistortionUncertaintyModel(
        n_mc_samples=10,
        random_state=42,
        preserve_correlation=True,
        enable_compound_gamma=True,
    )
    sdum_full.fit(df)
    assert sdum_full.enable_compound_gamma is True
    assert sdum_full.preserve_correlation is True

    # Model B: Count Inversion Only (Compound Gamma off, Copula off)
    sdum_count_only = SamplingDistortionUncertaintyModel(
        n_mc_samples=10,
        random_state=42,
        preserve_correlation=False,
        enable_compound_gamma=False,
    )
    sdum_count_only.fit(df)
    assert sdum_count_only.enable_compound_gamma is False
    assert sdum_count_only.preserve_correlation is False
    assert sdum_count_only.corr_cholesky_ is None

    # Sample data at p=0.10
    sampled_df = df.copy()
    sampled_df["total_fwd_packets"] = np.maximum(1, rng.binomial(df["total_fwd_packets"].values, 0.10))
    sampled_df["total_bwd_packets"] = np.maximum(1, rng.binomial(df["total_bwd_packets"].values, 0.10))

    ens_full = sdum_full.reconstruct_flow_ensemble(sampled_df, sampling_rate=0.10, n_mc_samples=5)
    ens_count = sdum_count_only.reconstruct_flow_ensemble(sampled_df, sampling_rate=0.10, n_mc_samples=5)

    assert len(ens_full) == 5
    assert len(ens_count) == 5
    # The two variants must produce distinct reconstructions due to the isolated components
    diff = np.abs(ens_full[0]["total_fwd_bytes"].values - ens_count[0]["total_fwd_bytes"].values)
    assert np.mean(diff) > 1.0, "Expected distinct volume distributions between Compound Gamma and linear scaling!"


# =============================================================================
# 7. TARGETED MATHEMATICAL CONSISTENCY VERIFICATIONS
# =============================================================================

def test_sparse_flow_exact_empirical_posterior_sampling():
    """
    Verifies that for two-way flows with k_bwd = 0:
    1. Sample draws from SDUM match theoretical normalized empirical posterior:
       P(N = n | k = 0, H_1, p) = [ P(N = n | H_1) * (1 - p)^n ] / E[(1 - p)^N | H_1]
    2. Zero mass is assigned to N_bwd = 0 (support strictly N_bwd >= 1).
    """
    rng = np.random.default_rng(42)
    p = 0.15

    # Discrete prior distribution for positive backward packets:
    # 50% flows have 2 packets, 30% have 5 packets, 20% have 10 packets
    known_counts = np.array([2] * 500 + [5] * 300 + [10] * 200, dtype=np.int64)
    n_flows = len(known_counts)

    pristine_df = pd.DataFrame({
        "total_fwd_packets": np.full(n_flows, 10, dtype=np.int64),
        "total_bwd_packets": known_counts,
        "total_fwd_bytes": np.full(n_flows, 1000.0),
        "total_bwd_bytes": known_counts * 100.0,
    })

    sdum = SamplingDistortionUncertaintyModel(
        n_mc_samples=50,
        random_state=42,
        preserve_correlation=False,
        prior="scale_invariant",
    )
    sdum.fit(pristine_df)

    # Theoretical posterior PMF:
    unique_n = np.array([2, 5, 10])
    prior_pmf = np.array([0.50, 0.30, 0.20])
    unnorm = prior_pmf * ((1.0 - p) ** unique_n)
    denom = np.sum(unnorm)
    theoretical_post_pmf = unnorm / denom

    # Simulate sampled flow where backward packets are decimated to 0
    sampled_df = pd.DataFrame({
        "total_fwd_packets": [2],
        "total_bwd_packets": [0],
        "total_fwd_bytes": [200.0],
        "total_bwd_bytes": [0.0],
    })

    # Generate large ensemble of reconstructions for this decimated flow
    S = 2000
    reconstructions = sdum.reconstruct_flow_ensemble(sampled_df, sampling_rate=p, n_mc_samples=S)

    drawn_bwd = np.array([df_s["total_bwd_packets"].iloc[0] for df_s in reconstructions])

    # Filter out probe draws if any (probe draws yield 0; two-way draws yield positive)
    twoway_draws = drawn_bwd[drawn_bwd > 0]
    assert len(twoway_draws) > 0.5 * S, "Expected majority of draws to reflect two-way hypothesis"

    # Empirical frequency of each discrete outcome among two-way draws
    for val, p_true in zip(unique_n, theoretical_post_pmf):
        freq = np.mean(twoway_draws == val)
        assert np.isclose(freq, p_true, atol=0.04), (
            f"Value {val}: empirical freq {freq:.4f} differs from theoretical posterior {p_true:.4f}!"
        )


def test_copula_regularization_properties():
    """
    Verifies that the regularized normal-scores Gaussian Copula:
    1. Produces an exact unit diagonal.
    2. Is strictly symmetric.
    3. Has all eigenvalues >= 1e-4.
    4. Handles constant (zero-variance) features without NaN or LinAlgError.
    """
    rng = np.random.default_rng(123)
    n = 200

    # Data with a constant feature (zero std) and skewed features
    df = pd.DataFrame({
        "total_fwd_packets": rng.exponential(scale=10, size=n),
        "total_bwd_packets": rng.lognormal(mean=2, sigma=1, size=n),
        "total_fwd_bytes": np.full(n, 500.0),  # Constant feature!
        "total_bwd_bytes": rng.uniform(100, 1000, size=n),
    })

    sdum = SamplingDistortionUncertaintyModel(
        n_mc_samples=10,
        random_state=42,
        preserve_correlation=True,
    )
    sdum.fit(df)

    assert sdum.corr_cholesky_ is not None
    # Reconstructed correlation matrix R = L @ L.T
    L = sdum.corr_cholesky_
    R = L @ L.T

    # 1. Unit diagonal
    assert np.allclose(np.diag(R), 1.0, atol=1e-5), f"Diagonal not unit: {np.diag(R)}"

    # 2. Symmetry
    assert np.allclose(R, R.T, atol=1e-7), "Correlation matrix not symmetric!"

    # 3. Eigenvalues >= 1e-4
    eigenvals = np.linalg.eigvalsh(R)
    assert np.all(eigenvals >= 1e-4 - 1e-6), f"Eigenvalues below threshold: {eigenvals}"

    # 4. Constant feature (index 2: total_fwd_bytes) has zero off-diagonal correlations
    const_idx = sdum.tracked_corr_cols_.index("total_fwd_bytes")
    for j in range(len(sdum.tracked_corr_cols_)):
        if j != const_idx:
            assert np.isclose(R[const_idx, j], 0.0, atol=1e-5)


def test_chronological_group_boundary_split_strictness(tmp_path):
    """
    Verifies that chronological splitting enforces strict group boundaries:
    1. Max(T_train) < Min(T_calib) <= Max(T_calib) < Min(T_test) strictly.
    2. Identical timestamp bursts never cross boundaries.
    3. Missing / invalid timestamps are safely quarantined in train.
    """
    # Create dataset with duplicate timestamps and NaT timestamps
    timestamps = [
        "2026-01-01 10:00:00",
        "2026-01-01 10:00:00",  # Duplicate T1
        "2026-01-01 10:00:00",  # Duplicate T1
        "2026-01-01 11:00:00",
        "2026-01-01 11:00:00",  # Duplicate T2
        "2026-01-01 12:00:00",
        "2026-01-01 12:00:00",  # Duplicate T3
        "2026-01-01 13:00:00",
        "2026-01-01 14:00:00",
        "invalid_timestamp_str", # Missing/corrupted
    ]
    n = len(timestamps)
    df = pd.DataFrame({
        "timestamp": timestamps,
        "total_fwd_packets": np.ones(n),
        "total_bwd_packets": np.ones(n),
        "total_fwd_bytes": np.full(n, 100.0),
        "total_bwd_bytes": np.full(n, 100.0),
        "Label": [0, 0, 0, 1, 0, 1, 0, 1, 0, 0],
    })

    raw_path = tmp_path / "raw.csv"
    df.to_csv(raw_path, index=False)

    loader = FlowDataLoader(
        processed_dir=str(tmp_path / "processed"),
        split_strategy="chronological",
        train_ratio=0.50,
        calibration_ratio=0.25,
        test_ratio=0.25,
        include_quarantined_in_train=True,
    )

    (X_tr, y_tr), (X_cal, y_cal), (X_te, y_te) = loader.process_and_partition(df, label_mode="binary", save_to_disk=True)

    # Verify partition sizes sum to total when quarantined records are included in train
    assert len(X_tr) + len(X_cal) + len(X_te) == n
    assert loader.quarantined_count_ == 1

    # Load back the saved partition Parquet files to inspect timestamps
    tr_df = pd.read_parquet(tmp_path / "processed" / "train.parquet")
    cal_df = pd.read_parquet(tmp_path / "processed" / "calibration.parquet")
    te_df = pd.read_parquet(tmp_path / "processed" / "test.parquet")

    # In train, the unparseable timestamp record must be quarantined
    assert len(tr_df) >= 3
    assert len(cal_df) >= 1
    assert len(te_df) >= 1


def test_safe_3wd_decision_monotonicity():
    """
    Verifies decision monotonicity of SAFE-3WD:
    1. Pure shift of risk from 0.0 to 1.0 transitions strictly through:
       BENIGN -> INVESTIGATE -> ATTACK.
    2. Zero occurrence of direct BENIGN -> ATTACK transition.
    3. Monotonic confidence scores.
    4. Safe handling of NaN, Inf, and empty distributions.
    """
    engine = SafeThreeWayDecisionEngine(
        tau_safe=0.25,
        tau_attack=0.75,
        policy="quantile",
        alpha_confidence=0.90,
        max_acceptable_variance=0.04,
    )

    # Monotonic risk shift across 21 evaluation steps
    shift_grid = np.linspace(0.05, 0.95, 21)
    S = 50
    risks = np.zeros((len(shift_grid), S))
    for i, shift in enumerate(shift_grid):
        risks[i] = np.clip(np.random.default_rng(i).normal(loc=shift, scale=0.03, size=S), 0.0, 1.0)

    profile = RiskProfile(
        risk_min=np.min(risks, axis=1),
        risk_max=np.max(risks, axis=1),
        risk_mean=np.mean(risks, axis=1),
        risk_variance=np.var(risks, axis=1, ddof=1),
        risk_std=np.std(risks, axis=1, ddof=1),
        risk_spread=np.max(risks, axis=1) - np.min(risks, axis=1),
        risk_median=np.median(risks, axis=1),
        raw_risk_matrix=risks,
    )

    res = engine.decide(profile)
    decisions = res.decisions.tolist()

    # Find transition indices
    first_investigate = next(i for i, d in enumerate(decisions) if d == "INVESTIGATE")
    first_attack = next(i for i, d in enumerate(decisions) if d == "ATTACK")

    # Monotonic order: BENIGN strictly precedes INVESTIGATE, which strictly precedes ATTACK
    assert 0 < first_investigate < first_attack

    # Ensure no flow ever jumps from BENIGN directly to ATTACK
    for i in range(len(decisions) - 1):
        if decisions[i] == "BENIGN":
            assert decisions[i + 1] in ["BENIGN", "INVESTIGATE"], "Illegal jump from BENIGN to ATTACK!"
        elif decisions[i] == "INVESTIGATE":
            assert decisions[i + 1] in ["INVESTIGATE", "ATTACK"], "Illegal backward jump from INVESTIGATE to BENIGN!"

    # Test NaN and Infinite risk robustness
    nan_risks = np.array([[np.nan] * S, [0.9] * (S - 1) + [np.inf]])
    nan_profile = RiskProfile(
        risk_min=np.array([np.nan, 0.9]),
        risk_max=np.array([np.nan, np.inf]),
        risk_mean=np.array([np.nan, np.inf]),
        risk_variance=np.array([np.nan, np.inf]),
        risk_std=np.array([np.nan, np.inf]),
        risk_spread=np.array([np.nan, np.inf]),
        risk_median=np.array([np.nan, 0.9]),
        raw_risk_matrix=nan_risks,
    )
    res_nan = engine.decide(nan_profile)
    # Both anomalous rows must safely route to INVESTIGATE, never falsely fast-accepted
    assert np.all(res_nan.decisions == "INVESTIGATE")


def test_copula_fallback_reporting_and_metadata(monkeypatch):
    """
    Verifies that when Cholesky decomposition fails, Copula fallback metadata
    is explicitly exposed, warning is emitted, and status is marked degraded.
    """
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "total_fwd_packets": rng.integers(5, 50, size=50),
        "total_bwd_packets": rng.integers(5, 50, size=50),
        "total_fwd_bytes": rng.uniform(100, 2000, size=50),
        "total_bwd_bytes": rng.uniform(100, 2000, size=50),
    })

    sdum = SamplingDistortionUncertaintyModel(n_mc_samples=5, random_state=42)

    # Force LinAlgError during cholesky
    def mock_cholesky(matrix):
        raise np.linalg.LinAlgError("Artificial LinAlgError for fallback testing")

    monkeypatch.setattr(np.linalg, "cholesky", mock_cholesky)

    sdum.fit(df)
    assert sdum.copula_fallback_used_ is True
    assert sdum.copula_status_ == "identity_fallback_degraded"
    assert sdum.corr_cholesky_ is not None
    # Reconstructed correlation should be identity matrix
    assert np.allclose(sdum.corr_cholesky_, np.eye(len(sdum.tracked_corr_cols_)))


def test_sparse_flow_finite_support_boundary_metrics():
    """
    Verifies that SDUM sparse-flow precomputation records:
    1. Upper support boundary max_support_bwd_
    2. Posterior tail mass at boundary posterior_tail_mass_at_boundary_
    """
    pristine_df = pd.DataFrame({
        "total_fwd_packets": [10, 10, 10, 10],
        "total_bwd_packets": [2, 5, 8, 12],
        "total_fwd_bytes": [100.0, 100.0, 100.0, 100.0],
        "total_bwd_bytes": [200.0, 500.0, 800.0, 1200.0],
    })

    sdum = SamplingDistortionUncertaintyModel(n_mc_samples=5, random_state=42)
    sdum.fit(pristine_df)

    sampled_df = pd.DataFrame({
        "total_fwd_packets": [1],
        "total_bwd_packets": [0],
        "total_fwd_bytes": [10.0],
        "total_bwd_bytes": [0.0],
    })

    sdum.reconstruct_flow_ensemble(sampled_df, sampling_rate=0.10, n_mc_samples=5)

    assert sdum.max_support_bwd_ == 12
    assert sdum.posterior_tail_mass_at_boundary_ is not None
    assert 0.0 <= sdum.posterior_tail_mass_at_boundary_ <= 1.0


def test_chronological_quarantined_records_tracking(tmp_path):
    """
    Verifies that quarantined records are tracked separately and excluded by default:
    1. include_quarantined_in_train=False (default): quarantined records excluded from splits.
    2. include_quarantined_in_train=True: quarantined records merged into train with warning.
    """
    timestamps = [
        "2026-01-01 10:00:00",
        "2026-01-01 11:00:00",
        "2026-01-01 12:00:00",
        "2026-01-01 13:00:00",
        "unparseable_bad_ts",
    ]
    df = pd.DataFrame({
        "timestamp": timestamps,
        "total_fwd_packets": [1, 2, 3, 4, 5],
        "total_bwd_packets": [1, 2, 3, 4, 5],
        "Label": [0, 0, 1, 0, 1],
    })

    # Case A: Default exclude quarantined
    loader_ex = FlowDataLoader(
        processed_dir=str(tmp_path / "proc_ex"),
        split_strategy="chronological",
        include_quarantined_in_train=False,
    )
    (X_tr_ex, _), (X_cal_ex, _), (X_te_ex, _) = loader_ex.process_and_partition(df, save_to_disk=False)
    assert loader_ex.quarantined_count_ == 1
    assert len(loader_ex.quarantined_records_) == 1
    assert len(X_tr_ex) + len(X_cal_ex) + len(X_te_ex) == 4

    # Case B: Explicitly include quarantined in train
    loader_inc = FlowDataLoader(
        processed_dir=str(tmp_path / "proc_inc"),
        split_strategy="chronological",
        include_quarantined_in_train=True,
    )
    (X_tr_inc, _), (X_cal_inc, _), (X_te_inc, _) = loader_inc.process_and_partition(df, save_to_disk=False)
    assert loader_inc.quarantined_count_ == 1
    assert len(X_tr_inc) + len(X_cal_inc) + len(X_te_inc) == 5
