r"""
FlowGuard Sampling Distortion & Uncertainty Model (SDUM)
========================================================

Mathematically inverts the packet sampling and decimation process to reconstruct
an ensemble of S plausible true flows {X_hat^(1), ..., X_hat^(S)} per observed flow.

Mathematical Foundations & Posterior Derivations:
-------------------------------------------------
Given true flow packet count N and binomial sampling probability p in (0, 1]:
  K | N ~ Binomial(N, p)
  P(K = k | N = k + u) = binom(k + u, k) * p^k * (1 - p)^u, where U = N - k >= 0.

By Bayes' theorem, the posterior of unobserved packets U given observed k packets is:
1. Under scale-invariant / Haldane prior P(N) \propto 1 / N (default, prior="scale_invariant"):
   P(U = u | k, p) \propto binom(k + u, k) * p^k * (1 - p)^u * (1 / (k + u))
                  = (1 / k) * binom(u + k - 1, u) * p^k * (1 - p)^u
   Normalizing over u >= 0:
     \sum_{u=0}^\infty binom(u + k - 1, u) * (1 - p)^u = p^(-k)
   Hence:
     P(U = u | k, p) = binom(u + k - 1, u) * p^k * (1 - p)^u  ==>  U ~ NegBin(k, p)
   Posterior moments:
     E[U | k, p] = k * (1 - p) / p
     E[N | k, p] = k + E[U] = k / p
     Var(N | k, p) = Var(U | k, p) = k * (1 - p) / (p^2)

2. Under improper uniform prior P(N) \propto 1 (prior="uniform"):
   P(U = u | k, p) \propto binom(k + u, k) * (1 - p)^u = binom(u + (k + 1) - 1, u) * (1 - p)^u
   Normalizing over u >= 0:
     \sum_{u=0}^\infty binom(k + u, k) * (1 - p)^u = p^(-(k + 1))
   Hence:
     P(U = u | k, p) = binom(u + (k + 1) - 1, u) * p^(k + 1) * (1 - p)^u  ==>  U ~ NegBin(k + 1, p)
   Posterior moments:
     E[U | k, p] = (k + 1) * (1 - p) / p
     E[N | k, p] = k + E[U] = k / p + (1 - p) / p
     Var(N | k, p) = Var(U | k, p) = (k + 1) * (1 - p) / (p^2)

Both priors are fully supported and mathematically validated.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import scipy.stats as stats
import scipy.special as special

logger = logging.getLogger(__name__)

from flowguard.sampling.distortion_math import (
    binomial_count_moments,
    inverted_count_posterior_moments,
    invert_sampled_iat_moments,
    sampled_iat_moments,
)

# Canonical additive volume features (strictly volume/header counts, never window sizes)
ADDITIVE_BYTE_FEATURES = {
    "total_fwd_bytes",
    "total_bwd_bytes",
    "subflow_fwd_bytes",
    "subflow_bwd_bytes",
    "fwd_header_len",
    "bwd_header_len",
}


class SamplingDistortionUncertaintyModel:
    """
    Sampling Distortion & Uncertainty Model (SDUM).

    Given an observed degraded flow record X_tilde under packet sampling rate p,
    SDUM samples from the analytical posterior distribution P(X | X_tilde, p, Sigma)
    to generate S plausible pristine flow realizations.
    """
    def __init__(
        self,
        n_mc_samples: int = 30,
        random_state: int = 42,
        preserve_correlation: bool = True,
        probe_prior: float = 0.40,
        prior: str = "scale_invariant",
        enable_compound_gamma: bool = True,
    ):
        if prior not in ["scale_invariant", "uniform"]:
            raise ValueError(f"Unknown prior '{prior}'. Must be 'scale_invariant' or 'uniform'.")

        self.n_mc_samples = n_mc_samples
        self.random_state = random_state
        self.preserve_correlation = preserve_correlation
        self.probe_prior = probe_prior
        self.prior = prior
        self.enable_compound_gamma = bool(enable_compound_gamma)
        self.rng = np.random.default_rng(random_state)

        # Learned empirical priors and correlation structures
        self.is_fitted: bool = False
        self.feature_names_: List[str] = []
        self.empirical_medians_: Dict[str, float] = {}
        self.empirical_stds_: Dict[str, float] = {}
        self.corr_cholesky_: Optional[np.ndarray] = None
        self.tracked_corr_cols_: List[str] = []

        # Empirical Bayes latent regime parameters
        self.empirical_probe_prior_: float = float(probe_prior)
        self.empirical_two_way_bwd_counts_: Optional[np.ndarray] = None
        self.empirical_bwd_ratio_median_: float = 0.50
        self.empirical_bwd_ratio_std_: float = 0.15

        # Compound Gamma parameters for packet payload lengths (shape alpha, scale beta)
        self.fwd_pkt_gamma_shape_: float = 2.0
        self.fwd_pkt_gamma_scale_: float = 100.0
        self.bwd_pkt_gamma_shape_: float = 2.0
        self.bwd_pkt_gamma_scale_: float = 200.0

        # Copula fallback and status tracking
        self.copula_fallback_used_: bool = False
        self.copula_status_: str = "not_fitted"
        self.copula_regularized_corr_: Optional[np.ndarray] = None

        # Sparse-flow finite-support and tail attributes
        self.max_support_bwd_: Optional[int] = None
        self.posterior_tail_mass_at_boundary_: Optional[float] = None

    def fit(self, X_pristine: pd.DataFrame) -> "SamplingDistortionUncertaintyModel":
        """
        Learns empirical flow priors, packet length distributions, and correlation structure
        from pristine complete network flows.
        """
        self.feature_names_ = X_pristine.columns.tolist()

        # Compute empirical baseline statistics
        for col in self.feature_names_:
            vals = X_pristine[col].values
            self.empirical_medians_[col] = float(np.nanmedian(vals))
            self.empirical_stds_[col] = float(np.nanstd(vals) + 1e-6)

        # Fit Empirical Bayes latent regime parameters from training data
        if "total_bwd_packets" in X_pristine.columns:
            bwd_vals = X_pristine["total_bwd_packets"].values
            fwd_vals = (
                X_pristine["total_fwd_packets"].values
                if "total_fwd_packets" in X_pristine.columns
                else np.ones_like(bwd_vals)
            )
            # Unidirectional flows (0 backward packets) correspond to scans/probes
            unidir_mask = (bwd_vals == 0)
            self.empirical_probe_prior_ = float(np.mean(unidir_mask))

            # Bidirectional flows: fit backward-to-forward packet ratio and preserve positive counts
            bidir_mask = (bwd_vals > 0)
            if bidir_mask.any():
                self.empirical_two_way_bwd_counts_ = bwd_vals[bidir_mask].astype(np.float64)
                fwd_positive = fwd_vals[bidir_mask]
                ratios = bwd_vals[bidir_mask] / np.maximum(1.0, fwd_positive)
                self.empirical_bwd_ratio_median_ = float(np.median(ratios))
                self.empirical_bwd_ratio_std_ = float(np.std(ratios) + 1e-4)
            else:
                self.empirical_two_way_bwd_counts_ = np.array([5.0])
                self.empirical_bwd_ratio_median_ = 0.50
                self.empirical_bwd_ratio_std_ = 0.15
        else:
            self.empirical_probe_prior_ = float(self.probe_prior)
            self.empirical_two_way_bwd_counts_ = np.array([5.0])
            self.empirical_bwd_ratio_median_ = 0.50
            self.empirical_bwd_ratio_std_ = 0.15

        # Fit Compound Gamma parameters for packet length distributions
        # Mean = shape * scale, Var = shape * scale^2  ==> scale = Var / Mean, shape = Mean / scale
        if "total_fwd_bytes" in X_pristine.columns and "total_fwd_packets" in X_pristine.columns:
            pkts = np.maximum(1.0, X_pristine["total_fwd_packets"].values)
            lens = np.maximum(20.0, X_pristine["total_fwd_bytes"].values / pkts)
            mu_l = float(np.nanmean(lens))
            var_l = float(np.nanvar(lens) + 1.0)
            self.fwd_pkt_gamma_scale_ = max(1.0, var_l / max(1.0, mu_l))
            self.fwd_pkt_gamma_shape_ = max(0.5, mu_l / self.fwd_pkt_gamma_scale_)

        if "total_bwd_bytes" in X_pristine.columns and "total_bwd_packets" in X_pristine.columns:
            bwd_mask = X_pristine["total_bwd_packets"].values > 0
            if bwd_mask.any():
                pkts_b = np.maximum(1.0, X_pristine["total_bwd_packets"].values[bwd_mask])
                lens_b = np.maximum(20.0, X_pristine["total_bwd_bytes"].values[bwd_mask] / pkts_b)
                mu_b = float(np.nanmean(lens_b))
                var_b = float(np.nanvar(lens_b) + 1.0)
                self.bwd_pkt_gamma_scale_ = max(1.0, var_b / max(1.0, mu_b))
                self.bwd_pkt_gamma_shape_ = max(0.5, mu_b / self.bwd_pkt_gamma_scale_)

        # Correlation preservation across core volume and temporal features
        # Focus on features whose marginal inversion uses the Gaussian Copula uniform variates
        candidate_cols = [
            "total_fwd_packets",
            "total_bwd_packets",
            "total_fwd_bytes",
            "total_bwd_bytes",
        ]
        self.tracked_corr_cols_ = [c for c in candidate_cols if c in self.feature_names_]

        if self.preserve_correlation and len(self.tracked_corr_cols_) > 1:
            sub_df = X_pristine[self.tracked_corr_cols_].fillna(0.0)
            n_p = len(sub_df)

            # Semiparametric Latent Gaussian Copula Estimation:
            # Map empirical sample marginals to standard normal scores Z = Phi^{-1}((rank - 0.5) / n).
            # This captures monotone rank dependence invariant to heavy-tailed non-linear distortions.
            p_cols = len(self.tracked_corr_cols_)
            z_scores = np.zeros((n_p, p_cols), dtype=np.float64)
            for j, c_name in enumerate(self.tracked_corr_cols_):
                vals_c = sub_df[c_name].values
                std_c = np.std(vals_c)
                if std_c < 1e-12:
                    z_scores[:, j] = 0.0
                else:
                    ranks_c = stats.rankdata(vals_c)
                    u_ranks_c = np.clip((ranks_c - 0.5) / n_p, 1e-5, 1.0 - 1e-5)
                    z_scores[:, j] = np.sqrt(2.0) * special.erfinv(2.0 * u_ranks_c - 1.0)

            stds = np.std(z_scores, axis=0)
            is_const = stds < 1e-12
            raw_corr = np.eye(p_cols)
            act_idx = np.where(~is_const)[0]
            if len(act_idx) > 1:
                sub_corr = np.corrcoef(z_scores[:, act_idx], rowvar=False)
                for i_s, i_a in enumerate(act_idx):
                    for j_s, j_a in enumerate(act_idx):
                        raw_corr[i_a, j_a] = sub_corr[i_s, j_s]

            # Symmetrize
            sym_corr = 0.5 * (raw_corr + raw_corr.T)
            # Eigenvalue clipping for strict positive definiteness
            eigenvals, eigenvecs = np.linalg.eigh(sym_corr)
            eigenvals = np.maximum(eigenvals, 1e-4)
            proj_corr = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T

            # Rescale back to exact unit diagonal correlation matrix
            d_inv = 1.0 / np.sqrt(np.maximum(1e-8, np.diag(proj_corr)))
            R_unit = proj_corr * np.outer(d_inv, d_inv)
            R_unit = 0.5 * (R_unit + R_unit.T)
            np.fill_diagonal(R_unit, 1.0)

            self.copula_regularized_corr_ = R_unit
            try:
                self.corr_cholesky_ = np.linalg.cholesky(R_unit)
                self.copula_fallback_used_ = False
                self.copula_status_ = "regularized_cholesky"
            except np.linalg.LinAlgError:
                logger.warning(
                    "Gaussian Copula Cholesky decomposition failed; falling back to identity matrix. "
                    "Cross-feature dependence preservation is degraded or unavailable."
                )
                self.corr_cholesky_ = np.eye(len(self.tracked_corr_cols_))
                self.copula_fallback_used_ = True
                self.copula_status_ = "identity_fallback_degraded"
        else:
            self.corr_cholesky_ = None
            self.copula_fallback_used_ = False
            self.copula_status_ = "disabled"

        self.is_fitted = True
        return self

    def reconstruct_flow_ensemble(
        self,
        X_sampled: pd.DataFrame,
        sampling_rate: float,
        n_mc_samples: Optional[int] = None,
    ) -> List[pd.DataFrame]:
        """
        Generates S plausible true flow realizations for each observed flow record.

        Parameters
        ----------
        X_sampled : pd.DataFrame
            Decimated/sampled network flow DataFrame of shape (N, D).
        sampling_rate : float
            Packet sampling probability p in (0, 1].
        n_mc_samples : Optional[int]
            Number of Monte Carlo realizations S (defaults to self.n_mc_samples).

        Returns
        -------
        List[pd.DataFrame]
            List of S reconstructed DataFrames, each of shape (N, D).
        """
        if not self.is_fitted:
            raise RuntimeError("SDUM must be fitted on pristine traffic before reconstruction!")

        p = float(np.clip(sampling_rate, 1e-4, 1.0))
        S = n_mc_samples if n_mc_samples is not None else self.n_mc_samples
        n_rows = len(X_sampled)

        # If complete traffic (p = 1.0), true flow is known exactly with 0 variance
        if p >= 1.0:
            return [X_sampled.copy() for _ in range(S)]

        fwd_col = "total_fwd_packets" if "total_fwd_packets" in X_sampled.columns else None
        bwd_col = "total_bwd_packets" if "total_bwd_packets" in X_sampled.columns else None

        k_fwd = X_sampled[fwd_col].values.astype(np.int64) if fwd_col else np.ones(n_rows, dtype=np.int64)
        k_bwd = X_sampled[bwd_col].values.astype(np.int64) if bwd_col else np.zeros(n_rows, dtype=np.int64)
        k_tot = k_fwd + k_bwd

        sparse_mask = k_tot <= 2
        multi_mask = ~sparse_mask

        # Parameter r for Negative Binomial draws based on chosen prior
        # scale_invariant: r = k; uniform: r = k + 1
        r_offset = 1 if self.prior == "uniform" else 0

        corr_col_idx = {c: i for i, c in enumerate(self.tracked_corr_cols_)} if self.tracked_corr_cols_ else {}
        use_copula = (
            self.preserve_correlation
            and self.corr_cholesky_ is not None
            and len(self.tracked_corr_cols_) > 1
        )

        # Precompute Empirical Bayes Posterior PMF and CDF for sampling rate p
        # P(N = n | k = 0, H_1, p) proportional to P(N = n | H_1) * (1 - p)^n
        # Normalized by E[(1 - p)^N | H_1] = sum_n P(N = n | H_1) * (1 - p)^n
        if self.empirical_two_way_bwd_counts_ is not None and len(self.empirical_two_way_bwd_counts_) > 0:
            unique_bwd, prior_counts = np.unique(self.empirical_two_way_bwd_counts_, return_counts=True)
            prior_pmf = prior_counts / float(len(self.empirical_two_way_bwd_counts_))
            unnorm_weights = prior_pmf * ((1.0 - p) ** unique_bwd)
            e_prob_zero_twoway = float(np.sum(unnorm_weights))
            post_pmf_bwd = unnorm_weights / max(1e-12, e_prob_zero_twoway)
            post_cdf_bwd = np.cumsum(post_pmf_bwd)
            post_cdf_bwd[-1] = 1.0

            # Record upper support boundary and boundary posterior tail mass
            self.max_support_bwd_ = int(unique_bwd[-1])
            self.posterior_tail_mass_at_boundary_ = float(post_pmf_bwd[-1])
            if self.posterior_tail_mass_at_boundary_ > 1e-3:
                logger.debug(
                    f"Sparse-flow posterior tail mass at upper boundary N={self.max_support_bwd_} is "
                    f"{self.posterior_tail_mass_at_boundary_:.5f} (p={p:.3f})."
                )
        else:
            unique_bwd = np.array([1, 2, 3, 5, 10], dtype=np.int64)
            mu_b = max(1.0, self.empirical_bwd_ratio_median_ * self.empirical_medians_.get("total_fwd_packets", 5.0))
            q_geom = 1.0 / mu_b
            e_prob_zero_twoway = float((q_geom * (1.0 - p)) / max(1e-6, 1.0 - (1.0 - q_geom) * (1.0 - p)))
            post_cdf_bwd = None
            self.max_support_bwd_ = 10
            self.posterior_tail_mass_at_boundary_ = 0.0

        e_prob_zero_twoway = float(np.clip(e_prob_zero_twoway, 1e-8, 1.0))
        pi_p = float(np.clip(self.empirical_probe_prior_, 1e-4, 1.0 - 1e-4))
        posterior_probe_prob = (pi_p * 1.0) / (pi_p * 1.0 + (1.0 - pi_p) * e_prob_zero_twoway)
        posterior_probe_prob = float(np.clip(posterior_probe_prob, 0.001, 0.999))

        reconstructed_dfs: List[pd.DataFrame] = []

        # Vectorized draws for S realizations
        for s in range(S):
            df_s = X_sampled.copy()

            # Latent hypothesis for flows with 0 observed backward packets:
            # Z = 1: Flow is a unidirectional probe / scan (true N_bwd = 0)
            # Z = 0: Flow is a two-way session with packets unobserved due to decimation (true N_bwd >= 1)
            z_sparse = np.zeros(n_rows, dtype=bool)
            bwd_zero_all = (k_bwd == 0)
            if bwd_zero_all.any():
                z_sparse[bwd_zero_all] = self.rng.binomial(
                    1, posterior_probe_prob, size=int(bwd_zero_all.sum())
                ).astype(bool)

            # Generate correlated uniform variates via Gaussian Copula if enabled
            if use_copula:
                z_std = self.rng.standard_normal(size=(n_rows, len(self.tracked_corr_cols_)))
                z_corr = z_std @ self.corr_cholesky_.T
                u_corr = np.clip(0.5 * (1.0 + special.erf(z_corr / np.sqrt(2.0))), 1e-4, 1.0 - 1e-4)
            else:
                u_corr = None

            # 1. Packet Count Reconstruction
            n_hat_fwd = np.zeros(n_rows, dtype=np.int64)
            n_hat_bwd = np.zeros(n_rows, dtype=np.int64)
            u_fwd = np.zeros(n_rows, dtype=np.int64)
            u_bwd = np.zeros(n_rows, dtype=np.int64)

            # Multi-packet flows: Negative Binomial draws for unobserved packets U
            if multi_mask.any():
                fwd_n_param = np.maximum(1, k_fwd[multi_mask] + r_offset)
                if use_copula and "total_fwd_packets" in corr_col_idx:
                    idx_c = corr_col_idx["total_fwd_packets"]
                    u_sub = u_corr[multi_mask, idx_c]
                    missed_fwd = stats.nbinom.ppf(u_sub, n=fwd_n_param, p=p).astype(np.int64)
                else:
                    missed_fwd = self.rng.negative_binomial(n=fwd_n_param, p=p)
                u_fwd[multi_mask] = missed_fwd
                n_hat_fwd[multi_mask] = k_fwd[multi_mask] + missed_fwd

                bwd_nonzero = (k_bwd[multi_mask] > 0)
                if bwd_nonzero.any():
                    idx_bwd = np.where(multi_mask)[0][bwd_nonzero]
                    bwd_n_param = np.maximum(1, k_bwd[idx_bwd] + r_offset)
                    if use_copula and "total_bwd_packets" in corr_col_idx:
                        idx_cb = corr_col_idx["total_bwd_packets"]
                        u_sub_b = u_corr[idx_bwd, idx_cb]
                        missed_bwd = stats.nbinom.ppf(u_sub_b, n=bwd_n_param, p=p).astype(np.int64)
                    else:
                        missed_bwd = self.rng.negative_binomial(n=bwd_n_param, p=p)
                    u_bwd[idx_bwd] = missed_bwd
                    n_hat_bwd[idx_bwd] = k_bwd[idx_bwd] + missed_bwd

                # Directional zero-observation handling:
                # When k_bwd == 0 in multi-packet flows, the flow is two-way if z_sparse == False.
                # Its conditional support MUST satisfy N_bwd >= 1.
                bwd_zero = (k_bwd[multi_mask] == 0)
                if bwd_zero.any():
                    idx_bwd_z_all = np.where(multi_mask)[0][bwd_zero]
                    # Filter for two-way flows (z_sparse == False)
                    is_twoway = ~z_sparse[idx_bwd_z_all]
                    if is_twoway.any():
                        idx_bwd_z = idx_bwd_z_all[is_twoway]
                        n_z = len(idx_bwd_z)
                        if post_cdf_bwd is not None and self.prior != "uniform":
                            # Exact Empirical Bayes posterior draw:
                            # P(N = n | k = 0, H_1, p) proportional to P(N = n | H_1) * (1 - p)^n
                            if use_copula and "total_bwd_packets" in corr_col_idx:
                                idx_cb = corr_col_idx["total_bwd_packets"]
                                u_sub_b = u_corr[idx_bwd_z, idx_cb]
                            else:
                                u_sub_b = self.rng.uniform(size=n_z)
                            samp_indices = np.searchsorted(post_cdf_bwd, u_sub_b)
                            n_draws = unique_bwd[np.clip(samp_indices, 0, len(unique_bwd) - 1)]
                            n_hat_bwd[idx_bwd_z] = np.maximum(1, n_draws)
                            u_bwd[idx_bwd_z] = n_hat_bwd[idx_bwd_z]
                        else:
                            # Uniform positive-integer prior: N = 1 + NegBin(1, p)
                            if use_copula and "total_bwd_packets" in corr_col_idx:
                                idx_cb = corr_col_idx["total_bwd_packets"]
                                missed_bwd_z = stats.nbinom.ppf(u_corr[idx_bwd_z, idx_cb], n=1, p=p).astype(np.int64)
                            else:
                                missed_bwd_z = self.rng.negative_binomial(n=1, p=p, size=n_z)
                            n_hat_bwd[idx_bwd_z] = np.maximum(1, 1 + missed_bwd_z)
                            u_bwd[idx_bwd_z] = n_hat_bwd[idx_bwd_z]

            # Sparse flows: Empirical Bayes Mixture draws
            if sparse_mask.any():
                # Case A: Probe hypothesis (Z = 1) -> small true count (preserved or +0-1 packets)
                probe_indices = sparse_mask & z_sparse
                if probe_indices.any():
                    n_hat_fwd[probe_indices] = np.maximum(1, k_fwd[probe_indices])
                    n_hat_bwd[probe_indices] = k_bwd[probe_indices]
                    u_fwd[probe_indices] = n_hat_fwd[probe_indices] - k_fwd[probe_indices]
                    u_bwd[probe_indices] = 0

                # Case B: Decimated session hypothesis (Z = 0) -> true count was substantial
                session_indices = sparse_mask & (~z_sparse)
                if session_indices.any():
                    sess_n_param = 1 + r_offset
                    n_sess = int(session_indices.sum())
                    if use_copula and "total_fwd_packets" in corr_col_idx:
                        idx_c = corr_col_idx["total_fwd_packets"]
                        missed_fwd_sess = stats.nbinom.ppf(
                            u_corr[session_indices, idx_c], n=sess_n_param, p=p
                        ).astype(np.int64)
                    else:
                        missed_fwd_sess = self.rng.negative_binomial(
                            n=sess_n_param, p=p, size=n_sess
                        )
                    n_hat_fwd[session_indices] = np.maximum(
                        2, np.maximum(1, k_fwd[session_indices]) + missed_fwd_sess
                    )
                    u_fwd[session_indices] = n_hat_fwd[session_indices] - k_fwd[session_indices]

                    # Backward packet reconstruction under Z = 0
                    if post_cdf_bwd is not None and self.prior != "uniform":
                        # Exact Empirical Bayes posterior sampling
                        if use_copula and "total_bwd_packets" in corr_col_idx:
                            idx_cb = corr_col_idx["total_bwd_packets"]
                            u_sub_b = u_corr[session_indices, idx_cb]
                        else:
                            u_sub_b = self.rng.uniform(size=n_sess)
                        samp_indices = np.searchsorted(post_cdf_bwd, u_sub_b)
                        n_hat_bwd[session_indices] = np.maximum(
                            1, unique_bwd[np.clip(samp_indices, 0, len(unique_bwd) - 1)]
                        )
                    else:
                        # Empirical Bayes backward packet ratio or uniform prior
                        bwd_ratio_draw = np.clip(
                            self.rng.normal(
                                loc=self.empirical_bwd_ratio_median_,
                                scale=self.empirical_bwd_ratio_std_,
                                size=n_sess,
                            ),
                            0.05,
                            3.0,
                        )
                        n_hat_bwd[session_indices] = np.maximum(
                            1, (n_hat_fwd[session_indices] * bwd_ratio_draw).astype(np.int64)
                        )
                    u_bwd[session_indices] = np.maximum(
                        0, n_hat_bwd[session_indices] - k_bwd[session_indices]
                    )

            if fwd_col:
                df_s[fwd_col] = n_hat_fwd
            if bwd_col:
                df_s[bwd_col] = n_hat_bwd

            n_hat_tot = n_hat_fwd + n_hat_bwd

            # 2. Volume Reconstruction (Compound Gamma vs Deterministic)
            # When enable_compound_gamma=True, added volume Delta_V ~ Gamma(U * alpha, scale),
            # strictly compounding individual packet payload lengths.
            # When enable_compound_gamma=False (ablation mode), volume is rescaled linearly without Gamma variance.
            u_fwd_pos = u_fwd > 0
            delta_bytes_fwd = np.zeros(n_rows, dtype=np.float64)
            if u_fwd_pos.any():
                if self.enable_compound_gamma:
                    shapes_fwd = np.maximum(0.1, u_fwd[u_fwd_pos] * self.fwd_pkt_gamma_shape_)
                    if use_copula and "total_fwd_bytes" in corr_col_idx:
                        idx_cf = corr_col_idx["total_fwd_bytes"]
                        delta_bytes_fwd[u_fwd_pos] = stats.gamma.ppf(
                            u_corr[u_fwd_pos, idx_cf],
                            a=shapes_fwd,
                            scale=self.fwd_pkt_gamma_scale_,
                        )
                    else:
                        delta_bytes_fwd[u_fwd_pos] = self.rng.gamma(
                            shape=shapes_fwd,
                            scale=self.fwd_pkt_gamma_scale_,
                        )
                else:
                    # Ablation baseline: linear rescaling without Gamma stochastic compounding
                    obs_fwd_b = df_s["total_fwd_bytes"].values[u_fwd_pos] if "total_fwd_bytes" in df_s.columns else 0.0
                    mean_len_fwd = obs_fwd_b / np.maximum(1, k_fwd[u_fwd_pos])
                    delta_bytes_fwd[u_fwd_pos] = u_fwd[u_fwd_pos] * mean_len_fwd

            u_bwd_pos = u_bwd > 0
            delta_bytes_bwd = np.zeros(n_rows, dtype=np.float64)
            if u_bwd_pos.any():
                if self.enable_compound_gamma:
                    shapes_bwd = np.maximum(0.1, u_bwd[u_bwd_pos] * self.bwd_pkt_gamma_shape_)
                    if use_copula and "total_bwd_bytes" in corr_col_idx:
                        idx_cb = corr_col_idx["total_bwd_bytes"]
                        delta_bytes_bwd[u_bwd_pos] = stats.gamma.ppf(
                            u_corr[u_bwd_pos, idx_cb],
                            a=shapes_bwd,
                            scale=self.bwd_pkt_gamma_scale_,
                        )
                    else:
                        delta_bytes_bwd[u_bwd_pos] = self.rng.gamma(
                            shape=shapes_bwd,
                            scale=self.bwd_pkt_gamma_scale_,
                        )
                else:
                    # Ablation baseline: linear rescaling without Gamma stochastic compounding
                    obs_bwd_b = df_s["total_bwd_bytes"].values[u_bwd_pos] if "total_bwd_bytes" in df_s.columns else 0.0
                    mean_len_bwd = obs_bwd_b / np.maximum(1, k_bwd[u_bwd_pos])
                    delta_bytes_bwd[u_bwd_pos] = u_bwd[u_bwd_pos] * mean_len_bwd

            # Apply volume updates strictly to additive volume features (excluding window sizes)
            for col in df_s.columns:
                col_l = col.lower()
                # STRICT check: Never scale TCP initial window sizes (init_win_bytes_fwd/bwd)
                if "init_win" in col_l or "win_bytes" in col_l:
                    continue

                if col in ADDITIVE_BYTE_FEATURES or ("subflow" in col_l and "bytes" in col_l):
                    if "fwd" in col_l:
                        df_s[col] = np.maximum(0.0, df_s[col].values + delta_bytes_fwd)
                    elif "bwd" in col_l:
                        df_s[col] = np.maximum(0.0, df_s[col].values + delta_bytes_bwd)
                elif "header_len" in col_l:
                    if "fwd" in col_l:
                        df_s[col] = np.maximum(20.0, n_hat_fwd * 20.0)
                    elif "bwd" in col_l:
                        df_s[col] = np.maximum(0.0, n_hat_bwd * 20.0)

            # 3. Flow Duration Reconstruction
            if "flow_duration" in df_s.columns:
                obs_dur = X_sampled["flow_duration"].values
                multi_pkt_obs = k_tot > 1
                single_pkt_obs = ~multi_pkt_obs

                if multi_pkt_obs.any():
                    dur_scale = (n_hat_tot[multi_pkt_obs] - 1.0) / np.maximum(1.0, k_tot[multi_pkt_obs] - 1.0)
                    jitter = np.exp(self.rng.normal(0.0, 0.15, size=int(multi_pkt_obs.sum())))
                    df_s.loc[multi_pkt_obs, "flow_duration"] = np.maximum(
                        10.0, obs_dur[multi_pkt_obs] * dur_scale * jitter
                    )

                if single_pkt_obs.any():
                    # For probes: short duration
                    probe_single = single_pkt_obs & z_sparse
                    if probe_single.any():
                        df_s.loc[probe_single, "flow_duration"] = self.rng.uniform(
                            100.0, 10_000.0, size=int(probe_single.sum())
                        )
                    # For decimated sessions: typical session duration
                    sess_single = single_pkt_obs & (~z_sparse)
                    if sess_single.any():
                        prior_median = self.empirical_medians_.get("flow_duration", 500_000.0)
                        df_s.loc[sess_single, "flow_duration"] = np.clip(
                            self.rng.lognormal(mean=np.log(max(100.0, prior_median)), sigma=1.0, size=int(sess_single.sum())),
                            50_000.0, 20_000_000.0,
                        )

            # 4. Temporal IAT Moments Inversion (Geometric Random Sum Inversion)
            for col in df_s.columns:
                col_l = col.lower()
                if "iat" in col_l:
                    if "iat_mean" in col_l or "iat_total" in col_l or "iat_max" in col_l or "iat_min" in col_l:
                        df_s[col] = np.maximum(0.0, df_s[col].values * p)
                    elif "iat_std" in col_l:
                        # Full geometric-sum moment inversion:
                        # Var(tau) = max(eps, p * Var(tau_tilde) - p * (1 - p) * E[tau_tilde]^2)
                        iat_mean_col = None
                        for c_cand in df_s.columns:
                            if "iat" in c_cand.lower() and "mean" in c_cand.lower():
                                iat_mean_col = c_cand
                                break
                        if iat_mean_col:
                            mu_sampled = np.maximum(0.0, X_sampled[iat_mean_col].values)
                        else:
                            mu_sampled = np.maximum(0.0, df_s[col].values)
                        var_sampled = np.maximum(0.0, X_sampled[col].values ** 2)
                        dispersion = p * (1.0 - p) * (mu_sampled ** 2)
                        var_true = np.maximum(1e-6, p * var_sampled - dispersion)
                        df_s[col] = np.sqrt(var_true)

            # 5. Dynamic Rate Recalculation
            if "flow_duration" in df_s.columns:
                dur_sec = np.maximum(1e-4, df_s["flow_duration"].values / 1e6)
                if "flow_packets_per_s" in df_s.columns:
                    df_s["flow_packets_per_s"] = n_hat_tot / dur_sec
                if "flow_bytes_per_s" in df_s.columns:
                    fwd_b = df_s["total_fwd_bytes"].values if "total_fwd_bytes" in df_s.columns else 0.0
                    bwd_b = df_s["total_bwd_bytes"].values if "total_bwd_bytes" in df_s.columns else 0.0
                    df_s["flow_bytes_per_s"] = (fwd_b + bwd_b) / dur_sec

            # 6. Packet Length & Variance Plausible Imputation
            if "pkt_len_std" in df_s.columns:
                sess_mask = (~z_sparse) & (n_hat_tot > 2)
                if sess_mask.any():
                    df_s.loc[sess_mask, "pkt_len_std"] = self.empirical_medians_.get("pkt_len_std", 180.0)
            if "pkt_len_max" in df_s.columns:
                sess_mask = (~z_sparse) & (n_hat_tot > 2)
                if sess_mask.any():
                    df_s.loc[sess_mask, "pkt_len_max"] = 1460.0
                probe_mask = z_sparse & sparse_mask
                if probe_mask.any():
                    mean_val = df_s.loc[probe_mask, "fwd_pkt_len_mean"].values if "fwd_pkt_len_mean" in df_s.columns else 64.0
                    df_s.loc[probe_mask, "pkt_len_max"] = np.clip(mean_val, 40.0, 100.0)

            reconstructed_dfs.append(df_s)

        return reconstructed_dfs

