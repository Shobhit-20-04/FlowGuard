"""
FlowGuard Sampling Distortion & Uncertainty Model (SDUM)
========================================================

Mathematically inverts the packet sampling and decimation process to reconstruct
an ensemble of S plausible true flows {X_hat^(1), ..., X_hat^(S)} per observed flow.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from flowguard.sampling.distortion_math import (
    binomial_count_moments,
    inverted_count_posterior_moments,
    sampled_iat_moments,
)


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
    ):
        self.n_mc_samples = n_mc_samples
        self.random_state = random_state
        self.preserve_correlation = preserve_correlation
        self.probe_prior = probe_prior
        self.rng = np.random.default_rng(random_state)

        # Learned empirical priors and correlation structures
        self.is_fitted: bool = False
        self.feature_names_: List[str] = []
        self.empirical_medians_: Dict[str, float] = {}
        self.empirical_stds_: Dict[str, float] = {}
        self.corr_cholesky_: Optional[np.ndarray] = None
        self.tracked_corr_cols_: List[str] = []

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

        # Correlation preservation across core volume and temporal features
        candidate_cols = [
            "total_fwd_packets",
            "total_bwd_packets",
            "total_fwd_bytes",
            "total_bwd_bytes",
            "flow_duration",
            "flow_bytes_per_s",
            "flow_packets_per_s",
            "flow_iat_mean",
        ]
        self.tracked_corr_cols_ = [c for c in candidate_cols if c in self.feature_names_]

        if self.preserve_correlation and len(self.tracked_corr_cols_) > 1:
            sub_df = X_pristine[self.tracked_corr_cols_].fillna(0.0)
            corr_mat = sub_df.corr().values
            reg_corr = corr_mat + np.eye(len(corr_mat)) * 1e-4
            reg_corr = reg_corr / np.sqrt(np.outer(np.diag(reg_corr), np.diag(reg_corr)))
            try:
                self.corr_cholesky_ = np.linalg.cholesky(reg_corr)
            except np.linalg.LinAlgError:
                self.corr_cholesky_ = np.eye(len(self.tracked_corr_cols_))
        else:
            self.corr_cholesky_ = None

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

        reconstructed_dfs: List[pd.DataFrame] = []

        # Vectorized draws for S realizations
        for s in range(S):
            df_s = X_sampled.copy()

            # Latent hypothesis for sparse flows:
            # Z = 1: Flow was originally a small probe / scan / short transaction
            # Z = 0: Flow was originally a multi-packet session decimated by sampling
            z_sparse = self.rng.binomial(1, self.probe_prior, size=n_rows).astype(bool)

            # 1. Packet Count Reconstruction
            # For multi-packet observations (k_tot > 2): unimodal Negative Binomial inversion
            # For sparse observations (k_tot <= 2): mixture posterior
            n_hat_fwd = np.zeros(n_rows, dtype=np.int64)
            n_hat_bwd = np.zeros(n_rows, dtype=np.int64)

            # Multi-packet flows: Negative Binomial draws
            if multi_mask.any():
                missed_fwd_multi = self.rng.negative_binomial(
                    n=np.maximum(1, k_fwd[multi_mask]), p=p
                )
                n_hat_fwd[multi_mask] = k_fwd[multi_mask] + missed_fwd_multi

                bwd_nonzero = (k_bwd[multi_mask] > 0)
                missed_bwd_multi = np.zeros(int(multi_mask.sum()), dtype=np.int64)
                if bwd_nonzero.any():
                    missed_bwd_multi[bwd_nonzero] = self.rng.negative_binomial(
                        n=k_bwd[multi_mask][bwd_nonzero], p=p
                    )
                n_hat_bwd[multi_mask] = k_bwd[multi_mask] + missed_bwd_multi

            # Sparse flows: Mixture draws
            if sparse_mask.any():
                # Case A: Probe hypothesis (Z = 1) -> small true count (1 to 3 packets)
                probe_indices = sparse_mask & z_sparse
                if probe_indices.any():
                    n_hat_fwd[probe_indices] = np.maximum(1, k_fwd[probe_indices])
                    n_hat_bwd[probe_indices] = k_bwd[probe_indices]

                # Case B: Decimated session hypothesis (Z = 0) -> true count was substantial
                session_indices = sparse_mask & (~z_sparse)
                if session_indices.any():
                    missed_fwd_sess = self.rng.negative_binomial(n=1, p=p, size=int(session_indices.sum()))
                    n_hat_fwd[session_indices] = np.maximum(2, 1 + missed_fwd_sess)

                    # Plausible backward response packets
                    bwd_ratio = self.rng.uniform(0.3, 0.8, size=int(session_indices.sum()))
                    n_hat_bwd[session_indices] = np.maximum(1, (n_hat_fwd[session_indices] * bwd_ratio).astype(np.int64))

            if fwd_col:
                df_s[fwd_col] = n_hat_fwd
                ratio_fwd = n_hat_fwd / np.maximum(1, k_fwd)
            else:
                ratio_fwd = np.full(n_rows, 1.0 / p)

            if bwd_col:
                df_s[bwd_col] = n_hat_bwd
                ratio_bwd = np.where(k_bwd > 0, n_hat_bwd / np.maximum(1, k_bwd), 1.0 / p)
            else:
                ratio_bwd = np.full(n_rows, 1.0 / p)

            n_hat_tot = n_hat_fwd + n_hat_bwd

            # 2. Byte Volume Reconstruction (Compound Gamma Scale)
            for col in df_s.columns:
                col_l = col.lower()
                if "byte" in col_l or "header_len" in col_l or "subflow" in col_l:
                    if "fwd" in col_l:
                        noise = self.rng.gamma(shape=10.0, scale=0.1, size=n_rows)
                        df_s[col] = np.maximum(0.0, df_s[col].values * ratio_fwd * noise)
                    elif "bwd" in col_l:
                        noise = self.rng.gamma(shape=10.0, scale=0.1, size=n_rows)
                        df_s[col] = np.maximum(0.0, df_s[col].values * ratio_bwd * noise)
                    else:
                        overall_ratio = (ratio_fwd + ratio_bwd) / 2.0
                        df_s[col] = np.maximum(0.0, df_s[col].values * overall_ratio)

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
                        contraction = np.sqrt(max(1e-4, p))
                        df_s[col] = np.maximum(0.0, df_s[col].values * contraction)

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
                    df_s.loc[probe_mask, "pkt_len_max"] = np.clip(df_s.loc[probe_mask, "fwd_pkt_len_mean"].values, 40.0, 100.0)

            reconstructed_dfs.append(df_s)

        return reconstructed_dfs
