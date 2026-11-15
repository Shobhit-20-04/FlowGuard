"""
FlowGuard Packet Sampling & Channel Loss Simulator
===================================================

Simulates network-level packet decimation and stochastic degradation on flow feature matrices:
1. Uniform Random Packet Sampling (sFlow / NetFlow 1:N)
2. Periodic / Systematic Packet Sampling (1 in M with phase randomization)
3. Gilbert-Elliott Burst Packet Loss (2-state Markov chain switch buffer overflow)
4. Direction-Biased Asymmetric Sampling (Ingress vs Egress path disparity)
"""

import logging
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from flowguard.sampling.distortion_math import gilbert_elliott_parameters

logger = logging.getLogger(__name__)

# Additive byte volume feature names to scale proportionally with sampled packet ratios
ADDITIVE_BYTE_FEATURES = {
    "total_fwd_bytes",
    "total_bwd_bytes",
    "subflow_fwd_bytes",
    "subflow_bwd_bytes",
    "fwd_header_len",
    "bwd_header_len",
}


class PacketSamplingSimulator:
    """
    Simulates packet-level decimation mechanisms on tabular flow feature vectors.
    """
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def sample_random(
        self,
        X: pd.DataFrame,
        sampling_rate: float,
        drop_unobserved: bool = False,
    ) -> pd.DataFrame:
        """
        Applies uniform random packet sampling (1-in-N decimation) with probability p in (0, 1].

        Parameters
        ----------
        X : pd.DataFrame
            Pristine flow DataFrame of shape (N, D).
        sampling_rate : float
            Packet sampling probability p in (0, 1].
        drop_unobserved : bool
            If True, flows where zero packets are captured (k_tot == 0) are dropped from the DataFrame,
            matching export-only NetFlow behavior. If False, retains zero-observation flows with k=0.
        """
        p = float(np.clip(sampling_rate, 1e-4, 1.0))
        if p >= 1.0:
            return X.copy()

        df = X.copy()

        # Identify forward and backward packet count columns
        fwd_col = "total_fwd_packets" if "total_fwd_packets" in df.columns else None
        bwd_col = "total_bwd_packets" if "total_bwd_packets" in df.columns else None

        if fwd_col is None:
            for col in df.columns:
                if "fwd" in col.lower() and "packet" in col.lower() and "total" in col.lower():
                    fwd_col = col
                    break

        n_rows = len(df)

        # 1. Decimate packet counts via Binomial sampling
        if fwd_col in df.columns:
            fwd_orig = df[fwd_col].values.astype(np.int64)
            fwd_sampled = self.rng.binomial(n=np.maximum(0, fwd_orig), p=p)
            fwd_ratio = np.where(fwd_orig > 0, fwd_sampled / np.maximum(1, fwd_orig), 0.0)
            df[fwd_col] = fwd_sampled
        else:
            fwd_orig = np.ones(n_rows, dtype=np.int64)
            fwd_sampled = np.ones(n_rows, dtype=np.int64)
            fwd_ratio = np.full(n_rows, p)

        if bwd_col in df.columns:
            bwd_orig = df[bwd_col].values.astype(np.int64)
            bwd_sampled = self.rng.binomial(n=np.maximum(0, bwd_orig), p=p)
            bwd_ratio = np.where(bwd_orig > 0, bwd_sampled / np.maximum(1, bwd_orig), 0.0)
            df[bwd_col] = bwd_sampled
        else:
            bwd_orig = np.ones(n_rows, dtype=np.int64)
            bwd_sampled = np.ones(n_rows, dtype=np.int64)
            bwd_ratio = np.full(n_rows, p)

        # Handle zero-observation flows honestly: NO artificial packet forcing
        if drop_unobserved and fwd_col in df.columns and bwd_col in df.columns:
            observed_mask = (df[fwd_col] > 0) | (df[bwd_col] > 0)
            df = df[observed_mask].copy()
            fwd_ratio = fwd_ratio[observed_mask]
            bwd_ratio = bwd_ratio[observed_mask]
            fwd_orig = fwd_orig[observed_mask]
            bwd_orig = bwd_orig[observed_mask]
            n_rows = len(df)

        # 2. Scale additive byte volume features proportionally (STRICTLY excluding window sizes)
        for col in df.columns:
            col_l = col.lower()
            if "init_win" in col_l or "win_bytes" in col_l:
                continue

            if col in ADDITIVE_BYTE_FEATURES or ("subflow" in col_l and "bytes" in col_l):
                if "fwd" in col_l:
                    df[col] = np.maximum(0.0, df[col].values * fwd_ratio)
                elif "bwd" in col_l:
                    df[col] = np.maximum(0.0, df[col].values * bwd_ratio)
                else:
                    overall_ratio = (fwd_ratio + bwd_ratio) / 2.0
                    df[col] = np.maximum(0.0, df[col].values * overall_ratio)
            elif "header_len" in col_l:
                if "fwd" in col_l:
                    df[col] = np.maximum(0.0, df[fwd_col].values * 20.0) if fwd_col else df[col].values * fwd_ratio
                elif "bwd" in col_l:
                    df[col] = np.maximum(0.0, df[bwd_col].values * 20.0) if bwd_col else df[col].values * bwd_ratio

        # 3. Distort temporal & Inter-Arrival Time (IAT) moments
        iat_inflation = 1.0 / p
        for col in df.columns:
            col_l = col.lower()
            if "iat" in col_l and ("mean" in col_l or "total" in col_l or "max" in col_l or "min" in col_l):
                df[col] = np.maximum(0.0, df[col].values * iat_inflation)
            elif "iat_std" in col_l:
                var_multiplier = np.sqrt(max(1.0, (2.0 - p) / (p ** 2)))
                df[col] = np.maximum(0.0, df[col].values * var_multiplier * 0.5)

        # 4. Decimate TCP flags
        for col in df.columns:
            col_l = col.lower()
            if "flag" in col_l and ("syn" in col_l or "fin" in col_l or "rst" in col_l or "psh" in col_l):
                flag_vals = df[col].values.astype(np.int64)
                df[col] = self.rng.binomial(n=np.maximum(0, flag_vals), p=p)

        # 5. Handle directional and packet length statistics under sparse sampling
        total_k = (df[fwd_col].values + df[bwd_col].values) if (fwd_col and bwd_col) else np.ones(n_rows)
        n_orig = fwd_orig + bwd_orig

        # When 0 backward packets sampled, backward packet statistics collapse to 0
        if bwd_col in df.columns:
            bwd_zero_mask = (df[bwd_col] == 0)
            for col in df.columns:
                col_l = col.lower()
                if "bwd_pkt_len" in col_l or "bwd_segment_size" in col_l or "bwd_header_len" in col_l:
                    df.loc[bwd_zero_mask, col] = 0.0
            if "down_up_ratio" in df.columns:
                df.loc[bwd_zero_mask, "down_up_ratio"] = 0.0

        # When 0 forward packets sampled, forward statistics collapse to 0
        if fwd_col in df.columns:
            fwd_zero_mask = (df[fwd_col] == 0)
            for col in df.columns:
                col_l = col.lower()
                if "fwd_pkt_len" in col_l or "fwd_segment_size" in col_l or "fwd_header_len" in col_l:
                    df.loc[fwd_zero_mask, col] = 0.0

        # Sparse variance & length collapse
        single_pkt_mask = total_k <= 1
        multi_pkt_mask = ~single_pkt_mask

        if fwd_col and "fwd_pkt_len_std" in df.columns:
            df.loc[df[fwd_col] <= 1, "fwd_pkt_len_std"] = 0.0
        if bwd_col and "bwd_pkt_len_std" in df.columns:
            df.loc[df[bwd_col] <= 1, "bwd_pkt_len_std"] = 0.0
        if "pkt_len_std" in df.columns:
            df.loc[single_pkt_mask, "pkt_len_std"] = 0.0
        if "pkt_len_var" in df.columns:
            df.loc[single_pkt_mask, "pkt_len_var"] = 0.0

        if "pkt_len_max" in df.columns and "fwd_pkt_len_mean" in df.columns:
            df.loc[single_pkt_mask, "pkt_len_max"] = df.loc[single_pkt_mask, "fwd_pkt_len_mean"]
        if "pkt_len_min" in df.columns and "fwd_pkt_len_mean" in df.columns:
            df.loc[single_pkt_mask, "pkt_len_min"] = df.loc[single_pkt_mask, "fwd_pkt_len_mean"]

        # 6. Recalculate duration & rate features
        if "flow_duration" in df.columns:
            df.loc[single_pkt_mask, "flow_duration"] = 0.0
            if multi_pkt_mask.any():
                dur_scale = (total_k[multi_pkt_mask] - 1.0) / np.maximum(1.0, n_orig[multi_pkt_mask] - 1.0)
                df.loc[multi_pkt_mask, "flow_duration"] = np.maximum(10.0, df.loc[multi_pkt_mask, "flow_duration"].values * dur_scale)

            if "flow_bytes_per_s" in df.columns:
                fwd_b = df["total_fwd_bytes"].values if "total_fwd_bytes" in df.columns else np.zeros(n_rows)
                bwd_b = df["total_bwd_bytes"].values if "total_bwd_bytes" in df.columns else np.zeros(n_rows)
                df.loc[single_pkt_mask, "flow_bytes_per_s"] = 0.0
                if multi_pkt_mask.any():
                    dur_sec = np.maximum(1e-4, df.loc[multi_pkt_mask, "flow_duration"].values / 1e6)
                    df.loc[multi_pkt_mask, "flow_bytes_per_s"] = (fwd_b[multi_pkt_mask] + bwd_b[multi_pkt_mask]) / dur_sec

            if "flow_packets_per_s" in df.columns:
                fwd_p = df[fwd_col].values if fwd_col else np.zeros(n_rows)
                bwd_p = df[bwd_col].values if bwd_col else np.zeros(n_rows)
                df.loc[single_pkt_mask, "flow_packets_per_s"] = 0.0
                if multi_pkt_mask.any():
                    dur_sec = np.maximum(1e-4, df.loc[multi_pkt_mask, "flow_duration"].values / 1e6)
                    df.loc[multi_pkt_mask, "flow_packets_per_s"] = (fwd_p[multi_pkt_mask] + bwd_p[multi_pkt_mask]) / dur_sec

        return df

    def sample_periodic(
        self,
        X: pd.DataFrame,
        sampling_interval: int = 10,
        phase_randomization: bool = True,
    ) -> pd.DataFrame:
        """
        Simulates deterministic periodic / systematic 1-in-M packet sampling.

        Parameters
        ----------
        X : pd.DataFrame
            Pristine flow DataFrame of shape (N, D).
        sampling_interval : int
            Sampling period M (captures 1 packet every M packets).
        phase_randomization : bool
            If True, randomly offsets the initial sampling phase phi in {0, ..., M - 1} per flow.
        """
        M = max(1, int(sampling_interval))
        if M == 1:
            return X.copy()

        df = X.copy()
        fwd_col = "total_fwd_packets" if "total_fwd_packets" in df.columns else None
        bwd_col = "total_bwd_packets" if "total_bwd_packets" in df.columns else None

        n_rows = len(df)
        fwd_orig = df[fwd_col].values.astype(np.int64) if fwd_col else np.ones(n_rows, dtype=np.int64)
        bwd_orig = df[bwd_col].values.astype(np.int64) if bwd_col else np.zeros(n_rows, dtype=np.int64)

        # Generate phase offsets
        if phase_randomization:
            phi_fwd = self.rng.integers(0, M, size=n_rows)
            phi_bwd = self.rng.integers(0, M, size=n_rows)
        else:
            phi_fwd = np.zeros(n_rows, dtype=np.int64)
            phi_bwd = np.zeros(n_rows, dtype=np.int64)

        # Number of captured packets in interval: max(0, (N - 1 - phi) // M + 1)
        fwd_sampled = np.where(fwd_orig > phi_fwd, (fwd_orig - 1 - phi_fwd) // M + 1, 0)
        bwd_sampled = np.where(bwd_orig > phi_bwd, (bwd_orig - 1 - phi_bwd) // M + 1, 0)

        if fwd_col:
            df[fwd_col] = fwd_sampled
            fwd_ratio = np.where(fwd_orig > 0, fwd_sampled / np.maximum(1, fwd_orig), 0.0)
        else:
            fwd_ratio = np.full(n_rows, 1.0 / M)

        if bwd_col:
            df[bwd_col] = bwd_sampled
            bwd_ratio = np.where(bwd_orig > 0, bwd_sampled / np.maximum(1, bwd_orig), 0.0)
        else:
            bwd_ratio = np.full(n_rows, 1.0 / M)

        # Scale additive volume features
        for col in df.columns:
            col_l = col.lower()
            if "init_win" in col_l or "win_bytes" in col_l:
                continue
            if col in ADDITIVE_BYTE_FEATURES or ("subflow" in col_l and "bytes" in col_l):
                if "fwd" in col_l:
                    df[col] = np.maximum(0.0, df[col].values * fwd_ratio)
                elif "bwd" in col_l:
                    df[col] = np.maximum(0.0, df[col].values * bwd_ratio)
                else:
                    df[col] = np.maximum(0.0, df[col].values * ((fwd_ratio + bwd_ratio) / 2.0))
            elif "header_len" in col_l:
                if "fwd" in col_l and fwd_col:
                    df[col] = np.maximum(0.0, df[fwd_col].values * 20.0)
                elif "bwd" in col_l and bwd_col:
                    df[col] = np.maximum(0.0, df[bwd_col].values * 20.0)

        # Distort temporal and rate features
        iat_inflation = float(M)
        for col in df.columns:
            col_l = col.lower()
            if "iat" in col_l and ("mean" in col_l or "total" in col_l or "max" in col_l or "min" in col_l):
                df[col] = np.maximum(0.0, df[col].values * iat_inflation)

        total_k = (fwd_sampled + bwd_sampled)
        single_pkt_mask = total_k <= 1
        multi_pkt_mask = ~single_pkt_mask

        if "flow_duration" in df.columns:
            df.loc[single_pkt_mask, "flow_duration"] = 0.0
            if multi_pkt_mask.any():
                n_orig = fwd_orig + bwd_orig
                dur_scale = (total_k[multi_pkt_mask] - 1.0) / np.maximum(1.0, n_orig[multi_pkt_mask] - 1.0)
                df.loc[multi_pkt_mask, "flow_duration"] = np.maximum(10.0, df.loc[multi_pkt_mask, "flow_duration"].values * dur_scale)

            if "flow_bytes_per_s" in df.columns:
                fwd_b = df["total_fwd_bytes"].values if "total_fwd_bytes" in df.columns else np.zeros(n_rows)
                bwd_b = df["total_bwd_bytes"].values if "total_bwd_bytes" in df.columns else np.zeros(n_rows)
                df.loc[single_pkt_mask, "flow_bytes_per_s"] = 0.0
                if multi_pkt_mask.any():
                    dur_sec = np.maximum(1e-4, df.loc[multi_pkt_mask, "flow_duration"].values / 1e6)
                    df.loc[multi_pkt_mask, "flow_bytes_per_s"] = (fwd_b[multi_pkt_mask] + bwd_b[multi_pkt_mask]) / dur_sec

            if "flow_packets_per_s" in df.columns:
                df.loc[single_pkt_mask, "flow_packets_per_s"] = 0.0
                if multi_pkt_mask.any():
                    dur_sec = np.maximum(1e-4, df.loc[multi_pkt_mask, "flow_duration"].values / 1e6)
                    df.loc[multi_pkt_mask, "flow_packets_per_s"] = (fwd_sampled[multi_pkt_mask] + bwd_sampled[multi_pkt_mask]) / dur_sec

        return df

    def sample_burst_loss(
        self,
        X: pd.DataFrame,
        loss_rate: float = 0.05,
        burst_length: float = 5.0,
    ) -> pd.DataFrame:
        """
        Simulates consecutive packet drops via the authentic 2-state Gilbert-Elliott Markov chain model.
        Good state (G): packet is retained.
        Bad state (B): packet is dropped.
        """
        params = gilbert_elliott_parameters(loss_rate=loss_rate, burst_length=burst_length)
        p_gb = params["p_GB"]
        p_bg = params["p_BG"]
        pi_b = params["pi_Bad"]

        df = X.copy()
        fwd_col = "total_fwd_packets" if "total_fwd_packets" in df.columns else None
        bwd_col = "total_bwd_packets" if "total_bwd_packets" in df.columns else None

        n_rows = len(df)
        fwd_orig = df[fwd_col].values.astype(np.int64) if fwd_col else np.ones(n_rows, dtype=np.int64)
        bwd_orig = df[bwd_col].values.astype(np.int64) if bwd_col else np.zeros(n_rows, dtype=np.int64)

        fwd_sampled = np.zeros(n_rows, dtype=np.int64)
        bwd_sampled = np.zeros(n_rows, dtype=np.int64)

        # Simulate 2-state Markov chain per flow
        # State 0: Good (retained), State 1: Bad (dropped)
        for i in range(n_rows):
            n_f = fwd_orig[i]
            if n_f > 0:
                # Initial state from stationary distribution
                curr_state = 1 if self.rng.random() < pi_b else 0
                retained_f = 0
                for _ in range(n_f):
                    if curr_state == 0:
                        retained_f += 1
                        curr_state = 1 if self.rng.random() < p_gb else 0
                    else:
                        curr_state = 0 if self.rng.random() < p_bg else 1
                fwd_sampled[i] = retained_f

            n_b = bwd_orig[i]
            if n_b > 0:
                curr_state = 1 if self.rng.random() < pi_b else 0
                retained_b = 0
                for _ in range(n_b):
                    if curr_state == 0:
                        retained_b += 1
                        curr_state = 1 if self.rng.random() < p_gb else 0
                    else:
                        curr_state = 0 if self.rng.random() < p_bg else 1
                bwd_sampled[i] = retained_b

        if fwd_col:
            df[fwd_col] = fwd_sampled
            fwd_ratio = np.where(fwd_orig > 0, fwd_sampled / np.maximum(1, fwd_orig), 0.0)
        else:
            fwd_ratio = np.full(n_rows, 1.0 - pi_b)

        if bwd_col:
            df[bwd_col] = bwd_sampled
            bwd_ratio = np.where(bwd_orig > 0, bwd_sampled / np.maximum(1, bwd_orig), 0.0)
        else:
            bwd_ratio = np.full(n_rows, 1.0 - pi_b)

        # Scale additive byte volume features (excluding window sizes)
        for col in df.columns:
            col_l = col.lower()
            if "init_win" in col_l or "win_bytes" in col_l:
                continue
            if col in ADDITIVE_BYTE_FEATURES or ("subflow" in col_l and "bytes" in col_l):
                if "fwd" in col_l:
                    df[col] = np.maximum(0.0, df[col].values * fwd_ratio)
                elif "bwd" in col_l:
                    df[col] = np.maximum(0.0, df[col].values * bwd_ratio)
                else:
                    df[col] = np.maximum(0.0, df[col].values * ((fwd_ratio + bwd_ratio) / 2.0))
            elif "header_len" in col_l:
                if "fwd" in col_l and fwd_col:
                    df[col] = np.maximum(0.0, df[fwd_col].values * 20.0)
                elif "bwd" in col_l and bwd_col:
                    df[col] = np.maximum(0.0, df[bwd_col].values * 20.0)

        return df

    def sample_asymmetric(
        self,
        X: pd.DataFrame,
        forward_rate: float = 0.10,
        backward_rate: float = 0.02,
    ) -> pd.DataFrame:
        """
        Simulates asymmetric directional packet sampling (disparate rates on ingress vs egress).
        """
        df = X.copy()
        p_fwd = float(np.clip(forward_rate, 1e-4, 1.0))
        p_bwd = float(np.clip(backward_rate, 1e-4, 1.0))

        fwd_col = "total_fwd_packets" if "total_fwd_packets" in df.columns else None
        bwd_col = "total_bwd_packets" if "total_bwd_packets" in df.columns else None

        if fwd_col in df.columns:
            orig = df[fwd_col].values.astype(np.int64)
            df[fwd_col] = self.rng.binomial(n=np.maximum(0, orig), p=p_fwd)
            ratio_fwd = np.where(orig > 0, df[fwd_col].values / np.maximum(1, orig), 0.0)
        else:
            ratio_fwd = np.full(len(df), p_fwd)

        if bwd_col in df.columns:
            orig = df[bwd_col].values.astype(np.int64)
            df[bwd_col] = self.rng.binomial(n=np.maximum(0, orig), p=p_bwd)
            ratio_bwd = np.where(orig > 0, df[bwd_col].values / np.maximum(1, orig), 0.0)
        else:
            ratio_bwd = np.full(len(df), p_bwd)

        for col in df.columns:
            col_l = col.lower()
            if "init_win" in col_l or "win_bytes" in col_l:
                continue
            if col in ADDITIVE_BYTE_FEATURES or ("subflow" in col_l and "bytes" in col_l):
                if "fwd" in col_l:
                    df[col] = np.maximum(0.0, df[col].values * ratio_fwd)
                elif "bwd" in col_l:
                    df[col] = np.maximum(0.0, df[col].values * ratio_bwd)
                else:
                    df[col] = np.maximum(0.0, df[col].values * ((ratio_fwd + ratio_bwd) / 2.0))
            elif "header_len" in col_l:
                if "fwd" in col_l and fwd_col:
                    df[col] = np.maximum(0.0, df[fwd_col].values * 20.0)
                elif "bwd" in col_l and bwd_col:
                    df[col] = np.maximum(0.0, df[bwd_col].values * 20.0)

        return df

    def degrade(
        self,
        X: pd.DataFrame,
        mechanism: str = "random",
        **kwargs,
    ) -> pd.DataFrame:
        """
        Master dispatch method for flow degradation.
        """
        if mechanism in ["random", "uniform"]:
            rate = kwargs.get("rate", kwargs.get("sampling_rate", 0.10))
            drop_unobs = kwargs.get("drop_unobserved", False)
            return self.sample_random(X, sampling_rate=rate, drop_unobserved=drop_unobs)
        elif mechanism in ["periodic", "systematic"]:
            interval = kwargs.get("sampling_interval", kwargs.get("interval", 10))
            phase_rand = kwargs.get("phase_randomization", True)
            return self.sample_periodic(X, sampling_interval=interval, phase_randomization=phase_rand)
        elif mechanism in ["burst", "gilbert_elliott"]:
            loss_rate = kwargs.get("loss_rate", 0.05)
            burst_len = kwargs.get("burst_length", 5.0)
            return self.sample_burst_loss(X, loss_rate=loss_rate, burst_length=burst_len)
        elif mechanism in ["asymmetric", "directional"]:
            fwd_rate = kwargs.get("forward_rate", 0.10)
            bwd_rate = kwargs.get("backward_rate", 0.02)
            return self.sample_asymmetric(X, forward_rate=fwd_rate, backward_rate=bwd_rate)
        else:
            raise ValueError(f"Unknown sampling mechanism: {mechanism}")

