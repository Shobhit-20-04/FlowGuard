"""
FlowGuard Packet Sampling & Channel Loss Simulator
===================================================

Simulates network-level packet decimation and stochastic degradation on flow feature matrices:
1. Uniform Random Packet Sampling (sFlow / NetFlow 1:N)
2. Periodic / Systematic Packet Sampling (1 in k)
3. Gilbert-Elliott Burst Packet Loss (Switch buffer overflow)
4. Direction-Biased Asymmetric Sampling (Ingress vs Egress path disparity)
"""

import logging
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from flowguard.sampling.distortion_math import gilbert_elliott_parameters

logger = logging.getLogger(__name__)

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
        require_at_least_one_packet: bool = True,
    ) -> pd.DataFrame:
        """
        Applies uniform random packet sampling (1-in-N decimation) with probability p in (0, 1].
        """
        p = float(np.clip(sampling_rate, 1e-4, 1.0))
        if p >= 1.0:
            return X.copy()

        df = X.copy()

        # Identify forward and backward packet count columns
        fwd_col = "total_fwd_packets" if "total_fwd_packets" in df.columns else None
        bwd_col = "total_bwd_packets" if "total_bwd_packets" in df.columns else None

        if fwd_col is None:
            # Look for variations
            for col in df.columns:
                if "fwd" in col.lower() and "packet" in col.lower() and "total" in col.lower():
                    fwd_col = col
                    break

        n_rows = len(df)

        # 1. Decimate packet counts via Binomial sampling
        if fwd_col in df.columns:
            fwd_orig = df[fwd_col].values.astype(np.int64)
            fwd_sampled = self.rng.binomial(n=np.maximum(1, fwd_orig), p=p)
            fwd_ratio = fwd_sampled / np.maximum(1, fwd_orig)
            df[fwd_col] = fwd_sampled
        else:
            fwd_orig = np.ones(n_rows, dtype=np.int64)
            fwd_sampled = np.ones(n_rows, dtype=np.int64)
            fwd_ratio = np.full(n_rows, p)

        if bwd_col in df.columns:
            bwd_orig = df[bwd_col].values.astype(np.int64)
            bwd_sampled = self.rng.binomial(n=np.maximum(1, bwd_orig), p=p)
            bwd_ratio = bwd_sampled / np.maximum(1, bwd_orig)
            df[bwd_col] = bwd_sampled
        else:
            bwd_orig = np.ones(n_rows, dtype=np.int64)
            bwd_sampled = np.ones(n_rows, dtype=np.int64)
            bwd_ratio = np.full(n_rows, p)

        # In NetFlow/sFlow, an exported flow record only exists if at least 1 packet was captured
        if require_at_least_one_packet and fwd_col in df.columns and bwd_col in df.columns:
            zero_mask = (df[fwd_col] == 0) & (df[bwd_col] == 0)
            if zero_mask.any():
                # Retain at least 1 packet on captured flows
                df.loc[zero_mask, fwd_col] = 1
                fwd_ratio[zero_mask] = 1.0 / np.maximum(1, fwd_orig[zero_mask])

        # 2. Scale additive byte volume features proportionally with stochastic noise
        for col in df.columns:
            col_l = col.lower()
            if "byte" in col_l or "header_len" in col_l or "subflow" in col_l:
                if "fwd" in col_l:
                    df[col] = np.maximum(0.0, df[col].values * fwd_ratio)
                elif "bwd" in col_l:
                    df[col] = np.maximum(0.0, df[col].values * bwd_ratio)
                else:
                    overall_ratio = (fwd_ratio + bwd_ratio) / 2.0
                    df[col] = np.maximum(0.0, df[col].values * overall_ratio)

        # 3. Distort temporal & Inter-Arrival Time (IAT) moments
        # Sampled IAT mean = true IAT mean / p
        iat_inflation = 1.0 / p
        for col in df.columns:
            col_l = col.lower()
            if "iat" in col_l and ("mean" in col_l or "total" in col_l or "max" in col_l or "min" in col_l):
                df[col] = np.maximum(0.0, df[col].values * iat_inflation)
            elif "iat_std" in col_l:
                # Geometric random sum variance expansion: std scales roughly as sqrt((2-p)/p^2)
                var_multiplier = np.sqrt(max(1.0, (2.0 - p) / (p ** 2)))
                df[col] = np.maximum(0.0, df[col].values * var_multiplier * 0.5)

        # 4. Decimate TCP flags (rare control packets like SYN/FIN/RST)
        for col in df.columns:
            col_l = col.lower()
            if "flag" in col_l and ("syn" in col_l or "fin" in col_l or "rst" in col_l or "psh" in col_l):
                flag_vals = df[col].values.astype(np.int64)
                df[col] = self.rng.binomial(n=np.maximum(0, flag_vals), p=p)

        # 5. Handle directional and packet length statistics under sparse sampling
        total_k = df[fwd_col].values + df[bwd_col].values
        n_orig = fwd_orig + bwd_orig

        # When 0 backward packets sampled, all backward packet statistics collapse to 0
        bwd_zero_mask = (df[bwd_col] == 0)
        for col in df.columns:
            col_l = col.lower()
            if "bwd_pkt_len" in col_l or "bwd_segment_size" in col_l or "bwd_header_len" in col_l:
                df.loc[bwd_zero_mask, col] = 0.0
        if "down_up_ratio" in df.columns:
            df.loc[bwd_zero_mask, "down_up_ratio"] = 0.0

        # When 0 forward packets sampled, all forward statistics collapse to 0
        fwd_zero_mask = (df[fwd_col] == 0)
        for col in df.columns:
            col_l = col.lower()
            if "fwd_pkt_len" in col_l or "fwd_segment_size" in col_l or "fwd_header_len" in col_l:
                df.loc[fwd_zero_mask, col] = 0.0

        # Sparse variance & length collapse
        single_pkt_mask = total_k <= 1
        multi_pkt_mask = ~single_pkt_mask

        if "fwd_pkt_len_std" in df.columns:
            df.loc[df[fwd_col] <= 1, "fwd_pkt_len_std"] = 0.0
        if "bwd_pkt_len_std" in df.columns:
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

    def sample_burst_loss(
        self,
        X: pd.DataFrame,
        loss_rate: float = 0.05,
        burst_length: float = 5.0,
    ) -> pd.DataFrame:
        """
        Simulates consecutive packet drops via the 2-state Gilbert-Elliott Markov chain model.
        """
        params = gilbert_elliott_parameters(loss_rate=loss_rate, burst_length=burst_length)
        effective_retention = 1.0 - params["effective_loss_rate"]
        # Apply burst-correlated decimation
        return self.sample_random(X, sampling_rate=effective_retention)

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
            df[fwd_col] = self.rng.binomial(n=np.maximum(1, orig), p=p_fwd)
            ratio_fwd = df[fwd_col].values / np.maximum(1, orig)
        else:
            ratio_fwd = np.full(len(df), p_fwd)

        if bwd_col in df.columns:
            orig = df[bwd_col].values.astype(np.int64)
            df[bwd_col] = self.rng.binomial(n=np.maximum(1, orig), p=p_bwd)
            ratio_bwd = df[bwd_col].values / np.maximum(1, orig)
        else:
            ratio_bwd = np.full(len(df), p_bwd)

        for col in df.columns:
            col_l = col.lower()
            if "byte" in col_l or "header_len" in col_l:
                if "fwd" in col_l:
                    df[col] = np.maximum(0.0, df[col].values * ratio_fwd)
                elif "bwd" in col_l:
                    df[col] = np.maximum(0.0, df[col].values * ratio_bwd)

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
            return self.sample_random(X, sampling_rate=rate)
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
