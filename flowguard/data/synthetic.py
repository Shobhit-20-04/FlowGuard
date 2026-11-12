"""
FlowGuard Realistic Synthetic Network Flow Generator
=====================================================

Generates synthetic network-flow records following authentic empirical distributions
from CIC-IDS2017/2018 for testing, offline experimentation, and controlled benchmark simulations.
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from flowguard.data.schema import RAW_COLUMN_MAP

class SyntheticFlowGenerator:
    """
    Generates realistic synthetic flow datasets matching the CICFlowMeter 80-feature schema.
    """
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def generate_flow_batch(
        self,
        n_samples: int = 10000,
        attack_ratio: float = 0.25,
        attack_distribution: Optional[Dict[str, float]] = None,
    ) -> pd.DataFrame:
        """
        Generates a DataFrame of n_samples flow records with authentic statistical properties.
        """
        if attack_distribution is None:
            attack_distribution = {
                "DoS": 0.40,
                "DDoS": 0.30,
                "PortScan": 0.20,
                "Botnet": 0.10,
            }

        n_attacks = int(n_samples * attack_ratio)
        n_benign = n_samples - n_attacks

        rows = []
        labels = []

        # 1. Generate Benign Traffic (Normal browsing, streaming, DNS, web API)
        for _ in range(n_benign):
            # Packet counts: Log-normal distribution
            fwd_pkts = int(np.clip(self.rng.lognormal(mean=2.2, sigma=0.8), 2, 2000))
            bwd_pkts = int(np.clip(self.rng.lognormal(mean=2.4, sigma=0.9), 1, 3000))
            
            # Packet sizes: Mix of ACKs (40-60B) and data MTU chunks (1460B)
            avg_fwd_len = float(self.rng.uniform(60, 800))
            avg_bwd_len = float(self.rng.uniform(100, 1400))
            total_fwd_bytes = fwd_pkts * avg_fwd_len
            total_bwd_bytes = bwd_pkts * avg_bwd_len

            duration_us = float(self.rng.exponential(scale=2_500_000)) # ~2.5s duration
            iat_mean = duration_us / max(1, (fwd_pkts + bwd_pkts - 1))
            iat_std = float(iat_mean * self.rng.uniform(0.5, 1.5))

            rows.append({
                "Flow Duration": duration_us,
                "Total Fwd Packets": fwd_pkts,
                "Total Backward Packets": bwd_pkts,
                "Total Length of Fwd Packets": total_fwd_bytes,
                "Total Length of Bwd Packets": total_bwd_bytes,
                "Fwd Packet Length Max": avg_fwd_len * 1.5,
                "Fwd Packet Length Min": 40.0,
                "Fwd Packet Length Mean": avg_fwd_len,
                "Fwd Packet Length Std": avg_fwd_len * 0.4,
                "Bwd Packet Length Max": avg_bwd_len * 1.2,
                "Bwd Packet Length Min": 40.0,
                "Bwd Packet Length Mean": avg_bwd_len,
                "Bwd Packet Length Std": avg_bwd_len * 0.3,
                "Flow Bytes/s": (total_fwd_bytes + total_bwd_bytes) / max(1.0, duration_us / 1e6),
                "Flow Packets/s": (fwd_pkts + bwd_pkts) / max(1.0, duration_us / 1e6),
                "Flow IAT Mean": iat_mean,
                "Flow IAT Std": iat_std,
                "Flow IAT Max": iat_mean * 3.0,
                "Flow IAT Min": max(1.0, iat_mean * 0.1),
                "Fwd IAT Total": duration_us * 0.9,
                "Fwd IAT Mean": duration_us / max(1, fwd_pkts),
                "Fwd IAT Std": iat_std,
                "Fwd IAT Max": iat_mean * 2.5,
                "Fwd IAT Min": 5.0,
                "Bwd IAT Total": duration_us * 0.8,
                "Bwd IAT Mean": duration_us / max(1, bwd_pkts),
                "Bwd IAT Std": iat_std,
                "Bwd IAT Max": iat_mean * 2.5,
                "Bwd IAT Min": 5.0,
                "Fwd PSH Flags": int(self.rng.binomial(1, 0.3)),
                "Bwd PSH Flags": 0,
                "Fwd URG Flags": 0,
                "Bwd URG Flags": 0,
                "Fwd Header Length": fwd_pkts * 20,
                "Bwd Header Length": bwd_pkts * 20,
                "Fwd Packets/s": fwd_pkts / max(1.0, duration_us / 1e6),
                "Bwd Packets/s": bwd_pkts / max(1.0, duration_us / 1e6),
                "Min Packet Length": 40.0,
                "Max Packet Length": 1460.0,
                "Packet Length Mean": (total_fwd_bytes + total_bwd_bytes) / max(1, (fwd_pkts + bwd_pkts)),
                "Packet Length Std": 250.0,
                "Packet Length Variance": 250.0 ** 2,
                "FIN Flag Count": 1,
                "SYN Flag Count": 1,
                "RST Flag Count": 0,
                "PSH Flag Count": 1,
                "ACK Flag Count": 1,
                "URG Flag Count": 0,
                "CWE Flag Count": 0,
                "ECE Flag Count": 0,
                "Down/Up Ratio": float(bwd_pkts / max(1, fwd_pkts)),
                "Average Packet Size": (total_fwd_bytes + total_bwd_bytes) / max(1, (fwd_pkts + bwd_pkts)),
                "Avg Fwd Segment Size": avg_fwd_len,
                "Avg Bwd Segment Size": avg_bwd_len,
                "Subflow Fwd Packets": fwd_pkts,
                "Subflow Fwd Bytes": total_fwd_bytes,
                "Subflow Bwd Packets": bwd_pkts,
                "Subflow Bwd Bytes": total_bwd_bytes,
                "Init_Win_bytes_forward": 29200,
                "Init_Win_bytes_backward": 65535,
                "act_data_pkt_fwd": max(1, int(fwd_pkts * 0.7)),
                "min_seg_size_forward": 20,
                "Active Mean": duration_us * 0.7,
                "Active Std": 1000.0,
                "Active Max": duration_us * 0.7,
                "Active Min": duration_us * 0.7,
                "Idle Mean": 0.0,
                "Idle Std": 0.0,
                "Idle Max": 0.0,
                "Idle Min": 0.0,
            })
            labels.append("BENIGN")

        # 2. Generate Attacks
        attack_types = list(attack_distribution.keys())
        attack_probs = list(attack_distribution.values())

        for _ in range(n_attacks):
            atk_type = self.rng.choice(attack_types, p=attack_probs)

            if atk_type in ["DoS", "DDoS"]:
                # Volumetric floods: high packets, tiny payloads (SYN/UDP flood), tiny IAT
                fwd_pkts = int(self.rng.uniform(500, 15000))
                bwd_pkts = int(self.rng.uniform(0, 5)) # Asymmetric response
                avg_fwd_len = float(self.rng.choice([40.0, 64.0, 128.0]))
                avg_bwd_len = 0.0 if bwd_pkts == 0 else 40.0
                duration_us = float(self.rng.uniform(10_000, 500_000)) # Very short burst
                iat_mean = duration_us / max(1, fwd_pkts)
                iat_std = float(iat_mean * 0.2) # Extremely regular packet pacing

                syn_flag = 1
                ack_flag = 0 if bwd_pkts == 0 else 1
            elif atk_type == "PortScan":
                # Reconnaissance: 1 to 3 packets per flow, high SYN flags, 0 response
                fwd_pkts = int(self.rng.integers(1, 4))
                bwd_pkts = int(self.rng.choice([0, 1], p=[0.85, 0.15]))
                avg_fwd_len = 40.0
                avg_bwd_len = 40.0 if bwd_pkts > 0 else 0.0
                duration_us = float(self.rng.uniform(100, 15_000))
                iat_mean = duration_us / max(1, fwd_pkts)
                iat_std = 0.0 if fwd_pkts <= 1 else float(iat_mean * 0.1)

                syn_flag = 1
                ack_flag = 0
            else: # Botnet / C2
                # Periodic heartbeat pulses
                fwd_pkts = int(self.rng.integers(4, 15))
                bwd_pkts = int(self.rng.integers(4, 15))
                avg_fwd_len = float(self.rng.uniform(150, 400))
                avg_bwd_len = float(self.rng.uniform(150, 400))
                duration_us = float(self.rng.uniform(1_000_000, 10_000_000))
                iat_mean = duration_us / max(1, fwd_pkts)
                iat_std = float(iat_mean * 0.05) # Strict periodic timer

                syn_flag = 1
                ack_flag = 1

            total_fwd_bytes = fwd_pkts * avg_fwd_len
            total_bwd_bytes = bwd_pkts * avg_bwd_len

            rows.append({
                "Flow Duration": duration_us,
                "Total Fwd Packets": fwd_pkts,
                "Total Backward Packets": bwd_pkts,
                "Total Length of Fwd Packets": total_fwd_bytes,
                "Total Length of Bwd Packets": total_bwd_bytes,
                "Fwd Packet Length Max": avg_fwd_len,
                "Fwd Packet Length Min": 40.0,
                "Fwd Packet Length Mean": avg_fwd_len,
                "Fwd Packet Length Std": 5.0,
                "Bwd Packet Length Max": avg_bwd_len,
                "Bwd Packet Length Min": 0.0 if bwd_pkts == 0 else 40.0,
                "Bwd Packet Length Mean": avg_bwd_len,
                "Bwd Packet Length Std": 2.0,
                "Flow Bytes/s": (total_fwd_bytes + total_bwd_bytes) / max(1.0, duration_us / 1e6),
                "Flow Packets/s": (fwd_pkts + bwd_pkts) / max(1.0, duration_us / 1e6),
                "Flow IAT Mean": iat_mean,
                "Flow IAT Std": iat_std,
                "Flow IAT Max": iat_mean * 1.5,
                "Flow IAT Min": max(1.0, iat_mean * 0.5),
                "Fwd IAT Total": duration_us,
                "Fwd IAT Mean": duration_us / max(1, fwd_pkts),
                "Fwd IAT Std": iat_std,
                "Fwd IAT Max": iat_mean * 1.5,
                "Fwd IAT Min": 1.0,
                "Bwd IAT Total": duration_us * 0.5,
                "Bwd IAT Mean": duration_us / max(1, max(1, bwd_pkts)),
                "Bwd IAT Std": iat_std,
                "Bwd IAT Max": iat_mean,
                "Bwd IAT Min": 1.0,
                "Fwd PSH Flags": 0,
                "Bwd PSH Flags": 0,
                "Fwd URG Flags": 0,
                "Bwd URG Flags": 0,
                "Fwd Header Length": fwd_pkts * 20,
                "Bwd Header Length": bwd_pkts * 20,
                "Fwd Packets/s": fwd_pkts / max(1.0, duration_us / 1e6),
                "Bwd Packets/s": bwd_pkts / max(1.0, duration_us / 1e6),
                "Min Packet Length": 40.0,
                "Max Packet Length": max(avg_fwd_len, avg_bwd_len, 40.0),
                "Packet Length Mean": (total_fwd_bytes + total_bwd_bytes) / max(1, (fwd_pkts + bwd_pkts)),
                "Packet Length Std": 10.0,
                "Packet Length Variance": 100.0,
                "FIN Flag Count": 0,
                "SYN Flag Count": syn_flag,
                "RST Flag Count": 0,
                "PSH Flag Count": 0,
                "ACK Flag Count": ack_flag,
                "URG Flag Count": 0,
                "CWE Flag Count": 0,
                "ECE Flag Count": 0,
                "Down/Up Ratio": float(bwd_pkts / max(1, fwd_pkts)),
                "Average Packet Size": (total_fwd_bytes + total_bwd_bytes) / max(1, (fwd_pkts + bwd_pkts)),
                "Avg Fwd Segment Size": avg_fwd_len,
                "Avg Bwd Segment Size": avg_bwd_len,
                "Subflow Fwd Packets": fwd_pkts,
                "Subflow Fwd Bytes": total_fwd_bytes,
                "Subflow Bwd Packets": bwd_pkts,
                "Subflow Bwd Bytes": total_bwd_bytes,
                "Init_Win_bytes_forward": 1024,
                "Init_Win_bytes_backward": 0 if bwd_pkts == 0 else 1024,
                "act_data_pkt_fwd": max(0, fwd_pkts - 1),
                "min_seg_size_forward": 20,
                "Active Mean": duration_us,
                "Active Std": 0.0,
                "Active Max": duration_us,
                "Active Min": duration_us,
                "Idle Mean": 0.0,
                "Idle Std": 0.0,
                "Idle Max": 0.0,
                "Idle Min": 0.0,
            })
            labels.append(atk_type)

        df = pd.DataFrame(rows)
        df["Label"] = labels

        # Shuffle deterministically
        df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
        return df
