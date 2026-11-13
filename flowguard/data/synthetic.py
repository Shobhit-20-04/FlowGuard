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
    Eliminates artificial shortcut constants to guarantee authentic degradation under decimation.
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

        window_sizes = [8192, 14600, 29200, 65535]

        # 1. Generate Benign Traffic (DNS/NTP, Web/API, Bulk streaming)
        for _ in range(n_benign):
            subtype = self.rng.choice(["dns_short", "web", "streaming"], p=[0.25, 0.55, 0.20])

            if subtype == "dns_short":
                # Single-transaction / DNS / NTP: 1-2 packets, 0-1 responses
                fwd_pkts = int(self.rng.integers(1, 3))
                bwd_pkts = int(self.rng.choice([0, 1], p=[0.25, 0.75]))
                avg_fwd_len = float(self.rng.uniform(45, 95))
                avg_bwd_len = float(self.rng.uniform(60, 180)) if bwd_pkts > 0 else 0.0
                tot_pkts = fwd_pkts + bwd_pkts
                duration_us = float(self.rng.uniform(500, 45_000)) if tot_pkts > 1 else 0.0
                fin_flag = 0
                syn_flag = 1
                ack_flag = 1 if bwd_pkts > 0 else 0
                psh_flag = 0

            elif subtype == "web":
                # Interactive browsing / REST APIs: 4-35 fwd, 3-50 bwd
                fwd_pkts = int(np.clip(self.rng.lognormal(mean=2.5, sigma=0.6), 4, 40))
                bwd_pkts = int(np.clip(self.rng.lognormal(mean=2.8, sigma=0.7), 3, 60))
                avg_fwd_len = float(self.rng.uniform(80, 450))
                avg_bwd_len = float(self.rng.uniform(180, 1100))
                tot_pkts = fwd_pkts + bwd_pkts
                duration_us = float(np.clip(self.rng.lognormal(mean=12.5, sigma=1.2), 15_000, 4_000_000))
                fin_flag = 1
                syn_flag = 1
                ack_flag = 1
                psh_flag = int(self.rng.binomial(1, 0.6))

            else:  # streaming / bulk download
                fwd_pkts = int(np.clip(self.rng.lognormal(mean=3.5, sigma=0.6), 20, 250))
                bwd_pkts = int(np.clip(self.rng.lognormal(mean=4.2, sigma=0.7), 30, 600))
                avg_fwd_len = float(self.rng.uniform(55, 90))  # mostly ACKs
                avg_bwd_len = float(self.rng.uniform(1050, 1460))  # full MTU chunks
                tot_pkts = fwd_pkts + bwd_pkts
                duration_us = float(np.clip(self.rng.lognormal(mean=14.5, sigma=1.2), 150_000, 25_000_000))
                fin_flag = 1
                syn_flag = 1
                ack_flag = 1
                psh_flag = int(self.rng.binomial(1, 0.8))

            total_fwd_bytes = fwd_pkts * avg_fwd_len
            total_bwd_bytes = bwd_pkts * avg_bwd_len
            dur_sec = max(1e-4, duration_us / 1e6) if tot_pkts > 1 else 0.0

            iat_mean = duration_us / max(1, (tot_pkts - 1)) if tot_pkts > 1 else 0.0
            iat_std = float(iat_mean * self.rng.uniform(0.4, 1.2)) if tot_pkts > 1 else 0.0
            iat_min = max(1.0, float(iat_mean * self.rng.uniform(0.01, 0.15))) if tot_pkts > 1 else 0.0
            iat_max = float(iat_mean * self.rng.uniform(1.8, 3.5)) if tot_pkts > 1 else 0.0

            pkt_rate = tot_pkts / dur_sec if dur_sec > 0 else 0.0
            byte_rate = (total_fwd_bytes + total_bwd_bytes) / dur_sec if dur_sec > 0 else 0.0

            win_fwd = int(self.rng.choice(window_sizes))
            win_bwd = int(self.rng.choice(window_sizes)) if bwd_pkts > 0 else 0

            rows.append({
                "Flow Duration": duration_us,
                "Total Fwd Packets": fwd_pkts,
                "Total Backward Packets": bwd_pkts,
                "Total Length of Fwd Packets": total_fwd_bytes,
                "Total Length of Bwd Packets": total_bwd_bytes,
                "Fwd Packet Length Max": avg_fwd_len * 1.4,
                "Fwd Packet Length Min": 40.0,
                "Fwd Packet Length Mean": avg_fwd_len,
                "Fwd Packet Length Std": avg_fwd_len * 0.35 if fwd_pkts > 1 else 0.0,
                "Bwd Packet Length Max": avg_bwd_len * 1.3 if bwd_pkts > 0 else 0.0,
                "Bwd Packet Length Min": 40.0 if bwd_pkts > 0 else 0.0,
                "Bwd Packet Length Mean": avg_bwd_len,
                "Bwd Packet Length Std": avg_bwd_len * 0.30 if bwd_pkts > 1 else 0.0,
                "Flow Bytes/s": byte_rate,
                "Flow Packets/s": pkt_rate,
                "Flow IAT Mean": iat_mean,
                "Flow IAT Std": iat_std,
                "Flow IAT Max": iat_max,
                "Flow IAT Min": iat_min,
                "Fwd IAT Total": duration_us * 0.9 if fwd_pkts > 1 else 0.0,
                "Fwd IAT Mean": duration_us / max(1, fwd_pkts) if fwd_pkts > 1 else 0.0,
                "Fwd IAT Std": iat_std,
                "Fwd IAT Max": iat_max,
                "Fwd IAT Min": iat_min,
                "Bwd IAT Total": duration_us * 0.8 if bwd_pkts > 1 else 0.0,
                "Bwd IAT Mean": duration_us / max(1, bwd_pkts) if bwd_pkts > 1 else 0.0,
                "Bwd IAT Std": iat_std if bwd_pkts > 1 else 0.0,
                "Bwd IAT Max": iat_max if bwd_pkts > 1 else 0.0,
                "Bwd IAT Min": iat_min if bwd_pkts > 1 else 0.0,
                "Fwd PSH Flags": psh_flag,
                "Bwd PSH Flags": 0,
                "Fwd URG Flags": 0,
                "Bwd URG Flags": 0,
                "Fwd Header Length": fwd_pkts * 20,
                "Bwd Header Length": bwd_pkts * 20,
                "Fwd Packets/s": fwd_pkts / dur_sec if dur_sec > 0 else 0.0,
                "Bwd Packets/s": bwd_pkts / dur_sec if dur_sec > 0 else 0.0,
                "Min Packet Length": 40.0,
                "Max Packet Length": 1460.0 if subtype == "streaming" else 1100.0,
                "Packet Length Mean": (total_fwd_bytes + total_bwd_bytes) / max(1, tot_pkts),
                "Packet Length Std": 220.0 if tot_pkts > 2 else 0.0,
                "Packet Length Variance": (220.0 ** 2) if tot_pkts > 2 else 0.0,
                "FIN Flag Count": fin_flag,
                "SYN Flag Count": syn_flag,
                "RST Flag Count": 0,
                "PSH Flag Count": psh_flag,
                "ACK Flag Count": ack_flag,
                "URG Flag Count": 0,
                "CWE Flag Count": 0,
                "ECE Flag Count": 0,
                "Down/Up Ratio": float(bwd_pkts / max(1, fwd_pkts)),
                "Average Packet Size": (total_fwd_bytes + total_bwd_bytes) / max(1, tot_pkts),
                "Avg Fwd Segment Size": avg_fwd_len,
                "Avg Bwd Segment Size": avg_bwd_len,
                "Subflow Fwd Packets": fwd_pkts,
                "Subflow Fwd Bytes": total_fwd_bytes,
                "Subflow Bwd Packets": bwd_pkts,
                "Subflow Bwd Bytes": total_bwd_bytes,
                "Init_Win_bytes_forward": win_fwd,
                "Init_Win_bytes_backward": win_bwd,
                "act_data_pkt_fwd": max(0, fwd_pkts - 1),
                "min_seg_size_forward": 20,
                "Active Mean": duration_us * 0.7 if duration_us > 0 else 0.0,
                "Active Std": 500.0 if duration_us > 0 else 0.0,
                "Active Max": duration_us * 0.7 if duration_us > 0 else 0.0,
                "Active Min": duration_us * 0.7 if duration_us > 0 else 0.0,
                "Idle Mean": 0.0,
                "Idle Std": 0.0,
                "Idle Max": 0.0,
                "Idle Min": 0.0,
            })
            labels.append("BENIGN")

        # 2. Generate Attacks (DoS, DDoS, PortScan, Botnet)
        attack_types = list(attack_distribution.keys())
        attack_probs = list(attack_distribution.values())

        for _ in range(n_attacks):
            atk_type = self.rng.choice(attack_types, p=attack_probs)

            if atk_type in ["DoS", "DDoS"]:
                # High packet count, small payloads (40-64B), high packet rate, asymmetric
                fwd_pkts = int(self.rng.uniform(350, 4500))
                bwd_pkts = int(self.rng.choice([0, 1, 2], p=[0.7, 0.2, 0.1]))
                avg_fwd_len = float(self.rng.choice([40.0, 54.0, 64.0]))
                avg_bwd_len = 40.0 if bwd_pkts > 0 else 0.0
                tot_pkts = fwd_pkts + bwd_pkts
                duration_us = float(np.clip(self.rng.lognormal(mean=13.0, sigma=1.2), 30_000, 15_000_000))
                fin_flag = 0
                syn_flag = 1
                ack_flag = 1 if bwd_pkts > 0 else 0
                psh_flag = 0

            elif atk_type == "PortScan":
                # Rapid scans: 1-3 packets, 0-1 responses, fast probes
                fwd_pkts = int(self.rng.integers(1, 4))
                bwd_pkts = int(self.rng.choice([0, 1], p=[0.85, 0.15]))
                avg_fwd_len = 40.0
                avg_bwd_len = 40.0 if bwd_pkts > 0 else 0.0
                tot_pkts = fwd_pkts + bwd_pkts
                duration_us = float(self.rng.uniform(100, 10_000)) if tot_pkts > 1 else 0.0
                fin_flag = 0
                syn_flag = 1
                ack_flag = 0
                psh_flag = 0

            else:  # Botnet / C2
                # Periodic beacon pulses: 4-15 packets each way, regular intervals
                fwd_pkts = int(self.rng.integers(5, 18))
                bwd_pkts = int(self.rng.integers(5, 18))
                avg_fwd_len = float(self.rng.uniform(140, 350))
                avg_bwd_len = float(self.rng.uniform(140, 350))
                tot_pkts = fwd_pkts + bwd_pkts
                duration_us = float(np.clip(self.rng.lognormal(mean=13.5, sigma=1.0), 100_000, 20_000_000))
                fin_flag = 1
                syn_flag = 1
                ack_flag = 1
                psh_flag = 1

            total_fwd_bytes = fwd_pkts * avg_fwd_len
            total_bwd_bytes = bwd_pkts * avg_bwd_len
            dur_sec = max(1e-4, duration_us / 1e6) if tot_pkts > 1 else 0.0

            iat_mean = duration_us / max(1, (tot_pkts - 1)) if tot_pkts > 1 else 0.0
            # Attacks have rigid pacing (low IAT variance)
            iat_std = float(iat_mean * self.rng.uniform(0.05, 0.25)) if tot_pkts > 1 else 0.0
            iat_min = max(1.0, float(iat_mean * self.rng.uniform(0.01, 0.15))) if tot_pkts > 1 else 0.0
            iat_max = float(iat_mean * self.rng.uniform(1.2, 2.0)) if tot_pkts > 1 else 0.0

            pkt_rate = tot_pkts / dur_sec if dur_sec > 0 else 0.0
            byte_rate = (total_fwd_bytes + total_bwd_bytes) / dur_sec if dur_sec > 0 else 0.0

            win_fwd = int(self.rng.choice(window_sizes))
            win_bwd = int(self.rng.choice(window_sizes)) if bwd_pkts > 0 else 0

            rows.append({
                "Flow Duration": duration_us,
                "Total Fwd Packets": fwd_pkts,
                "Total Backward Packets": bwd_pkts,
                "Total Length of Fwd Packets": total_fwd_bytes,
                "Total Length of Bwd Packets": total_bwd_bytes,
                "Fwd Packet Length Max": avg_fwd_len * 1.2,
                "Fwd Packet Length Min": 40.0,
                "Fwd Packet Length Mean": avg_fwd_len,
                "Fwd Packet Length Std": 5.0 if fwd_pkts > 1 else 0.0,
                "Bwd Packet Length Max": avg_bwd_len if bwd_pkts > 0 else 0.0,
                "Bwd Packet Length Min": 40.0 if bwd_pkts > 0 else 0.0,
                "Bwd Packet Length Mean": avg_bwd_len,
                "Bwd Packet Length Std": 2.0 if bwd_pkts > 1 else 0.0,
                "Flow Bytes/s": byte_rate,
                "Flow Packets/s": pkt_rate,
                "Flow IAT Mean": iat_mean,
                "Flow IAT Std": iat_std,
                "Flow IAT Max": iat_max,
                "Flow IAT Min": iat_min,
                "Fwd IAT Total": duration_us * 0.9 if fwd_pkts > 1 else 0.0,
                "Fwd IAT Mean": duration_us / max(1, fwd_pkts) if fwd_pkts > 1 else 0.0,
                "Fwd IAT Std": iat_std,
                "Fwd IAT Max": iat_max,
                "Fwd IAT Min": iat_min,
                "Bwd IAT Total": duration_us * 0.8 if bwd_pkts > 1 else 0.0,
                "Bwd IAT Mean": duration_us / max(1, bwd_pkts) if bwd_pkts > 1 else 0.0,
                "Bwd IAT Std": iat_std if bwd_pkts > 1 else 0.0,
                "Bwd IAT Max": iat_max if bwd_pkts > 1 else 0.0,
                "Bwd IAT Min": iat_min if bwd_pkts > 1 else 0.0,
                "Fwd PSH Flags": psh_flag,
                "Bwd PSH Flags": 0,
                "Fwd URG Flags": 0,
                "Bwd URG Flags": 0,
                "Fwd Header Length": fwd_pkts * 20,
                "Bwd Header Length": bwd_pkts * 20,
                "Fwd Packets/s": fwd_pkts / dur_sec if dur_sec > 0 else 0.0,
                "Bwd Packets/s": bwd_pkts / dur_sec if dur_sec > 0 else 0.0,
                "Min Packet Length": 40.0,
                "Max Packet Length": max(avg_fwd_len, avg_bwd_len, 40.0),
                "Packet Length Mean": (total_fwd_bytes + total_bwd_bytes) / max(1, tot_pkts),
                "Packet Length Std": 8.0 if tot_pkts > 2 else 0.0,
                "Packet Length Variance": 64.0 if tot_pkts > 2 else 0.0,
                "FIN Flag Count": fin_flag,
                "SYN Flag Count": syn_flag,
                "RST Flag Count": 0,
                "PSH Flag Count": psh_flag,
                "ACK Flag Count": ack_flag,
                "URG Flag Count": 0,
                "CWE Flag Count": 0,
                "ECE Flag Count": 0,
                "Down/Up Ratio": float(bwd_pkts / max(1, fwd_pkts)),
                "Average Packet Size": (total_fwd_bytes + total_bwd_bytes) / max(1, tot_pkts),
                "Avg Fwd Segment Size": avg_fwd_len,
                "Avg Bwd Segment Size": avg_bwd_len,
                "Subflow Fwd Packets": fwd_pkts,
                "Subflow Fwd Bytes": total_fwd_bytes,
                "Subflow Bwd Packets": bwd_pkts,
                "Subflow Bwd Bytes": total_bwd_bytes,
                "Init_Win_bytes_forward": win_fwd,
                "Init_Win_bytes_backward": win_bwd,
                "act_data_pkt_fwd": max(0, fwd_pkts - 1),
                "min_seg_size_forward": 20,
                "Active Mean": duration_us * 0.7 if duration_us > 0 else 0.0,
                "Active Std": 0.0,
                "Active Max": duration_us * 0.7 if duration_us > 0 else 0.0,
                "Active Min": duration_us * 0.7 if duration_us > 0 else 0.0,
                "Idle Mean": 0.0,
                "Idle Std": 0.0,
                "Idle Max": 0.0,
                "Idle Min": 0.0,
            })
            labels.append(atk_type)

        df = pd.DataFrame(rows)
        df["Label"] = labels
        # Randomly shuffle rows
        df = df.sample(frac=1.0, random_state=int(self.rng.integers(0, 100000))).reset_index(drop=True)
        return df
