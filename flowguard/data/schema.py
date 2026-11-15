"""
FlowGuard Canonical Feature Schema & Definitions
=================================================

Defines canonical feature classifications, raw column name mappings,
and statistical properties of network-flow features under packet sampling.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Set

class FeatureGroup(Enum):
    """
    Categorizes flow features based on their theoretical behavior under packet sampling.
    """
    GROUP_A_ADDITIVE = "group_a_additive"   # Scales roughly proportionally with sampling rate p (pkts, bytes)
    GROUP_B_TEMPORAL = "group_b_temporal"   # Non-linear temporal & IAT moments (duration, IAT mean/std)
    GROUP_C_INTRINSIC = "group_c_intrinsic" # Ratios, flag distributions, packet size moments

@dataclass(frozen=True)
class FeatureMetadata:
    name: str
    group: FeatureGroup
    description: str
    min_value: float = 0.0
    is_count: bool = False

# Socket identifiers and metadata that MUST be dropped to prevent shortcut learning
IDENTIFIER_COLUMNS: Set[str] = {
    "flow_id",
    "source_ip",
    "src_ip",
    "destination_ip",
    "dst_ip",
    "source_port",
    "src_port",
    "destination_port",
    "dst_port",
    "timestamp",
    "date",
    "time",
    "simillarhttp",
}

# Canonical Mapping from raw CIC-IDS / CICFlowMeter column names to clean snake_case
RAW_COLUMN_MAP: Dict[str, str] = {
    "Flow ID": "flow_id",
    "Source IP": "source_ip",
    "Src IP": "source_ip",
    "Source Port": "source_port",
    "Src Port": "source_port",
    "Destination IP": "destination_ip",
    "Dst IP": "destination_ip",
    "Destination Port": "destination_port",
    "Dst Port": "destination_port",
    "Timestamp": "timestamp",
    "Flow Duration": "flow_duration",
    "Total Fwd Packets": "total_fwd_packets",
    "Total Backward Packets": "total_bwd_packets",
    "Total Length of Fwd Packets": "total_fwd_bytes",
    "Total Length of Bwd Packets": "total_bwd_bytes",
    "Fwd Packet Length Max": "fwd_pkt_len_max",
    "Fwd Packet Length Min": "fwd_pkt_len_min",
    "Fwd Packet Length Mean": "fwd_pkt_len_mean",
    "Fwd Packet Length Std": "fwd_pkt_len_std",
    "Bwd Packet Length Max": "bwd_pkt_len_max",
    "Bwd Packet Length Min": "bwd_pkt_len_min",
    "Bwd Packet Length Mean": "bwd_pkt_len_mean",
    "Bwd Packet Length Std": "bwd_pkt_len_std",
    "Flow Bytes/s": "flow_bytes_per_s",
    "Flow Packets/s": "flow_packets_per_s",
    "Flow IAT Mean": "flow_iat_mean",
    "Flow IAT Std": "flow_iat_std",
    "Flow IAT Max": "flow_iat_max",
    "Flow IAT Min": "flow_iat_min",
    "Fwd IAT Total": "fwd_iat_total",
    "Fwd IAT Mean": "fwd_iat_mean",
    "Fwd IAT Std": "fwd_iat_std",
    "Fwd IAT Max": "fwd_iat_max",
    "Fwd IAT Min": "fwd_iat_min",
    "Bwd IAT Total": "bwd_iat_total",
    "Bwd IAT Mean": "bwd_iat_mean",
    "Bwd IAT Std": "bwd_iat_std",
    "Bwd IAT Max": "bwd_iat_max",
    "Bwd IAT Min": "bwd_iat_min",
    "Fwd PSH Flags": "fwd_psh_flags",
    "Bwd PSH Flags": "bwd_psh_flags",
    "Fwd URG Flags": "fwd_urg_flags",
    "Bwd URG Flags": "bwd_urg_flags",
    "Fwd Header Length": "fwd_header_len",
    "Bwd Header Length": "bwd_header_len",
    "Fwd Packets/s": "fwd_packets_per_s",
    "Bwd Packets/s": "bwd_packets_per_s",
    "Min Packet Length": "pkt_len_min",
    "Max Packet Length": "pkt_len_max",
    "Packet Length Mean": "pkt_len_mean",
    "Packet Length Std": "pkt_len_std",
    "Packet Length Variance": "pkt_len_var",
    "FIN Flag Count": "fin_flag_count",
    "SYN Flag Count": "syn_flag_count",
    "RST Flag Count": "rst_flag_count",
    "PSH Flag Count": "psh_flag_count",
    "ACK Flag Count": "ack_flag_count",
    "URG Flag Count": "urg_flag_count",
    "CWE Flag Count": "cwe_flag_count",
    "ECE Flag Count": "ece_flag_count",
    "Down/Up Ratio": "down_up_ratio",
    "Average Packet Size": "avg_packet_size",
    "Avg Fwd Segment Size": "avg_fwd_segment_size",
    "Avg Bwd Segment Size": "avg_bwd_segment_size",
    "Fwd Avg Bytes/Bulk": "fwd_avg_bytes_bulk",
    "Fwd Avg Packets/Bulk": "fwd_avg_packets_bulk",
    "Fwd Avg Bulk Rate": "fwd_avg_bulk_rate",
    "Bwd Avg Bytes/Bulk": "bwd_avg_bytes_bulk",
    "Bwd Avg Packets/Bulk": "bwd_avg_packets_bulk",
    "Bwd Avg Bulk Rate": "bwd_avg_bulk_rate",
    "Subflow Fwd Packets": "subflow_fwd_pkts",
    "Subflow Fwd Bytes": "subflow_fwd_bytes",
    "Subflow Bwd Packets": "subflow_bwd_pkts",
    "Subflow Bwd Bytes": "subflow_bwd_bytes",
    "Init_Win_bytes_forward": "init_win_bytes_fwd",
    "Init_Win_bytes_backward": "init_win_bytes_bwd",
    "act_data_pkt_fwd": "act_data_pkt_fwd",
    "min_seg_size_forward": "min_seg_size_fwd",
    "Active Mean": "active_mean",
    "Active Std": "active_std",
    "Active Max": "active_max",
    "Active Min": "active_min",
    "Idle Mean": "idle_mean",
    "Idle Std": "idle_std",
    "Idle Max": "idle_max",
    "Idle Min": "idle_min",
    "Label": "label",
}

# Group A Features (Additive counts and volumes)
GROUP_A_FEATURES: List[str] = [
    "total_fwd_packets",
    "total_bwd_packets",
    "total_fwd_bytes",
    "total_bwd_bytes",
    "fwd_header_len",
    "bwd_header_len",
    "subflow_fwd_pkts",
    "subflow_fwd_bytes",
    "subflow_bwd_pkts",
    "subflow_bwd_bytes",
    "act_data_pkt_fwd",
]

# Strict set of additive byte volume and header length features
# (Strictly distinguishes volume features from TCP window sizes init_win_bytes_fwd/bwd)
ADDITIVE_BYTE_FEATURES: Set[str] = {
    "total_fwd_bytes",
    "total_bwd_bytes",
    "subflow_fwd_bytes",
    "subflow_bwd_bytes",
    "fwd_header_len",
    "bwd_header_len",
}

# Group B Features (Temporal & Inter-Arrival Time moments)
GROUP_B_FEATURES: List[str] = [
    "flow_duration",
    "flow_iat_mean",
    "flow_iat_std",
    "flow_iat_max",
    "flow_iat_min",
    "fwd_iat_total",
    "fwd_iat_mean",
    "fwd_iat_std",
    "fwd_iat_max",
    "fwd_iat_min",
    "bwd_iat_total",
    "bwd_iat_mean",
    "bwd_iat_std",
    "bwd_iat_max",
    "bwd_iat_min",
    "active_mean",
    "active_std",
    "active_max",
    "active_min",
    "idle_mean",
    "idle_std",
    "idle_max",
    "idle_min",
]

# Group C Features (Intrinsic ratios, packet length distributions, TCP flags)
GROUP_C_FEATURES: List[str] = [
    "fwd_pkt_len_max",
    "fwd_pkt_len_min",
    "fwd_pkt_len_mean",
    "fwd_pkt_len_std",
    "bwd_pkt_len_max",
    "bwd_pkt_len_min",
    "bwd_pkt_len_mean",
    "bwd_pkt_len_std",
    "pkt_len_min",
    "pkt_len_max",
    "pkt_len_mean",
    "pkt_len_std",
    "pkt_len_var",
    "flow_bytes_per_s",
    "flow_packets_per_s",
    "fwd_packets_per_s",
    "bwd_packets_per_s",
    "fin_flag_count",
    "syn_flag_count",
    "rst_flag_count",
    "psh_flag_count",
    "ack_flag_count",
    "urg_flag_count",
    "cwe_flag_count",
    "ece_flag_count",
    "down_up_ratio",
    "avg_packet_size",
    "avg_fwd_segment_size",
    "avg_bwd_segment_size",
    "init_win_bytes_fwd",
    "init_win_bytes_bwd",
    "min_seg_size_fwd",
]

# Mapping of raw string attack labels into standardized binary and multiclass taxonomy
LABEL_TAXONOMY: Dict[str, Dict[str, int]] = {
    "binary": {
        "BENIGN": 0,
        "ATTACK": 1,
    },
    "multiclass": {
        "BENIGN": 0,
        "DOS": 1,
        "DDOS": 2,
        "PORTSCAN": 3,
        "BRUTE_FORCE": 4,
        "BOTNET": 5,
        "INFILTRATION": 6,
        "WEB_ATTACK": 7,
        "OTHER": 8,
    }
}
