"""
FlowGuard Sampling & Loss Simulation
====================================
Simulates packet-level decimation mechanisms:
- Uniform Random Sampling (1/N)
- Periodic Systematic Sampling (1 in k)
- Burst Packet Loss (Gilbert-Elliott Markov Chain)
- Direction-Biased Asymmetric Sampling
"""

from flowguard.sampling.distortion_math import (
    binomial_count_moments,
    gilbert_elliott_parameters,
    inverted_count_posterior_moments,
    sampled_iat_moments,
)
from flowguard.sampling.simulator import PacketSamplingSimulator

__all__ = [
    "PacketSamplingSimulator",
    "binomial_count_moments",
    "inverted_count_posterior_moments",
    "sampled_iat_moments",
    "gilbert_elliott_parameters",
]
