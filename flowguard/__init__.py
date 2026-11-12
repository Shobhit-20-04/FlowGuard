"""
FlowGuard: Sampling-Aware Network Intrusion Detection
=====================================================

A modular framework connecting packet-sampling mechanisms, flow-feature
uncertainty estimation, and risk-based three-way decisions (SAFE-3WD).
"""

__version__ = "0.1.0"
__author__ = "FlowGuard Research Project"

from typing import List

__all__: List[str] = [
    "data",
    "sampling",
    "uncertainty",
    "decision",
    "models",
    "evaluation",
]
