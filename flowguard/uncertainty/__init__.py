"""
FlowGuard Uncertainty Quantification & SDUM
============================================
Implements the Sampling Distortion & Uncertainty Model (SDUM):
Inverts sampling distortion, generates plausible true flow ensembles,
and propagates observational variance to compute risk intervals [R_min, R_max].
"""

from flowguard.uncertainty.sdum import SamplingDistortionUncertaintyModel
from flowguard.uncertainty.propagator import RiskProfile, UncertaintyPropagator

__all__ = [
    "SamplingDistortionUncertaintyModel",
    "UncertaintyPropagator",
    "RiskProfile",
]
