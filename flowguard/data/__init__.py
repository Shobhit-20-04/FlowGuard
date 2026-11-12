"""
FlowGuard Data Pipeline
=======================

Handles feature schema definitions, data hygiene, synthetic traffic generation,
and leak-free stratified train/calibration/test partitioning.
"""

from flowguard.data.loader import FlowDataLoader
from flowguard.data.preprocessor import FlowDataPreprocessor
from flowguard.data.schema import (
    GROUP_A_FEATURES,
    GROUP_B_FEATURES,
    GROUP_C_FEATURES,
    IDENTIFIER_COLUMNS,
    FeatureGroup,
)
from flowguard.data.synthetic import SyntheticFlowGenerator

__all__ = [
    "FeatureGroup",
    "GROUP_A_FEATURES",
    "GROUP_B_FEATURES",
    "GROUP_C_FEATURES",
    "IDENTIFIER_COLUMNS",
    "FlowDataPreprocessor",
    "SyntheticFlowGenerator",
    "FlowDataLoader",
]
