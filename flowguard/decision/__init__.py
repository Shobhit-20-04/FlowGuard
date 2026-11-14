"""
FlowGuard Three-Way Decision Policy (SAFE-3WD)
==============================================
Partitions predictions into three calibrated regions:
1. Benign (Fast Accept)
2. Attack (Immediate Block)
3. Uncertain / Investigate (Routed to deep inspection / analyst triage)
"""

from flowguard.decision.safe_3wd import (
    SafeThreeWayDecisionEngine,
    ThreeWayDecisionResult,
)
from flowguard.decision.calibrator import CostSensitiveThresholdCalibrator

__all__ = [
    "SafeThreeWayDecisionEngine",
    "ThreeWayDecisionResult",
    "CostSensitiveThresholdCalibrator",
]
