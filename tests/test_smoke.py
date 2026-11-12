"""
Smoke test to verify that flowguard package and configs import cleanly.
"""
import os
import yaml

def test_imports():
    import flowguard
    import flowguard.data
    import flowguard.sampling
    import flowguard.uncertainty
    import flowguard.decision
    import flowguard.models
    import flowguard.evaluation
    assert flowguard.__version__ == "0.1.0"

def test_config_loads():
    config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "default_config.yaml")
    assert os.path.exists(config_path), f"Config file not found at {config_path}"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    assert cfg["project"]["name"] == "FlowGuard"
    assert "sampling_degradation" in cfg
    assert "uncertainty_model" in cfg
    assert "three_way_decision_policy" in cfg
