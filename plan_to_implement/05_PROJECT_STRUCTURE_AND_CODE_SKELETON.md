# FlowGuard: Project Structure & Code Skeleton

## 1. Directory Tree Architecture

When implementing FlowGuard, the project should follow standard Python packaging conventions for modularity, reproducibility, and unit testing:

```
FlowGuard/
├── FlowGuard_Faculty_Presentation.pptx          # Safe: Faculty presentation
├── FlowGuard_Project_and_Slides_Explanation.pdf # Safe: 9-page project guide
├── README.md                                    # Root overview
│
├── paper/                                       # Safe: Research repository
│   ├── metadata/
│   │   ├── download_summary.md
│   │   ├── paper_tracking.csv
│   │   └── paper_tracking.xlsx
│   └── pdfs/                                    # 82 verified research papers
│
├── plan_to_implement/                           # Implementation blueprint
│   ├── 01_ARCHITECTURE_AND_SPECS.md
│   ├── 02_PROJECT_ROADMAP_AND_PHASES.md
│   ├── 03_DATASET_AND_FEATURE_PIPELINE.md
│   ├── 04_BASELINE_COMPARISONS_AND_METRICS.md
│   ├── 05_PROJECT_STRUCTURE_AND_CODE_SKELETON.md
│   ├── config_template.yaml
│   ├── requirements.txt
│   └── README.md
│
├── configs/                                     # Active YAML experiment configs
│   └── default_config.yaml
│
├── data/                                        # Dataset storage
│   ├── raw/                                     # Raw CSV / Parquet
│   └── processed/                               # Cleaned, normalized parquet
│
├── flowguard/                                   # Core implementation package
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py                            # Parquet loader & splitting
│   │   └── preprocessor.py                      # Outlier removal, scaler, encoding
│   ├── sampling/
│   │   ├── __init__.py
│   │   ├── simulator.py                         # Random, Periodic, Gilbert-Elliott
│   │   └── distortion_math.py                   # Theoretical moments & variances
│   ├── uncertainty/
│   │   ├── __init__.py
│   │   ├── sdum.py                              # Sampling Distortion & Uncertainty Model
│   │   ├── plausible_generator.py               # Monte Carlo true flow sampler
│   │   └── propagator.py                        # Risk distribution propagator
│   ├── decision/
│   │   ├── __init__.py
│   │   ├── safe_3wd.py                          # 3-Way Decision rule engine
│   │   └── calibrator.py                        # Cost-sensitive threshold tuning
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base_classifier.py                   # LightGBM, RF, MLP wrappers
│   │   └── baselines.py                         # B1 (Oblivious), B2 (Impute), B3 (Augment)
│   └── evaluation/
│       ├── __init__.py
│       ├── metrics.py                           # DR, FPR, FNR, ECE, Coverage
│       └── visualizer.py                        # Risk interval plots, ROC, Reliability diagrams
│
├── experiments/                                 # Executable experiment scripts
│   ├── 01_train_base_model.py
│   ├── 02_evaluate_oblivious.py
│   ├── 03_calibrate_safe_3wd.py
│   ├── 04_run_benchmarks.py
│   └── 05_generate_paper_artifacts.py
│
└── tests/                                       # Unit and integration tests
    ├── test_sampling.py
    ├── test_sdum.py
    ├── test_decision.py
    └── test_metrics.py
```

---

## 2. Core Python Class Interfaces

### 2.1 Sampling Simulator (`flowguard/sampling/simulator.py`)
```python
import numpy as np

class PacketSamplingSimulator:
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def random_sample_flow(self, flow_vector: np.ndarray, rate: float) -> np.ndarray:
        """
        Decimates packet counts and scales extrinsic features by binomial sampling.
        """
        sampled = flow_vector.copy()
        # count indices: e.g. fwd_pkts (idx 0), bwd_pkts (idx 1)
        sampled[0] = self.rng.binomial(n=int(flow_vector[0]), p=rate)
        sampled[1] = self.rng.binomial(n=int(flow_vector[1]), p=rate)
        # scale byte volumes proportionally with stochastic noise
        sampled[2] = flow_vector[2] * (sampled[0] / max(1, flow_vector[0]))
        return sampled

    def gilbert_elliott_burst_loss(self, flow_vector: np.ndarray, loss_rate: float, burst_len: float) -> np.ndarray:
        """
        Simulates consecutive burst drops using a two-state Markov chain.
        """
        ...
```

### 2.2 SDUM True Flow Generator (`flowguard/uncertainty/sdum.py`)
```python
import numpy as np
from typing import List

class SamplingDistortionUncertaintyModel:
    def __init__(self, num_draws: int = 30, seed: int = 42):
        self.num_draws = num_draws
        self.rng = np.random.default_rng(seed)

    def generate_plausible_flows(self, sampled_flow: np.ndarray, sampling_rate: float) -> np.ndarray:
        """
        Draws S plausible true flow representations using negative binomial posteriors
        for counts and Gamma distributions for volumes.
        Returns: matrix of shape (num_draws, num_features)
        """
        S = self.num_draws
        D = len(sampled_flow)
        plausible_matrix = np.tile(sampled_flow, (S, 1))

        k_fwd = max(1, int(sampled_flow[0]))
        # Negative binomial draw for true packet count
        plausible_matrix[:, 0] = self.rng.negative_binomial(n=k_fwd, p=sampling_rate, size=S) + k_fwd
        # Gamma draw for byte volume
        scale = max(1.0, sampled_flow[2]) / (k_fwd * sampling_rate)
        plausible_matrix[:, 2] = self.rng.gamma(shape=k_fwd, scale=scale, size=S)

        return plausible_matrix
```

### 2.3 SAFE-3WD Decision Engine (`flowguard/decision/safe_3wd.py`)
```python
import numpy as np
from enum import Enum

class DecisionOutcome(Enum):
    BENIGN = 0
    ATTACK = 1
    INVESTIGATE = 2

class SAFE3WDPolicy:
    def __init__(self, tau_safe: float = 0.20, tau_attack: float = 0.80, max_variance: float = 0.04):
        self.tau_safe = tau_safe
        self.tau_attack = tau_attack
        self.max_variance = max_variance

    def decide(self, risk_distribution: np.ndarray) -> dict:
        """
        Evaluates risk array R = [r_1, ..., r_S] and emits a three-way decision.
        """
        r_min = float(np.min(risk_distribution))
        r_max = float(np.max(risk_distribution))
        mu_r = float(np.mean(risk_distribution))
        var_r = float(np.var(risk_distribution))

        if r_max < self.tau_safe and var_r <= self.max_variance:
            outcome = DecisionOutcome.BENIGN
        elif r_min > self.tau_attack and var_r <= self.max_variance:
            outcome = DecisionOutcome.ATTACK
        else:
            outcome = DecisionOutcome.INVESTIGATE

        return {
            "decision": outcome.name,
            "r_min": r_min,
            "r_max": r_max,
            "mean_risk": mu_r,
            "risk_variance": var_r
        }
```

---

## 3. Immediate Smoke Test Setup

To verify installation and initial imports once coding begins, run:
```powershell
pytest tests/
```
All unit tests should validate:
1. Determinism across seeds.
2. Inverted bounds ($\mathbb{E}[\hat{N}] \approx N$).
3. Boundary edge cases in decision logic ($R \in [0, 1]$).
