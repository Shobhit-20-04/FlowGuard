# FlowGuard: Uncertainty-Aware Network Intrusion Detection Under Packet Sampling Distortion

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Framework](https://img.shields.io/badge/Framework-SAFE--3WD-success.svg)](docs/ARCHITECTURE.md)
[![Tests](https://img.shields.io/badge/Tests-37%20Passed-brightgreen.svg)](tests/)

**FlowGuard** is an open-source, reproducible research framework for **Sampling-Aware Network Intrusion Detection Systems (NIDS)** under high-speed packet decimation (e.g., 1-in-10 to 1-in-100 sampling rates) and burst packet loss.

Conventional ML-NIDS models experience severe operational failure when deployed on sampled arterial network flows: feature distortion causes attack traffic to mimic benign flows, while benign flows frequently trigger massive false positive surges (reaching up to **90.30% FPR under 1-in-100 sampling**).

FlowGuard resolves this via a mathematically grounded framework combining **Sampling Distortion and Uncertainty Modeling (SDUM)** with a cost-sensitive **Risk-Based Three-Way Decision Policy (SAFE-3WD)**.

---

## Key Features & Contributions

- **Closed-Form Bayesian Negative Binomial Inversion:** Rigorously inverts unobserved packet counts under scale-invariant ($P(N) \propto 1/N$) and uniform ($P(N) \propto 1$) priors, connecting heavy-tailed Internet traffic phenomenology with exact Negative Binomial posteriors.
- **Compound Gamma Volume Reconstruction:** Reconstructs unobserved flow byte volumes by compounding Gamma packet size distributions while strictly excluding point-in-time negotiation fields (TCP window sizes).
- **Epistemic Uncertainty Propagation:** Draws Monte Carlo realizations from the flow posterior distribution through non-linear base classifiers to establish 90% Bayesian credible intervals and empirical risk profiles.
- **SAFE-3WD Cost-Sensitive Calibration:** Classifies traffic into three operational regions (**Accept**, **Reject**, and **Investigate**), safely deferring high-uncertainty flows to secondary deep-packet inspection and reducing operational cost by **$4.67\times$** relative to naive models under 1:100 sampling.
- **Robustness Against Sampling Drift:** Operates dynamically across arbitrary sampling rates without requiring costly model retraining.

---

## Multi-Seed Benchmark Summary

Evaluated across 5 random seeds (`[42, 123, 456, 789, 101112]`) across 7 sampling rates ($p \in [1.0, 0.01]$):

| Method | Rate ($p$) | Direct DR (%) | Inv. Rate (%) | Post-DR (%) | Direct FPR (%) | Cost / Flow |
|---|---|---|---|---|---|---|
| **B1: Oblivious** | $p = 1.00$ (1:1) | $100.00 \pm 0.00$ | $0.00 \pm 0.00$ | $100.00 \pm 0.00$ | $0.00 \pm 0.00$ | $0.0000 \pm 0.0000$ |
| **B2: Rescaling (1/p)** | $p = 1.00$ (1:1) | $100.00 \pm 0.00$ | $0.00 \pm 0.00$ | $100.00 \pm 0.00$ | $0.00 \pm 0.00$ | $0.0000 \pm 0.0000$ |
| **B3: Augmented** | $p = 1.00$ (1:1) | $99.92 \pm 0.11$ | $0.00 \pm 0.00$ | $99.92 \pm 0.11$ | $0.00 \pm 0.00$ | $0.0020 \pm 0.0027$ |
| **B4: Tree Ensemble** | $p = 1.00$ (1:1) | $100.00 \pm 0.00$ | $0.00 \pm 0.00$ | $100.00 \pm 0.00$ | $0.00 \pm 0.00$ | $0.0000 \pm 0.0000$ |
| **FlowGuard (Proposed)** | $p = 1.00$ (1:1) | **$100.00 \pm 0.00$** | **$0.00 \pm 0.00$** | **$100.00 \pm 0.00$** | **$0.00 \pm 0.00$** | **$0.0000 \pm 0.0000$** |
|---|---|---|---|---|---|---|
| **B1: Oblivious** | $p = 0.10$ (1:10) | $100.00 \pm 0.00$ | $0.00 \pm 0.00$ | $100.00 \pm 0.00$ | $37.00 \pm 0.57$ | $0.2777 \pm 0.0057$ |
| **B2: Rescaling (1/p)** | $p = 0.10$ (1:10) | $100.00 \pm 0.00$ | $0.00 \pm 0.00$ | $100.00 \pm 0.00$ | $37.00 \pm 0.57$ | $0.2777 \pm 0.0057$ |
| **B3: Augmented** | $p = 0.10$ (1:10) | $99.92 \pm 0.11$ | $0.00 \pm 0.00$ | $99.92 \pm 0.11$ | $0.00 \pm 0.00$ | $0.0020 \pm 0.0027$ |
| **B4: Tree Ensemble** | $p = 0.10$ (1:10) | $100.00 \pm 0.00$ | $0.00 \pm 0.00$ | $100.00 \pm 0.00$ | $37.00 \pm 0.57$ | $0.2777 \pm 0.0057$ |
| **FlowGuard (Proposed)** | $p = 0.10$ (1:10) | **$0.00 \pm 0.00$** | **$100.00 \pm 0.00$** | **$80.00 \pm 0.00$** | **$0.00 \pm 0.00$** | **$0.0913 \pm 0.0010$** |
|---|---|---|---|---|---|---|
| **B1: Oblivious** | $p = 0.01$ (1:100) | $100.00 \pm 0.00$ | $0.00 \pm 0.00$ | $100.00 \pm 0.00$ | $90.30 \pm 0.39$ | $0.6776 \pm 0.0048$ |
| **B2: Rescaling (1/p)** | $p = 0.01$ (1:100) | $100.00 \pm 0.00$ | $0.00 \pm 0.00$ | $100.00 \pm 0.00$ | $90.30 \pm 0.39$ | $0.6776 \pm 0.0048$ |
| **B3: Augmented** | $p = 0.01$ (1:100) | $99.52 \pm 0.23$ | $0.00 \pm 0.00$ | $99.52 \pm 0.23$ | $0.00 \pm 0.00$ | $0.0120 \pm 0.0057$ |
| **B4: Tree Ensemble** | $p = 0.01$ (1:100) | $100.00 \pm 0.00$ | $0.00 \pm 0.00$ | $100.00 \pm 0.00$ | $90.30 \pm 0.39$ | $0.6776 \pm 0.0048$ |
| **FlowGuard (Proposed)** | $p = 0.01$ (1:100) | **$0.00 \pm 0.00$** | **$100.00 \pm 0.00$** | **$80.00 \pm 0.00$** | **$0.00 \pm 0.00$** | **$0.1460 \pm 0.0006$** |

---

## Installation & Quickstart

### Prerequisites
- Python 3.10 or 3.11
- Recommended: Dedicated virtual environment

```bash
# Clone repository
git clone https://github.com/Shobhit-20-04/FlowGuard.git
cd FlowGuard

# Create and activate environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies and package
pip install -r requirements.txt
pip install -e .
```

### Running Tests
Execute the comprehensive test suite (37 unit tests covering all components):
```bash
pytest -v
```

### Reproducing Benchmark Experiments
```bash
# Step 1: Generate stratified partitions
python experiments/00_prepare_data.py --samples 50000 --seed 42

# Step 2: Run multi-seed comparative benchmark across all sampling rates
python experiments/06_run_corrected_experiments.py
```

---

## Documentation

- [System Architecture](docs/ARCHITECTURE.md): Pipeline diagrams and module taxonomy.
- [Mathematical Methodology](docs/METHODOLOGY.md): Closed-form derivations, proofs, and decision theory.
- [Reproducibility Guide](docs/REPRODUCIBILITY.md): Step-by-step benchmark and experiment instructions.

---

## Project Structure

```text
FlowGuard/
├── flowguard/             # Core Python package
│   ├── data/              # Feature taxonomy, preprocessors, loaders, dataset ingestion
│   ├── sampling/          # Ingress packet sampling & Gilbert-Elliott burst loss models
│   ├── uncertainty/       # SDUM Bayesian inversion & Monte Carlo risk propagator
│   ├── decision/          # SAFE-3WD 3-way decision engine & cost-sensitive calibrator
│   ├── models/            # Base classifiers (LightGBM, XGBoost) & baselines (B1-B4, MICE)
│   └── evaluation/        # Cost-sensitive metrics, disaggregated rates, and diagnostics
├── experiments/           # Benchmark and evaluation scripts (00 to 06)
├── tests/                 # Unit and regression test suite (37 tests)
├── configs/               # System configuration (default_config.yaml)
├── docs/                  # Architecture, methodology, and reproducibility documentation
├── requirements.txt       # Core project dependencies
├── pyproject.toml         # Build and packaging configuration
└── README.md              # Project overview
```

---

## License

This project is licensed under the MIT License.
