# FlowGuard: Reproducibility & Benchmark Runbook

This guide provides instructions for replicating all benchmarks, model training, and evaluation reported in the FlowGuard project.

## 1. Environment Setup

```bash
# Create and activate Python 3.11 virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install package and dependencies
pip install -r requirements.txt
pip install -e .
```

## 2. Test Suite Verification

Run the complete 37-test suite to verify the mathematical and engineering foundations:
```bash
pytest -v
```

## 3. End-to-End Pipeline Execution

### Step 1: Preprocess & Partition Data
```bash
python experiments/00_prepare_data.py --samples 50000 --seed 42
```
Outputs stratified partitions to `data/processed/` (`train.parquet`, `calibration.parquet`, `test.parquet`).

### Step 2: Oblivious Baseline Evaluation
```bash
python experiments/01_evaluate_oblivious.py
```

### Step 3: SDUM Inversion & Risk Propagation
```bash
python experiments/02_evaluate_sdum_propagation.py
```

### Step 4: Calibrate SAFE-3WD Thresholds
```bash
python experiments/03_calibrate_and_evaluate_safe_3wd.py
```

### Step 5: Multi-Seed Comparative Benchmark & Ablation Study
```bash
python experiments/06_run_corrected_experiments.py
```
Executes evaluation over 5 random seeds (`[42, 123, 456, 789, 101112]`) across 7 sampling rates ($p \in [1.0, 0.01]$). Generates comparative summary tables and LaTeX publication tables.
