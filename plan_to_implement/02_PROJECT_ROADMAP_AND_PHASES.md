# FlowGuard: Implementation Roadmap & Execution Phases

This document provides the structured, phase-by-phase implementation plan for FlowGuard. Each phase has explicit prerequisites, core implementation tasks, test criteria, and deliverables.

---

## Roadmap Timeline Overview

```
Phase 1: Environment Setup & Data Pipeline
   └── Verification: Clean parquet dataset, train/val/test splits, benchmark labels

Phase 2: Sampling & Degradation Simulator (SDUM)
   └── Verification: Unit tests pass for 4 sampling modes; synthetic flow distortion verified

Phase 3: Base Classifier Training & Oblivious Baseline
   └── Verification: LightGBM, Random Forest, XGBoost trained on pristine flows; baseline drop quantified

Phase 4: Feature Uncertainty Propagation & Monte Carlo Sampler
   └── Verification: S plausible flows generated per flow; risk interval [R_min, R_max] computed

Phase 5: SAFE-3WD Policy Implementation & Calibration
   └── Verification: tau_safe and tau_attack calibrated on validation set; 3-way decisions generated

Phase 6: Comprehensive Benchmarking, Ablations & Paper Results
   └── Verification: 5 baseline models evaluated; tables, ROC curves, ECE plots, and runtime profiles exported
```

---

## Detailed Phase Breakdown

### Phase 1: Environment Setup & Data Pipeline
- [ ] **Task 1.1:** Setup Python virtual environment (`python -m venv .venv`) and install `requirements.txt`.
- [ ] **Task 1.2:** Acquire target benchmark dataset (CIC-IDS2017 or CSE-CIC-IDS2018).
- [ ] **Task 1.3:** Build feature extraction & cleaning script:
  - Remove infinite, NaN, and negative timestamp rows.
  - Drop zero-variance and non-informative identifier features (Source IP, Destination IP, Timestamp).
  - Normalize packet lengths, flow bytes/s, flow packets/s, and duration.
- [ ] **Task 1.4:** Partition data into stratified splits:
  - **Train Set (65%):** For base classifier training on complete flows.
  - **Calibration Set (15%):** For tuning SAFE-3WD thresholds $\tau_{\text{safe}}, \tau_{\text{attack}}$.
  - **Test Set (20%):** Pristine test set to be subjected to synthetic sampling degradation.
- **Deliverable:** `data/processed/clean_cic_ids_2017.parquet` and `flowguard/data/loader.py`.

---

### Phase 2: Sampling & Degradation Engine
- [ ] **Task 2.1:** Implement **Random Packet Sampling Engine**:
  - Decimate packet counts by binomial sampling with rates $p \in \{0.10, 0.05, 0.02, 0.01\}$.
- [ ] **Task 2.2:** Implement **Periodic / Systematic Sampling**:
  - Sample 1 packet every $k$ packets ($k \in \{10, 20, 50, 100\}$).
- [ ] **Task 2.3:** Implement **Burst Packet Loss Simulator (Gilbert-Elliott Markov Chain)**:
  - Two states: *Good* (no loss) and *Bad* (loss with probability $1 - h$).
  - Configurable loss rates (2%, 5%, 10%) and burst duration (3 to 8 packets).
- [ ] **Task 2.4:** Implement **Direction-Biased Sampling**:
  - Asymmetric forward vs. backward sampling rates ($p_{\text{fwd}} = 0.10, p_{\text{bwd}} = 0.02$).
- **Deliverable:** `flowguard/sampling/simulator.py` with 100% test coverage in `tests/test_sampling.py`.

---

### Phase 3: Base Intrusion Classifier & Oblivious Benchmark
- [ ] **Task 3.1:** Implement modular classifier wrapper supporting:
  - LightGBM (`LGBMClassifier`)
  - Random Forest (`RandomForestClassifier`)
  - Multi-Layer Perceptron (`MLPClassifier` / PyTorch)
- [ ] **Task 3.2:** Train baseline models on pristine complete flow data.
- [ ] **Task 3.3:** Evaluate model performance on degraded traffic (the **Oblivious Baseline**):
  - Measure performance collapse as $p$ drops from $1.0$ down to $0.01$.
  - Record metric drops: False Negative Rate (FNR) spike, F1 score degradation, Calibration drift.
- **Deliverable:** `flowguard/models/base_classifier.py` and `experiments/01_oblivious_evaluation.py`.

---

### Phase 4: Feature Uncertainty Estimation & Plausible True Flow Sampler
- [ ] **Task 4.1:** Formulate statistical moment inversion formulas for:
  - Flow packet count ($N_{\text{fwd}}, N_{\text{bwd}}$)
  - Flow byte volume ($V_{\text{fwd}}, V_{\text{bwd}}$)
  - Flow duration ($T_{\text{dur}}$) and inter-arrival time statistics (mean, std, max IAT)
- [ ] **Task 4.2:** Implement Monte Carlo generator drawing $S=30$ plausible flow feature representations per sampled flow:
  - Parametric draws (Negative Binomial for counts, Gamma for volume).
  - Correlation preservation via empirical covariance Cholesky decomposition.
- [ ] **Task 4.3:** Implement Risk Propagation Engine:
  - Vectorized batch inference through $f_\theta$ for the $S \times D$ reconstructed flow matrix.
  - Compute output risk profile: $[R_{\min}, R_{\max}]$, mean $\mu_R$, and variance $\sigma_R^2$.
- **Deliverable:** `flowguard/uncertainty/sdum.py` and `flowguard/uncertainty/propagator.py`.

---

### Phase 5: SAFE-3WD Policy & Threshold Calibration
- [ ] **Task 5.1:** Implement the 3-Way Decision Rule:
  - **Region 1 (Benign / Fast-Pass):** $R_{\max} < \tau_{\text{safe}}$
  - **Region 2 (Attack / Block):** $R_{\min} > \tau_{\text{attack}}$
  - **Region 3 (Investigate / Defer):** Interval overlaps boundary or $\sigma_R^2 > \sigma_{\text{thresh}}^2$.
- [ ] **Task 5.2:** Build automated threshold calibration optimizer on the Calibration Split:
  - Cost-sensitive risk minimization balancing false negatives ($C_{\text{FN}}$), false positives ($C_{\text{FP}}$), and inspection budget ($C_{\text{INV}}$):
    $$\min_{\tau_{\text{safe}}, \tau_{\text{attack}}} \mathcal{L}_{\text{cost}} = C_{\text{FN}} \cdot \text{FN} + C_{\text{FP}} \cdot \text{FP} + C_{\text{INV}} \cdot \text{INV}$$
- [ ] **Task 5.3:** Validate stability of calibrated thresholds under varying sampling rates.
- **Deliverable:** `flowguard/decision/safe_3wd.py` and `flowguard/decision/calibrator.py`.

---

### Phase 6: Comprehensive Benchmarking & Paper Artifacts
- [ ] **Task 6.1:** Implement and execute all 4 comparative baselines:
  - **Baseline 1:** Oblivious Model (Standard classifier on raw sampled features).
  - **Baseline 2:** Missing Value Imputation (MICE, Mean, k-NN, Autoencoder reconstruction).
  - **Baseline 3:** Sampling-Augmented Training (Model retrained on multi-rate sampled data).
  - **Baseline 4:** Traditional Uncertainty Estimation (MC-Dropout / Deep Ensemble on sampled data).
- [ ] **Task 6.2:** Compute comparative metrics:
  - Detection Rate, FNR, FPR, F1-Score
  - Selective Accuracy (accuracy on definite decisions)
  - Coverage Rate vs. Investigation Rate
  - Expected Calibration Error (ECE) and Brier Score
  - Runtime Latency per flow (microseconds)
- [ ] **Task 6.3:** Run Ablation Studies:
  - Impact of Monte Carlo sample size $S \in \{5, 10, 20, 30, 50\}$.
  - Single-threshold binary decision vs. SAFE-3WD three-way decision.
  - Sensitivity to mismatched sampling rate estimation.
- [ ] **Task 6.4:** Generate publication-ready figures & LaTeX tables.
- **Deliverable:** Full results suite in `experiments/results/` and publication tables.
