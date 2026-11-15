# FlowGuard: System Architecture

FlowGuard is an uncertainty-aware network intrusion detection framework designed to operate reliably under severe ingress packet sampling (e.g., 1-in-10 to 1-in-100 sampling rates).

```
                            FlowGuard Pipeline
                            
  Raw Network Packets
          │
          ▼
   [ Packet Sampling ]  ── (Random, Periodic, or Gilbert-Elliott Burst Loss)
          │
          ▼
   Sampled Flow Records
          │
          ▼
  ┌────────────────────────────────────────────────────────┐
  │ 1. Feature Preprocessing & Hygiene (flowguard.data)   │
  │    - 8 Disjoint feature groups                         │
  │    - Robust median/IQR scaling & infinity clipping     │
  │    - Socket identifier stripping                       │
  └───────────────────────────┬────────────────────────────┘
                              │
                              ▼
  ┌────────────────────────────────────────────────────────┐
  │ 2. SDUM: Sampling Distortion Model (flowguard.uncertainty)│
  │    - Closed-form Bayes Negative Binomial inversion     │
  │    - Compound Gamma flow volume reconstruction         │
  │    - Harmonic mean inter-arrival time scaling          │
  └───────────────────────────┬────────────────────────────┘
                              │
                              ▼
  ┌────────────────────────────────────────────────────────┐
  │ 3. Monte Carlo Uncertainty Propagation                 │
  │    - S posterior draws per sampled flow               │
  │    - Base classifier inference (LightGBM / XGBoost)   │
  │    - Epistemic risk profile: R_mean, sigma_R, quantiles│
  └───────────────────────────┬────────────────────────────┘
                              │
                              ▼
  ┌────────────────────────────────────────────────────────┐
  │ 4. SAFE-3WD Decision Engine (flowguard.decision)       │
  │    - 3-Way Partition: Accept, Reject, Investigate      │
  │    - Cost-sensitive multi-threshold calibration        │
  │    - Secondary deep-inspection triage modeling         │
  └────────────────────────────────────────────────────────┘
```

## Module Taxonomy

- `flowguard.data`: Feature taxonomy schema, robust preprocessing, synthetic flow generation, real dataset ingestion (CIC-IDS2017 & CSE-CIC-IDS2018).
- `flowguard.sampling`: Ingress packet sampling simulation (uniform random, periodic, Gilbert-Elliott 2-state Markov burst loss, asymmetric directional sampling).
- `flowguard.uncertainty`: Sampling Distortion and Uncertainty Model (SDUM) and Monte Carlo risk propagation into Bayesian credible intervals.
- `flowguard.decision`: SAFE-3WD decision engine, cost-sensitive threshold calibration, and selective triage accounting.
- `flowguard.models`: Base gradient boosting classifiers (LightGBM, XGBoost) and comparative baselines (B1 Oblivious, B2 Rescaling, B3 Sampling-Augmented, B4 Tree Ensemble, MICE Imputation).
- `flowguard.evaluation`: Cost-sensitive metrics, disaggregated rates, and selective classification metrics.
