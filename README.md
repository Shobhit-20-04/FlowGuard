# FlowGuard: Sampling-Aware Network Intrusion Detection

**M.Tech Solo Research Project**

## 1. Overview

**FlowGuard** addresses the fundamental challenge of network intrusion detection in high-speed, high-bandwidth networks where complete packet inspection is computationally prohibitive. Real-world network monitoring architectures (such as NetFlow, IPFIX, and sFlow) rely on **packet sampling**, which systematically introduces incompleteness, distorted feature distributions, and flow-statistic uncertainty.

Traditional machine learning (ML) and deep learning (DL) Network Intrusion Detection Systems (NIDS) are trained on complete, unsampled network traces (e.g., standard CSE-CIC-IDS2018, UNSW-NB15, or CIC-IDS2017) and experience severe performance degradation, high false alarm rates, and missed intrusions when deployed on sampled traffic.

FlowGuard investigates **sampling-aware decision algorithms**, **uncertainty quantification (UQ)**, and **selective classification (abstention / reject options)** to build a trustworthy and risk-calibrated intrusion detection framework.

---

## 2. Core Research Pillars

1. **Packet Sampling & Incomplete Traffic Dynamics:**
   - Modeling feature distortion across uniform random, deterministic, and adaptive packet sampling regimes.
   - Flow-level statistics under sampling: packet count, byte volume, duration, and inter-arrival time (IAT) variance.

2. **Flow-Feature Uncertainty & Missing Data:**
   - Formal characterization of epistemic and aleatoric uncertainty resulting from incomplete traffic.
   - Missing data estimation and robust imputation algorithms tailored for streaming network flows.

3. **Uncertainty Quantification (UQ):**
   - Calibrated confidence estimation using Bayesian deep learning, Monte Carlo Dropout, Evidential Deep Learning, and Conformal Prediction.
   - Distinguishing out-of-distribution traffic shifts from sampling-induced noise.

4. **Selective Classification & Risk-Aware Decisioning:**
   - Abstention mechanisms ("reject option") for low-confidence or high-uncertainty flow classifications.
   - Hierarchical escalation pathways: autonomous low-risk mitigation vs. routing uncertain high-risk flows to targeted Deep Packet Inspection (DPI) or security analyst triage.

5. **Cross-Dataset Generalization & Concept Drift Adaptation:**
   - Evaluating temporal stability and label-shift robustness under continuous traffic evolution.

---

## 3. Project Structure

```text
FlowGuard/
├── src/                  # Source code for sampling, feature extraction, and models
├── scripts/              # Reproducible experiment runners and simulation benchmarks
├── configs/              # Model and experiment hyperparameter configurations
├── tests/                # Unit and integration test suites
├── docs/                 # Architectural specifications, technical reports, and notes
├── .gitignore            # Git exclusion rules (private paper collection excluded)
└── README.md             # Project overview and research methodology
```

> **Note on Research Literature:**  
> The internal literature review collection, paper metadata spreadsheets, and downloaded research PDFs are maintained in a dedicated, private local directory (`paper/`) and are strictly excluded from version control to respect publisher licensing and copyright agreements.
