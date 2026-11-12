# FlowGuard: Implementation Blueprint & Execution Guide

Welcome to the implementation master directory for **FlowGuard: Sampling-Aware Network Intrusion Detection with Feature Uncertainty & Risk-Based Three-Way Decisions (SAFE-3WD)**.

This directory contains the complete technical blueprint, architectural specifications, mathematical formulations, experimental protocols, and project roadmap designed to guide the development of your research codebase.

---

## Document Index & Reading Sequence

| File | Title | Description |
| :--- | :--- | :--- |
| **[01_ARCHITECTURE_AND_SPECS.md](file:///D:/FlowGuard/plan_to_implement/01_ARCHITECTURE_AND_SPECS.md)** | System Architecture & Mathematical Specs | Formal problem formulation, system flowchart, distortion models, Monte Carlo flow generator, Algorithm 1 (SDUM), and Algorithm 2 (SAFE-3WD). |
| **[02_PROJECT_ROADMAP_AND_PHASES.md](file:///D:/FlowGuard/plan_to_implement/02_PROJECT_ROADMAP_AND_PHASES.md)** | Phase-by-Phase Execution Roadmap | 6 detailed phases covering environment setup, degradation simulator, base classifier, uncertainty engine, 3-way calibration, and benchmarks. |
| **[03_DATASET_AND_FEATURE_PIPELINE.md](file:///D:/FlowGuard/plan_to_implement/03_DATASET_AND_FEATURE_PIPELINE.md)** | Dataset Acquisition & Feature Pipeline | Benchmark dataset selection (CIC-IDS2017, CSE-CIC-IDS2018), 3 feature groups (Additive, Temporal/IAT, Intrinsic), and cleaning checklists. |
| **[04_BASELINE_COMPARISONS_AND_METRICS.md](file:///D:/FlowGuard/plan_to_implement/04_BASELINE_COMPARISONS_AND_METRICS.md)** | Baselines & Evaluation Metrics | Precise definitions of the 4 baselines (Oblivious, Imputation, Augmented, Deep Ensemble), 3WD selective metrics (Coverage, Investigation Rate), ECE, and LaTeX tables. |
| **[05_PROJECT_STRUCTURE_AND_CODE_SKELETON.md](file:///D:/FlowGuard/plan_to_implement/05_PROJECT_STRUCTURE_AND_CODE_SKELETON.md)** | Directory Layout & Class Skeletons | Production-grade Python package structure (`flowguard/`), class interfaces, function signatures, and unit testing strategy. |
| **[config_template.yaml](file:///D:/FlowGuard/plan_to_implement/config_template.yaml)** | Master Configuration Template | Centralized YAML config with sampling rates, loss models, Monte Carlo draws ($S=30$), classifier hyperparameters, and decision thresholds. |
| **[requirements.txt](file:///D:/FlowGuard/plan_to_implement/requirements.txt)** | Python Dependency Manifest | Clean list of required libraries (`lightgbm`, `xgboost`, `torch`, `scikit-learn`, `pypdf`, `pyyaml`, `pytest`). |

---

## Core Assets Preserved & Referenced

Your core project deliverables remain completely safe and untouched in their designated locations:

1. **Faculty Slide Deck:** [FlowGuard_Faculty_Presentation.pptx](file:///D:/FlowGuard/FlowGuard_Faculty_Presentation.pptx)  
   *(10-slide academic presentation with speaker notes and SAFE-3WD novelty).*
2. **Project & Slide Explanation Guide:** [FlowGuard_Project_and_Slides_Explanation.pdf](file:///D:/FlowGuard/FlowGuard_Project_and_Slides_Explanation.pdf)  
   *(9-page comprehensive layman guide explaining the project using intuitive analogies with faculty defense Q&A).*
3. **Research Papers Library:** [paper/pdfs/](file:///D:/FlowGuard/paper/pdfs)  
   *(82 full-text verified peer-reviewed articles and preprints).*
4. **Master Literature Metadata:**  
   - Excel Tracker: [paper/metadata/paper_tracking.xlsx](file:///D:/FlowGuard/paper/metadata/paper_tracking.xlsx)  
   - Master CSV: [paper/metadata/paper_tracking.csv](file:///D:/FlowGuard/paper/metadata/paper_tracking.csv)  
   - Markdown Catalog: [paper/metadata/download_summary.md](file:///D:/FlowGuard/paper/metadata/download_summary.md)

---

## How to Begin Coding

When you are ready to initiate Phase 1:
```powershell
# 1. Activate your virtual environment
.\.venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r plan_to_implement/requirements.txt

# 3. Create active config from template
Copy-Item plan_to_implement\config_template.yaml configs\default_config.yaml
```
You can now start implementing `flowguard/data/loader.py` and `flowguard/sampling/simulator.py` following **Phase 1** and **Phase 2** of the roadmap!
