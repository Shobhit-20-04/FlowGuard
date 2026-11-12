# FlowGuard: Baselines & Evaluation Metrics Specification

## 1. Comparative Baselines

To rigorously establish FlowGuard's scientific validity and novelty, it must be benchmarked against four representative paradigms:

| Baseline ID | Paradigm | Training Data | Inference Mechanism | Decision Policy | Known Limitation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **B1: Oblivious** | Standard NIDS | Pristine Complete Flows | Feed degraded flow $\tilde{X}$ directly into $f_\theta(\tilde{X})$ | Single threshold $\tau = 0.5$ | Blind to sampling; catastrophic FNR spike as rate $p \downarrow$. |
| **B2: Imputation** | Feature Imputation | Pristine Complete Flows | Impute missing values via MICE / Autoencoder / Mean scaling: $\hat{X} = \tilde{X}/p$ | Single threshold $\tau = 0.5$ | Point-estimate only; ignores feature variance & correlation distortion. |
| **B3: Data Augmented** | Multi-Condition Retraining | Multi-Rate Sampled Flows ($p \in [0.01, 1.0]$) | Feed degraded flow $\tilde{X}$ into retrained robust model $f_{\theta,\text{aug}}(\tilde{X})$ | Single threshold $\tau = 0.5$ | Model learns blurry averages; high false alarm rate under low sampling rates. |
| **B4: Deep Uncertainty** | MC-Dropout / Deep Ensemble | Sampled Flows | Draw $K$ predictions using dropout / weight ensembles | Rejection based on output entropy / variance | Captures model epistemic uncertainty, but does NOT model physical sampling distortion. |
| **Proposed: FlowGuard** | Sampling-Aware 3WD (SAFE-3WD) | Pristine Complete Flows | SDUM generates $S$ plausible true flows $\hat{X}^{(s)}$; propagates through $f_\theta$ | **Risk-based Three-Way Decision:** Benign / Attack / Investigate | **Explicitly connects sampling rate $p$, feature distortion, and calibrated risk bounds.** |

---

## 2. Evaluation Metrics

### 2.1 Standard Classification Metrics
Evaluated on flows where a definitive decision is made ($D \in \{\text{Benign}, \text{Attack}\}$):
- **Detection Rate (Recall / Sensitivity):**
  $$\text{DR} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$
- **False Positive Rate (FPR):**
  $$\text{FPR} = \frac{\text{FP}}{\text{FP} + \text{TN}}$$
- **False Negative Rate (FNR):**
  $$\text{FNR} = \frac{\text{FN}}{\text{TP} + \text{FN}} = 1 - \text{DR}$$
- **Precision & F1-Score:**
  $$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}, \quad \text{F1} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

---

### 2.2 Selective Classification & Three-Way Decision Metrics
For a three-way decision system, evaluation must quantify both the quality of accepted decisions and the inspection burden:

- **Coverage Rate ($\mathcal{C}$):** Proportion of flows classified immediately without manual/deep investigation:
  $$\mathcal{C} = \frac{|\mathcal{D}_{\text{Benign}}| + |\mathcal{D}_{\text{Attack}}|}{N_{\text{total}}}$$

- **Investigation Rate ($\mathcal{I}$):** Proportion of flows routed to Region 3 (Uncertain – Investigate):
  $$\mathcal{I} = \frac{|\mathcal{D}_{\text{Investigate}}|}{N_{\text{total}}} = 1 - \mathcal{C}$$

- **Selective Accuracy ($\text{Acc}_{\text{sel}}$):** Accuracy computed strictly over flows accepted with high confidence:
  $$\text{Acc}_{\text{sel}} = \frac{\text{TP}_{\text{definite}} + \text{TN}_{\text{definite}}}{|\mathcal{D}_{\text{Benign}}| + |\mathcal{D}_{\text{Attack}}|}$$

- **Selective Risk ($\mathcal{R}_{\text{sel}}$):** The empirical error penalty of accepted flows:
  $$\mathcal{R}_{\text{sel}} = 1 - \text{Acc}_{\text{sel}}$$

---

### 2.3 Model Calibration Metrics
Uncertainty must reflect true statistical probabilities, not overconfident heuristics:

- **Expected Calibration Error (ECE):**
  Partition the predicted probability range $[0, 1]$ into $M$ equal-width bins $B_m$:
  $$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} \cdot |\text{acc}(B_m) - \text{conf}(B_m)|$$
  where $\text{acc}(B_m)$ is the empirical accuracy of bin $m$ and $\text{conf}(B_m)$ is the average confidence.

- **Brier Score:** Mean squared error between predicted probability and true binary label:
  $$\text{BS} = \frac{1}{N} \sum_{i=1}^N (P(Y_i = 1 \mid \tilde{X}_i) - y_i)^2$$

---

### 2.4 Computational Efficiency
- **Inference Latency:** Average time required to process a flow through SDUM + propagation + decision ($\mu s/\text{flow}$).
- **Throughput:** Processed flows per second (Kpps / Mpps).

---

## 3. Publication LaTeX Table Template

```latex
\begin{table*}[t]
\centering
\caption{Performance Comparison under Packet Sampling Rates $p \in \{0.10, 0.02, 0.01\}$ on CIC-IDS2017.}
\label{tab:main_results}
\resizebox{\textwidth}{!}{
\begin{tabular}{llccccccc}
\toprule
\textbf{Sampling Rate} & \textbf{Method} & \textbf{Recall / DR (\%)} & \textbf{FPR (\%)} & \textbf{F1-Score (\%)} & \textbf{Coverage (\%)} & \textbf{Inv. Rate (\%)} & \textbf{ECE} & \textbf{Latency ($\mu$s)} \\
\midrule
\multirow{5}{*}{$p = 0.10$ (1:10)} 
 & Oblivious (B1)         & -- & -- & -- & 100.0 & 0.0 & -- & -- \\
 & Imputation (B2)        & -- & -- & -- & 100.0 & 0.0 & -- & -- \\
 & Augmented (B3)         & -- & -- & -- & 100.0 & 0.0 & -- & -- \\
 & Deep Ensemble (B4)     & -- & -- & -- & --    & --  & -- & -- \\
 & \textbf{FlowGuard (Ours)} & \textbf{--} & \textbf{--} & \textbf{--} & \textbf{--} & \textbf{--} & \textbf{--} & \textbf{--} \\
\midrule
\multirow{5}{*}{$p = 0.02$ (1:50)} 
 & Oblivious (B1)         & -- & -- & -- & 100.0 & 0.0 & -- & -- \\
 & Imputation (B2)        & -- & -- & -- & 100.0 & 0.0 & -- & -- \\
 & Augmented (B3)         & -- & -- & -- & 100.0 & 0.0 & -- & -- \\
 & Deep Ensemble (B4)     & -- & -- & -- & --    & --  & -- & -- \\
 & \textbf{FlowGuard (Ours)} & \textbf{--} & \textbf{--} & \textbf{--} & \textbf{--} & \textbf{--} & \textbf{--} & \textbf{--} \\
\midrule
\multirow{5}{*}{$p = 0.01$ (1:100)} 
 & Oblivious (B1)         & -- & -- & -- & 100.0 & 0.0 & -- & -- \\
 & Imputation (B2)        & -- & -- & -- & 100.0 & 0.0 & -- & -- \\
 & Augmented (B3)         & -- & -- & -- & 100.0 & 0.0 & -- & -- \\
 & Deep Ensemble (B4)     & -- & -- & -- & --    & --  & -- & -- \\
 & \textbf{FlowGuard (Ours)} & \textbf{--} & \textbf{--} & \textbf{--} & \textbf{--} & \textbf{--} & \textbf{--} & \textbf{--} \\
\bottomrule
\end{tabular}
}
\end{table*}
```
