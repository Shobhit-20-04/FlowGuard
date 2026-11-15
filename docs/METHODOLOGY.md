# FlowGuard: Mathematical Methodology & Formal Foundations

## 1. Closed-Form Bayesian Negative Binomial Inversion

Under packet sampling probability $p \in (0, 1]$, the observed packet count $k$ of a flow with true size $N \ge 1$ follows a Binomial distribution:
$$P(k \mid N, p) = \binom{N}{k} p^k (1-p)^{N-k}$$

Conditioned on observing $k \ge 1$ packets, the unobserved packet deficit $U = N - k$ is inverted via Bayes' theorem:

- **Scale-Invariant Prior ($P(N) \propto 1/N$):** Grounded in Internet heavy-tail flow phenomenology:
  $$U \mid k, p \sim \text{NegBin}(r = k, p), \quad \mathbb{E}[N \mid k, p] = \frac{k}{p}$$
- **Uniform Prior ($P(N) \propto 1$):**
  $$U \mid k, p \sim \text{NegBin}(r = k + 1, p), \quad \mathbb{E}[N \mid k, p] = \frac{k}{p} + \frac{1-p}{p}$$
- **Boundary Condition ($p = 1.0$):**
  $$\text{Var}(U \mid k, 1.0) = 0 \implies P(U = 0) = 1.0$$

## 2. Compound Gamma Flow Volume Reconstruction

Assuming individual packet sizes follow $S_i \sim \text{Gamma}(\alpha, \beta)$, the unobserved flow volume is compounded over the unobserved count $U$:
$$V_{\text{unobs}} \mid U \sim \text{Gamma}(U \cdot \alpha, \beta), \quad V_{\text{rec}} = V_{\text{obs}} + V_{\text{unobs}}$$
*Strict Scoping:* Applied strictly to additive volume features (`totlen_fwd_pkts`, `totlen_bwd_pkts`, `tot_fwd_bytes`, `tot_bwd_bytes`). Point-in-time negotiation fields (TCP window sizes `init_win_bytes_fwd/bwd`) are strictly excluded.

## 3. Harmonic Temporal Inversion

Inter-arrival times between sampled packets compound over geometric skips:
$$\mathbb{E}[\Delta t_{\text{orig}}] = p \cdot \mathbb{E}[\Delta t_{\text{sampled}}]$$

## 4. Epistemic Uncertainty Propagation

For each sampled flow feature vector $\tilde{X}$, SDUM generates $S$ Monte Carlo posterior draws:
$$X^{(s)} \sim P(X \mid \tilde{X}, p), \quad s = 1, \dots, S$$
Evaluating through the base classifier $f_\theta$ yields risk distribution $\{R^{(1)}, \dots, R^{(S)}\}$:
- **Mean Risk:** $\bar{R} = \frac{1}{S} \sum_{s=1}^S R^{(s)}$
- **Epistemic Spread:** $\sigma_R = \text{std}(R^{(s)})$
- **90% Bayesian Credible Interval:** $[\hat{q}_{0.05}, \hat{q}_{0.95}]$
- **Sample Extrema:** $[R_{\min}, R_{\max}]$

## 5. SAFE-3WD Cost-Sensitive Decision Calibration

Flows are partitioned into three disjoint operational actions:
1. **Accept (Pass):** $\bar{R} \le \tau_L \;\land\; \sigma_R \le \tau_U$
2. **Reject (Block):** $\bar{R} \ge \tau_H \;\land\; \sigma_R \le \tau_U$
3. **Investigate (Triage):** $\sigma_R > \tau_U \;\lor\; (\tau_L < \bar{R} < \tau_H)$

### Strict Disaggregated Metric Accounting:
$$\text{Direct DR} + \text{Attack Inv Rate} + \text{Definite FNR} = 1.0$$
$$\text{Direct FPR} + \text{Benign Inv Rate} + \text{Direct TNR} = 1.0$$
$$\text{Post-Triage DR} = \text{Direct DR} + \rho_{\text{sec}} \cdot \text{Attack Inv Rate}$$
$$\mathcal{C} = c_{\text{FP}} \cdot \text{Direct FPR} + c_{\text{FN}} \cdot \text{Definite FNR} + c_{\text{inv}} \cdot \text{Inv Rate}$$
where $\rho_{\text{sec}} \in [0, 1]$ represents secondary deep-inspection triage efficacy.
