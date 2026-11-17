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

## 3. Semiparametric Latent Gaussian Copula & Sparse-Flow Posterior

- **Gaussian Copula:** Preserves non-linear cross-feature rank dependence using semiparametric normal scores $Z_j = \Phi^{-1}((\text{rank}(X_j) - 0.5) / N)$. Symmetrized and regularized via spectral eigenvalue clipping ($\lambda_i \ge 10^{-4}$) and diagonal rescaling to guarantee positive definiteness.
- **Sparse-Flow Posterior ($k=0$):** Under the empirical prior $P(N=n \mid H_1)$, two-way flows with decimated counts are reconstructed from the exact normalized posterior:
  $$P(N = n \mid k = 0, H_1, p) = \frac{P(N = n \mid H_1) (1 - p)^n}{\sum_{m \ge 1} P(N = m \mid H_1) (1 - p)^m}$$

## 4. Constrained Geometric Temporal Inversion

Inter-arrival times between sampled packets compound over geometric skips. Moments are recovered via Wald's identity and the law of total variance:
$$\mathbb{E}[\tau_{\text{orig}}] = p \cdot \mathbb{E}[\tau_{\text{sampled}}]$$
$$\text{Var}(\tau_{\text{orig}}) = \max\left(\epsilon, p \cdot \text{Var}(\tau_{\text{sampled}}) - p (1 - p) (\mathbb{E}[\tau_{\text{sampled}}])^2\right)$$

## 5. Epistemic Uncertainty Propagation

For each sampled flow feature vector $\tilde{X}$, SDUM generates $S$ Monte Carlo posterior draws:
$$X^{(s)} \sim P(X \mid \tilde{X}, p), \quad s = 1, \dots, S$$
Evaluating through the base classifier $f_\theta$ yields risk distribution $\{R^{(1)}, \dots, R^{(S)}\}$:
- **Mean Risk:** $\bar{R} = \frac{1}{S} \sum_{s=1}^S R^{(s)}$
- **Epistemic Variance:** $\sigma_R^2 = \text{Var}(R^{(s)})$
- **Central Bayesian Credible Interval ($\alpha = 0.90$):** $[R_{q_{0.05}}, R_{q_{0.95}}]$
- **Sample Extrema:** $[R_{\min}, R_{\max}]$

## 6. SAFE-3WD Cost-Sensitive Decision Calibration

Flows are partitioned into three mutually disjoint operational actions:
1. **Accept Benign ($\Omega_A$):** $R_{q_{0.95}} < \tau_{\text{safe}} \;\land\; \sigma_R^2 \le \sigma_{\max}^2$
2. **Immediate Block ($\Omega_D$):** $R_{q_{0.05}} > \tau_{\text{attack}} \;\land\; \sigma_R^2 \le \sigma_{\max}^2$
3. **Investigate / Defer ($\Omega_R$):** $\text{Complement of } (\Omega_A \cup \Omega_D)$

### Continuous Expected Cost Accounting:
Decouples primary classification from secondary triage performance:
$$\mathbb{E}[\text{FN}_{\text{residual}}] = (1 - \text{TPR}_{\text{sec}}) \cdot N_{\text{inv}}^{\text{attack}}$$
$$\mathbb{E}[\text{FP}_{\text{secondary}}] = \text{FPR}_{\text{sec}} \cdot N_{\text{inv}}^{\text{benign}}$$
$$\mathcal{C}_{\text{total}} = \frac{1}{N} \left[ C_{\text{FN}} (\text{FN}_{\text{direct}} + \mathbb{E}[\text{FN}_{\text{residual}}]) + C_{\text{FP}} (\text{FP}_{\text{direct}} + \mathbb{E}[\text{FP}_{\text{secondary}}]) + C_{\text{INV}} N_{\text{inv}} \right]$$
where $\text{TPR}_{\text{sec}}, \text{FPR}_{\text{sec}} \in [0, 1]$ represent secondary deep-inspection triage efficacy.
