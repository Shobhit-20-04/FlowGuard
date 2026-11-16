"""
FlowGuard Theoretical Sampling Distortion Mathematics
======================================================

Mathematical formulations of feature degradation, moments, and variance
under packet decimation (sFlow/NetFlow) and Gilbert-Elliott burst packet loss.
"""

from typing import Dict, Tuple
import numpy as np

def binomial_count_moments(true_count: float, sampling_rate: float) -> Tuple[float, float]:
    """
    Computes theoretical expected value and variance of observed packet count
    under uniform binomial sampling with rate p in (0, 1].
    E[k] = p * N
    Var(k) = N * p * (1 - p)
    """
    p = float(sampling_rate)
    expected_k = p * true_count
    var_k = true_count * p * (1.0 - p)
    return expected_k, var_k

def inverted_count_posterior_moments(observed_count: float, sampling_rate: float) -> Tuple[float, float]:
    """
    Computes posterior mean and variance of unobserved true packet count N
    given observed count k under binomial likelihood and negative binomial posterior.
    E[N | k, p] = k / p
    Var(N | k, p) = k * (1 - p) / (p^2)
    """
    p = max(1e-5, float(sampling_rate))
    k = max(0.0, float(observed_count))
    expected_n = k / p
    var_n = (k * (1.0 - p)) / (p ** 2)
    return expected_n, var_n

def sampled_iat_moments(
    mean_true_iat: float, std_true_iat: float, sampling_rate: float
) -> Tuple[float, float]:
    """
    Computes the theoretical mean and variance of sampled Inter-Arrival Times (IAT).
    When packets are sampled with probability p, the observed IAT tau_tilde is the sum
    of M independent true IATs, where M ~ Geometric(p).
    By Wald's Identity and the law of total variance:
    E[tau_tilde] = E[tau] / p
    Var(tau_tilde) = Var(tau) / p + ((1 - p) / p^2) * (E[tau])^2
    """
    p = max(1e-5, float(sampling_rate))
    mu_tau = float(mean_true_iat)
    var_tau = float(std_true_iat ** 2)

    expected_sampled_iat = mu_tau / p
    var_sampled_iat = (var_tau / p) + (((1.0 - p) / (p ** 2)) * (mu_tau ** 2))
    return expected_sampled_iat, max(0.0, var_sampled_iat)

def invert_sampled_iat_moments(
    mean_sampled_iat: float,
    std_sampled_iat: float,
    sampling_rate: float,
    min_std: float = 1e-4,
) -> Tuple[float, float]:
    """
    Inverts the geometric-sum variance and mean formulas to estimate the original
    pristine inter-arrival time (IAT) mean and standard deviation:
      E[tau] = p * E[tau_tilde]
      Var(tau) = max(min_std^2, p * Var(tau_tilde) - p * (1 - p) * (E[tau_tilde])^2)
      std[tau] = sqrt(Var(tau))
    """
    p = float(np.clip(sampling_rate, 1e-5, 1.0))
    mu_sampled = max(0.0, float(mean_sampled_iat))
    var_sampled = max(0.0, float(std_sampled_iat) ** 2)

    mu_true = p * mu_sampled
    dispersion_term = p * (1.0 - p) * (mu_sampled ** 2)
    var_true = max(float(min_std) ** 2, p * var_sampled - dispersion_term)
    std_true = float(np.sqrt(var_true))

    return mu_true, std_true

def gilbert_elliott_parameters(loss_rate: float, burst_length: float) -> Dict[str, float]:
    """
    Derives Gilbert-Elliott 2-state Markov chain transition probabilities
    from target average loss rate rho and average burst length b.
    States:
      Good (G): packet loss probability = 0
      Bad (B): packet loss probability = 1
    Transition:
      p_BG = 1 / b (probability of recovering from Bad to Good)
      p_GB = rho / (b * (1 - rho)) (probability of entering burst loss)
    """
    rho = float(np.clip(loss_rate, 1e-4, 0.95))
    b = max(1.0, float(burst_length))

    p_bg = 1.0 / b
    p_gb = (rho * p_bg) / (1.0 - rho)
    # Clip to valid probability range
    p_gb = float(np.clip(p_gb, 1e-5, 0.99))

    pi_g = p_bg / (p_gb + p_bg)
    pi_b = p_gb / (p_gb + p_bg)

    return {
        "p_GB": p_gb,
        "p_BG": p_bg,
        "pi_Good": pi_g,
        "pi_Bad": pi_b,
        "effective_loss_rate": pi_b,
        "mean_burst_length": b,
    }
