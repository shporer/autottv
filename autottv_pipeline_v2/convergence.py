"""
MCMC Convergence Diagnostics for AutoTTV Pipeline v2.0

Provides tools for checking MCMC chain convergence including:
- Gelman-Rubin R-hat statistic
- Effective Sample Size (ESS)
- Autocorrelation time estimation
"""

import numpy as np
import logging
from typing import Dict, Tuple, Optional

from . import config

logger = logging.getLogger(__name__)


def compute_rhat_split(chains: np.ndarray) -> np.ndarray:
    """
    Compute Gelman-Rubin R-hat statistic using split-chain approach.

    This is the standard approach used by emcee and other MCMC packages.
    Each walker's chain is split in half to create 2*n_walkers chains,
    then R-hat is computed across all split chains.

    Parameters
    ----------
    chains : np.ndarray
        Shape (n_steps, n_walkers, n_params) - emcee default format

    Returns
    -------
    np.ndarray
        R-hat values for each parameter (shape: n_params)
    """
    n_steps, n_walkers, n_params = chains.shape

    if n_steps < config.CONVERGENCE_MIN_STEPS:
        return np.ones(n_params) * np.inf

    # Split each walker chain in half — use views instead of concatenate
    split_point = n_steps // 2
    first_half = chains[:split_point, :, :]   # view, shape (n_steps//2, n_walkers, n_params)
    second_half = chains[split_point:2*split_point, :, :]  # view, same shape

    n_steps_half = split_point
    n_chains = 2 * n_walkers

    # Vectorized across all params: compute means per chain-half per param
    # Each has shape (n_walkers, n_params)
    means_first = np.mean(first_half, axis=0)
    means_second = np.mean(second_half, axis=0)
    # Stack to get (n_chains, n_params)
    chain_means = np.concatenate([means_first, means_second], axis=0)

    # Between-chain variance B for each param: shape (n_params,)
    B = n_steps_half * np.var(chain_means, axis=0, ddof=1)

    # Within-chain variance: var per walker per param, then average
    vars_first = np.var(first_half, axis=0, ddof=1)   # (n_walkers, n_params)
    vars_second = np.var(second_half, axis=0, ddof=1)  # (n_walkers, n_params)
    W = (np.sum(vars_first, axis=0) + np.sum(vars_second, axis=0)) / n_chains  # (n_params,)

    # Pooled variance and R-hat, vectorized
    var_plus = ((n_steps_half - 1) / n_steps_half) * W + (1 / n_steps_half) * B
    rhat = np.where(W > 0, np.sqrt(var_plus / W), np.inf)

    return rhat


def compute_autocorr_time_simple(samples: np.ndarray, c: float = config.SOKAL_WINDOW_PARAM) -> np.ndarray:
    """
    Estimate integrated autocorrelation time using Sokal's automatic windowing.

    Parameters
    ----------
    samples : np.ndarray
        Shape (n_samples, n_params) - flattened chain
    c : float
        Window parameter for Sokal's method (default: 5.0)

    Returns
    -------
    np.ndarray
        Autocorrelation time for each parameter
    """
    n_samples, n_params = samples.shape
    tau = np.zeros(n_params)

    for p in range(n_params):
        x = samples[:, p]
        x = x - np.mean(x)

        # Compute autocorrelation using FFT
        n = len(x)
        f = np.fft.fft(x, n=2*n)
        acf = np.fft.ifft(f * np.conjugate(f))[:n].real
        acf /= acf[0]

        # Integrate until autocorrelation drops below threshold
        # Use Sokal's automatic windowing
        tau_est = 1.0
        for m in range(1, n):
            tau_est = 1 + 2 * np.sum(acf[1:m+1])
            if m >= c * tau_est:
                tau[p] = tau_est
                break
        else:
            tau[p] = tau_est  # Use last estimate if window not found

    return tau


def gelman_rubin(chains: np.ndarray) -> np.ndarray:
    """
    Calculate Gelman-Rubin R-hat statistic for convergence diagnosis.

    Parameters
    ----------
    chains : np.ndarray
        Shape (n_walkers, n_steps, n_params) or (n_chains, n_steps, n_params)

    Returns
    -------
    np.ndarray
        R-hat values for each parameter (shape: n_params)

    Notes
    -----
    R-hat < 1.1 generally indicates convergence.
    """
    n_chains, n_steps, n_params = chains.shape

    if n_steps < 2:
        return np.ones(n_params) * np.inf

    # Chain means
    chain_means = np.mean(chains, axis=1)  # (n_chains, n_params)

    # Overall mean
    overall_mean = np.mean(chain_means, axis=0)  # (n_params,)

    # Between-chain variance (B)
    B = n_steps * np.var(chain_means, axis=0, ddof=1)  # (n_params,)

    # Within-chain variance (W)
    chain_vars = np.var(chains, axis=1, ddof=1)  # (n_chains, n_params)
    W = np.mean(chain_vars, axis=0)  # (n_params,)

    # Pooled variance estimate
    var_hat = ((n_steps - 1) / n_steps) * W + (1 / n_steps) * B

    # R-hat
    r_hat = np.sqrt(var_hat / (W + 1e-10))

    return r_hat


def effective_sample_size(chains: np.ndarray) -> np.ndarray:
    """
    Calculate effective sample size (ESS) for MCMC chains.

    Uses the method from Gelman et al. (2013) BDA3.

    Parameters
    ----------
    chains : np.ndarray
        Shape (n_walkers, n_steps, n_params) or flattened (n_samples, n_params)

    Returns
    -------
    np.ndarray
        ESS for each parameter
    """
    if chains.ndim == 3:
        n_chains, n_steps, n_params = chains.shape
        # Flatten chains
        flat_chains = chains.reshape(-1, n_params)
    else:
        flat_chains = chains
        n_steps = len(flat_chains)

    n_samples, n_params = flat_chains.shape
    ess = np.zeros(n_params)

    for i in range(n_params):
        x = flat_chains[:, i]
        # Compute autocorrelation
        n = len(x)
        mean = np.mean(x)
        var = np.var(x)

        if var < 1e-10:
            ess[i] = n
            continue

        # Compute autocorrelation using FFT
        x_centered = x - mean
        fft_result = np.fft.fft(x_centered, n=2*n)
        acf = np.fft.ifft(fft_result * np.conj(fft_result))[:n].real
        acf = acf / acf[0]

        # Find where autocorrelation drops below threshold
        # Sum positive autocorrelations (Geyer's method)
        tau = 1.0
        for k in range(1, n // 2):
            if acf[k] + acf[k+1] < 0:
                break
            tau += 2 * acf[k]

        ess[i] = n / tau

    return ess


def autocorrelation_time(chains: np.ndarray, c: float = config.SOKAL_WINDOW_PARAM) -> Tuple[np.ndarray, bool]:
    """
    Estimate integrated autocorrelation time.

    Parameters
    ----------
    chains : np.ndarray
        Shape (n_walkers, n_steps, n_params)
    c : float
        Truncation threshold (typically 5-10)

    Returns
    -------
    tau : np.ndarray
        Autocorrelation time for each parameter
    converged : bool
        Whether the chain is long enough for reliable tau estimation
    """
    if chains.ndim == 3:
        n_chains, n_steps, n_params = chains.shape
    else:
        return np.ones(chains.shape[-1]) * np.inf, False

    tau = np.zeros(n_params)
    converged = True

    for i in range(n_params):
        # Use emcee's method if available
        try:
            import emcee
            param_chains = chains[:, :, i].T  # (n_steps, n_walkers)
            tau[i] = emcee.autocorr.integrated_time(param_chains, c=c, quiet=True)[0]
        except Exception:
            # Fallback: estimate from ESS
            ess = effective_sample_size(chains[:, :, i:i+1])
            n_total = n_chains * n_steps
            tau[i] = n_total / (ess[0] + 1e-10)

        # Check if chain is long enough
        if n_steps < c * tau[i]:
            converged = False

    return tau, converged


def check_convergence(sampler, min_steps: int = 100,
                     rhat_threshold: float = config.CONVERGENCE_RHAT,
                     ess_threshold: int = config.CONVERGENCE_ESS,
                     autocorr_threshold: int = config.CONVERGENCE_AUTOCORR) -> Dict:
    """
    Check MCMC convergence using multiple diagnostics.

    Parameters
    ----------
    sampler : emcee.EnsembleSampler
        The emcee sampler object
    min_steps : int
        Minimum number of steps before checking convergence
    rhat_threshold : float
        Maximum acceptable R-hat value (typically 1.1)
    ess_threshold : int
        Minimum effective sample size
    autocorr_threshold : int
        Minimum number of autocorrelation times

    Returns
    -------
    dict
        Convergence diagnostics including:
        - converged: bool
        - rhat: array of R-hat values
        - ess: array of effective sample sizes
        - autocorr_time: array of autocorrelation times
        - n_steps: current number of steps
        - messages: list of diagnostic messages
    """
    chains = sampler.get_chain()  # (n_steps, n_walkers, n_params)
    n_steps, n_walkers, n_params = chains.shape

    # Transpose to (n_walkers, n_steps, n_params) for our functions
    chains = chains.transpose(1, 0, 2)

    result = {
        'converged': False,
        'n_steps': n_steps,
        'n_walkers': n_walkers,
        'n_params': n_params,
        'messages': []
    }

    if n_steps < min_steps:
        result['messages'].append(f"Not enough steps ({n_steps} < {min_steps})")
        return result

    # Compute R-hat (cheapest diagnostic — array means/variances, no FFT)
    try:
        rhat = gelman_rubin(chains)
        result['rhat'] = rhat
        rhat_ok = np.all(rhat <= rhat_threshold)
        if not rhat_ok:
            max_rhat = np.max(rhat)
            result['messages'].append(f"R-hat not converged: max={max_rhat:.3f} > {rhat_threshold}")
    except Exception as e:
        result['messages'].append(f"R-hat calculation failed: {e}")
        rhat_ok = False

    # Short-circuit: skip expensive FFT-based diagnostics if R-hat already fails
    if not rhat_ok:
        result['converged'] = False
        return result

    # Compute ESS (requires FFT per parameter)
    try:
        ess = effective_sample_size(chains)
        result['ess'] = ess
        ess_ok = np.all(ess > ess_threshold)
        if not ess_ok:
            min_ess = np.min(ess)
            result['messages'].append(f"ESS too low: min={min_ess:.0f} < {ess_threshold}")
    except Exception as e:
        result['messages'].append(f"ESS calculation failed: {e}")
        ess_ok = False

    # Short-circuit: skip autocorrelation if ESS already fails
    if not ess_ok:
        result['converged'] = False
        return result

    # Compute autocorrelation time (requires FFT per parameter via emcee)
    try:
        tau, tau_converged = autocorrelation_time(chains)
        result['autocorr_time'] = tau
        autocorr_ok = tau_converged and np.all(n_steps > autocorr_threshold * tau)
        if not autocorr_ok:
            max_tau = np.max(tau)
            result['messages'].append(
                f"Autocorr not converged: n_steps={n_steps}, max_tau={max_tau:.1f}, "
                f"need {autocorr_threshold}*tau={autocorr_threshold*max_tau:.1f}"
            )
    except Exception as e:
        result['messages'].append(f"Autocorr calculation failed: {e}")
        autocorr_ok = False

    # Overall convergence
    result['converged'] = rhat_ok and ess_ok and autocorr_ok

    if result['converged']:
        result['messages'].append("MCMC converged successfully")

    # Acceptance fraction
    try:
        result['acceptance_fraction'] = float(np.mean(sampler.acceptance_fraction))
    except Exception:
        pass

    return result


def run_until_converged(sampler, initial_state, max_steps: int = config.N_STEPS_MAX,
                        check_interval: int = config.CONVERGENCE_CHECK_INTERVAL,
                        **convergence_kwargs) -> Dict:
    """
    Run MCMC sampler until convergence criteria are met.

    Parameters
    ----------
    sampler : emcee.EnsembleSampler
        The emcee sampler (should be reset before calling)
    initial_state : np.ndarray
        Initial walker positions
    max_steps : int
        Maximum total steps before giving up
    check_interval : int
        Check convergence every N steps
    **convergence_kwargs
        Additional arguments passed to check_convergence()

    Returns
    -------
    dict
        Final convergence diagnostics
    """
    n_steps_run = 0
    state = initial_state

    while n_steps_run < max_steps:
        # Run for check_interval steps
        steps_this_round = min(check_interval, max_steps - n_steps_run)
        state = sampler.run_mcmc(state, steps_this_round, progress=False)
        n_steps_run += steps_this_round

        # Check convergence
        diagnostics = check_convergence(sampler, **convergence_kwargs)

        if diagnostics['converged']:
            logger.info(f"MCMC converged after {n_steps_run} steps")
            return diagnostics

        if n_steps_run >= max_steps:
            logger.warning(f"MCMC did not converge after {max_steps} steps")
            diagnostics['messages'].append(f"Max steps ({max_steps}) reached without convergence")
            return diagnostics

    return diagnostics


if __name__ == "__main__":
    # Test convergence diagnostics
    import emcee

    # Create a simple test case
    np.random.seed(42)
    n_dim = 3
    n_walkers = 32
    n_steps = 2000

    # Simple Gaussian target
    def log_prob(x):
        return -0.5 * np.sum(x**2)

    sampler = emcee.EnsembleSampler(n_walkers, n_dim, log_prob)
    initial = np.random.randn(n_walkers, n_dim)

    # Run sampler
    sampler.run_mcmc(initial, n_steps, progress=False)

    # Check convergence
    result = check_convergence(sampler)

    print("Convergence Diagnostics:")
    print(f"  Converged: {result['converged']}")
    print(f"  R-hat: {result.get('rhat', 'N/A')}")
    print(f"  ESS: {result.get('ess', 'N/A')}")
    print(f"  Autocorr time: {result.get('autocorr_time', 'N/A')}")
    print(f"  Messages: {result['messages']}")
