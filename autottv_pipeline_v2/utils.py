"""
Utility Functions for AutoTTV Pipeline v2.0

Provides shared utility functions used across multiple modules:
- Batman transit model computation with exposure time integration
- JSON serialization for numpy types
- Chi-squared calculation
- MCMC parameter extraction from samples
"""

import math
from collections import namedtuple
import numpy as np
from scipy import stats
from typing import Dict, Any, Tuple, Optional

try:
    import batman
    BATMAN_AVAILABLE = True
except ImportError:
    BATMAN_AVAILABLE = False

from . import config

# =============================================================================
# Batman Transit Model Utilities
# =============================================================================

# Import from centralized config
LONG_CADENCE_THRESHOLD = config.LONG_CADENCE_THRESHOLD

# Precomputed constant to avoid per-call math.degrees overhead
_RAD2DEG = 180.0 / math.pi

# Structured container for cached transit models
CachedTransitModel = namedtuple('CachedTransitModel', ['model', 'is_mixed', 'buffer'])


def update_batman_params(params: 'batman.TransitParams', period: float, t0: float,
                         rp_rs: float, a_rs: float, b: float,
                         u1: float, u2: float) -> None:
    """
    Update batman TransitParams in-place instead of creating a new object.

    Parameters
    ----------
    params : batman.TransitParams
        Existing TransitParams object to update
    period, t0, rp_rs, a_rs, b, u1, u2 : float
        Transit parameters to set
    """
    params.per = period
    params.t0 = t0
    params.rp = rp_rs
    params.a = a_rs
    params.inc = math.acos(max(0.0, min(b / a_rs, 1.0))) * _RAD2DEG if a_rs > 0 else 90.0
    params.u[0] = u1
    params.u[1] = u2


def create_cached_transit_model(time: np.ndarray, params: 'batman.TransitParams',
                                cadence=None,
                                long_cadence_threshold: float = LONG_CADENCE_THRESHOLD):
    """
    Create batman TransitModel object(s) that can be reused across MCMC iterations.

    The returned object should be passed as ``cached_model`` to ``compute_batman_model()``.

    Parameters
    ----------
    time : np.ndarray
        Time array in days
    params : batman.TransitParams
        Batman transit parameters (used only for initial model creation)
    cadence : np.ndarray or float, optional
        Cadence in seconds. If None, assumes short cadence.
    long_cadence_threshold : float
        Cadence above this threshold triggers exposure integration.

    Returns
    -------
    batman.TransitModel or list of (np.ndarray, batman.TransitModel)
        Single model for uniform cadence, or list of (mask, model) tuples for mixed cadences.
    """
    if not BATMAN_AVAILABLE:
        raise ImportError("batman-package is required for transit model computation")

    if cadence is None:
        m = batman.TransitModel(params, time)
        return CachedTransitModel(model=m, is_mixed=False, buffer=None)

    if np.isscalar(cadence):
        cadence = np.full(len(time), cadence)

    unique_cads = np.unique(cadence)

    if len(unique_cads) == 1:
        cad_sec = unique_cads[0]
        if cad_sec > long_cadence_threshold:
            exp_time_days = cad_sec / config.SECONDS_PER_DAY
            supersample = max(config.BATMAN_MIN_SUPERSAMPLE,
                              int(cad_sec / config.BATMAN_SUPERSAMPLE_REF_CADENCE))
            m = batman.TransitModel(params, time,
                                       supersample_factor=supersample,
                                       exp_time=exp_time_days)
        else:
            m = batman.TransitModel(params, time)
        return CachedTransitModel(model=m, is_mixed=False, buffer=None)

    # Mixed cadences — return CachedTransitModel with list of (mask, model) tuples
    parts = []
    for cad_sec in unique_cads:
        mask = cadence == cad_sec
        time_subset = time[mask]
        if cad_sec > long_cadence_threshold:
            exp_time_days = cad_sec / config.SECONDS_PER_DAY
            supersample = max(config.BATMAN_MIN_SUPERSAMPLE,
                              int(cad_sec / config.BATMAN_SUPERSAMPLE_REF_CADENCE))
            m = batman.TransitModel(params, time_subset,
                                    supersample_factor=supersample,
                                    exp_time=exp_time_days)
        else:
            m = batman.TransitModel(params, time_subset)
        parts.append((mask, m))
    return CachedTransitModel(model=parts, is_mixed=True, buffer=np.empty(len(time)))


def compute_batman_model(time: np.ndarray, params: 'batman.TransitParams',
                         cadence: Optional[np.ndarray] = None,
                         long_cadence_threshold: float = LONG_CADENCE_THRESHOLD,
                         cached_model=None) -> np.ndarray:
    """
    Compute batman transit model with optional exposure time integration.

    For long-cadence data (>200s by default), the model is supersampled and
    integrated over the exposure time to accurately model the light curve.

    Parameters
    ----------
    time : np.ndarray
        Time array in days
    params : batman.TransitParams
        Batman transit parameters object (already configured)
    cadence : np.ndarray or float, optional
        Cadence in seconds for each time point. If None, assumes short cadence.
        Can be a single value or array matching time length.
    long_cadence_threshold : float
        Cadence above this threshold (in seconds) triggers exposure integration.
        Default: 200.0 seconds.
    cached_model : batman.TransitModel or list, optional
        Pre-created TransitModel from ``create_cached_transit_model()``.
        When provided, skips TransitModel construction (major speedup for MCMC).

    Returns
    -------
    np.ndarray
        Model flux values
    """
    if not BATMAN_AVAILABLE:
        raise ImportError("batman-package is required for transit model computation")

    # Fast path: use cached model
    if cached_model is not None:
        if hasattr(cached_model, 'is_mixed'):
            # New CachedTransitModel namedtuple
            if cached_model.is_mixed:
                buf = cached_model.buffer
                buf[:] = 0.0
                for mask, m in cached_model.model:
                    buf[mask] = m.light_curve(params)
                return buf
            else:
                return cached_model.model.light_curve(params)
        elif isinstance(cached_model, list):
            # Legacy: list of (mask, model) tuples
            model_flux = np.zeros(len(time))
            for mask, m in cached_model:
                model_flux[mask] = m.light_curve(params)
            return model_flux
        else:
            # Legacy: single TransitModel
            return cached_model.light_curve(params)

    # Original path: create TransitModel each call
    # Handle cadence input
    if cadence is None:
        # No cadence info - assume short cadence, no integration needed
        m = batman.TransitModel(params, time)
        return m.light_curve(params)

    if np.isscalar(cadence):
        cadence = np.full(len(time), cadence)

    # Get unique cadences
    unique_cads = np.unique(cadence)

    # If all same cadence, compute once
    if len(unique_cads) == 1:
        cad_sec = unique_cads[0]
        if cad_sec > long_cadence_threshold:
            exp_time_days = cad_sec / config.SECONDS_PER_DAY
            supersample = max(config.BATMAN_MIN_SUPERSAMPLE,
                            int(cad_sec / config.BATMAN_SUPERSAMPLE_REF_CADENCE))
            m = batman.TransitModel(params, time,
                                    supersample_factor=supersample,
                                    exp_time=exp_time_days)
        else:
            m = batman.TransitModel(params, time)
        return m.light_curve(params)

    # Mixed cadences - compute separately for each cadence group
    model_flux = np.zeros(len(time))
    for cad_sec in unique_cads:
        mask = cadence == cad_sec
        time_subset = time[mask]

        if cad_sec > long_cadence_threshold:
            exp_time_days = cad_sec / config.SECONDS_PER_DAY
            supersample = max(config.BATMAN_MIN_SUPERSAMPLE,
                            int(cad_sec / config.BATMAN_SUPERSAMPLE_REF_CADENCE))
            m = batman.TransitModel(params, time_subset,
                                    supersample_factor=supersample,
                                    exp_time=exp_time_days)
        else:
            m = batman.TransitModel(params, time_subset)

        model_flux[mask] = m.light_curve(params)

    return model_flux


def setup_batman_params(period: float, t0: float, rp_rs: float, a_rs: float,
                        b: float, u1: float, u2: float,
                        ecc: float = 0.0, omega: float = 90.0) -> 'batman.TransitParams':
    """
    Create and configure batman.TransitParams object.

    Parameters
    ----------
    period : float
        Orbital period in days
    t0 : float
        Mid-transit time in BJD
    rp_rs : float
        Planet-to-star radius ratio
    a_rs : float
        Semi-major axis in stellar radii
    b : float
        Impact parameter
    u1, u2 : float
        Quadratic limb darkening coefficients
    ecc : float
        Eccentricity (default: 0.0)
    omega : float
        Argument of periastron in degrees (default: 90.0)

    Returns
    -------
    batman.TransitParams
        Configured batman parameters object
    """
    if not BATMAN_AVAILABLE:
        raise ImportError("batman-package is required")

    params = batman.TransitParams()
    params.per = period
    params.t0 = t0
    params.rp = rp_rs
    params.a = a_rs

    # Compute inclination from impact parameter
    if a_rs > 0:
        cos_i = np.clip(b / a_rs, 0, 1)
        params.inc = np.degrees(np.arccos(cos_i))
    else:
        params.inc = 90.0

    params.ecc = ecc
    params.w = omega
    params.u = [u1, u2]
    params.limb_dark = "quadratic"

    return params


# =============================================================================
# JSON Serialization Utilities
# =============================================================================

def json_serializer(obj: Any) -> Any:
    """
    JSON serializer for numpy types and objects with to_dict method.

    Use as the `default` argument to json.dump/dumps:
        json.dump(data, f, default=json_serializer)

    Parameters
    ----------
    obj : Any
        Object to serialize

    Returns
    -------
    Any
        JSON-serializable representation

    Raises
    ------
    TypeError
        If object is not serializable
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, 'to_dict'):
        return obj.to_dict()
    else:
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


# =============================================================================
# Statistical Utilities
# =============================================================================

def chauvenet_criterion(data: np.ndarray, max_iterations: int = 10) -> Tuple[np.ndarray, int]:
    """
    Identify outliers using Chauvenet's criterion with iterative rejection.

    Chauvenet's criterion rejects data points where the probability of obtaining
    their deviation from the mean is less than 1/(2N), where N is the sample size.
    This corresponds to an expected number of outliers of 0.5 for a normal distribution.

    The sigma threshold is calculated as: threshold = Φ⁻¹(1 - 1/(4N))
    where Φ⁻¹ is the inverse normal CDF.

    Example thresholds:
    - N = 100: threshold ≈ 2.58σ
    - N = 1000: threshold ≈ 3.29σ
    - N = 10000: threshold ≈ 3.89σ

    Uses robust statistics: median and σ = 1.48 × MAD (Median Absolute Deviation)

    Parameters
    ----------
    data : np.ndarray
        Data array to check for outliers
    max_iterations : int
        Maximum number of iterations (default: 10)

    Returns
    -------
    good_mask : np.ndarray
        Boolean mask where True indicates non-outlier (good) data
    n_rejected : int
        Total number of points rejected as outliers
    """
    # Start with all points marked as good
    good_mask = np.isfinite(data)
    total_rejected = np.sum(~good_mask)

    for iteration in range(max_iterations):
        # Get current good data
        good_data = data[good_mask]
        n_good = len(good_data)

        if n_good < 3:
            # Not enough points to compute statistics
            break

        # Compute robust statistics
        median = np.median(good_data)
        mad = np.median(np.abs(good_data - median))
        sigma = 1.48 * mad  # Convert MAD to sigma equivalent

        if sigma == 0 or np.isnan(sigma):
            # No scatter, can't reject outliers
            break

        # Calculate Chauvenet threshold for current sample size
        # Threshold = inverse_normal_CDF(1 - 1/(4N))
        # This gives expected number of outliers = 0.5
        chauvenet_prob = 1.0 - 1.0 / (4.0 * n_good)
        threshold = stats.norm.ppf(chauvenet_prob)

        # Identify outliers (deviation from median exceeds threshold)
        deviations = np.abs(data - median) / sigma
        new_mask = good_mask & (deviations <= threshold)

        # Count newly rejected points
        n_newly_rejected = np.sum(good_mask) - np.sum(new_mask)

        if n_newly_rejected == 0:
            # No new outliers found, converged
            break

        total_rejected += n_newly_rejected
        good_mask = new_mask

    return good_mask, total_rejected


def chauvenet_threshold(n_samples: int) -> float:
    """
    Calculate the sigma threshold for Chauvenet's criterion.

    The threshold is chosen such that the expected number of outliers
    from a normal distribution is 0.5.

    Parameters
    ----------
    n_samples : int
        Number of samples in the dataset

    Returns
    -------
    threshold : float
        Sigma threshold for outlier rejection
    """
    if n_samples < 2:
        return np.inf

    chauvenet_prob = 1.0 - 1.0 / (4.0 * n_samples)
    return stats.norm.ppf(chauvenet_prob)


def chi_squared(residuals: np.ndarray, errors: np.ndarray) -> float:
    """
    Compute chi-squared statistic.

    Parameters
    ----------
    residuals : np.ndarray
        Residuals (observed - model)
    errors : np.ndarray
        Measurement uncertainties

    Returns
    -------
    float
        Chi-squared value: sum((residuals / errors)^2)
    """
    return float(np.sum((residuals / errors) ** 2))


def reduced_chi_squared(residuals: np.ndarray, errors: np.ndarray,
                        n_params: int) -> float:
    """
    Compute reduced chi-squared statistic.

    Parameters
    ----------
    residuals : np.ndarray
        Residuals (observed - model)
    errors : np.ndarray
        Measurement uncertainties
    n_params : int
        Number of fitted parameters

    Returns
    -------
    float
        Reduced chi-squared: chi2 / (n_data - n_params)
    """
    chi2 = chi_squared(residuals, errors)
    dof = len(residuals) - n_params
    if dof > 0:
        return chi2 / dof
    return np.inf


def bic(chi2: float, n_params: int, n_data: int) -> float:
    """
    Compute Bayesian Information Criterion.

    BIC = chi2 + k * ln(n)

    Parameters
    ----------
    chi2 : float
        Chi-squared value
    n_params : int
        Number of fitted parameters (k)
    n_data : int
        Number of data points (n)

    Returns
    -------
    float
        BIC value
    """
    return chi2 + n_params * np.log(n_data)


# =============================================================================
# MCMC Parameter Extraction Utilities
# =============================================================================

def extract_percentiles(samples: np.ndarray,
                        percentiles: Tuple[float, float, float] = config.MCMC_PERCENTILES
                        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract percentile-based statistics from MCMC samples.

    Parameters
    ----------
    samples : np.ndarray
        Shape (n_samples, n_params) - flattened MCMC chain
    percentiles : tuple
        Percentiles to compute (default: 16, 50, 84 for 1-sigma)

    Returns
    -------
    lower : np.ndarray
        Lower percentile values for each parameter
    median : np.ndarray
        Median values for each parameter
    upper : np.ndarray
        Upper percentile values for each parameter
    """
    p = np.percentile(samples, percentiles, axis=0)
    return p[0], p[1], p[2]


def extract_parameter_stats(samples: np.ndarray, param_names: list = None
                            ) -> Dict[str, Dict[str, float]]:
    """
    Extract parameter statistics from MCMC samples.

    Parameters
    ----------
    samples : np.ndarray
        Shape (n_samples, n_params) - flattened MCMC chain
    param_names : list, optional
        Names for each parameter. If None, uses 'param_0', 'param_1', etc.

    Returns
    -------
    dict
        Dictionary with parameter statistics:
        {
            'param_name': {
                'value': median,
                'err_lower': median - p16,
                'err_upper': p84 - median,
                'err': (err_lower + err_upper) / 2,
                'percentile_16': p16,
                'percentile_84': p84
            },
            ...
        }
    """
    n_params = samples.shape[1]

    if param_names is None:
        param_names = [f'param_{i}' for i in range(n_params)]

    lower, median, upper = extract_percentiles(samples)

    result = {}
    for i, name in enumerate(param_names):
        err_lower = median[i] - lower[i]
        err_upper = upper[i] - median[i]
        result[name] = {
            'value': float(median[i]),
            'err_lower': float(err_lower),
            'err_upper': float(err_upper),
            'err': float((err_lower + err_upper) / 2),
            'percentile_16': float(lower[i]),
            'percentile_84': float(upper[i])
        }

    return result


# =============================================================================
# Light Curve Utilities
# =============================================================================

def normalize_flux(flux: np.ndarray, flux_err: np.ndarray,
                   method: str = 'median',
                   outlier_rejection: bool = True) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Normalize flux and flux errors to a reference level.

    Optionally uses Chauvenet's criterion for outlier rejection before
    computing the normalization factor.

    Parameters
    ----------
    flux : np.ndarray
        Raw flux values
    flux_err : np.ndarray
        Raw flux errors
    method : str
        Normalization method: 'median' (default) or 'mean'
    outlier_rejection : bool
        If True, use Chauvenet's criterion to reject outliers before
        computing the normalization factor (default: True)

    Returns
    -------
    flux_norm : np.ndarray
        Normalized flux (divided by reference)
    flux_err_norm : np.ndarray
        Normalized flux errors (divided by reference)
    norm_factor : float
        The normalization factor used (median or mean of non-outlier flux)
    """
    # Get finite values mask
    finite_mask = np.isfinite(flux)

    if outlier_rejection and np.sum(finite_mask) >= 3:
        # Use Chauvenet's criterion to identify outliers
        # Only apply to finite values
        flux_finite = flux.copy()
        flux_finite[~finite_mask] = np.nan

        good_mask, n_rejected = chauvenet_criterion(flux_finite)

        # Use only non-outlier points for computing normalization factor
        clean_flux = flux[good_mask]
    else:
        clean_flux = flux[finite_mask]

    # Compute normalization factor from clean data
    if len(clean_flux) == 0:
        norm_factor = 1.0
    elif method == 'median':
        norm_factor = np.nanmedian(clean_flux)
    elif method == 'mean':
        norm_factor = np.nanmean(clean_flux)
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    if norm_factor == 0 or np.isnan(norm_factor):
        norm_factor = 1.0

    # Normalize ALL flux values (not just clean ones)
    flux_norm = flux / norm_factor
    flux_err_norm = flux_err / norm_factor

    return flux_norm, flux_err_norm, float(norm_factor)


# =============================================================================
# Weighted Statistics Utilities
# =============================================================================

def weighted_mean(values: np.ndarray, errors: np.ndarray) -> Tuple[float, float]:
    """
    Compute inverse-variance weighted mean and its uncertainty.

    Parameters
    ----------
    values : np.ndarray
        Data values
    errors : np.ndarray
        Uncertainties on values

    Returns
    -------
    wmean : float
        Weighted mean
    wmean_err : float
        Uncertainty on weighted mean
    """
    # Inverse-variance weights
    weights = 1.0 / (errors ** 2)
    sum_weights = np.sum(weights)

    if sum_weights == 0:
        return float(np.nanmean(values)), float(np.nanstd(values))

    wmean = np.sum(weights * values) / sum_weights
    wmean_err = 1.0 / np.sqrt(sum_weights)

    return float(wmean), float(wmean_err)


def weighted_rms(values: np.ndarray, errors: np.ndarray,
                 reference: float = None) -> float:
    """
    Compute weighted root-mean-square deviation.

    Parameters
    ----------
    values : np.ndarray
        Data values
    errors : np.ndarray
        Uncertainties on values
    reference : float, optional
        Reference value to compute deviations from.
        If None, uses weighted mean.

    Returns
    -------
    wrms : float
        Weighted RMS
    """
    weights = 1.0 / (errors ** 2)
    sum_weights = np.sum(weights)

    if sum_weights == 0:
        return float(np.nanstd(values))

    if reference is None:
        reference = np.sum(weights * values) / sum_weights

    wrms = np.sqrt(np.sum(weights * (values - reference) ** 2) / sum_weights)
    return float(wrms)
