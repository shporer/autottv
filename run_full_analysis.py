#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Full Combined Sector Analysis for TOI 109.01

Includes:
- Phase-folded MCMC fitting
- Corner plot
- Individual mid-transit fitting
- O-C (Observed minus Calculated) plot
- TTV periodogram
- Chain saving
"""

import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='lightkurve')
warnings.filterwarnings('ignore', category=FutureWarning, module='arviz')

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for thread-safe plotting
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import json
import batman
import emcee
from scipy.optimize import minimize
from scipy import signal
from scipy.stats import norm
from multiprocessing import Pool, cpu_count
from concurrent.futures import ThreadPoolExecutor

# Add the pipeline to path
sys.path.insert(0, str(Path(__file__).parent))

from autottv_pipeline_v2.data_loader import DataLoader, load_toi_catalog, get_toi_parameters
from autottv_pipeline_v2.limb_darkening import get_limb_darkening
from autottv_pipeline_v2.convergence import check_convergence, run_until_converged, compute_rhat_split
from autottv_pipeline_v2 import config
from autottv_pipeline_v2.phase_fold_fitter import PhaseFoldFitter
from autottv_pipeline_v2.individual_transit_fitter import IndividualTransitFitter as ModularIndividualTransitFitter
from autottv_pipeline_v2.ephemeris_analysis import EphemerisAnalyzer, linear_ephemeris, quadratic_ephemeris
from autottv_pipeline_v2.utils import compute_batman_model, create_cached_transit_model, update_batman_params, setup_batman_params, chi_squared, bic as compute_bic

# Try to import corner
try:
    import corner
    HAS_CORNER = True
except ImportError:
    HAS_CORNER = False
    print("Warning: corner package not installed. Corner plots will be skipped.")

# Parameters
TIC_ID = None  # Will be retrieved from TOI catalog
TOI = sys.argv[1] if len(sys.argv) > 1 else "109.01"  # Accept TOI from command line
SECTORS_TO_USE = None  # None = use all available sectors

# MCMC parameters - import from centralized config
N_WALKERS = config.N_WALKERS
N_BURN = config.N_BURN
N_STEPS_MAX = config.N_STEPS_MAX

# CPU configuration: default 16, or cpu_count()-1 if machine has <=16 cores. Override with --cpus=N
_DEFAULT_CPUS = max(1, cpu_count() - 1) if cpu_count() <= 16 else 16
N_CPUS = _DEFAULT_CPUS
USE_CACHE = True
PERIOD_OVERRIDE = None   # Override period prior center
T0_OVERRIDE = None       # Override T0 prior center
PERIOD_ERR_OVERRIDE = None  # Override period prior width (sigma)
T0_ERR_OVERRIDE = None      # Override T0 prior width (sigma)
DURATION_OVERRIDE = None    # Override transit duration (hours) for window calculation
DUR_PRIOR = False           # Add Gaussian prior on a/Rs from catalog duration
for arg in sys.argv[2:]:
    if arg.startswith('--cpus='):
        try:
            N_CPUS = int(arg.split('=')[1])
            N_CPUS = max(1, N_CPUS)
        except ValueError:
            pass
    elif arg.startswith('--sectors='):
        try:
            SECTORS_TO_USE = [int(s.strip()) for s in arg.split('=')[1].split(',')]
        except ValueError:
            pass
    elif arg == '--no-cache':
        USE_CACHE = False
    elif arg.startswith('--period='):
        try:
            PERIOD_OVERRIDE = float(arg.split('=')[1])
        except ValueError:
            pass
    elif arg.startswith('--t0='):
        try:
            T0_OVERRIDE = float(arg.split('=')[1])
        except ValueError:
            pass
    elif arg.startswith('--period-err='):
        try:
            PERIOD_ERR_OVERRIDE = float(arg.split('=')[1])
        except ValueError:
            pass
    elif arg.startswith('--t0-err='):
        try:
            T0_ERR_OVERRIDE = float(arg.split('=')[1])
        except ValueError:
            pass
    elif arg.startswith('--duration='):
        try:
            DURATION_OVERRIDE = float(arg.split('=')[1])
        except ValueError:
            pass
    elif arg == '--dur-prior':
        DUR_PRIOR = True

# Global state for multiprocessing (needed because Pool can't pickle class methods)
_MCMC_SHARED_DATA = {}
_MCMC_CACHED_MODELS = {}  # Per-worker lazy-init cache for batman TransitModel
_MCMC_POOL = None  # Reference to multiprocessing Pool for vectorized dispatch

# Module-level cached variables for hot-path MCMC (set by _init_mcmc_worker)
_MCMC_TIME = None
_MCMC_FLUX = None
_MCMC_INV_VAR = None
_MCMC_FIX_LD = False
_MCMC_U1_PRIOR = 0.0
_MCMC_U2_PRIOR = 0.0
_MCMC_PERIOD_PRIOR = 0.0
_MCMC_T0_PRIOR = 0.0
_MCMC_RP_RS_PRIOR = 0.0
_MCMC_A_RS_PRIOR = 0.0
_MCMC_PERIOD_SIGMA = 0.0
_MCMC_T0_SIGMA = 0.0
_MCMC_RP_RS_SIGMA = 0.0
_MCMC_A_RS_SIGMA = 0.0
_MCMC_LD_SIGMA_U1 = 0.0
_MCMC_LD_SIGMA_U2 = 0.0
_MCMC_PERIOD_LO = 0.0
_MCMC_PERIOD_HI = 0.0
_MCMC_T0_LO = 0.0
_MCMC_T0_HI = 0.0

def _mcmc_log_probability(theta):
    """Module-level log probability function for multiprocessing.

    Includes bounds checking for use as standalone function.
    Note: parameter index 4 is b_sq (= b^2), sampled uniformly in b^2, which
    gives p(b) proportional to b. Isotropic orbits (uniform in cos(i)) would
    give a uniform prior in b.
    """
    if _MCMC_FIX_LD:
        period, t0, rp_rs, a_rs, b_sq, baseline = theta
    else:
        period, t0, rp_rs, a_rs, b_sq, baseline, u1, u2 = theta

    # Hard bounds using precomputed limits
    if not (_MCMC_PERIOD_LO < period < _MCMC_PERIOD_HI):
        return -np.inf
    if not (_MCMC_T0_LO < t0 < _MCMC_T0_HI):
        return -np.inf
    if not (config.RP_RS_MIN < rp_rs < config.RP_RS_MAX):
        return -np.inf
    if not (config.A_RS_MIN < a_rs < config.A_RS_MAX):
        return -np.inf
    if not (config.B_MIN <= b_sq < (config.B_MAX + rp_rs)**2):
        return -np.inf
    if not (config.BASELINE_MIN < baseline < config.BASELINE_MAX):
        return -np.inf
    if not _MCMC_FIX_LD:
        if not (config.LD_U1_MIN < u1 < config.LD_U1_MAX):
            return -np.inf
        if not (config.LD_U2_MIN < u2 < config.LD_U2_MAX):
            return -np.inf
        if not (u1 + u2 < config.LD_SUM_MAX):
            return -np.inf
        if not (u1 + 2.0 * u2 >= config.LD_U1_2U2_MIN):
            return -np.inf

    return _mcmc_log_prob_inner(theta)


def _mcmc_log_prob_inner(theta):
    """Log probability without bounds checking — for walkers already validated
    by _mcmc_log_probability_vectorized. Skips redundant bounds checks.
    Note: parameter index 4 is b_sq (= b^2); convert to b before batman calls."""
    global _MCMC_CACHED_MODELS

    if _MCMC_FIX_LD:
        period, t0, rp_rs, a_rs, b_sq, baseline = theta
        u1 = _MCMC_U1_PRIOR
        u2 = _MCMC_U2_PRIOR
    else:
        period, t0, rp_rs, a_rs, b_sq, baseline, u1, u2 = theta

    # Convert b_sq to b for batman
    b = np.sqrt(b_sq)

    # Gaussian priors on P, T0, Rp/Rs, a/Rs, and (optionally) LD. b_sq and baseline
    # remain uniform within their hard bounds.
    log_prior = 0.0
    log_prior += -0.5 * ((period - _MCMC_PERIOD_PRIOR) / _MCMC_PERIOD_SIGMA) ** 2
    log_prior += -0.5 * ((t0 - _MCMC_T0_PRIOR) / _MCMC_T0_SIGMA) ** 2
    log_prior += -0.5 * ((rp_rs - _MCMC_RP_RS_PRIOR) / _MCMC_RP_RS_SIGMA) ** 2
    log_prior += -0.5 * ((a_rs - _MCMC_A_RS_PRIOR) / _MCMC_A_RS_SIGMA) ** 2
    if not _MCMC_FIX_LD:
        log_prior += -0.5 * ((u1 - _MCMC_U1_PRIOR) / _MCMC_LD_SIGMA_U1) ** 2
        log_prior += -0.5 * ((u2 - _MCMC_U2_PRIOR) / _MCMC_LD_SIGMA_U2) ** 2

    # Duration prior: constrain a/Rs given b, Rp/Rs, and catalog duration
    if _MCMC_DUR_PRIOR and _MCMC_DUR_PRIOR_HOURS > 0:
        dur_days = _MCMC_DUR_PRIOR_HOURS / 24.0
        sin_arg = np.pi * dur_days / period
        if 0 < sin_arg < 1 and (1 + rp_rs)**2 > b**2:
            a_rs_expected = np.sqrt((1 + rp_rs)**2 - b**2) / np.sin(sin_arg)
            sigma_a = _MCMC_DUR_PRIOR_SIGMA_FRAC * a_rs_expected
            log_prior += -0.5 * ((a_rs - a_rs_expected) / sigma_a) ** 2

    # Lazy-init cached TransitModel and TransitParams on first call in this worker
    if 'batman_params' not in _MCMC_CACHED_MODELS:
        bp = setup_batman_params(
            period=period, t0=t0, rp_rs=rp_rs, a_rs=a_rs, b=b, u1=u1, u2=u2
        )
        cached_model = create_cached_transit_model(
            _MCMC_TIME, bp, cadence=_MCMC_SHARED_DATA['cadence'],
            long_cadence_threshold=_MCMC_SHARED_DATA['long_cadence_threshold']
        )
        _MCMC_CACHED_MODELS['batman_params'] = bp
        _MCMC_CACHED_MODELS['transit_model'] = cached_model

    # Update existing TransitParams in-place
    bp = _MCMC_CACHED_MODELS['batman_params']
    update_batman_params(bp, period, t0, rp_rs, a_rs, b, u1, u2)

    # Compute model using cached TransitModel
    cached = _MCMC_CACHED_MODELS['transit_model']
    if hasattr(cached, 'is_mixed'):
        if cached.is_mixed:
            buf = cached.buffer
            buf[:] = 0.0
            for mask, m in cached.model:
                buf[mask] = m.light_curve(bp)
            transit_flux = buf
        else:
            transit_flux = cached.model.light_curve(bp)
    elif isinstance(cached, list):
        transit_flux = np.zeros(len(_MCMC_TIME))
        for mask, m in cached:
            transit_flux[mask] = m.light_curve(bp)
    else:
        transit_flux = cached.light_curve(bp)
    model = transit_flux * baseline

    residuals = _MCMC_FLUX - model
    chi2 = np.dot(residuals, residuals * _MCMC_INV_VAR)

    return log_prior - 0.5 * chi2


def _init_mcmc_worker(shared_data):
    """Initialize worker with shared data and precomputed constants."""
    global _MCMC_SHARED_DATA, _MCMC_CACHED_MODELS
    global _MCMC_TIME, _MCMC_FLUX, _MCMC_INV_VAR, _MCMC_FIX_LD
    global _MCMC_U1_PRIOR, _MCMC_U2_PRIOR
    global _MCMC_PERIOD_PRIOR, _MCMC_T0_PRIOR, _MCMC_RP_RS_PRIOR, _MCMC_A_RS_PRIOR
    global _MCMC_PERIOD_SIGMA, _MCMC_T0_SIGMA, _MCMC_RP_RS_SIGMA, _MCMC_A_RS_SIGMA
    global _MCMC_LD_SIGMA_U1, _MCMC_LD_SIGMA_U2
    global _MCMC_PERIOD_LO, _MCMC_PERIOD_HI, _MCMC_T0_LO, _MCMC_T0_HI
    global _MCMC_DUR_PRIOR, _MCMC_DUR_PRIOR_HOURS, _MCMC_DUR_PRIOR_SIGMA_FRAC

    _MCMC_SHARED_DATA = shared_data
    _MCMC_CACHED_MODELS = {}  # Reset cache so each worker creates its own TransitModel

    # Extract frequently-accessed values into module-level variables
    _MCMC_TIME = shared_data['time']
    _MCMC_FLUX = shared_data['flux']
    _MCMC_INV_VAR = shared_data['inv_var']
    _MCMC_FIX_LD = shared_data.get('fix_ld', False)
    _MCMC_U1_PRIOR = shared_data['u1_prior']
    _MCMC_U2_PRIOR = shared_data['u2_prior']
    _MCMC_PERIOD_PRIOR = shared_data['period_prior']
    _MCMC_T0_PRIOR = shared_data['t0_prior']
    _MCMC_RP_RS_PRIOR = shared_data['rp_rs_prior']
    _MCMC_A_RS_PRIOR = shared_data['a_rs_prior']

    # Precompute prior sigmas
    _MCMC_PERIOD_SIGMA = shared_data.get('period_sigma', config.PERIOD_PRIOR_WIDTH * _MCMC_PERIOD_PRIOR)
    _MCMC_T0_SIGMA = shared_data.get('t0_sigma', config.T0_PRIOR_WIDTH * _MCMC_PERIOD_PRIOR)
    _MCMC_RP_RS_SIGMA = config.RP_RS_PRIOR_WIDTH * _MCMC_RP_RS_PRIOR
    _MCMC_A_RS_SIGMA = config.A_RS_PRIOR_WIDTH * _MCMC_A_RS_PRIOR
    _MCMC_LD_SIGMA_U1 = shared_data['ld_prior_width_u1']
    _MCMC_LD_SIGMA_U2 = shared_data['ld_prior_width_u2']

    # Precompute hard bounds
    _MCMC_PERIOD_LO = _MCMC_PERIOD_PRIOR * (1 - config.PERIOD_BOUND_FRACTION)
    _MCMC_PERIOD_HI = _MCMC_PERIOD_PRIOR * (1 + config.PERIOD_BOUND_FRACTION)
    _MCMC_T0_LO = _MCMC_T0_PRIOR - config.T0_BOUND_FRACTION * _MCMC_PERIOD_PRIOR
    _MCMC_T0_HI = _MCMC_T0_PRIOR + config.T0_BOUND_FRACTION * _MCMC_PERIOD_PRIOR

    # Duration prior on a/Rs
    _MCMC_DUR_PRIOR = shared_data.get('dur_prior', False)
    _MCMC_DUR_PRIOR_HOURS = shared_data.get('dur_prior_hours', 0.0)
    _MCMC_DUR_PRIOR_SIGMA_FRAC = shared_data.get('dur_prior_sigma_frac', 0.15)


def _mcmc_log_probability_vectorized(positions):
    """Vectorized log probability — processes all walkers in one call.

    Called by emcee with vectorize=True. Takes (n_walkers, ndim) array,
    returns (n_walkers,) array. Performs vectorized prior rejection to
    skip batman computation for ~30% of walkers that fail bounds checks.
    """
    n_walkers = positions.shape[0]
    result = np.full(n_walkers, -np.inf)

    # Extract parameter columns (index 4 is b_sq = b^2)
    if _MCMC_FIX_LD:
        period, t0, rp_rs, a_rs, b_sq, baseline = positions.T
        u1 = np.full(n_walkers, _MCMC_U1_PRIOR)
        u2 = np.full(n_walkers, _MCMC_U2_PRIOR)
    else:
        period, t0, rp_rs, a_rs, b_sq, baseline, u1, u2 = positions.T

    # Vectorized prior bounds (numpy, all walkers at once)
    valid = np.ones(n_walkers, dtype=bool)
    valid &= (period > _MCMC_PERIOD_LO) & (period < _MCMC_PERIOD_HI)
    valid &= (t0 > _MCMC_T0_LO) & (t0 < _MCMC_T0_HI)
    valid &= (rp_rs > config.RP_RS_MIN) & (rp_rs < config.RP_RS_MAX)
    valid &= (a_rs > config.A_RS_MIN) & (a_rs < config.A_RS_MAX)
    valid &= (b_sq >= config.B_MIN) & (b_sq < (config.B_MAX + rp_rs)**2)
    valid &= (baseline > config.BASELINE_MIN) & (baseline < config.BASELINE_MAX)
    if not _MCMC_FIX_LD:
        valid &= (u1 > config.LD_U1_MIN) & (u1 < config.LD_U1_MAX)
        valid &= (u2 > config.LD_U2_MIN) & (u2 < config.LD_U2_MAX)
        valid &= (u1 + u2 < config.LD_SUM_MAX)
        valid &= (u1 + 2.0 * u2 >= config.LD_U1_2U2_MIN)

    idx_valid = np.where(valid)[0]
    if len(idx_valid) == 0:
        return result

    # Dispatch only valid walkers — use bounds-free inner function since
    # vectorized bounds checking above already filtered invalid walkers
    valid_thetas = [positions[i] for i in idx_valid]
    if _MCMC_POOL is not None:
        log_probs = list(_MCMC_POOL.map(_mcmc_log_prob_inner, valid_thetas))
    else:
        log_probs = [_mcmc_log_prob_inner(theta) for theta in valid_thetas]

    for i, lp in zip(idx_valid, log_probs):
        result[i] = lp

    return result


def _fit_single_transit(args):
    """
    Fit a single transit - module-level function for multiprocessing.

    args: tuple of (transit_dict, fitter_params)
    """
    transit, fitter_params = args

    # Get median cadence for this transit (if available)
    cadence_array = transit.get('cadence', None)
    if cadence_array is not None:
        cadence = float(np.median(cadence_array))
    else:
        cadence = None

    # Build step1_results dict for modular IndividualTransitFitter
    step1_results = {
        'parameters': {
            'period': {'value': fitter_params['period']},
            't0': {'value': transit['t_expected']},  # Use expected time as reference
            'rp_rs': {'value': fitter_params['rp_rs']},
            'a_rs': {'value': fitter_params['a_rs']},
            'b': {'value': fitter_params['b']},
            'baseline': {'value': 1.0},
            'u1': {'value': fitter_params['u1']},
            'u2': {'value': fitter_params['u2']}
        },
        'derived': {
            'duration_hr': config.DEFAULT_TRANSIT_DURATION_HR
        }
    }

    # Create individual transit fitter using modular class
    ind_fitter = ModularIndividualTransitFitter(step1_results, cadence=cadence)

    t_data = transit['t_data']
    f_data = transit['f_data']
    f_err = transit['f_err']

    # Initialize T0 at the flux minimum (smoothed) if the dip is significant
    # This helps for systems with large TTVs where the expected time is shifted
    t0_guess = transit['t_expected']
    try:
        n_smooth = max(5, len(f_data) // 20)
        from scipy.ndimage import median_filter
        f_smooth = median_filter(f_data, size=n_smooth)
        f_min = np.min(f_smooth)
        f_med = np.median(f_smooth)
        f_std = 1.48 * np.median(np.abs(f_smooth - f_med))
        # Only use flux minimum if the dip is > 3 sigma below the median
        if f_std > 0 and (f_med - f_min) > 3 * f_std:
            t0_guess = t_data[np.argmin(f_smooth)]
    except:
        pass

    # Fit the transit
    result = ind_fitter.fit_single_transit(
        t_data, f_data, f_err, t0_guess, transit['epoch'],
        cadence=cadence
    )

    return {
        'epoch': transit['epoch'],
        't_expected': transit['t_expected'],
        't0_fit': result.t_mid,
        't0_err': result.t_mid_err,
        'baseline_fit': result.baseline,
        'baseline_err': result.baseline_err,
        'slope_fit': result.slope,
        'slope_err': result.slope_err,
        'chi2': getattr(result, 'chi2', 0.0),
        'n_points': transit['n_points'],
        'converged': getattr(result, 'converged', True),
        'n_steps': getattr(result, 'n_steps', 1000),
        'max_rhat': getattr(result, 'max_rhat', 1.0),
        'rhat': getattr(result, 'rhat', {}),
        'autocorr_time': getattr(result, 'autocorr_time', {}),
        'ess': getattr(result, 'ess', {})
    }

# Output directory
OUTPUT_DIR = Path(__file__).parent / "autottv_results_v2" / f"TOI_{TOI.replace('.', '_')}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Parameter names (now includes limb darkening)
PARAM_NAMES = ['period', 't0', 'rp_rs', 'a_rs', 'b', 'baseline', 'u1', 'u2']

# Limb darkening options - import defaults from centralized config
LD_PRIOR_WIDTH_U1 = config.LD_PRIOR_WIDTH_U1
LD_PRIOR_WIDTH_U2 = config.LD_PRIOR_WIDTH_U2
FIX_LD = config.FIX_LD
for arg in sys.argv[2:]:
    if arg.startswith('--ld-width='):
        # Set both to same value for backwards compatibility
        try:
            width = float(arg.split('=')[1])
            LD_PRIOR_WIDTH_U1 = width
            LD_PRIOR_WIDTH_U2 = width
        except ValueError:
            pass
    elif arg.startswith('--ld-width-u1='):
        try:
            LD_PRIOR_WIDTH_U1 = float(arg.split('=')[1])
        except ValueError:
            pass
    elif arg.startswith('--ld-width-u2='):
        try:
            LD_PRIOR_WIDTH_U2 = float(arg.split('=')[1])
        except ValueError:
            pass
    elif arg == '--fix-ld':
        FIX_LD = True

# Parameter names depend on whether LD is fixed or free
if FIX_LD:
    PARAM_NAMES = ['period', 't0', 'rp_rs', 'a_rs', 'b', 'baseline']
else:
    PARAM_NAMES = ['period', 't0', 'rp_rs', 'a_rs', 'b', 'baseline', 'u1', 'u2']


def get_oot_excluding_eclipse_mask(time, period, t0, transit_width=config.TRANSIT_MASK_PHASE_WIDTH, eclipse_width=config.ECLIPSE_MASK_PHASE_WIDTH):
    """Create mask for out-of-transit AND out-of-secondary-eclipse points."""
    phase = ((time - t0) / period) % 1.0
    phase[phase > 0.5] -= 1.0
    transit_mask = np.abs(phase) < transit_width / 2
    eclipse_phase = np.abs(np.abs(phase) - 0.5)
    eclipse_mask = eclipse_phase < eclipse_width / 2
    oot_mask = ~transit_mask & ~eclipse_mask
    return oot_mask, transit_mask, eclipse_mask


def normalize_to_oot_no_eclipse(time, flux, flux_err, period, t0,
                                 transit_width=config.TRANSIT_MASK_PHASE_WIDTH, eclipse_width=config.ECLIPSE_MASK_PHASE_WIDTH):
    """Normalize flux to out-of-transit, out-of-eclipse level."""
    oot_mask, transit_mask, eclipse_mask = get_oot_excluding_eclipse_mask(
        time, period, t0, transit_width, eclipse_width
    )
    n_oot = np.sum(oot_mask)
    if n_oot < config.MIN_POINTS_LIGHTCURVE:
        oot_level = np.median(flux)
    else:
        oot_level = np.median(flux[oot_mask])
    flux_norm = flux / oot_level
    flux_err_norm = flux_err / oot_level
    info = {
        'oot_level': float(oot_level),
        'n_oot': int(n_oot),
        'n_transit': int(np.sum(transit_mask)),
        'n_eclipse': int(np.sum(eclipse_mask)),
        'n_total': len(time)
    }
    return flux_norm, flux_err_norm, info


def weighted_bin_with_outlier_rejection(values, errors):
    """
    Calculate weighted mean with one iteration of outlier rejection using Chauvenet's criterion.

    Parameters
    ----------
    values : array-like
        Data values to bin
    errors : array-like
        Uncertainties on the values

    Returns
    -------
    weighted_mean : float
        Weighted mean after outlier rejection
    weighted_err : float
        Error on the weighted mean
    n_rejected : int
        Number of points rejected as outliers
    """
    values = np.asarray(values)
    errors = np.asarray(errors)

    if len(values) == 0:
        return np.nan, np.nan, 0

    if len(values) == 1:
        return values[0], errors[0], 0

    # Initial weighted mean
    weights = 1.0 / errors**2
    weighted_mean = np.sum(values * weights) / np.sum(weights)

    # Calculate residuals from weighted mean
    residuals = values - weighted_mean

    # Calculate sigma using 1.48 * MAD (robust estimator)
    mad = np.median(np.abs(residuals - np.median(residuals)))
    sigma = 1.48 * mad

    # If sigma is zero (all points identical), no outliers
    if sigma == 0:
        weighted_err = 1.0 / np.sqrt(np.sum(weights))
        return weighted_mean, weighted_err, 0

    # Chauvenet's criterion: threshold = Φ⁻¹(1 - 1/(4N))
    N = len(values)
    chauvenet_threshold = norm.ppf(1.0 - 1.0 / (4 * N))

    # Identify outliers (one iteration)
    good_mask = np.abs(residuals) <= chauvenet_threshold * sigma
    n_rejected = np.sum(~good_mask)

    # Recalculate weighted mean without outliers
    if np.sum(good_mask) > 0:
        weights_good = weights[good_mask]
        weighted_mean = np.sum(values[good_mask] * weights_good) / np.sum(weights_good)
        weighted_err = 1.0 / np.sqrt(np.sum(weights_good))
    else:
        # All points rejected - fall back to original
        weighted_err = 1.0 / np.sqrt(np.sum(weights))

    return weighted_mean, weighted_err, n_rejected


class FullAnalysisFitter:
    """MCMC fitter with full analysis capabilities."""

    def __init__(self, time, flux, flux_err, planet_params, stellar_params=None, cadence=None):
        self.time = time
        self.flux = flux
        self.flux_err = flux_err
        self.inv_var = 1.0 / flux_err**2

        # Store cadence information for exposure time integration
        # cadence is in seconds; if not provided, assume 120s (SPOC 2-min)
        if cadence is None:
            self.cadence = np.full(len(time), config.DEFAULT_CADENCE_SPOC)
        else:
            self.cadence = np.array(cadence)

        # Identify unique cadences and create masks for each
        self.unique_cadences = np.unique(self.cadence)
        self.cadence_masks = {cad: self.cadence == cad for cad in self.unique_cadences}

        # Long cadence threshold (seconds) - integrate for cadences above this
        self.long_cadence_threshold = config.LONG_CADENCE_THRESHOLD

        self.period_prior = planet_params['period']
        self.t0_prior = planet_params['t0']
        self.depth_prior = planet_params.get('depth_ppm', 10000) / 1e6

        planet_radius = planet_params.get('planet_radius')
        stellar_radius = planet_params.get('stellar_radius')

        if (planet_radius and not np.isnan(planet_radius) and planet_radius > 0 and
            stellar_radius and not np.isnan(stellar_radius) and stellar_radius > 0):
            self.rp_rs_prior = (planet_radius * config.R_EARTH_TO_R_SUN) / stellar_radius
        else:
            self.rp_rs_prior = np.sqrt(self.depth_prior)

        stellar_mass = planet_params.get('stellar_mass')
        if (stellar_mass and not np.isnan(stellar_mass) and stellar_mass > 0 and
            stellar_radius and not np.isnan(stellar_radius) and stellar_radius > 0):
            self.a_rs_prior = config.KEPLER_CONSTANT * (stellar_mass ** (1/3)) * (self.period_prior ** (2/3)) / stellar_radius
        else:
            self.a_rs_prior = config.DEFAULT_A_RS

        if stellar_params is None:
            stellar_params = {
                'teff': planet_params.get('stellar_teff', config.SOLAR_TEFF),
                'logg': planet_params.get('stellar_logg', config.SOLAR_LOGG)
            }
        self.u1, self.u2 = get_limb_darkening(
            stellar_params.get('teff', config.SOLAR_TEFF),
            stellar_params.get('logg', config.SOLAR_LOGG)
        )

        self.batman_params = batman.TransitParams()
        self.batman_params.per = self.period_prior
        self.batman_params.t0 = self.t0_prior
        self.batman_params.rp = self.rp_rs_prior
        self.batman_params.a = self.a_rs_prior
        self.batman_params.inc = config.DEFAULT_OMEGA  # 90 degrees (face-on)
        self.batman_params.limb_dark = "quadratic"
        self.batman_params.u = [self.u1, self.u2]
        self.batman_params.ecc = config.DEFAULT_ECCENTRICITY
        self.batman_params.w = config.DEFAULT_OMEGA

        self.sampler = None
        self.samples = None
        self.results = None
        self.burnin_chain = None
        self.burnin_log_prob = None
        self.n_bad_walkers = 0
        self.bad_walker_indices = []
        self._cached_transit_model = None

        # Precompute prior sigmas and hard bounds for log_prior
        period_err = planet_params.get('period_err')
        if period_err and not np.isnan(period_err) and period_err > 0:
            self._period_sigma = config.CATALOG_ERROR_PRIOR_MULTIPLIER * period_err
        else:
            self._period_sigma = config.PERIOD_PRIOR_WIDTH * self.period_prior
        t0_err = planet_params.get('t0_err')
        if t0_err and not np.isnan(t0_err) and t0_err > 0:
            self._t0_sigma = config.CATALOG_ERROR_PRIOR_MULTIPLIER * t0_err
        else:
            self._t0_sigma = config.T0_PRIOR_WIDTH * self.period_prior
        self._rp_rs_sigma = config.RP_RS_PRIOR_WIDTH * self.rp_rs_prior
        self._a_rs_sigma = config.A_RS_PRIOR_WIDTH * self.a_rs_prior
        self._period_lo = self.period_prior * (1 - config.PERIOD_BOUND_FRACTION)
        self._period_hi = self.period_prior * (1 + config.PERIOD_BOUND_FRACTION)
        self._t0_lo = self.t0_prior - config.T0_BOUND_FRACTION * self.period_prior
        self._t0_hi = self.t0_prior + config.T0_BOUND_FRACTION * self.period_prior

        print(f"  Priors: Period={self.period_prior:.6f}, Rp/Rs={self.rp_rs_prior:.4f}, a/Rs={self.a_rs_prior:.2f}")
        if FIX_LD:
            print(f"  Limb darkening: u1={self.u1:.3f}, u2={self.u2:.3f} (FIXED)")
        else:
            print(f"  Limb darkening: u1={self.u1:.3f} (σ={LD_PRIOR_WIDTH_U1:.2f}), u2={self.u2:.3f} (σ={LD_PRIOR_WIDTH_U2:.2f})")

    def transit_model(self, time, period, t0, rp_rs, a_rs, b, baseline, u1, u2, cadence=None):
        """
        Compute transit model with variable limb darkening.

        For long-cadence data (QLP 10-min or 30-min), integrates the model over
        the exposure time using batman's supersample feature.

        Parameters
        ----------
        time : array
            Time array in BJD
        period, t0, rp_rs, a_rs, b, baseline, u1, u2 : float
            Transit parameters
        cadence : array, optional
            Cadence in seconds for each time point. If None, uses self.cadence
            or assumes 120s for external calls.
        """
        # Update batman params in-place
        update_batman_params(self.batman_params, period, t0, rp_rs, a_rs, b, u1, u2)

        # Use cached model for MCMC (same time array), fresh model for plotting etc.
        if time is self.time and self._cached_transit_model is not None:
            cached = self._cached_transit_model
            if hasattr(cached, 'is_mixed'):
                if cached.is_mixed:
                    buf = cached.buffer
                    buf[:] = 0.0
                    for mask, m in cached.model:
                        buf[mask] = m.light_curve(self.batman_params)
                    transit = buf
                else:
                    transit = cached.model.light_curve(self.batman_params)
            elif isinstance(cached, list):
                transit = np.zeros(len(time))
                for mask, m in cached:
                    transit[mask] = m.light_curve(self.batman_params)
            else:
                transit = cached.light_curve(self.batman_params)
            return transit * baseline

        # Determine cadence for each point
        if cadence is None:
            if hasattr(self, 'cadence') and len(time) == len(self.cadence):
                cadence = self.cadence
            else:
                cadence = np.full(len(time), config.DEFAULT_CADENCE_SPOC)

        # Use utility function for cadence-aware model computation
        transit = compute_batman_model(time, self.batman_params, cadence=cadence,
                                       long_cadence_threshold=self.long_cadence_threshold)

        return transit * baseline

    def log_likelihood(self, theta):
        if FIX_LD:
            period, t0, rp_rs, a_rs, b_sq, baseline = theta
            u1, u2 = self.u1, self.u2
        else:
            period, t0, rp_rs, a_rs, b_sq, baseline, u1, u2 = theta
        b = np.sqrt(b_sq)
        model = self.transit_model(self.time, period, t0, rp_rs, a_rs, b, baseline, u1, u2)
        residuals = self.flux - model
        return -0.5 * np.dot(residuals, residuals * self.inv_var)

    def log_prior(self, theta):
        if FIX_LD:
            period, t0, rp_rs, a_rs, b_sq, baseline = theta
        else:
            period, t0, rp_rs, a_rs, b_sq, baseline, u1, u2 = theta
        if not (self._period_lo < period < self._period_hi):
            return -np.inf
        if not (self._t0_lo < t0 < self._t0_hi):
            return -np.inf
        if not (config.RP_RS_MIN < rp_rs < config.RP_RS_MAX):
            return -np.inf
        if not (config.A_RS_MIN < a_rs < config.A_RS_MAX):
            return -np.inf
        if not (config.B_MIN <= b_sq < (config.B_MAX + rp_rs)**2):
            return -np.inf
        if not (config.BASELINE_MIN < baseline < config.BASELINE_MAX):
            return -np.inf
        if not FIX_LD:
            if not (config.LD_U1_MIN < u1 < config.LD_U1_MAX):
                return -np.inf
            if not (config.LD_U2_MIN < u2 < config.LD_U2_MAX):
                return -np.inf
            if not (u1 + u2 < config.LD_SUM_MAX):
                return -np.inf
            if not (u1 + 2.0 * u2 >= config.LD_U1_2U2_MIN):
                return -np.inf
        log_prior = 0.0
        log_prior += -0.5 * ((period - self.period_prior) / self._period_sigma) ** 2
        log_prior += -0.5 * ((t0 - self.t0_prior) / self._t0_sigma) ** 2
        log_prior += -0.5 * ((rp_rs - self.rp_rs_prior) / self._rp_rs_sigma) ** 2
        log_prior += -0.5 * ((a_rs - self.a_rs_prior) / self._a_rs_sigma) ** 2
        if not FIX_LD:
            log_prior += -0.5 * ((u1 - self.u1) / LD_PRIOR_WIDTH_U1) ** 2
            log_prior += -0.5 * ((u2 - self.u2) / LD_PRIOR_WIDTH_U2) ** 2
        return log_prior

    def log_probability(self, theta):
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        ll = self.log_likelihood(theta)
        if not np.isfinite(ll):
            return -np.inf
        return lp + ll

    def _initialize_walkers(self, n_walkers):
        ndim = 6 if FIX_LD else 8
        p0 = np.zeros((n_walkers, ndim))
        for i in range(n_walkers):
            p0[i, 0] = self.period_prior + np.random.uniform(-self._period_sigma, self._period_sigma)
            p0[i, 1] = self.t0_prior + np.random.uniform(-self._t0_sigma, self._t0_sigma)
            p0[i, 2] = self.rp_rs_prior * (1 + np.random.uniform(-config.WALKER_INIT_RP_RS_FRAC, config.WALKER_INIT_RP_RS_FRAC))
            p0[i, 3] = self.a_rs_prior * (1 + np.random.uniform(-config.WALKER_INIT_A_RS_FRAC, config.WALKER_INIT_A_RS_FRAC))
            p0[i, 4] = np.random.uniform(0.0, (1.0 + self.rp_rs_prior)**2)
            p0[i, 5] = 1.0 + np.random.uniform(-config.WALKER_INIT_BASELINE_FRAC, config.WALKER_INIT_BASELINE_FRAC)
            if not FIX_LD:
                p0[i, 6] = self.u1 + np.random.uniform(-config.WALKER_INIT_LD_WIDTH_U1, config.WALKER_INIT_LD_WIDTH_U1)
                p0[i, 7] = self.u2 + np.random.uniform(-config.WALKER_INIT_LD_WIDTH_U2, config.WALKER_INIT_LD_WIDTH_U2)
        return p0

    def _identify_bad_walkers(self, log_prob):
        n_steps, n_walkers = log_prob.shape
        # Use the last 10% of burn-in steps (minimum 100) to detect bad walkers,
        # so walkers that migrated to a good mode during burn-in are not penalized
        # by their early (poor) log-prob values
        tail_steps = max(100, n_steps // 10)
        log_prob_tail = log_prob[-tail_steps:, :]
        walker_medians = np.median(log_prob_tail, axis=0)
        median_of_medians = np.median(walker_medians)
        mad = np.median(np.abs(walker_medians - median_of_medians))
        sigma = 1.48 * mad
        if sigma > 0:
            threshold = median_of_medians - config.BAD_WALKER_SIGMA * sigma
            bad_indices = np.where(walker_medians < threshold)[0].tolist()
        else:
            bad_indices = []
        print(f"  Bad walker detection (using last {tail_steps} of {n_steps} burn-in steps):", flush=True)
        print(f"    Median of medians: {median_of_medians:.2f}, Sigma: {sigma:.2f}", flush=True)
        print(f"    Bad walkers found: {len(bad_indices)}", flush=True)
        return bad_indices, sigma, median_of_medians

    def _reinitialize_bad_walkers(self, chains, bad_indices, sigma):
        n_steps, n_walkers, n_params = chains.shape
        final_positions = chains[-1, :, :]
        good_indices = [i for i in range(n_walkers) if i not in bad_indices]
        good_chains = chains[:, good_indices, :]
        param_medians = np.zeros(n_params)
        param_mads = np.zeros(n_params)
        for p in range(n_params):
            all_values = good_chains[:, :, p].flatten()
            param_medians[p] = np.median(all_values)
            param_mads[p] = np.median(np.abs(all_values - param_medians[p]))
        param_sigmas = 1.48 * param_mads
        new_positions = final_positions.copy()
        for idx in bad_indices:
            for p in range(n_params):
                new_positions[idx, p] = param_medians[p] + np.random.uniform(-1, 1) * param_sigmas[p]
        return new_positions

    def fit(self, n_walkers=64, n_burn=4000, check_convergence_flag=True, n_cpus=1):
        global _MCMC_SHARED_DATA, _MCMC_POOL
        ndim = 6 if FIX_LD else 8
        print(f"\n  Initializing {n_walkers} walkers...", flush=True)
        p0 = self._initialize_walkers(n_walkers)

        # Set up shared data for multiprocessing
        _MCMC_SHARED_DATA = {
            'time': self.time,
            'flux': self.flux,
            'flux_err': self.flux_err,
            'inv_var': 1.0 / self.flux_err**2,
            'cadence': self.cadence,
            'unique_cadences': self.unique_cadences,
            'cadence_masks': self.cadence_masks,
            'long_cadence_threshold': self.long_cadence_threshold,
            'period_prior': self.period_prior,
            'period_sigma': self._period_sigma,
            't0_prior': self.t0_prior,
            't0_sigma': self._t0_sigma,
            'rp_rs_prior': self.rp_rs_prior,
            'a_rs_prior': self.a_rs_prior,
            'u1_prior': self.u1,
            'u2_prior': self.u2,
            'ld_prior_width_u1': LD_PRIOR_WIDTH_U1,
            'ld_prior_width_u2': LD_PRIOR_WIDTH_U2,
            'fix_ld': FIX_LD,
            'dur_prior': DUR_PRIOR,
            'dur_prior_hours': DURATION_OVERRIDE if (DUR_PRIOR and DURATION_OVERRIDE) else 0.0,
            'dur_prior_sigma_frac': 0.15,
        }

        # Create cached TransitModel for single-CPU path
        self._cached_transit_model = create_cached_transit_model(
            self.time, self.batman_params, cadence=self.cadence,
            long_cadence_threshold=self.long_cadence_threshold
        )

        # Initialize module-level MCMC variables in main process
        _init_mcmc_worker(_MCMC_SHARED_DATA)

        # Use multiprocessing pool if n_cpus > 1
        if n_cpus > 1:
            print(f"  Using {n_cpus} CPUs for parallel MCMC (vectorized)", flush=True)
            _MCMC_POOL = Pool(processes=n_cpus, initializer=_init_mcmc_worker,
                              initargs=(_MCMC_SHARED_DATA,))
            self.pool = _MCMC_POOL
        else:
            _MCMC_POOL = None
            self.pool = None

        self.sampler = emcee.EnsembleSampler(
            n_walkers, ndim, _mcmc_log_probability_vectorized, vectorize=True
        )

        # Adaptive burn-in: run in chunks, stop early when R-hat converges
        burnin_check_interval = config.CONVERGENCE_CHECK_INTERVAL  # 500 steps
        burnin_min = config.N_BURN_MIN  # 2000 steps minimum
        burnin_max = n_burn  # 4000 steps maximum
        print(f"\n  Running adaptive burn-in (min={burnin_min}, max={burnin_max}, check every {burnin_check_interval})...", flush=True)

        burnin_steps_done = 0
        state = p0
        while burnin_steps_done < burnin_max:
            chunk = min(burnin_check_interval, burnin_max - burnin_steps_done)
            state = self.sampler.run_mcmc(state, chunk, progress=False)
            burnin_steps_done += chunk

            # Check R-hat after minimum burn-in reached
            if burnin_steps_done >= burnin_min:
                chains = self.sampler.get_chain()  # (n_steps, n_walkers, n_params)
                rhat = compute_rhat_split(chains)
                max_rhat = float(np.max(rhat))
                if max_rhat < config.CONVERGENCE_RHAT:
                    print(f"    Burn-in converged at {burnin_steps_done} steps (R-hat={max_rhat:.4f})", flush=True)
                    break
                else:
                    print(f"    {burnin_steps_done} steps: R-hat={max_rhat:.4f} (not converged)", flush=True)

        if burnin_steps_done >= burnin_max:
            print(f"    Burn-in reached maximum {burnin_max} steps", flush=True)

        burnin_chains_1 = self.sampler.get_chain()
        burnin_log_prob_1 = self.sampler.get_log_prob()

        print(f"\n  Checking for bad walkers...", flush=True)
        bad_indices, sigma, median_of_medians = self._identify_bad_walkers(burnin_log_prob_1)
        self.n_bad_walkers = len(bad_indices)
        self.bad_walker_indices = bad_indices

        if len(bad_indices) > 0:
            print(f"\n  Re-initializing {len(bad_indices)} bad walkers...", flush=True)
            new_positions = self._reinitialize_bad_walkers(burnin_chains_1, bad_indices, sigma)
            self.sampler.reset()
            # Second burn-in is shorter — just enough for re-initialized walkers to mix
            second_burnin = burnin_min
            print(f"\n  Running second burn-in ({second_burnin} steps)...", flush=True)
            state = self.sampler.run_mcmc(new_positions, second_burnin, progress=False)
            burnin_chains_2 = self.sampler.get_chain()
            burnin_log_prob_2 = self.sampler.get_log_prob()
            self.burnin_chain = np.concatenate([burnin_chains_1, burnin_chains_2], axis=0)
            self.burnin_log_prob = np.concatenate([burnin_log_prob_1, burnin_log_prob_2], axis=0)
            self.burnin_break = burnin_steps_done
        else:
            self.burnin_chain = burnin_chains_1
            self.burnin_log_prob = burnin_log_prob_1
            self.burnin_break = None

        self.sampler.reset()

        if check_convergence_flag:
            print(f"\n  Running production with convergence checking...", flush=True)
            diagnostics = run_until_converged(
                self.sampler, state,
                max_steps=N_STEPS_MAX,
                check_interval=config.CONVERGENCE_CHECK_INTERVAL,
                rhat_threshold=config.CONVERGENCE_RHAT,
                ess_threshold=config.CONVERGENCE_ESS,
                autocorr_threshold=config.CONVERGENCE_AUTOCORR
            )
        else:
            self.sampler.run_mcmc(state, N_STEPS_MAX, progress=False)
            diagnostics = {'converged': True, 'n_steps': N_STEPS_MAX}

        self.samples = self.sampler.get_chain(flat=True)

        # Outlier rejection using Chauvenet's criterion
        # Compute model from median parameters, find outliers, mask them, and refit
        preliminary_medians = np.median(self.samples, axis=0)
        if FIX_LD:
            p_period, p_t0, p_rp, p_ars, p_bsq, p_bl = preliminary_medians
            p_u1, p_u2 = self.u1, self.u2
        else:
            p_period, p_t0, p_rp, p_ars, p_bsq, p_bl, p_u1, p_u2 = preliminary_medians
        p_b = np.sqrt(p_bsq)

        bp_check = setup_batman_params(p_period, p_t0, p_rp, p_ars, p_b, p_u1, p_u2)
        model_check = create_cached_transit_model(
            self.time, bp_check, cadence=self.cadence,
            long_cadence_threshold=self.long_cadence_threshold)
        if hasattr(model_check, 'is_mixed'):
            if model_check.is_mixed:
                buf = model_check.buffer
                buf[:] = 0.0
                for mask_m, m in model_check.model:
                    buf[mask_m] = m.light_curve(bp_check)
                model_flux_check = buf
            else:
                model_flux_check = model_check.model.light_curve(bp_check)
        elif isinstance(model_check, list):
            model_flux_check = np.zeros(len(self.time))
            for mask_m, m in model_check:
                model_flux_check[mask_m] = m.light_curve(bp_check)
        else:
            model_flux_check = model_check.light_curve(bp_check)
        model_flux_check *= p_bl

        residuals_check = self.flux - model_flux_check
        residuals_sigma = residuals_check / self.flux_err
        N_pts = len(self.flux)
        chauvenet_threshold = norm.ppf(1.0 - 1.0 / (4 * N_pts))
        outlier_mask = np.abs(residuals_sigma) > chauvenet_threshold
        n_outliers = np.sum(outlier_mask)

        outlier_iter = getattr(self, '_outlier_iteration', 0)
        max_outlier_iter = 5
        total_rejected = getattr(self, '_total_outliers_rejected', 0)

        if n_outliers > 0 and n_outliers < 0.1 * N_pts and outlier_iter < max_outlier_iter:
            print(f"\n  Chauvenet outlier rejection (iteration {outlier_iter + 1}): "
                  f"{n_outliers} / {N_pts} points (threshold = {chauvenet_threshold:.1f} sigma)", flush=True)
            good_mask = ~outlier_mask
            self.time = self.time[good_mask]
            self.flux = self.flux[good_mask]
            self.flux_err = self.flux_err[good_mask]
            if self.cadence is not None:
                self.cadence = self.cadence[good_mask]

            # Refit with cleaned data
            self._outlier_iteration = outlier_iter + 1
            self._total_outliers_rejected = total_rejected + n_outliers
            return self.fit(n_walkers=n_walkers, n_burn=n_burn,
                           check_convergence_flag=check_convergence_flag, n_cpus=n_cpus)
        else:
            self.n_outliers_rejected = total_rejected
            if n_outliers > 0 and outlier_iter >= max_outlier_iter:
                print(f"\n  Chauvenet: max iterations ({max_outlier_iter}) reached, "
                      f"total rejected = {total_rejected}", flush=True)
            elif n_outliers > 0:
                print(f"\n  Chauvenet: {n_outliers} outliers found but > 10% of data, "
                      f"skipping rejection", flush=True)
            elif total_rejected > 0:
                print(f"\n  Chauvenet: converged after {outlier_iter} iterations, "
                      f"total rejected = {total_rejected}", flush=True)

        params = {}
        for i, name in enumerate(PARAM_NAMES):
            values = self.samples[:, i]
            median = np.median(values)
            p16, p84 = np.percentile(values, [config.MCMC_PERCENTILES[0], config.MCMC_PERCENTILES[2]])
            params[name] = {
                'value': median,
                'err_lower': median - p16,
                'err_upper': p84 - median,
                'err': (p84 - p16) / 2,
                'percentile_16': p16,
                'percentile_84': p84
            }

        # Convert b_sq samples to b for reporting (MCMC samples b^2, report b)
        b_idx = PARAM_NAMES.index('b')
        b_samples = np.sqrt(self.samples[:, b_idx])
        b_med = np.median(b_samples)
        b_p16, b_p84 = np.percentile(b_samples, [config.MCMC_PERCENTILES[0], config.MCMC_PERCENTILES[2]])
        params['b'] = {
            'value': b_med,
            'err_lower': b_med - b_p16,
            'err_upper': b_p84 - b_med,
            'err': (b_p84 - b_p16) / 2,
            'percentile_16': b_p16,
            'percentile_84': b_p84
        }

        # If limb darkening was fixed, add it to params with zero uncertainty
        if FIX_LD:
            params['u1'] = {
                'value': self.u1,
                'err_lower': 0.0,
                'err_upper': 0.0,
                'err': 0.0,
                'percentile_16': self.u1,
                'percentile_84': self.u1,
                'fixed': True
            }
            params['u2'] = {
                'value': self.u2,
                'err_lower': 0.0,
                'err_upper': 0.0,
                'err': 0.0,
                'percentile_16': self.u2,
                'percentile_84': self.u2,
                'fixed': True
            }

        self.results = {'parameters': params, 'diagnostics': diagnostics}

        # Clean up multiprocessing pool and MCMC cache
        if self.pool is not None:
            self.pool.close()
            self.pool.join()
            self.pool = None
            _MCMC_POOL = None
        self._cached_transit_model = None

        return self.results

    def get_phase_folded_data(self):
        period = self.results['parameters']['period']['value']
        t0 = self.results['parameters']['t0']['value']
        phase = ((self.time - t0) / period) % 1.0
        phase[phase > 0.5] -= 1.0
        return phase, self.flux, self.flux_err

    def get_binned_phase_folded(self, n_bins=200):
        """Bin phase-folded data with Chauvenet's criterion outlier rejection."""
        phase, flux, flux_err = self.get_phase_folded_data()
        bins = np.linspace(-0.5, 0.5, n_bins + 1)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        bin_flux = np.zeros(n_bins)
        bin_err = np.zeros(n_bins)
        for i in range(n_bins):
            mask = (phase >= bins[i]) & (phase < bins[i+1])
            if np.sum(mask) > 0:
                bin_flux[i], bin_err[i], _ = weighted_bin_with_outlier_rejection(
                    flux[mask], flux_err[mask])
            else:
                bin_flux[i] = np.nan
                bin_err[i] = np.nan
        valid = ~np.isnan(bin_flux)
        return bin_centers[valid], bin_flux[valid], bin_err[valid]


def filter_outlier_transits(transit_fits, sigma_threshold=config.OUTLIER_SIGMA_T0_ERR):
    """
    Filter out transits with t0 errors > sigma_threshold * median error.

    Returns: filtered_fits, excluded_fits, median_err, threshold
    """
    if len(transit_fits) == 0:
        return [], [], None, None

    t0_errors = np.array([tf['t0_err'] for tf in transit_fits])
    median_err = np.median(t0_errors)
    threshold = sigma_threshold * median_err

    filtered_fits = [tf for tf in transit_fits if tf['t0_err'] <= threshold]
    excluded_fits = [tf for tf in transit_fits if tf['t0_err'] > threshold]

    return filtered_fits, excluded_fits, median_err, threshold


def filter_oc_outliers(transit_fits, period, t0_ref, sigma_threshold=config.OUTLIER_SIGMA_OC):
    """
    Filter out transits with O-C values > sigma_threshold * sigma from median O-C.

    Sigma is calculated as 1.48 * MAD (median absolute deviation).

    Returns: filtered_fits, excluded_fits, oc_stats
    """
    if len(transit_fits) == 0:
        return [], [], {}

    # Calculate O-C values for all transits
    oc_values = []
    for tf in transit_fits:
        t_expected = t0_ref + tf['epoch'] * period
        oc = (tf['t0_fit'] - t_expected) * 24 * 60  # Convert to minutes
        oc_values.append(oc)

    oc_values = np.array(oc_values)

    # Calculate median and MAD-based sigma
    median_oc = np.median(oc_values)
    mad = np.median(np.abs(oc_values - median_oc))
    sigma = 1.48 * mad

    # Filter based on deviation from median
    if sigma > 0:
        threshold = sigma_threshold * sigma
        filtered_fits = []
        excluded_fits = []

        for i, tf in enumerate(transit_fits):
            deviation = np.abs(oc_values[i] - median_oc)
            if deviation <= threshold:
                filtered_fits.append(tf)
            else:
                # Add O-C info to excluded transit
                tf_excluded = tf.copy()
                tf_excluded['oc_minutes'] = oc_values[i]
                tf_excluded['oc_deviation_sigma'] = deviation / sigma if sigma > 0 else np.inf
                excluded_fits.append(tf_excluded)
    else:
        # If sigma is 0, all values are the same - keep all
        filtered_fits = transit_fits
        excluded_fits = []
        threshold = 0

    oc_stats = {
        'median_oc_minutes': float(median_oc),
        'mad_minutes': float(mad),
        'sigma_minutes': float(sigma),
        'threshold_minutes': float(threshold) if sigma > 0 else None
    }

    return filtered_fits, excluded_fits, oc_stats


def get_min_points_for_cadence(cadence_sec):
    """
    Get minimum number of points required for transit fitting based on cadence.

    Parameters:
    -----------
    cadence_sec : float
        Cadence in seconds

    Returns:
    --------
    int
        Minimum number of points required
    """
    if cadence_sec >= 1800:  # 30-min cadence
        return config.MIN_POINTS_30MIN
    elif cadence_sec >= 600:  # 10-min cadence
        return config.MIN_POINTS_10MIN
    else:  # 200s or 2-min cadence
        return config.MIN_POINTS_2MIN


def identify_transits(time, period, t0_ref, duration_days=0.2, require_full_coverage=True,
                      coverage_factor=config.TRANSIT_COVERAGE_FACTOR, cadence=None):
    """
    Identify individual transit windows in the data.

    Parameters:
    -----------
    time : array
        Time array of observations
    period : float
        Orbital period in days
    t0_ref : float
        Reference mid-transit time
    duration_days : float
        Transit duration in days
    require_full_coverage : bool
        If True, exclude partial transits that don't have data coverage
        from -coverage_factor*duration to +coverage_factor*duration
    coverage_factor : float
        Factor for coverage requirement (default 1.5 means 3x duration total)
    cadence : array, optional
        Cadence array (in seconds) for each time point. If provided, uses
        cadence-dependent minimum points requirement.

    Returns:
    --------
    transits : list of dicts
        List of transit info dictionaries
    partial_transits : list of dicts
        List of excluded partial transits (if require_full_coverage=True)
    """
    # Calculate expected transit times
    t_min, t_max = time.min(), time.max()

    # Find epoch range
    n_min = int(np.floor((t_min - t0_ref) / period))
    n_max = int(np.ceil((t_max - t0_ref) / period))

    # Coverage window size (1.5 * duration on each side = 3 * duration total)
    coverage_half_width = coverage_factor * duration_days

    transits = []
    partial_transits = []

    for n in range(n_min, n_max + 1):
        t_expected = t0_ref + n * period
        # Check if we have data around this transit
        window_mask = np.abs(time - t_expected) < coverage_half_width
        n_points = np.sum(window_mask)

        # Determine minimum points based on cadence
        if cadence is not None and n_points > 0:
            # Use median cadence in this window
            median_cadence = np.median(cadence[window_mask])
            min_points = get_min_points_for_cadence(median_cadence)
        else:
            min_points = 20  # Default fallback

        if n_points >= min_points:
            transit_info = {
                'epoch': n,
                't_expected': t_expected,
                'mask': window_mask,
                'n_points': n_points,
                'min_points_required': min_points,
                'median_cadence_sec': float(median_cadence) if cadence is not None else None
            }

            if require_full_coverage:
                # Check data coverage: need data from t_expected - coverage_half_width
                # to t_expected + coverage_half_width
                t_transit = time[window_mask]
                t_start_required = t_expected - coverage_half_width
                t_end_required = t_expected + coverage_half_width

                # Check if we have data near both edges of the coverage window.
                # Tolerance is max(one cadence, 10% of coverage window): at long
                # cadence (e.g. 30-min) the 10% rule alone is tighter than the
                # data spacing, which causes well-sampled transits to be
                # misflagged as partial.
                cadence_days = (median_cadence / 86400.0) if cadence is not None else 0.0
                tolerance = max(cadence_days, 0.1 * coverage_half_width)
                has_early_data = np.any(t_transit <= t_start_required + tolerance)
                has_late_data = np.any(t_transit >= t_end_required - tolerance)

                # Require out-of-transit baseline data before ingress and after egress
                # Check that the data extent provides baseline on both sides of the transit.
                # The data must extend at least half_dur + baseline_margin beyond T_expected
                # on both sides, ensuring baseline exists regardless of where the transit falls.
                half_dur = duration_days / 2.0
                baseline_margin = 0.25 * duration_days
                min_baseline_extent = half_dur + baseline_margin
                # Use the actual data range to check baseline coverage
                t_data_min = t_transit.min()
                t_data_max = t_transit.max()
                has_pre_ingress = (t_expected - t_data_min) >= min_baseline_extent
                has_post_egress = (t_data_max - t_expected) >= min_baseline_extent

                if has_early_data and has_late_data and has_pre_ingress and has_post_egress:
                    # Full coverage - include this transit
                    transit_info['coverage'] = 'full'
                    transit_info['t_data_min'] = float(t_transit.min())
                    transit_info['t_data_max'] = float(t_transit.max())
                    transits.append(transit_info)
                else:
                    # Partial coverage - exclude this transit
                    transit_info['coverage'] = 'partial'
                    transit_info['t_data_min'] = float(t_transit.min())
                    transit_info['t_data_max'] = float(t_transit.max())
                    transit_info['t_required_min'] = float(t_start_required)
                    transit_info['t_required_max'] = float(t_end_required)
                    transit_info['has_early_data'] = has_early_data
                    transit_info['has_late_data'] = has_late_data
                    transit_info['has_pre_ingress'] = has_pre_ingress
                    transit_info['has_post_egress'] = has_post_egress
                    partial_transits.append(transit_info)
            else:
                # No coverage check - include all transits with enough points
                transits.append(transit_info)

    return transits, partial_transits


def plot_chains(fitter, output_path):
    """Plot MCMC chains."""
    prod_chains = fitter.sampler.get_chain()
    prod_log_prob = fitter.sampler.get_log_prob()

    if fitter.burnin_chain is not None:
        n_burnin = fitter.burnin_chain.shape[0]
        chains = np.concatenate([fitter.burnin_chain, prod_chains], axis=0)
        log_prob = np.concatenate([fitter.burnin_log_prob, prod_log_prob], axis=0)
    else:
        chains = prod_chains
        log_prob = prod_log_prob
        n_burnin = None

    n_steps, n_walkers, n_params = chains.shape
    n_panels = n_params + 1

    fig, axes = plt.subplots(n_panels, 1, figsize=(12, 2.5 * n_panels), sharex=True)

    for i, (ax, name) in enumerate(zip(axes[:-1], PARAM_NAMES)):
        ax.plot(chains[:, :, i], alpha=0.3, linewidth=0.5)
        ax.set_ylabel(name)
        if n_burnin is not None:
            ax.axvline(n_burnin, color='orange', linestyle='--', linewidth=2,
                      label='Burn-in end' if i == 0 else None)
        if fitter.burnin_break is not None:
            ax.axvline(fitter.burnin_break, color='red', linestyle=':', linewidth=2,
                      label='Re-init' if i == 0 else None)

    ax_logp = axes[-1]
    ax_logp.plot(log_prob, alpha=0.3, linewidth=0.5)
    ax_logp.set_ylabel('log(prob)')
    ax_logp.set_xlabel('Step')
    if n_burnin is not None:
        ax_logp.axvline(n_burnin, color='orange', linestyle='--', linewidth=2)
    if fitter.burnin_break is not None:
        ax_logp.axvline(fitter.burnin_break, color='red', linestyle=':', linewidth=2)

    if n_burnin is not None or fitter.burnin_break is not None:
        axes[0].legend(loc='upper right')

    fig.suptitle(f'TOI {TOI} - MCMC Chains', fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=config.PLOT_DPI, bbox_inches='tight')
    plt.close(fig)


def plot_corner_plot(samples, output_path):
    """Generate corner plot."""
    if not HAS_CORNER:
        print("  Skipping corner plot (corner package not installed)")
        return

    fig = corner.corner(
        samples[::10],
        labels=PARAM_NAMES,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_kwargs={"fontsize": 10}
    )
    fig.suptitle(f'TOI {TOI} - Parameter Covariance', fontsize=14, y=1.02)
    fig.savefig(output_path, dpi=config.PLOT_DPI, bbox_inches='tight')
    plt.close(fig)


def plot_phase_folded(fitter, output_path, bin_minutes=config.BIN_WIDTH_MINUTES, phase_window=None,
                      full_time=None, full_flux=None, full_flux_err=None, full_cadence=None):
    """
    Plot phase-folded light curve with specified bin width in minutes.

    If phase_window is None, it is calculated as 4× transit duration (2× T14 on each side).
    Only includes data within ±phase_window of transit center.

    If full_time/full_flux/full_flux_err/full_cadence are provided, re-selects data from the
    full time series using fitted T0/period. This avoids asymmetric clipping when catalog
    ephemeris differs from fitted ephemeris (the initial data selection used catalog values,
    which can shift phases at late epochs).

    Residuals are calculated by first computing the exposure-time-integrated
    model for each individual measurement, then binning the residuals.
    """
    params = fitter.results['parameters']
    period = params['period']['value']
    t0 = params['t0']['value']
    rp_rs = params['rp_rs']['value']
    a_rs = params['a_rs']['value']
    b = params['b']['value']

    # Calculate phase_window from fitted parameters if not specified
    # Window = 4× T14 = ±2× T14 from mid-transit
    if phase_window is None:
        # Calculate transit duration: T14 = (1/π) × arcsin(√((1+k)² - b²) / a)
        if b < (1 + rp_rs) and a_rs > 0:
            sin_arg = np.sqrt((1 + rp_rs)**2 - b**2) / a_rs
            if sin_arg <= 1:
                t14_phase = np.arcsin(sin_arg) / np.pi
            else:
                t14_phase = config.TRANSIT_WINDOW_PHASE_MIN  # Fallback
        else:
            # Grazing transit - use approximate duration
            t14_phase = rp_rs / (np.pi * a_rs) if a_rs > 0 else config.TRANSIT_WINDOW_PHASE_MIN
        # Transit window: 4× T14 = ±2× T14
        phase_window = config.TRANSIT_WINDOW_MULTIPLIER * t14_phase
        phase_window = max(config.TRANSIT_WINDOW_PHASE_MIN, min(phase_window, config.TRANSIT_WINDOW_PHASE_MAX))

    # Use full time series folded with fitted ephemeris (avoids asymmetric clipping
    # when catalog T0/period differ from fitted values)
    if full_time is not None and full_flux is not None:
        phase_full = ((full_time - t0) / period) % 1.0
        phase_full = np.where(phase_full > 0.5, phase_full - 1.0, phase_full)
        full_flux_err_safe = full_flux_err if full_flux_err is not None else np.ones_like(full_flux) * config.DEFAULT_FLUX_ERROR
        full_cadence_safe = full_cadence if full_cadence is not None else np.full(len(full_time), config.DEFAULT_CADENCE_SPOC)

        in_window = np.abs(phase_full) <= phase_window
        phase = phase_full[in_window]
        flux = full_flux[in_window]
        flux_err = full_flux_err_safe[in_window]
        cadence = full_cadence_safe[in_window]
        individual_time = full_time[in_window]

        sort_idx = np.argsort(phase)
        phase = phase[sort_idx]
        flux = flux[sort_idx]
        flux_err = flux_err[sort_idx]
        cadence = cadence[sort_idx]
        individual_time = individual_time[sort_idx]
    else:
        # Fall back to fitter's data if full time series not provided
        phase_fitter, flux_fitter, flux_err_fitter = fitter.get_phase_folded_data()
        cadence_fitter = fitter.cadence
        time_fitter = fitter.time
        in_window_fitter = np.abs(phase_fitter) <= phase_window
        phase = phase_fitter[in_window_fitter]
        flux = flux_fitter[in_window_fitter]
        flux_err = flux_err_fitter[in_window_fitter]
        cadence = cadence_fitter[in_window_fitter]
        individual_time = time_fitter[in_window_fitter]

    # Calculate n_bins for desired bin width in minutes (only for transit window)
    transit_duration_minutes = 2 * phase_window * period * 24 * 60
    n_bins = max(20, int(transit_duration_minutes / bin_minutes))

    # Bin the filtered data with outlier rejection
    bins = np.linspace(-phase_window, phase_window, n_bins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    bin_flux = np.zeros(n_bins)
    bin_err = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (phase >= bins[i]) & (phase < bins[i+1])
        if np.sum(mask) > 0:
            bin_flux[i], bin_err[i], _ = weighted_bin_with_outlier_rejection(
                flux[mask], flux_err[mask])
        else:
            bin_flux[i] = np.nan
            bin_err[i] = np.nan
    valid = ~np.isnan(bin_flux)
    bin_phase = bin_centers[valid]
    bin_flux = bin_flux[valid]
    bin_err = bin_err[valid]

    baseline = params['baseline']['value']
    u1 = params['u1']['value']
    u2 = params['u2']['value']

    model_phase = np.linspace(-phase_window, phase_window, 500)
    model_time = t0 + model_phase * period
    model_flux = fitter.transit_model(model_time, period, t0, rp_rs, a_rs, b, baseline, u1, u2)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8),
                             sharex=True, gridspec_kw={'hspace': 0.05, 'height_ratios': [3, 1]})
    ax_main, ax_resid = axes

    ax_main.scatter(phase, flux, s=1, alpha=0.1, color='gray', label='Data')
    ax_main.errorbar(bin_phase, bin_flux, yerr=bin_err, fmt='o', ms=4,
                     color='blue', capsize=2, label=f'Binned ({bin_minutes:.0f} min)')
    ax_main.plot(model_phase, model_flux, 'r-', linewidth=2, label='Model')
    ax_main.set_ylabel('Normalized Flux')
    ax_main.legend(loc='lower right')
    ax_main.set_title(f'TOI {TOI} - Phase-Folded Light Curve ({n_bins} bins, {bin_minutes:.0f} min each)')
    ax_main.set_xlim(-phase_window, phase_window)

    # Set Y-axis limits to zoom on transit: 2× depth below baseline, 1× depth above
    depth = rp_rs ** 2
    ax_main.set_ylim(baseline - 2 * depth, baseline + depth)

    # Calculate exposure-time-integrated model for each individual measurement
    individual_model = fitter.transit_model(individual_time, period, t0, rp_rs, a_rs, b, baseline, u1, u2, cadence=cadence)

    # Calculate individual residuals
    individual_residuals = flux - individual_model

    # Bin the residuals using weighted mean with Chauvenet's criterion outlier rejection
    bin_residuals = np.zeros(n_bins)
    bin_residuals_err = np.zeros(n_bins)

    for i in range(n_bins):
        mask = (phase >= bins[i]) & (phase < bins[i+1])
        if np.sum(mask) > 0:
            bin_residuals[i], bin_residuals_err[i], _ = weighted_bin_with_outlier_rejection(
                individual_residuals[mask], flux_err[mask])
        else:
            bin_residuals[i] = np.nan
            bin_residuals_err[i] = np.nan

    # Keep only bins that match bin_phase (valid bins from get_binned_phase_folded)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    valid = ~np.isnan(bin_residuals)
    bin_residuals = bin_residuals[valid] * 1e6  # Convert to ppm
    bin_residuals_err = bin_residuals_err[valid] * 1e6

    ax_resid.errorbar(bin_phase, bin_residuals, yerr=bin_residuals_err, fmt='o', ms=4,
                      color='blue', capsize=2)
    ax_resid.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax_resid.set_xlabel('Phase')
    ax_resid.set_ylabel('Residuals (ppm)')

    # Calculate and display RMS
    rms_ppm = np.std(bin_residuals)
    ax_resid.text(0.02, 0.95, f'RMS = {rms_ppm:.0f} ppm', transform=ax_resid.transAxes,
                  fontsize=10, verticalalignment='top')

    fig.tight_layout()
    fig.savefig(output_path, dpi=config.PLOT_DPI, bbox_inches='tight')
    plt.close(fig)


def plot_full_phase_folded(fitter, output_path, bin_minutes=config.BIN_WIDTH_MINUTES,
                           full_time=None, full_flux=None, full_flux_err=None, full_cadence=None):
    """
    Plot full phase-folded light curve showing entire orbital phase (0 to 1).

    Transit at phase 0.25, secondary eclipse at phase 0.75.

    If full orbit data (full_time, full_flux, etc.) is provided, uses that for plotting.
    Otherwise falls back to fitter's data (which may be filtered to transit window only).

    Residuals are calculated by first computing the exposure-time-integrated
    model for each individual measurement, then binning the residuals.
    """
    params = fitter.results['parameters']
    period = params['period']['value']
    t0 = params['t0']['value']

    # Use full orbit data if provided, otherwise fall back to fitter's data
    if full_time is not None and full_flux is not None:
        time_data = full_time
        flux = full_flux
        flux_err = full_flux_err if full_flux_err is not None else np.ones_like(full_flux) * 0.001
        cadence = full_cadence if full_cadence is not None else np.full(len(full_time), config.DEFAULT_CADENCE_SPOC)

        # Phase fold the full orbit data
        phase_orig = ((time_data - t0) / period) % 1.0
        phase_orig[phase_orig > 0.5] -= 1.0  # Center on transit at phase 0
    else:
        # Fall back to fitter's data (may be transit-window only)
        phase_orig, flux, flux_err = fitter.get_phase_folded_data()
        time_data = fitter.time
        cadence = fitter.cadence

    rp_rs = params['rp_rs']['value']
    a_rs = params['a_rs']['value']
    b = params['b']['value']
    baseline = params['baseline']['value']
    u1 = params['u1']['value']
    u2 = params['u2']['value']

    # Shift phase so transit is at 0.25 and eclipse at 0.75
    # Original: phase in [-0.5, 0.5] with transit at 0
    # New: phase in [0, 1] with transit at 0.25
    phase = (phase_orig + 0.25) % 1.0

    # Calculate n_bins for desired bin width in minutes
    n_bins = max(50, int(period * 24 * 60 / bin_minutes))

    # Bin the data in [0, 1] range with Chauvenet's criterion outlier rejection
    bins = np.linspace(0, 1, n_bins + 1)
    bin_phase_all = (bins[:-1] + bins[1:]) / 2
    bin_flux = np.zeros(n_bins)
    bin_err = np.zeros(n_bins)

    for i in range(n_bins):
        mask = (phase >= bins[i]) & (phase < bins[i+1])
        if np.sum(mask) > 0:
            bin_flux[i], bin_err[i], _ = weighted_bin_with_outlier_rejection(
                flux[mask], flux_err[mask])
        else:
            bin_flux[i] = np.nan
            bin_err[i] = np.nan

    # Remove NaN bins
    valid = ~np.isnan(bin_flux)
    bin_phase = bin_phase_all[valid]
    bin_flux = bin_flux[valid]
    bin_err = bin_err[valid]

    # Model in [0, 1] phase range using MCMC baseline (same as transit plot)
    # Phase 0.25 corresponds to transit (t0)
    model_phase = np.linspace(0, 1, 1000)
    model_time = t0 + (model_phase - 0.25) * period
    model_flux_vals = fitter.transit_model(model_time, period, t0, rp_rs, a_rs, b, baseline, u1, u2)

    # Calculate exposure-time-integrated model for each individual measurement
    individual_time = time_data
    individual_model = fitter.transit_model(individual_time, period, t0, rp_rs, a_rs, b, baseline, u1, u2, cadence=cadence)

    # Calculate individual residuals
    individual_residuals = flux - individual_model

    # Bin the residuals with Chauvenet's criterion outlier rejection
    bin_residuals = np.zeros(n_bins)
    bin_residuals_err = np.zeros(n_bins)

    for i in range(n_bins):
        mask = (phase >= bins[i]) & (phase < bins[i+1])
        if np.sum(mask) > 0:
            bin_residuals[i], bin_residuals_err[i], _ = weighted_bin_with_outlier_rejection(
                individual_residuals[mask], flux_err[mask])
        else:
            bin_residuals[i] = np.nan
            bin_residuals_err[i] = np.nan

    # Keep only valid bins
    bin_residuals = bin_residuals[valid] * 1e6  # Convert to ppm
    bin_residuals_err = bin_residuals_err[valid] * 1e6

    # Create figure with 2 panels: full phase and residuals
    fig, axes = plt.subplots(2, 1, figsize=(12, 8),
                             gridspec_kw={'hspace': 0.15, 'height_ratios': [3, 1]})
    ax_full, ax_resid = axes

    # Full orbital phase
    ax_full.scatter(phase, flux, s=1, alpha=0.1, color='gray', label='Data')
    ax_full.errorbar(bin_phase, bin_flux, yerr=bin_err, fmt='o', ms=4,
                     color='blue', capsize=2, label=f'Binned ({bin_minutes:.0f} min)')
    ax_full.plot(model_phase, model_flux_vals, 'r-', linewidth=2, label='Transit Model')
    ax_full.set_ylabel('Normalized Flux')
    ax_full.legend(loc='lower right')
    ax_full.set_title(f'TOI {TOI} - Full Phase Curve ({n_bins} bins, {bin_minutes:.0f} min each)')
    ax_full.set_xlim(0, 1)

    # Add vertical lines for transit and eclipse
    ax_full.axvline(0.25, color='green', linestyle='--', alpha=0.5)
    ax_full.axvline(0.75, color='orange', linestyle='--', alpha=0.5)

    # Residuals
    ax_resid.errorbar(bin_phase, bin_residuals, yerr=bin_residuals_err, fmt='o', ms=4,
                      color='blue', capsize=2)
    ax_resid.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax_resid.set_xlabel('Orbital Phase')
    ax_resid.set_ylabel('Residuals (ppm)')
    ax_resid.set_xlim(0, 1)

    # Calculate and display RMS
    rms_ppm = np.std(bin_residuals)
    ax_resid.text(0.02, 0.95, f'RMS = {rms_ppm:.0f} ppm', transform=ax_resid.transAxes,
                  fontsize=10, verticalalignment='top')

    fig.savefig(output_path, dpi=config.PLOT_DPI, bbox_inches='tight')
    plt.close(fig)


def calculate_binned_residual_statistics(fitter, bin_minutes=config.BIN_WIDTH_MINUTES):
    """
    Calculate statistics on the binned phase-folded residuals.

    Residuals are calculated by first computing the exposure-time-integrated
    model for each individual measurement, then binning the residuals.

    Parameters
    ----------
    fitter : PhaseFoldFitter
        Fitted phase-folded fitter object
    bin_minutes : float
        Bin width in minutes (default: 10 minutes)

    Returns a dictionary with:
    - reduced_chi2: Reduced chi-squared of residuals
    - rms_over_median_err: RMS of residuals divided by median bin error
    - neumann_ratio: Neumann ratio (mean square successive difference ratio)
    - neumann_ratio_random: Neumann ratio on randomly permuted residuals
    - alarm_statistic: Alarm statistic (Tamuz, Mazeh & Zucker 2006)
    - alarm_statistic_random: Alarm statistic on randomly permuted residuals
    """
    # Get model parameters
    params = fitter.results['parameters']
    period = params['period']['value']

    # Calculate n_bins for desired bin width in minutes
    n_bins = max(50, int(period * 24 * 60 / bin_minutes))  # At least 50 bins

    # Get phase-folded data
    phase, flux, flux_err = fitter.get_phase_folded_data()
    cadence = fitter.cadence  # Exposure time in seconds for each point

    t0 = params['t0']['value']
    rp_rs = params['rp_rs']['value']
    a_rs = params['a_rs']['value']
    b = params['b']['value']
    baseline = params['baseline']['value']
    u1 = params['u1']['value']
    u2 = params['u2']['value']

    # Calculate exposure-time-integrated model for each individual measurement
    # Use utility function for cadence-aware model computation
    from autottv_pipeline_v2.utils import setup_batman_params, compute_batman_model

    individual_time = fitter.time
    batman_params = setup_batman_params(
        period=period, t0=t0, rp_rs=rp_rs, a_rs=a_rs, b=b, u1=u1, u2=u2
    )
    individual_model = compute_batman_model(individual_time, batman_params, cadence=cadence) * baseline

    # Calculate individual residuals
    individual_residuals = flux - individual_model

    # Bin the residuals with Chauvenet's criterion outlier rejection
    bins = np.linspace(-0.5, 0.5, n_bins + 1)
    bin_residuals = np.zeros(n_bins)
    bin_err = np.zeros(n_bins)

    for i in range(n_bins):
        mask = (phase >= bins[i]) & (phase < bins[i+1])
        if np.sum(mask) > 0:
            bin_residuals[i], bin_err[i], _ = weighted_bin_with_outlier_rejection(
                individual_residuals[mask], flux_err[mask])
        else:
            bin_residuals[i] = np.nan
            bin_err[i] = np.nan

    # Remove NaN bins
    valid = ~np.isnan(bin_residuals)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_centers_valid = bin_centers[valid]
    residuals = bin_residuals[valid]
    errors = bin_err[valid]
    n = len(residuals)

    # Calculate in-transit vs out-of-transit residual statistics
    # Estimate transit half-duration in phase units
    # T14 ≈ (P/π) * arcsin(sqrt((1+k)² - b²) / a) for non-grazing transits
    try:
        if b < (1 + rp_rs) and a_rs > 0:
            sin_arg = np.sqrt((1 + rp_rs)**2 - b**2) / a_rs
            if sin_arg <= 1:
                transit_half_duration_phase = np.arcsin(sin_arg) / np.pi
            else:
                transit_half_duration_phase = config.TRANSIT_HALF_DURATION_PHASE_FALLBACK
        else:
            # Grazing transit - use approximate duration from Rp/Rs
            transit_half_duration_phase = rp_rs / (np.pi * a_rs) if a_rs > 0 else config.TRANSIT_HALF_DURATION_PHASE_FALLBACK
    except:
        transit_half_duration_phase = config.TRANSIT_HALF_DURATION_PHASE_FALLBACK

    # Use multiplier × transit duration as "in-transit" region to capture ingress/egress
    in_transit_half_width = config.IN_TRANSIT_DURATION_MULTIPLIER * transit_half_duration_phase
    in_transit_half_width = max(config.IN_TRANSIT_HALF_WIDTH_MIN, min(in_transit_half_width, config.IN_TRANSIT_HALF_WIDTH_MAX))

    # Separate bins into in-transit and out-of-transit
    in_transit_mask = np.abs(bin_centers_valid) < in_transit_half_width
    out_of_transit_mask = ~in_transit_mask

    # Calculate RMS for each region
    n_in_transit = np.sum(in_transit_mask)
    n_out_of_transit = np.sum(out_of_transit_mask)

    if n_in_transit > 0:
        in_transit_rms = np.sqrt(np.mean(residuals[in_transit_mask]**2))
        in_transit_rms_ppm = float(in_transit_rms * 1e6)
    else:
        in_transit_rms_ppm = None

    if n_out_of_transit > 0:
        out_of_transit_rms = np.sqrt(np.mean(residuals[out_of_transit_mask]**2))
        out_of_transit_rms_ppm = float(out_of_transit_rms * 1e6)
    else:
        out_of_transit_rms_ppm = None

    # Calculate ratio (in-transit / out-of-transit)
    # Ratio > 1 indicates poorer fit during transit
    if in_transit_rms_ppm is not None and out_of_transit_rms_ppm is not None and out_of_transit_rms_ppm > 0:
        in_out_rms_ratio = in_transit_rms_ppm / out_of_transit_rms_ppm
    else:
        in_out_rms_ratio = None

    # 1. Reduced chi-squared
    # Number of fitted parameters in phase-folded model: 6 if LD fixed, 8 if LD free
    n_params = 6 if FIX_LD else 8
    chi2 = chi_squared(residuals, errors)
    reduced_chi2 = chi2 / (n - n_params) if n > n_params else None

    # 2. RMS of residuals divided by median bin error
    rms = np.sqrt(np.mean(residuals**2))
    median_err = np.median(errors)
    rms_over_median_err = rms / median_err

    # 3. Neumann ratio (mean square successive difference ratio)
    # η = Σ(r_{i+1} - r_i)² / Σ(r_i - mean(r))²
    # For uncorrelated residuals, η ≈ 2
    mean_resid = np.mean(residuals)
    successive_diff_sq = np.sum(np.diff(residuals)**2)
    variance_sum = np.sum((residuals - mean_resid)**2)
    neumann_ratio = successive_diff_sq / variance_sum if variance_sum > 0 else np.nan

    # 4. Neumann ratio for random permutation
    np.random.seed(42)  # For reproducibility
    random_residuals = np.random.permutation(residuals)
    successive_diff_sq_random = np.sum(np.diff(random_residuals)**2)
    mean_random = np.mean(random_residuals)
    variance_sum_random = np.sum((random_residuals - mean_random)**2)
    neumann_ratio_random = successive_diff_sq_random / variance_sum_random if variance_sum_random > 0 else np.nan

    # 5. Alarm statistic (Tamuz, Mazeh & Zucker 2006, MNRAS 367, 1521)
    # A = Σ(r_{i+1} - r_i)² / (2 * Σ(σ_i² + σ_{i+1}²))
    # For white noise, A ≈ 1. A < 1 indicates red (correlated) noise.
    diff_sq = np.diff(residuals)**2
    sigma_sum = errors[:-1]**2 + errors[1:]**2
    alarm_statistic = np.sum(diff_sq) / np.sum(sigma_sum) if np.sum(sigma_sum) > 0 else np.nan

    # 6. Alarm statistic on random permutation
    random_errors = np.random.permutation(errors)
    diff_sq_random = np.diff(random_residuals)**2
    sigma_sum_random = random_errors[:-1]**2 + random_errors[1:]**2
    alarm_statistic_random = np.sum(diff_sq_random) / np.sum(sigma_sum_random) if np.sum(sigma_sum_random) > 0 else np.nan

    return {
        'n_bins': n,
        'bin_minutes': float(bin_minutes),
        'reduced_chi2': float(reduced_chi2) if reduced_chi2 is not None else None,
        'rms_ppm': float(rms * 1e6),
        'median_bin_err_ppm': float(median_err * 1e6),
        'rms_over_median_err': float(rms_over_median_err),
        'neumann_ratio': float(neumann_ratio),
        'neumann_ratio_random': float(neumann_ratio_random),
        'alarm_statistic': float(alarm_statistic),
        'alarm_statistic_random': float(alarm_statistic_random),
        'in_transit_rms_ppm': in_transit_rms_ppm,
        'out_of_transit_rms_ppm': out_of_transit_rms_ppm,
        'in_out_rms_ratio': float(in_out_rms_ratio) if in_out_rms_ratio is not None else None,
        'transit_phase_half_width': float(in_transit_half_width),
        'n_bins_in_transit': int(n_in_transit),
        'n_bins_out_of_transit': int(n_out_of_transit)
    }


def plot_oc_with_ephemeris(transit_fits, linear_eph, quadratic_eph, output_dir):
    """
    Plot O-C diagram after fitting linear and quadratic ephemerides.

    Creates two plots:
    1. O-C after subtracting linear ephemeris
    2. If quadratic is better: O-C after subtracting linear component,
       with quadratic curve overplotted
    """
    if len(transit_fits) == 0:
        return

    epochs = np.array([tf['epoch'] for tf in transit_fits])
    t_obs = np.array([tf['t0_fit'] for tf in transit_fits])
    t_err = np.array([tf['t0_err'] for tf in transit_fits])

    T0_lin, P_lin = linear_eph['T0'], linear_eph['P']
    T0_quad, P_quad, Q_quad = quadratic_eph['T0'], quadratic_eph['P'], quadratic_eph['Q']

    # --- Plot 1: O-C after linear ephemeris ---
    t_calc_linear = T0_lin + P_lin * epochs
    oc_linear = (t_obs - t_calc_linear) * 24 * 60  # minutes
    oc_err = t_err * 24 * 60

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.errorbar(epochs, oc_linear, yerr=oc_err, fmt='o', ms=6, capsize=3,
                color='blue', label='O-C (linear subtracted)')
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('O-C (minutes)', fontsize=12)
    ax.set_title(f'TOI {TOI} - O-C Diagram (Linear Ephemeris Subtracted)', fontsize=14)

    # Add ephemeris info
    rms_linear = np.sqrt(np.mean(oc_linear**2))
    info_text = (f'Linear Ephemeris:\n'
                 f'T0 = {T0_lin:.6f} ± {linear_eph["T0_err"]:.6f} BJD\n'
                 f'P = {P_lin:.8f} ± {linear_eph["P_err"]:.8f} d\n'
                 f'χ² = {linear_eph["chi2"]:.1f}, BIC = {linear_eph["bic"]:.1f}\n'
                 f'RMS = {rms_linear:.2f} min')
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    fig.tight_layout()
    fig.savefig(output_dir / f"oc_linear_ephemeris.{config.PLOT_FORMAT}", dpi=config.PLOT_DPI, bbox_inches='tight')
    plt.close(fig)

    # --- Plot 2: O-C after linear component of quadratic (if quadratic fit is available) ---
    # Skip quadratic plot if there weren't enough transits for a quadratic fit
    if T0_quad is None or Q_quad is None:
        # Create a placeholder plot indicating insufficient data
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.errorbar(epochs, oc_linear, yerr=oc_err, fmt='o', ms=6, capsize=3,
                    color='blue', label='O-C')
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('O-C (minutes)', fontsize=12)
        ax.set_title(f'TOI {TOI} - O-C Diagram (Insufficient data for quadratic fit)', fontsize=14)
        ax.text(0.5, 0.5, f'Quadratic fit requires >= 3 transits\n(Only {len(epochs)} available)',
                transform=ax.transAxes, fontsize=14, ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        fig.tight_layout()
        fig.savefig(output_dir / f"oc_quadratic_ephemeris.{config.PLOT_FORMAT}", dpi=config.PLOT_DPI, bbox_inches='tight')
        plt.close(fig)
        return

    # O-C after subtracting only the linear part of quadratic ephemeris
    t_calc_linear_part = T0_quad + P_quad * epochs
    oc_quad_linear_subtracted = (t_obs - t_calc_linear_part) * 24 * 60  # minutes

    # Quadratic component for overplotting
    epoch_range = np.linspace(epochs.min(), epochs.max(), 500)
    quad_component = (Q_quad * epoch_range**2) * 24 * 60  # minutes

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.errorbar(epochs, oc_quad_linear_subtracted, yerr=oc_err, fmt='o', ms=6, capsize=3,
                color='blue', label='O-C (linear part subtracted)')

    # Overplot quadratic curve
    ax.plot(epoch_range, quad_component, 'r-', linewidth=2,
            label=f'Quadratic: Q×E² (Q = {Q_quad:.2e} d/epoch²)')

    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('O-C (minutes)', fontsize=12)
    ax.set_title(f'TOI {TOI} - O-C Diagram (Quadratic Ephemeris)', fontsize=14)
    ax.legend(loc='upper right')

    # Add ephemeris info
    rms_quad = np.sqrt(np.mean((quadratic_eph['residuals'] * 24 * 60)**2))
    info_text = (f'Quadratic Ephemeris:\n'
                 f'T0 = {T0_quad:.6f} ± {quadratic_eph["T0_err"]:.6f} BJD\n'
                 f'P = {P_quad:.8f} ± {quadratic_eph["P_err"]:.8f} d\n'
                 f'Q = {Q_quad:.2e} ± {quadratic_eph["Q_err"]:.2e} d/epoch²\n'
                 f'dP/dE = {quadratic_eph["dPdE"]:.2e} ± {quadratic_eph["dPdE_err"]:.2e} d/epoch\n'
                 f'χ² = {quadratic_eph["chi2"]:.1f}, BIC = {quadratic_eph["bic"]:.1f}\n'
                 f'RMS = {rms_quad:.2f} min')
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Indicate which model is preferred
    if quadratic_eph['bic'] < linear_eph['bic']:
        delta_bic = linear_eph['bic'] - quadratic_eph['bic']
        ax.text(0.98, 0.02, f'Quadratic preferred (ΔBIC = {delta_bic:.1f})',
                transform=ax.transAxes, fontsize=11, color='red', fontweight='bold',
                horizontalalignment='right', verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    else:
        delta_bic = quadratic_eph['bic'] - linear_eph['bic']
        ax.text(0.98, 0.02, f'Linear preferred (ΔBIC = {delta_bic:.1f})',
                transform=ax.transAxes, fontsize=11, color='green', fontweight='bold',
                horizontalalignment='right', verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    fig.tight_layout()
    fig.savefig(output_dir / f"oc_quadratic_ephemeris.{config.PLOT_FORMAT}", dpi=config.PLOT_DPI, bbox_inches='tight')
    plt.close(fig)


def plot_individual_transits(time, flux, flux_err, transit_fits, excluded_fits, fitter, output_path, max_per_figure=100):
    """Plot individual transit fits with baseline and slope, including rejected transits.

    If there are more than max_per_figure transits, they are split into multiple figures
    with filenames like output_part1.png, output_part2.png, etc.
    """
    # Combine all transits and sort by epoch
    all_transits = []
    for tf in transit_fits:
        tf_copy = tf.copy()
        tf_copy['rejected'] = False
        all_transits.append(tf_copy)
    for tf in excluded_fits:
        tf_copy = tf.copy()
        tf_copy['rejected'] = True
        all_transits.append(tf_copy)

    # Sort by epoch
    all_transits.sort(key=lambda x: x['epoch'])

    n_transits_total = len(all_transits)
    if n_transits_total == 0:
        return

    params = fitter.results['parameters']
    period = params['period']['value']
    rp_rs = params['rp_rs']['value']
    a_rs = params['a_rs']['value']
    b = params['b']['value']

    # Calculate number of figures needed
    n_figures = int(np.ceil(n_transits_total / max_per_figure))

    # Prepare output path parts
    output_path = Path(output_path)
    base_name = output_path.stem
    suffix = output_path.suffix
    output_dir = output_path.parent

    # Count statistics for legend (across all transits)
    n_unconverged_total = sum(1 for tf in all_transits if not tf.get('converged', True) and not tf['rejected'])
    n_rejected_total = len(excluded_fits)
    n_converged_total = len(transit_fits) - n_unconverged_total

    for fig_idx in range(n_figures):
        # Get transits for this figure
        start_idx = fig_idx * max_per_figure
        end_idx = min((fig_idx + 1) * max_per_figure, n_transits_total)
        transits_subset = all_transits[start_idx:end_idx]
        n_transits = len(transits_subset)

        n_cols = min(4, n_transits)
        n_rows = int(np.ceil(n_transits / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
        if n_transits == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)

        for i, tf in enumerate(transits_subset):
            row, col = i // n_cols, i % n_cols
            ax = axes[row, col]

            mask = tf['mask']
            t_data = time[mask]
            f_data = flux[mask]
            f_err = flux_err[mask]
            t0_fit = tf['t0_fit']
            baseline_fit = tf['baseline_fit']
            slope_fit = tf['slope_fit']
            is_rejected = tf['rejected']

            # Center on transit
            t_centered = (t_data - t0_fit) * 24  # hours

            # Use different colors for rejected transits
            data_color = 'lightcoral' if is_rejected else 'gray'
            model_color = 'darkred' if is_rejected else 'blue'

            ax.errorbar(t_centered, f_data, yerr=f_err, fmt='.', ms=3, alpha=0.5, color=data_color)

            # Model with fitted baseline and slope
            t_model = np.linspace(t_data.min(), t_data.max(), 200)
            # Build step1_results for modular IndividualTransitFitter
            step1_for_plot = {
                'parameters': {
                    'period': {'value': period},
                    't0': {'value': t0_fit},
                    'rp_rs': {'value': rp_rs},
                    'a_rs': {'value': a_rs},
                    'b': {'value': b},
                    'baseline': {'value': 1.0},
                    'u1': {'value': fitter.u1},
                    'u2': {'value': fitter.u2}
                },
                'derived': {'duration_hr': 3.0}
            }
            ind_fitter = ModularIndividualTransitFitter(step1_for_plot)
            f_model = ind_fitter.transit_model_with_trend(t_model, t0_fit, baseline_fit, slope_fit)
            t_model_centered = (t_model - t0_fit) * 24

            ax.plot(t_model_centered, f_model, '-', color=model_color, linewidth=1.5)

            # Title with rejection and convergence status
            title = f'Epoch {tf["epoch"]}'
            title_color = 'black'
            if is_rejected:
                title += ' [REJECTED]'
                title_color = 'red'
                ax.set_facecolor('#fff0f0')  # Light red background for rejected
            elif not tf.get('converged', True):
                title += ' [UNCONVERGED]'
                title_color = 'orange'
                ax.set_facecolor('#fff8e0')  # Light yellow background for unconverged
            ax.set_title(title, fontsize=10, color=title_color)
            ax.set_xlabel('Hours from mid-transit')
            ax.set_ylabel('Flux')

        # Hide empty subplots
        for i in range(n_transits, n_rows * n_cols):
            row, col = i // n_cols, i % n_cols
            axes[row, col].set_visible(False)

        # Add legend
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
        legend_elements = [
            Line2D([0], [0], color='blue', linewidth=2, label=f'Converged ({n_converged_total})'),
            Patch(facecolor='#fff8e0', edgecolor='orange', label=f'Unconverged ({n_unconverged_total})'),
            Patch(facecolor='#fff0f0', edgecolor='red', label=f'Rejected ({n_rejected_total})')
        ]
        fig.legend(handles=legend_elements, loc='upper right', fontsize=10)

        # Title with part number if multiple figures
        if n_figures > 1:
            fig.suptitle(f'TOI {TOI} - Individual Transit Fits (Part {fig_idx + 1}/{n_figures}, transits {start_idx + 1}-{end_idx})', fontsize=14)
            fig_output_path = output_dir / f"{base_name}_part{fig_idx + 1}{suffix}"
        else:
            fig.suptitle(f'TOI {TOI} - Individual Transit Fits', fontsize=14)
            fig_output_path = output_path

        fig.tight_layout()
        fig.savefig(fig_output_path, dpi=config.PLOT_DPI, bbox_inches='tight')
        plt.close(fig)

        if n_figures > 1:
            print(f"    Saved: {fig_output_path.name}", flush=True)


def plot_oc_diagram(transit_fits, period, t0_ref, output_path):
    """Plot O-C (Observed minus Calculated) diagram."""
    if len(transit_fits) == 0:
        return

    epochs = np.array([tf['epoch'] for tf in transit_fits])
    t_obs = np.array([tf['t0_fit'] for tf in transit_fits])
    t_err = np.array([tf['t0_err'] for tf in transit_fits])

    # Calculate expected times
    t_calc = t0_ref + epochs * period

    # O-C in minutes
    oc = (t_obs - t_calc) * 24 * 60  # minutes
    oc_err = t_err * 24 * 60

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.errorbar(epochs, oc, yerr=oc_err, fmt='o', ms=8, capsize=4, color='blue')
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('O-C (minutes)')
    ax.set_title(f'TOI {TOI} - Transit Timing Variations')

    # Add RMS annotation
    rms = np.sqrt(np.mean(oc**2))
    ax.text(0.02, 0.98, f'RMS = {rms:.2f} min', transform=ax.transAxes,
            verticalalignment='top', fontsize=12,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fig.tight_layout()
    fig.savefig(output_path, dpi=config.PLOT_DPI, bbox_inches='tight')
    plt.close(fig)

    return oc, oc_err, epochs


def _bootstrap_worker(args):
    """Worker for parallel bootstrap FAP computation.

    Uses astropy.timeseries.LombScargle with error weighting (dy=oc_err) so
    the bootstrap matches the same statistic as the full-sample periodogram.
    Each permutation re-pairs (oc, oc_err) at fixed time stamps.
    """
    from astropy.timeseries import LombScargle
    worker_id, n_iter, time_epochs, oc, oc_err, frequencies, peak_power = args
    rng = np.random.default_rng(seed=42 + worker_id)
    n_exceed = 0
    use_err = oc_err is not None and len(oc_err) == len(oc)
    for _ in range(n_iter):
        perm = rng.permutation(len(oc))
        oc_shuf = oc[perm]
        if use_err:
            err_shuf = oc_err[perm]
            power_shuf = LombScargle(time_epochs, oc_shuf, dy=err_shuf).power(
                frequencies, normalization='standard')
        else:
            power_shuf = LombScargle(time_epochs, oc_shuf).power(
                frequencies, normalization='standard')
        if power_shuf.max() >= peak_power:
            n_exceed += 1
    return n_exceed


def plot_periodogram(oc, epochs, period, output_path, bootstrap_n_iter=100000, oc_err=None):
    """Plot Lomb-Scargle periodogram of O-C values with bootstrap FAP."""
    if len(oc) < 5:
        print("  Not enough transits for periodogram")
        return None

    # Convert epochs to time
    time_epochs = epochs * period  # days

    # Frequency range (1/day)
    f_min = 2.0 / (time_epochs.max() - time_epochs.min())  # Require at least 2 cycles
    f_max = 0.5 / period  # Nyquist-like limit

    frequencies = np.linspace(f_min, f_max, 1000)

    # Lomb-Scargle periodogram — astropy with error weighting when oc_err is
    # provided, so the LS power matches the same statistic used by the C4 LOO
    # robustness check in find_ttv_candidates.py. See
    # docs/notes/2026-04-23-c4-loo-periodogram-inconsistency.md.
    from astropy.timeseries import LombScargle
    if oc_err is not None and len(oc_err) == len(oc):
        ls = LombScargle(time_epochs, oc, dy=oc_err)
    else:
        ls = LombScargle(time_epochs, oc)
    power = ls.power(frequencies, normalization='standard')

    # Find peak
    peak_idx = int(np.argmax(power))
    peak_freq = frequencies[peak_idx]
    peak_period = 1.0 / peak_freq
    peak_power = float(power[peak_idx])

    # Compute FWHM-based period error
    half_power = peak_power / 2.0
    f_left = frequencies[0]
    for i in range(peak_idx, 0, -1):
        if power[i - 1] <= half_power:
            frac = (half_power - power[i - 1]) / (power[i] - power[i - 1])
            f_left = frequencies[i - 1] + frac * (frequencies[i] - frequencies[i - 1])
            break
    f_right = frequencies[-1]
    for i in range(peak_idx, len(power) - 1):
        if power[i + 1] <= half_power:
            frac = (half_power - power[i]) / (power[i + 1] - power[i])
            f_right = frequencies[i] + frac * (frequencies[i + 1] - frequencies[i])
            break
    fwhm_freq = max(f_right - f_left, 0.0)
    peak_period_error = (fwhm_freq / 2.0) * peak_period**2 if fwhm_freq > 0 else 0.0

    # Bootstrap FAP: shuffle (oc, oc_err) pairs among fixed time stamps (parallelized)
    n_workers = N_CPUS
    chunk_size = bootstrap_n_iter // n_workers
    remainder = bootstrap_n_iter % n_workers
    use_err = oc_err is not None and len(oc_err) == len(oc)
    err_arg = oc_err if use_err else None
    chunks = [(i, chunk_size + (1 if i < remainder else 0),
               time_epochs, oc, err_arg, frequencies, peak_power)
              for i in range(n_workers)]

    if n_workers > 1 and bootstrap_n_iter >= 1000:
        with Pool(processes=n_workers) as pool:
            results = pool.map(_bootstrap_worker, chunks)
        n_exceed = sum(results)
    else:
        n_exceed = 0
        rng = np.random.default_rng(seed=42)
        for _ in range(bootstrap_n_iter):
            perm = rng.permutation(len(oc))
            oc_shuf = oc[perm]
            if use_err:
                err_shuf = oc_err[perm]
                power_shuf = LombScargle(time_epochs, oc_shuf, dy=err_shuf).power(
                    frequencies, normalization='standard')
            else:
                power_shuf = LombScargle(time_epochs, oc_shuf).power(
                    frequencies, normalization='standard')
            if power_shuf.max() >= peak_power:
                n_exceed += 1
    bootstrap_fap = (n_exceed + 1) / (bootstrap_n_iter + 1)

    # Format FAP string
    if bootstrap_fap < 0.001:
        fap_str = f'{bootstrap_fap:.1e}'
    elif bootstrap_fap < 0.01:
        fap_str = f'{bootstrap_fap:.3f}'
    elif bootstrap_fap < 0.10:
        fap_str = f'{bootstrap_fap:.2f}'
    else:
        fap_str = f'{bootstrap_fap*100:.1f}%'

    # Create figure: 1 panel if FAP >= 1%, 2 panels if FAP < 1%
    if bootstrap_fap < 0.01:
        fig, (ax, ax2) = plt.subplots(2, 1, figsize=(10, 9), gridspec_kw={'height_ratios': [2, 1]})
    else:
        fig, ax = plt.subplots(figsize=(10, 6))

    # Plot frequency on x-axis (1/day)
    ax.plot(frequencies, power, 'b-', linewidth=1)

    ax.set_xlabel('Frequency (1/day)')
    ax.set_ylabel('Lomb-Scargle Power')
    ax.set_title(f'TOI {TOI} - TTV Periodogram')

    # Mark peak
    ax.axvline(peak_freq, color='red', linestyle=':', alpha=0.5)

    period_err_str = f' ± {peak_period_error:.2f}' if peak_period_error > 0 else ''
    ax.text(0.98, 0.98,
            f'Peak frequency: {peak_freq:.4f} 1/day\n'
            f'Peak period: {peak_period:.2f}{period_err_str} days\n'
            f'Power: {peak_power:.3f}\n'
            f'Bootstrap FAP: {fap_str}\n'
            f'({bootstrap_n_iter} iterations)',
            transform=ax.transAxes, verticalalignment='top', horizontalalignment='right',
            fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Bottom panel: O-C phase-folded on peak period (only if FAP < 1%)
    if bootstrap_fap < 0.01:
        phase_oc = ((time_epochs - time_epochs.min()) / peak_period) % 1.0
        phase_oc[phase_oc > 0.5] -= 1.0
        if oc_err is not None:
            ax2.errorbar(phase_oc, oc, yerr=oc_err, fmt='o', ms=5, color='blue', ecolor='blue', capsize=3, zorder=5)
        else:
            ax2.errorbar(phase_oc, oc, fmt='o', ms=5, color='blue', capsize=3, zorder=5)
        # Fit sine: oc = A * sin(2*pi*phase + phi) + C
        try:
            from scipy.optimize import curve_fit as _curve_fit
            def _sine(ph, A, phi, C): return A * np.sin(2 * np.pi * ph + phi) + C
            _p0 = [np.std(oc), 0.0, np.mean(oc)]
            _sigma = oc_err if oc_err is not None and len(oc_err) == len(oc) else None
            _popt, _pcov = _curve_fit(_sine, phase_oc, oc, p0=_p0, sigma=_sigma, absolute_sigma=True)
            _A_err = np.sqrt(_pcov[0, 0])
            _ph_model = np.linspace(-0.5, 0.5, 200)
            ax2.plot(_ph_model, _sine(_ph_model, *_popt), 'r-', lw=2, zorder=10,
                     label=f'A={abs(_popt[0]):.1f}±{_A_err:.1f} min')
            ax2.legend(fontsize=9)
        except:
            pass
        ax2.axhline(0, color='gray', ls='--')
        ax2.set_xlabel(f'Phase (P = {peak_period:.2f} d)')
        ax2.set_ylabel('O-C (minutes)')
        ax2.set_title('O-C phase-folded on periodogram peak')
        ax2.set_xlim(-0.5, 0.5)

    fig.tight_layout()
    fig.savefig(output_path, dpi=config.PLOT_DPI, bbox_inches='tight')
    plt.close(fig)

    return {
        'peak_frequency': peak_freq,
        'peak_period': peak_period,
        'peak_period_error': peak_period_error,
        'peak_fwhm_freq': fwhm_freq,
        'peak_power': peak_power,
        'peak_fap': bootstrap_fap,
        'bootstrap_fap': bootstrap_fap,
        'bootstrap_n_iter': bootstrap_n_iter,
        'n_exceed': n_exceed
    }


def save_chains(fitter, output_dir):
    """Save MCMC chains to files."""
    # Production chains
    prod_chains = fitter.sampler.get_chain()
    prod_log_prob = fitter.sampler.get_log_prob()

    np.save(output_dir / "production_chains.npy", prod_chains)
    np.save(output_dir / "production_log_prob.npy", prod_log_prob)

    # Burn-in chains
    if fitter.burnin_chain is not None:
        np.save(output_dir / "burnin_chains.npy", fitter.burnin_chain)
        np.save(output_dir / "burnin_log_prob.npy", fitter.burnin_log_prob)

    print(f"  Saved chains to {output_dir}")


def main():
    global FIX_LD, PARAM_NAMES
    print("=" * 60, flush=True)
    print(f"Full Analysis for TOI {TOI}", flush=True)
    print(f"Sectors: {'All available' if SECTORS_TO_USE is None else SECTORS_TO_USE}", flush=True)
    print(f"CPUs: {N_CPUS} of {cpu_count()} available", flush=True)
    print("=" * 60, flush=True)

    # Load catalog
    print("\nLoading catalog parameters...", flush=True)
    catalog = load_toi_catalog()
    planet_params = get_toi_parameters(catalog, toi=TOI)

    if planet_params is None:
        raise RuntimeError(f"TOI {TOI} not found in catalog")

    # Apply CLI overrides for period and T0
    if PERIOD_OVERRIDE is not None:
        print(f"  Overriding period: {planet_params['period']:.7f} -> {PERIOD_OVERRIDE:.7f}", flush=True)
        planet_params['period'] = PERIOD_OVERRIDE
    if T0_OVERRIDE is not None:
        print(f"  Overriding T0: {planet_params['t0']:.6f} -> {T0_OVERRIDE:.6f}", flush=True)
        planet_params['t0'] = T0_OVERRIDE
    if PERIOD_ERR_OVERRIDE is not None:
        print(f"  Overriding period_err: {planet_params.get('period_err', 'N/A')} -> {PERIOD_ERR_OVERRIDE:.7f}", flush=True)
        planet_params['period_err'] = PERIOD_ERR_OVERRIDE
    if T0_ERR_OVERRIDE is not None:
        print(f"  Overriding t0_err: {planet_params.get('t0_err', 'N/A')} -> {T0_ERR_OVERRIDE:.7f}", flush=True)
        planet_params['t0_err'] = T0_ERR_OVERRIDE

    # Get TIC_ID from catalog if not specified
    tic_id = TIC_ID if TIC_ID is not None else planet_params.get('tic_id')
    if tic_id is None:
        raise RuntimeError(f"TIC ID not found for TOI {TOI}")

    period = planet_params['period']
    t0 = planet_params['t0']

    if period is None or period <= 0:
        print(f"\n  ERROR: TOI {TOI} has invalid period ({period}). Skipping.", flush=True)
        return

    print(f"  TIC ID: {tic_id}", flush=True)
    print(f"  Period: {period:.6f} days", flush=True)
    print(f"  T0: {t0:.6f}", flush=True)

    # Load data (try cache first, fall back to MAST download)
    loader = DataLoader(tic_id)
    success = False
    if USE_CACHE:
        success = loader.load_from_npz_cache()
        if success:
            print(f"\n  Loaded from cache ({len(loader.lightcurves)} sectors)", flush=True)
    if not success:
        print("\nDownloading from MAST...", flush=True)
        success = loader.download_from_mast()
        if success and USE_CACHE:
            loader.save_to_npz_cache()
            print(f"  Saved to cache", flush=True)

    if not success:
        print(f"\n  ERROR: Failed to load TESS data for TIC {tic_id} (TOI {TOI}). Skipping.", flush=True)
        return

    sector_data = {}
    for lc in loader.lightcurves:
        # Use all sectors if SECTORS_TO_USE is None, otherwise filter
        if SECTORS_TO_USE is None or lc.sector in SECTORS_TO_USE:
            sector_data[lc.sector] = {
                'time': lc.time,
                'flux': lc.flux,
                'flux_err': lc.flux_err,
                'cadence': lc.cadence,  # Cadence in seconds
                'source': lc.source     # SPOC or QLP
            }
            cad_name = config.CADENCE_NAMES.get(int(lc.cadence), f"{lc.cadence}s")
            print(f"  Sector {lc.sector}: {len(lc.time)} points ({lc.source}, {cad_name})", flush=True)

    # Skip sectors that don't overlap with any transit window
    # Include sectors where any part of the transit window (±1.5× duration) falls within the sector
    transit_hw = max(duration_hr / 24.0 * 1.5, 0.5) if 'duration_hr' in dir() else 0.5  # days
    # Estimate duration from catalog if not yet computed
    dur_est = planet_params.get('duration_hr', 3.0) / 24.0
    transit_hw = max(dur_est * 1.5, 0.5)
    sectors_with_transit = set()
    for sec, data in sector_data.items():
        t_min, t_max = data['time'].min(), data['time'].max()
        n_min = int(np.floor((t_min - transit_hw - t0) / period))
        n_max = int(np.ceil((t_max + transit_hw - t0) / period))
        for n in range(n_min, n_max + 1):
            t_transit = t0 + n * period
            if (t_transit + transit_hw) >= t_min and (t_transit - transit_hw) <= t_max:
                sectors_with_transit.add(sec)
                break

    n_all = len(sector_data)
    n_skip = n_all - len(sectors_with_transit)
    if n_skip > 0:
        for sec in list(sector_data.keys()):
            if sec not in sectors_with_transit:
                del sector_data[sec]
        print(f"\n  Skipped {n_skip} sectors with no transits", flush=True)

    # Get list of sectors being used
    sectors_used = sorted(sector_data.keys())
    print(f"\n  Using {len(sectors_used)} sectors: {sectors_used}", flush=True)

    # Normalize and combine
    print("\nNormalizing sectors...", flush=True)
    combined_time = []
    combined_flux = []
    combined_flux_err = []
    combined_cadence = []
    normalization_info = {}

    for sector in sectors_used:
        data = sector_data[sector]
        flux_norm, flux_err_norm, info = normalize_to_oot_no_eclipse(
            data['time'], data['flux'], data['flux_err'], period, t0
        )
        print(f"  Sector {sector}: OOT={info['n_oot']}, Transit={info['n_transit']}, Eclipse={info['n_eclipse']}")
        combined_time.append(data['time'])
        combined_flux.append(flux_norm)
        combined_flux_err.append(flux_err_norm)
        # Create cadence array (same value for all points in sector)
        combined_cadence.append(np.full(len(data['time']), data['cadence']))
        normalization_info[sector] = info

    time_combined = np.concatenate(combined_time)
    flux_combined = np.concatenate(combined_flux)
    flux_err_combined = np.concatenate(combined_flux_err)
    cadence_combined = np.concatenate(combined_cadence)

    sort_idx = np.argsort(time_combined)
    time_combined = time_combined[sort_idx]
    flux_combined = flux_combined[sort_idx]
    flux_err_combined = flux_err_combined[sort_idx]
    cadence_combined = cadence_combined[sort_idx]

    # Auto-detect and mask sibling TOIs in multi-planet systems
    NO_MASK_SIBLINGS = any(arg == '--no-mask' for arg in sys.argv)
    toi_number = TOI.split('.')[0]  # e.g., '125' from '125.01'
    sibling_rows = catalog[catalog['TOI'].apply(lambda x: str(x).split('.')[0] == toi_number)]
    sibling_tois = [str(row['TOI']) for _, row in sibling_rows.iterrows()
                    if str(row['TOI']) != TOI and not np.isnan(row.get('Period (days)', float('nan')))]
    if sibling_tois and not NO_MASK_SIBLINGS:
        print(f"\n  Multi-planet system: masking {len(sibling_tois)} sibling TOIs: {', '.join(sibling_tois)}", flush=True)
        sibling_mask = np.ones(len(time_combined), dtype=bool)
        for sib_toi in sibling_tois:
            sib_pp = get_toi_parameters(catalog, toi=sib_toi)
            if sib_pp is None:
                continue
            sib_period = sib_pp['period']
            sib_t0 = sib_pp['t0']
            sib_dur = sib_pp.get('duration_hr', 3.0) / 24.0
            sib_hw = 1.5 * sib_dur
            n_min = int(np.floor((time_combined.min() - sib_t0) / sib_period))
            n_max = int(np.ceil((time_combined.max() - sib_t0) / sib_period))
            n_masked = 0
            for n in range(n_min, n_max + 1):
                t_transit = sib_t0 + n * sib_period
                in_transit = np.abs(time_combined - t_transit) < sib_hw
                sibling_mask &= ~in_transit
                n_masked += np.sum(in_transit)
            print(f"    Masked TOI {sib_toi}: {n_masked} points (P={sib_period:.4f} d, dur={sib_dur*24:.1f} hr)", flush=True)
        n_before = len(time_combined)
        time_combined = time_combined[sibling_mask]
        flux_combined = flux_combined[sibling_mask]
        flux_err_combined = flux_err_combined[sibling_mask]
        cadence_combined = cadence_combined[sibling_mask]
        print(f"    After masking: {len(time_combined)} points (removed {n_before - len(time_combined)})", flush=True)

    # Report cadence distribution
    unique_cadences = np.unique(cadence_combined)
    print(f"\n  Cadence distribution:", flush=True)
    for cad in unique_cadences:
        n_points = np.sum(cadence_combined == cad)
        cad_name = config.CADENCE_NAMES.get(int(cad), f"{cad}s")
        print(f"    {cad_name}: {n_points} points ({100*n_points/len(cadence_combined):.1f}%)", flush=True)

    print(f"\nCombined: {len(time_combined)} points", flush=True)

    # Filter data to transit window (4× transit duration centered on mid-transit)
    # This includes 1.5× T14 before ingress and 1.5× T14 after egress
    print("\n  Filtering to transit window (4× transit duration)...", flush=True)

    # Estimate transit duration from catalog parameters
    planet_radius = planet_params.get('planet_radius')
    stellar_radius = planet_params.get('stellar_radius')
    stellar_mass = planet_params.get('stellar_mass')
    depth_ppm = planet_params.get('depth_ppm', 10000)

    # Estimate Rp/Rs
    if (planet_radius and not np.isnan(planet_radius) and planet_radius > 0 and
        stellar_radius and not np.isnan(stellar_radius) and stellar_radius > 0):
        rp_rs_est = (planet_radius * config.R_EARTH_TO_R_SUN) / stellar_radius
    else:
        rp_rs_est = np.sqrt(depth_ppm / 1e6)

    # Estimate a/Rs
    if (stellar_mass and not np.isnan(stellar_mass) and stellar_mass > 0 and
        stellar_radius and not np.isnan(stellar_radius) and stellar_radius > 0):
        a_rs_est = config.KEPLER_CONSTANT * (stellar_mass ** (1/3)) * (period ** (2/3)) / stellar_radius
    else:
        a_rs_est = config.DEFAULT_A_RS

    # Estimate transit duration
    if DURATION_OVERRIDE is not None:
        # Use command-line override (e.g., from BLS)
        t14_phase = (DURATION_OVERRIDE / 24.0) / period  # Convert hours to phase units
    else:
        # Compute from Rp/Rs and a/Rs (assuming b=0 for initial estimate)
        # T14 = (P/π) × arcsin((1+Rp/Rs) / a_Rs)
        sin_arg = min((1 + rp_rs_est) / a_rs_est, 1.0)  # Clamp to avoid arcsin domain error
        t14_phase = np.arcsin(sin_arg) / np.pi  # Transit duration in phase units

    # Transit window: 4× T14 = ±2× T14 from mid-transit
    transit_window_half_phase = 2.0 * t14_phase
    transit_window_half_phase = max(config.TRANSIT_WINDOW_PHASE_MIN, min(transit_window_half_phase, config.TRANSIT_WINDOW_PHASE_MAX))

    print(f"    Estimated Rp/Rs: {rp_rs_est:.4f}", flush=True)
    print(f"    Estimated a/Rs: {a_rs_est:.2f}", flush=True)
    print(f"    Estimated T14 (phase): {t14_phase:.4f} ({t14_phase * period * 24:.2f} hours)", flush=True)
    print(f"    Transit window: ±{transit_window_half_phase:.4f} phase (4× T14)", flush=True)

    # Phase fold the data
    phase_combined = ((time_combined - t0) / period) % 1.0
    phase_combined[phase_combined > 0.5] -= 1.0  # Shift to [-0.5, 0.5]

    # Filter to transit window
    in_transit_window = np.abs(phase_combined) <= transit_window_half_phase
    time_filtered = time_combined[in_transit_window]
    flux_filtered = flux_combined[in_transit_window]
    flux_err_filtered = flux_err_combined[in_transit_window]
    cadence_filtered = cadence_combined[in_transit_window]

    n_total = len(time_combined)
    n_filtered = len(time_filtered)
    print(f"    Points in transit window: {n_filtered} / {n_total} ({100*n_filtered/n_total:.1f}%)", flush=True)

    # Phase-folded MCMC fit
    print("\n" + "="*60, flush=True)
    print("PHASE-FOLDED MCMC FIT", flush=True)
    print("="*60, flush=True)

    fitter = FullAnalysisFitter(time_filtered, flux_filtered, flux_err_filtered, planet_params,
                                 cadence=cadence_filtered)
    results = fitter.fit(n_walkers=N_WALKERS, n_burn=N_BURN, check_convergence_flag=True, n_cpus=N_CPUS)

    params = results['parameters']
    diagnostics = results.get('diagnostics', {})

    # --- Auto-retry on poor convergence (R-hat > 1.05) ---
    rhat_values = diagnostics.get('rhat', [])
    if len(rhat_values) > 0:
        max_rhat = float(np.max(rhat_values))
        if max_rhat > 1.05:
            worst_idx = int(np.argmax(rhat_values))
            worst_param = PARAM_NAMES[worst_idx]
            print(f"\n  WARNING: Poor convergence (max R-hat={max_rhat:.4f} for {worst_param})", flush=True)

            shape_params = {'rp_rs', 'a_rs', 'b'}
            ephemeris_params = {'period', 't0'}

            if worst_param in shape_params and not FIX_LD:
                # Remedy 1: fix limb darkening
                print(f"  -> Re-running with fixed limb darkening...", flush=True)
                FIX_LD = True
                PARAM_NAMES = ['period', 't0', 'rp_rs', 'a_rs', 'b', 'baseline']
                fitter = FullAnalysisFitter(time_filtered, flux_filtered, flux_err_filtered,
                                            planet_params, cadence=cadence_filtered)
                results = fitter.fit(n_walkers=N_WALKERS, n_burn=N_BURN,
                                     check_convergence_flag=True, n_cpus=N_CPUS)
                params = results['parameters']
                diagnostics = results.get('diagnostics', {})

            elif worst_param in ephemeris_params:
                # Remedy 2: tighten priors (multiplier 2.0 -> 1.0)
                print(f"  -> Re-running with tighter ephemeris priors (multiplier=1.0)...", flush=True)
                original_mult = config.CATALOG_ERROR_PRIOR_MULTIPLIER
                config.CATALOG_ERROR_PRIOR_MULTIPLIER = 1.0
                fitter = FullAnalysisFitter(time_filtered, flux_filtered, flux_err_filtered,
                                            planet_params, cadence=cadence_filtered)
                results = fitter.fit(n_walkers=N_WALKERS, n_burn=N_BURN,
                                     check_convergence_flag=True, n_cpus=N_CPUS)
                params = results['parameters']
                diagnostics = results.get('diagnostics', {})
                config.CATALOG_ERROR_PRIOR_MULTIPLIER = original_mult  # restore

    print(f"\nResults:", flush=True)
    print(f"  Period: {params['period']['value']:.6f} +/- {params['period']['err']:.6f} days", flush=True)
    print(f"  T0: {params['t0']['value']:.6f} +/- {params['t0']['err']:.6f}", flush=True)
    print(f"  Rp/Rs: {params['rp_rs']['value']:.4f} +/- {params['rp_rs']['err']:.4f}", flush=True)
    print(f"  a/Rs: {params['a_rs']['value']:.2f} +/- {params['a_rs']['err']:.2f}", flush=True)
    print(f"  b: {params['b']['value']:.3f} +/- {params['b']['err']:.3f}", flush=True)
    print(f"  Baseline: {params['baseline']['value']:.6f} +/- {params['baseline']['err']:.6f}", flush=True)
    if FIX_LD:
        print(f"  u1: {params['u1']['value']:.4f} (FIXED)", flush=True)
        print(f"  u2: {params['u2']['value']:.4f} (FIXED)", flush=True)
    else:
        print(f"  u1: {params['u1']['value']:.4f} +/- {params['u1']['err']:.4f} (prior: {fitter.u1:.4f})", flush=True)
        print(f"  u2: {params['u2']['value']:.4f} +/- {params['u2']['err']:.4f} (prior: {fitter.u2:.4f})", flush=True)
    print(f"  Converged: {diagnostics.get('converged', 'Unknown')}", flush=True)

    # Individual transit fitting
    print("\n" + "="*60, flush=True)
    print("INDIVIDUAL TRANSIT FITTING", flush=True)
    print("="*60, flush=True)

    period_fit = params['period']['value']
    t0_fit = params['t0']['value']
    rp_rs_fit = params['rp_rs']['value']
    a_rs_fit = params['a_rs']['value']
    b_fit = params['b']['value']

    # Estimate transit duration (handle grazing transits where b >= 1)
    if DURATION_OVERRIDE is not None:
        duration_days = DURATION_OVERRIDE / 24.0
        if b_fit >= 1.0:
            print(f"  WARNING: Grazing transit detected (b={b_fit:.3f} >= 1)", flush=True)
        print(f"  Using override duration: {DURATION_OVERRIDE:.2f} hours", flush=True)
    elif b_fit >= 1.0:
        # Grazing transit - use full planet disk duration: sqrt((1+k)^2 - b^2)
        print(f"  WARNING: Grazing transit detected (b={b_fit:.3f} >= 1)", flush=True)
        if (1 + rp_rs_fit)**2 > b_fit**2:
            duration_days = period_fit / (np.pi * a_rs_fit) * np.sqrt((1 + rp_rs_fit)**2 - b_fit**2)
        else:
            duration_days = period_fit / (np.pi * a_rs_fit) * rp_rs_fit
    else:
        duration_days = period_fit / (np.pi * a_rs_fit) * np.sqrt((1 + rp_rs_fit)**2 - b_fit**2)
    window_size = max(0.2, 3 * duration_days)  # At least 3x duration

    # Identify transits, filtering out partial transits
    transits, partial_transits = identify_transits(
        time_combined, period_fit, t0_fit, duration_days,
        require_full_coverage=True, coverage_factor=1.5,
        cadence=cadence_combined
    )
    print(f"  Found {len(transits)} full transits, {len(partial_transits)} partial transits excluded", flush=True)

    # Show cadence-dependent minimum points summary
    if len(transits) > 0:
        cadences_used = [t.get('median_cadence_sec') for t in transits if t.get('median_cadence_sec')]
        if cadences_used:
            min_pts_used = [t.get('min_points_required', 20) for t in transits]
            print(f"  Cadence-dependent minimum points: {min(min_pts_used)}-{max(min_pts_used)} pts", flush=True)

    if len(partial_transits) > 0:
        print(f"  Partial transits excluded (insufficient coverage):", flush=True)
        for pt in partial_transits:
            early = "YES" if pt['has_early_data'] else "NO"
            late = "YES" if pt['has_late_data'] else "NO"
            pre = "YES" if pt.get('has_pre_ingress', True) else "NO"
            post = "YES" if pt.get('has_post_egress', True) else "NO"
            print(f"    Epoch {pt['epoch']:4d}: early={early}, late={late}, pre_ingress={pre}, post_egress={post}, {pt['n_points']} pts", flush=True)

    # Prepare data for parallel processing
    fitter_params = {
        'period': period_fit,
        'rp_rs': rp_rs_fit,
        'a_rs': a_rs_fit,
        'b': b_fit,
        'u1': fitter.u1,
        'u2': fitter.u2
    }

    # Prepare transit data for parallel processing (can't pass masks directly)
    transit_data = []
    for transit in transits:
        mask = transit['mask']
        transit_data.append({
            'epoch': transit['epoch'],
            't_expected': transit['t_expected'],
            'n_points': transit['n_points'],
            't_data': time_combined[mask],
            'f_data': flux_combined[mask],
            'f_err': flux_err_combined[mask],
            'cadence': cadence_combined[mask],  # Cadence for each point
            'mask': mask  # Keep for later use
        })

    # Fit individual transits in parallel — each transit MCMC runs on 1 CPU,
    # multiple transits run simultaneously using all available cores
    n_transit_workers = N_CPUS
    fit_args = [(td, fitter_params) for td in transit_data]

    if len(transits) > 1:
        print(f"  Fitting {len(transits)} transits in parallel ({n_transit_workers} workers, 1 CPU each)...", flush=True)

        with Pool(processes=n_transit_workers) as pool:
            fit_results = pool.map(_fit_single_transit, fit_args)
    else:
        print(f"  Fitting {len(transits)} transit...", flush=True)
        fit_results = [_fit_single_transit(a) for a in fit_args]

    # Add mask back to results and print
    all_transit_fits = []
    for i, fit_result in enumerate(fit_results):
        fit_result['mask'] = transit_data[i]['mask']
        all_transit_fits.append(fit_result)
        conv_str = "" if fit_result['converged'] else " [NOT CONVERGED]"
        print(f"  Epoch {fit_result['epoch']:4d}: T0 = {fit_result['t0_fit']:.6f} +/- {fit_result['t0_err']:.6f}, "
              f"R-hat = {fit_result['max_rhat']:.3f}{conv_str} ({fit_result['n_points']} pts)", flush=True)

    # Re-center extraction on fitted T0 and refit transits that shifted significantly
    coverage_half_width = 1.5 * duration_days
    refit_transit_data = []
    refit_indices = []
    for i, fit_result in enumerate(all_transit_fits):
        t0_fitted = fit_result['t0_fit']
        t_expected = transit_data[i]['t_expected']
        shift = abs(t0_fitted - t_expected)
        # Refit if T0 shifted by more than 20% of the coverage window
        if shift > 0.2 * coverage_half_width:
            # Re-extract data centered on fitted T0
            new_mask = np.abs(time_combined - t0_fitted) < coverage_half_width
            n_new = np.sum(new_mask)
            if n_new >= 10:
                refit_transit_data.append({
                    'epoch': fit_result['epoch'],
                    't_expected': t0_fitted,  # Use fitted T0 as new center
                    'n_points': n_new,
                    't_data': time_combined[new_mask],
                    'f_data': flux_combined[new_mask],
                    'f_err': flux_err_combined[new_mask],
                    'cadence': cadence_combined[new_mask],
                    'mask': new_mask
                })
                refit_indices.append(i)

    if refit_indices:
        print(f"\n  Re-centering {len(refit_indices)} transits on fitted T0 and refitting...", flush=True)
        refit_args = [(td, fitter_params) for td in refit_transit_data]
        if len(refit_args) > 1:
            with Pool(processes=n_transit_workers) as pool:
                refit_results = pool.map(_fit_single_transit, refit_args)
        else:
            refit_results = [_fit_single_transit(a) for a in refit_args]

        for j, refit_result in enumerate(refit_results):
            idx = refit_indices[j]
            refit_result['mask'] = refit_transit_data[j]['mask']
            old_t0 = all_transit_fits[idx]['t0_fit']
            all_transit_fits[idx] = refit_result
            conv_str = "" if refit_result['converged'] else " [NOT CONVERGED]"
            print(f"  Epoch {refit_result['epoch']:4d}: T0 = {refit_result['t0_fit']:.6f} +/- {refit_result['t0_err']:.6f}, "
                  f"R-hat = {refit_result['max_rhat']:.3f}{conv_str} ({refit_result['n_points']} pts) [re-centered]", flush=True)

    n_unconverged = sum(1 for f in all_transit_fits if not f['converged'])

    # Summary of convergence
    if n_unconverged > 0:
        print(f"\n  WARNING: {n_unconverged}/{len(transits)} transits did not converge (R-hat >= {config.CONVERGENCE_RHAT_INDIVIDUAL})", flush=True)
    else:
        print(f"\n  All {len(transits)} transits converged (R-hat < {config.CONVERGENCE_RHAT_INDIVIDUAL})", flush=True)

    # Handle case with no transits
    if len(all_transit_fits) == 0:
        print("\n  No transits to filter - skipping outlier filtering", flush=True)
        transit_fits = []
        excluded_by_t0err = []
        excluded_by_oc = []
        n_excluded_t0err = 0
        n_excluded_oc = 0
        median_t0_err = None
        t0_err_threshold = None
        oc_filter_stats = {'median_oc_minutes': None, 'mad_minutes': None, 'sigma_minutes': None, 'threshold_minutes': None}
    else:
        # Filter unconverged transits first
        print("\n  Filtering outlier transits...", flush=True)
        converged_fits = [tf for tf in all_transit_fits if tf.get('converged', True)]
        n_unconverged = len(all_transit_fits) - len(converged_fits)
        if n_unconverged > 0:
            print(f"    Removed {n_unconverged} unconverged transits", flush=True)

        # Filter outlier transits - Step 1: t0_err > 10x median error
        print("\n  Step 1: Filter by T0 error (> 10x median)...", flush=True)
        transit_fits_step1, excluded_by_t0err, median_t0_err, t0_err_threshold = \
            filter_outlier_transits(converged_fits, sigma_threshold=10.0)

        n_excluded_t0err = len(excluded_by_t0err)
        if n_excluded_t0err > 0:
            print(f"    Median T0 error: {median_t0_err*24*60:.2f} minutes", flush=True)
            print(f"    Threshold (10x median): {t0_err_threshold*24*60:.2f} minutes", flush=True)
            print(f"    Excluded {n_excluded_t0err} transits with large T0 errors:", flush=True)
            for tf in excluded_by_t0err:
                print(f"      Epoch {tf['epoch']}: T0 error = {tf['t0_err']*24*60:.2f} minutes", flush=True)
        else:
            print(f"    No transits excluded (median error: {median_t0_err*24*60:.2f} min)", flush=True)

        # Filter outlier transits - Step 2: O-C > 10 sigma from median O-C
        print("\n  Step 2: Filter by O-C outliers (> 10 sigma from median)...", flush=True)
        transit_fits, excluded_by_oc, oc_filter_stats = \
            filter_oc_outliers(transit_fits_step1, period_fit, t0_fit, sigma_threshold=10.0)

        n_excluded_oc = len(excluded_by_oc)
        if n_excluded_oc > 0:
            print(f"    Median O-C: {oc_filter_stats['median_oc_minutes']:.2f} minutes", flush=True)
            print(f"    Sigma (1.48*MAD): {oc_filter_stats['sigma_minutes']:.2f} minutes", flush=True)
            print(f"    Threshold (10 sigma): {oc_filter_stats['threshold_minutes']:.2f} minutes", flush=True)
            print(f"    Excluded {n_excluded_oc} transits with O-C outliers:", flush=True)
            for tf in excluded_by_oc:
                print(f"      Epoch {tf['epoch']}: O-C = {tf['oc_minutes']:.2f} min "
                      f"({tf['oc_deviation_sigma']:.1f} sigma from median)", flush=True)
        else:
            print(f"    No transits excluded (sigma: {oc_filter_stats['sigma_minutes']:.2f} min)", flush=True)

    # Combine excluded transits list
    excluded_fits = excluded_by_t0err + excluded_by_oc

    print(f"\n  Summary: {len(all_transit_fits)} full transits ({len(partial_transits)} partial excluded) -> "
          f"{len(transit_fits)} used ({n_excluded_t0err} by T0 err, {n_excluded_oc} by O-C)", flush=True)

    # Generate plots (parallelized — independent plots run concurrently)
    print("\n" + "="*60, flush=True)
    print("GENERATING PLOTS (parallel)", flush=True)
    print("="*60, flush=True)

    # Calculate binned residual statistics (needed for results, not a plot)
    binned_residual_stats = calculate_binned_residual_statistics(fitter, bin_minutes=config.BIN_WIDTH_MINUTES)

    # O-C diagram must run first (returns data needed for ephemeris)
    print("  O-C diagram...", flush=True)
    oc_result = plot_oc_diagram(transit_fits, period_fit, t0_fit, OUTPUT_DIR / f"oc_diagram.{config.PLOT_FORMAT}")
    if oc_result is not None:
        oc, oc_err, epochs = oc_result
    else:
        oc, oc_err, epochs = np.array([]), np.array([]), np.array([])

    # Run remaining independent plots in parallel threads
    def _plot_chains():
        plot_chains(fitter, OUTPUT_DIR / f"chain_plot.{config.PLOT_FORMAT}")
        return "chain_plot"

    def _plot_corner():
        plot_corner_plot(fitter.samples, OUTPUT_DIR / f"corner_plot.{config.PLOT_FORMAT}")
        return "corner_plot"

    def _plot_phase_folded():
        plot_phase_folded(fitter, OUTPUT_DIR / f"phase_folded_lightcurve.{config.PLOT_FORMAT}",
                          full_time=time_combined, full_flux=flux_combined,
                          full_flux_err=flux_err_combined, full_cadence=cadence_combined)
        return "phase_folded"

    def _plot_full_phase():
        plot_full_phase_folded(fitter, OUTPUT_DIR / f"full_phase_curve.{config.PLOT_FORMAT}",
                               full_time=time_combined, full_flux=flux_combined,
                               full_flux_err=flux_err_combined, full_cadence=cadence_combined)
        return "full_phase_curve"

    def _plot_individual():
        plot_individual_transits(time_combined, flux_combined, flux_err_combined,
                                 transit_fits, excluded_fits, fitter, OUTPUT_DIR / f"individual_transits.{config.PLOT_FORMAT}")
        return "individual_transits"

    print("  Generating 5 plots in parallel...", flush=True)
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [
            executor.submit(_plot_chains),
            executor.submit(_plot_corner),
            executor.submit(_plot_phase_folded),
            executor.submit(_plot_full_phase),
            executor.submit(_plot_individual),
        ]
        for future in futures:
            try:
                name = future.result()
                print(f"    Done: {name}", flush=True)
            except Exception as e:
                print(f"    Plot failed: {e}", flush=True)

    # Ephemeris fitting
    print("\n" + "="*60, flush=True)
    print("EPHEMERIS FITTING", flush=True)
    print("="*60, flush=True)

    # Get transit times for ephemeris fitting
    epochs_fit = np.array([tf['epoch'] for tf in transit_fits])
    t_obs_fit = np.array([tf['t0_fit'] for tf in transit_fits])
    t_err_fit = np.array([tf['t0_err'] for tf in transit_fits])

    # Initialize ephemeris variables with defaults
    linear_eph = {'T0': None, 'P': None, 'T0_err': None, 'P_err': None, 'chi2': None, 'bic': None, 'residuals': None}
    quadratic_eph = {'T0': None, 'P': None, 'Q': None, 'T0_err': None, 'P_err': None, 'Q_err': None,
                     'chi2': None, 'bic': None, 'residuals': None, 'dPdE': None, 'dPdE_err': None}
    Q_quad = None
    Q_quad_err = None
    preferred_model = None
    delta_bic = None
    periodogram_result = None

    if len(transit_fits) < 2:
        print(f"\n  Insufficient transits for ephemeris fitting ({len(transit_fits)} < 2)", flush=True)
        print(f"  Skipping ephemeris, O-C ephemeris plots, and periodogram", flush=True)
    else:
        # Use EphemerisAnalyzer from modular pipeline
        print("\n  Using EphemerisAnalyzer for ephemeris fitting...", flush=True)
        eph_analyzer = EphemerisAnalyzer(epochs_fit, t_obs_fit, t_err_fit)
        eph_result = eph_analyzer.analyze()

        # Print linear ephemeris results
        print("\n  Linear ephemeris: T(E) = T0 + P × E", flush=True)
        print(f"    T0 = {eph_result.t0_linear:.6f} ± {eph_result.t0_linear_err:.6f} BJD", flush=True)
        print(f"    P  = {eph_result.period_linear:.8f} ± {eph_result.period_linear_err:.8f} days", flush=True)
        print(f"    χ² = {eph_result.chi2_linear:.2f} (dof = {len(epochs_fit) - 2})", flush=True)
        print(f"    BIC = {eph_result.bic_linear:.2f}", flush=True)

        # Print quadratic ephemeris results (if available)
        print("\n  Quadratic ephemeris: T(E) = T0 + P × E + Q × E²", flush=True)
        if eph_result.t0_quadratic is not None:
            Q_quad = 0.5 * eph_result.dP_dE  # Q = 0.5 * dP/dE
            Q_quad_err = 0.5 * eph_result.dP_dE_err
            print(f"    T0 = {eph_result.t0_quadratic:.6f} ± {eph_result.t0_quadratic_err:.6f} BJD", flush=True)
            print(f"    P  = {eph_result.period_quadratic:.8f} ± {eph_result.period_quadratic_err:.8f} days", flush=True)
            print(f"    Q  = {Q_quad:.2e} ± {Q_quad_err:.2e} days/epoch²", flush=True)
            print(f"    dP/dE = {eph_result.dP_dE:.2e} ± {eph_result.dP_dE_err:.2e} days/epoch", flush=True)
            print(f"    χ² = {eph_result.chi2_quadratic:.2f} (dof = {len(epochs_fit) - 3})", flush=True)
            print(f"    BIC = {eph_result.bic_quadratic:.2f}", flush=True)
        else:
            print(f"    Insufficient transits for quadratic fit (need >= 3)", flush=True)

        # Model comparison
        print("\n  Model comparison:", flush=True)
        if eph_result.t0_quadratic is not None:
            print(f"    ΔBIC (linear - quadratic) = {eph_result.delta_bic:.2f}", flush=True)
            if eph_result.preferred_model == "quadratic":
                print(f"    >> Quadratic ephemeris preferred (ΔBIC > 0)", flush=True)
            else:
                print(f"    >> Linear ephemeris preferred (ΔBIC < 0)", flush=True)
        else:
            print(f"    >> Linear ephemeris used by default", flush=True)

        # Compute residuals for plotting (EphemerisResult stores RMS, not arrays)
        resid_lin = t_obs_fit - linear_ephemeris(epochs_fit, eph_result.t0_linear, eph_result.period_linear)
        if eph_result.t0_quadratic is not None:
            resid_quad = t_obs_fit - quadratic_ephemeris(
                epochs_fit, eph_result.t0_quadratic, eph_result.period_quadratic, Q_quad)
        else:
            resid_quad = None

        # Store ephemeris results in dictionaries for plot function compatibility
        linear_eph = {
            'T0': eph_result.t0_linear, 'P': eph_result.period_linear,
            'T0_err': eph_result.t0_linear_err, 'P_err': eph_result.period_linear_err,
            'chi2': eph_result.chi2_linear, 'bic': eph_result.bic_linear,
            'residuals': resid_lin
        }

        quadratic_eph = {
            'T0': eph_result.t0_quadratic, 'P': eph_result.period_quadratic, 'Q': Q_quad,
            'T0_err': eph_result.t0_quadratic_err, 'P_err': eph_result.period_quadratic_err, 'Q_err': Q_quad_err,
            'chi2': eph_result.chi2_quadratic, 'bic': eph_result.bic_quadratic,
            'residuals': resid_quad,
            'dPdE': eph_result.dP_dE, 'dPdE_err': eph_result.dP_dE_err
        }
        preferred_model = eph_result.preferred_model
        delta_bic = eph_result.delta_bic if eph_result.t0_quadratic is not None else None

        # Recompute O-C against the Stage-3 linear ephemeris (T0_lin, P_lin)
        # so the live periodogram sees exactly the same residuals that get
        # stored in results.json under individual_transits.transit_times[*].oc
        # and oc_values[*].oc_minutes, and that recompute_periodogram_fap.py
        # operates on. Without this step, the periodogram uses the Stage-1
        # phase-fold MCMC ephemeris (period_fit, t0_fit), which differs from
        # the stored Stage-3 linear ephemeris by sub-minute amounts and would
        # produce a slightly different peak than any later recompute.
        if linear_eph.get('T0') is not None and linear_eph.get('P') is not None:
            _t_obs_periodo = np.array([tf['t0_fit'] for tf in transit_fits])
            _ep_periodo    = np.array([tf['epoch']  for tf in transit_fits])
            oc = (_t_obs_periodo - (linear_eph['T0'] + _ep_periodo * linear_eph['P'])) * 24 * 60
            epochs = _ep_periodo
            # oc_err is unchanged — it's just t0_err scaled to minutes

        # Plot O-C with ephemeris and periodogram in parallel
        print("\n  Generating ephemeris plots + periodogram in parallel...", flush=True)
        _periodogram_result = [None]

        def _plot_oc_eph():
            plot_oc_with_ephemeris(transit_fits, linear_eph, quadratic_eph, OUTPUT_DIR)
            return "oc_ephemeris"

        def _plot_periodo():
            # Use linear-ephemeris period for the time grid so a stale Stage-1
            # period_fit doesn't change the LS frequencies. linear_eph['P'] is
            # the same period that anchors the O-C used here.
            P_periodo = linear_eph.get('P', period_fit)
            _periodogram_result[0] = plot_periodogram(oc, epochs, P_periodo, OUTPUT_DIR / f"periodogram.{config.PLOT_FORMAT}", oc_err=oc_err)
            return "periodogram"

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(_plot_oc_eph), executor.submit(_plot_periodo)]
            for future in futures:
                try:
                    name = future.result()
                    print(f"    Done: {name}", flush=True)
                except Exception as e:
                    print(f"    Plot failed: {e}", flush=True)

        periodogram_result = _periodogram_result[0]

    # Save chains
    print("\n  Saving chains...", flush=True)
    save_chains(fitter, OUTPUT_DIR)

    # Prepare results JSON
    rhat_values = diagnostics.get('rhat', [])
    autocorr_values = diagnostics.get('autocorr_time', [])
    ess_values = diagnostics.get('ess', [])
    chain_size = diagnostics.get('n_steps', 0)
    max_rhat = float(np.max(rhat_values)) if len(rhat_values) > 0 else None
    max_autocorr = float(np.max(autocorr_values)) if len(autocorr_values) > 0 else None
    min_ess = float(np.min(ess_values)) if len(ess_values) > 0 else None

    # Parameter names for diagnostics
    if FIX_LD:
        param_names = ['period', 't0', 'rp_rs', 'a_rs', 'b', 'baseline']
    else:
        param_names = ['period', 't0', 'rp_rs', 'a_rs', 'b', 'baseline', 'u1', 'u2']

    # Build per-parameter diagnostics
    per_param_diagnostics = {}
    for i, name in enumerate(param_names):
        per_param_diagnostics[name] = {
            'rhat': round(float(rhat_values[i]), 4) if i < len(rhat_values) else None,
            'autocorr_time': round(float(autocorr_values[i]), 1) if i < len(autocorr_values) else None,
            'ess': round(float(ess_values[i]), 0) if i < len(ess_values) else None
        }

    # Determine why convergence failed (if it did)
    convergence_message = None
    if not diagnostics.get('converged', True):
        reasons = []
        # Check R-hat
        if max_rhat is not None and max_rhat > config.CONVERGENCE_RHAT:
            failing_params = [f"{param_names[i]} ({rhat_values[i]:.4f})"
                           for i in range(len(rhat_values))
                           if rhat_values[i] > config.CONVERGENCE_RHAT]
            reasons.append(f"R-hat > {config.CONVERGENCE_RHAT} for: {', '.join(failing_params)}")
        # Check ESS
        if min_ess is not None and min_ess < config.CONVERGENCE_ESS:
            failing_params = [f"{param_names[i]} ({ess_values[i]:.0f})"
                           for i in range(len(ess_values))
                           if ess_values[i] < config.CONVERGENCE_ESS]
            reasons.append(f"ESS < {config.CONVERGENCE_ESS} for: {', '.join(failing_params)}")
        # Check autocorrelation
        if max_autocorr is not None and chain_size < config.CONVERGENCE_AUTOCORR * max_autocorr:
            reasons.append(f"chain length ({chain_size}) < {config.CONVERGENCE_AUTOCORR} × max_autocorr ({max_autocorr:.0f})")

        if reasons:
            convergence_message = "; ".join(reasons)
        else:
            # Check diagnostic messages from convergence checker
            messages = diagnostics.get('messages', [])
            if messages:
                convergence_message = "; ".join([m for m in messages if 'converged' not in m.lower()])

    # Defensive: ensure no unconverged transits make it into transit_times[].
    # The Stage-2 → Stage-3 filter at line ~2856 should already drop them,
    # but historically some TOIs ended up with unconverged entries in
    # transit_times[] (~16% of the catalog). This second-pass filter
    # guarantees the JSON output is clean.
    leaked_unconverged = sum(1 for tf in transit_fits if not tf.get('converged', True))
    if leaked_unconverged > 0:
        print(f"  WARNING: {leaked_unconverged} unconverged transit(s) leaked past "
              f"the Stage-2 filter; dropping them now", flush=True)
        transit_fits = [tf for tf in transit_fits if tf.get('converged', True)]
        # Also recompute oc/oc_err/epochs for the now-cleaner set so the
        # periodogram and stored stats see only converged data
        if linear_eph.get('T0') is not None and linear_eph.get('P') is not None:
            _t = np.array([tf['t0_fit'] for tf in transit_fits])
            _e = np.array([tf['epoch']  for tf in transit_fits])
            _err = np.array([tf['t0_err'] for tf in transit_fits])
            oc = (_t - (linear_eph['T0'] + _e * linear_eph['P'])) * 24 * 60
            oc_err = _err * 24 * 60
            epochs = _e

    # Transit times for JSON (including baseline, slope, and convergence diagnostics)
    transit_times_json = []
    n_unconverged_used = 0
    for tf in transit_fits:
        if not tf.get('converged', True):
            n_unconverged_used += 1
        # Calculate O-C in minutes: (observed - calculated) * 24 * 60
        oc_minutes = (tf['t0_fit'] - tf['t_expected']) * 24 * 60
        transit_times_json.append({
            'epoch': int(tf['epoch']),
            't_expected': float(tf['t_expected']),
            't0_fit': float(tf['t0_fit']),
            't0_err': float(tf['t0_err']),
            'oc': float(oc_minutes),  # O-C in minutes
            'baseline_fit': float(tf['baseline_fit']),
            'baseline_err': float(tf['baseline_err']),
            'slope_fit': float(tf['slope_fit']),
            'slope_err': float(tf['slope_err']),
            'n_points': int(tf['n_points']),
            'mcmc_diagnostics': {
                'converged': tf.get('converged', None),
                'n_steps': tf.get('n_steps', None),
                'max_rhat': tf.get('max_rhat', None),
                'rhat': tf.get('rhat', None),
                'autocorr_time': tf.get('autocorr_time', None),
                'ess': tf.get('ess', None)
            }
        })

    # Excluded transits for JSON - by T0 error
    excluded_by_t0err_json = []
    for tf in excluded_by_t0err:
        excluded_by_t0err_json.append({
            'epoch': int(tf['epoch']),
            't0_fit': float(tf['t0_fit']),
            't0_err': float(tf['t0_err']),
            't0_err_minutes': float(tf['t0_err'] * 24 * 60),
            'baseline_fit': float(tf['baseline_fit']),
            'slope_fit': float(tf['slope_fit']),
            'n_points': int(tf['n_points']),
            'reason': 't0_err > 10x median',
            'converged': tf.get('converged', None),
            'max_rhat': tf.get('max_rhat', None)
        })

    # Excluded transits for JSON - by O-C outlier
    excluded_by_oc_json = []
    for tf in excluded_by_oc:
        excluded_by_oc_json.append({
            'epoch': int(tf['epoch']),
            't0_fit': float(tf['t0_fit']),
            't0_err': float(tf['t0_err']),
            'oc_minutes': float(tf['oc_minutes']),
            'oc_deviation_sigma': float(tf['oc_deviation_sigma']),
            'baseline_fit': float(tf['baseline_fit']),
            'slope_fit': float(tf['slope_fit']),
            'n_points': int(tf['n_points']),
            'reason': 'O-C > 10 sigma from median',
            'converged': tf.get('converged', None),
            'max_rhat': tf.get('max_rhat', None)
        })

    # Recompute O-C from Step 3 linear ephemeris (not Step 1 MCMC ephemeris)
    if linear_eph.get('T0') is not None and linear_eph.get('P') is not None:
        t_obs_arr = np.array([tf['t0_fit'] for tf in transit_fits])
        epochs_arr = np.array([tf['epoch'] for tf in transit_fits])
        t_err_arr = np.array([tf['t0_err'] for tf in transit_fits])
        oc = (t_obs_arr - (linear_eph['T0'] + epochs_arr * linear_eph['P'])) * 24 * 60
        oc_err = t_err_arr * 24 * 60

    # O-C values
    oc_json = []
    for i, tf in enumerate(transit_fits):
        oc_json.append({
            'epoch': int(tf['epoch']),
            'oc_minutes': float(oc[i]),
            'oc_err_minutes': float(oc_err[i])
        })

    results_json = {
        'toi': TOI,
        'tic_id': tic_id,
        'sectors': sectors_used,
        'n_points_total': len(time_combined),
        'normalization': normalization_info,
        'parameters': params,
        'convergence': {
            'converged': diagnostics.get('converged', None),
            'message': convergence_message,
            'criteria': {
                'rhat_threshold': config.CONVERGENCE_RHAT,
                'ess_threshold': config.CONVERGENCE_ESS,
                'autocorr_threshold': config.CONVERGENCE_AUTOCORR
            },
            'summary': {
                'max_rhat': round(max_rhat, 4) if max_rhat is not None else None,
                'min_ess': round(min_ess, 0) if min_ess is not None else None,
                'max_autocorr_time': round(max_autocorr, 1) if max_autocorr is not None else None,
                'chain_size': chain_size,
                'chain_over_autocorr': round(chain_size / max_autocorr, 1) if max_autocorr and max_autocorr > 0 else None,
                'acceptance_rate': round(float(np.mean(fitter.sampler.acceptance_fraction)), 3)
            },
            'per_parameter': per_param_diagnostics
        },
        'bad_walkers': {
            'n_bad_walkers': fitter.n_bad_walkers,
            'bad_walker_indices': fitter.bad_walker_indices
        },
        'mcmc_settings': {
            'n_walkers': N_WALKERS,
            'n_burn': N_BURN,
            'n_steps_max': N_STEPS_MAX
        },
        'binned_residuals': binned_residual_stats,
        'individual_transits': {
            'n_transits_full': len(all_transit_fits),
            'n_transits_partial_excluded': len(partial_transits),
            'n_transits_used': len(transit_fits),
            'n_excluded_by_t0err': len(excluded_by_t0err),
            'n_excluded_by_oc': len(excluded_by_oc),
            'n_unconverged': n_unconverged_used,
            'all_converged': n_unconverged_used == 0,
            'partial_transits_excluded': [
                {
                    'epoch': int(pt['epoch']),
                    't_expected': float(pt['t_expected']),
                    'n_points': int(pt['n_points']),
                    'has_early_data': pt['has_early_data'],
                    'has_late_data': pt['has_late_data']
                } for pt in partial_transits
            ],
            'filtering': {
                't0_err_filter': {
                    'method': 't0_err > 10x median',
                    'median_t0_err_minutes': float(median_t0_err * 24 * 60) if median_t0_err else None,
                    'threshold_minutes': float(t0_err_threshold * 24 * 60) if t0_err_threshold else None
                },
                'oc_filter': {
                    'method': 'O-C > 10 sigma from median (sigma = 1.48*MAD)',
                    'median_oc_minutes': oc_filter_stats.get('median_oc_minutes'),
                    'mad_minutes': oc_filter_stats.get('mad_minutes'),
                    'sigma_minutes': oc_filter_stats.get('sigma_minutes'),
                    'threshold_minutes': oc_filter_stats.get('threshold_minutes')
                }
            },
            'transit_times': transit_times_json,
            'excluded_by_t0err': excluded_by_t0err_json,
            'excluded_by_oc': excluded_by_oc_json,
            'oc_values': oc_json,
            # Weighted RMS about the weighted mean (inverse-variance weights
            # w_i = 1/oc_err_i²). This is the canonical scatter used by C3 in
            # find_ttv_candidates.py — small per-transit errors dominate the
            # scatter measure, large-error transits are downweighted.
            'oc_rms_minutes': (
                float(np.sqrt(
                    np.sum((1.0 / oc_err**2) * (oc - np.sum((1.0/oc_err**2) * oc) / np.sum(1.0/oc_err**2))**2)
                    / np.sum(1.0 / oc_err**2)
                )) if len(oc) > 0 and len(oc_err) == len(oc) and np.all(oc_err > 0) else None
            ),
            'oc_weighted_mean_minutes': (
                float(np.sum((1.0 / oc_err**2) * oc) / np.sum(1.0 / oc_err**2))
                if len(oc) > 0 and len(oc_err) == len(oc) and np.all(oc_err > 0) else None
            ),
            'oc_mean_err_minutes': float(np.mean(oc_err)) if len(oc_err) > 0 else None,
            'oc_median_err_minutes': float(np.median(oc_err)) if len(oc_err) > 0 else None,
            # Legacy kept for backward compatibility with any consumer that
            # still reads it. New canonical ratio is computed by
            # find_ttv_candidates.py on the fly: weighted_rms / median(oc_err).
            'oc_rms_over_mean_err': (
                float(np.sqrt(np.sum((1.0/oc_err**2)*(oc - np.sum((1.0/oc_err**2)*oc)/np.sum(1.0/oc_err**2))**2)
                              / np.sum(1.0/oc_err**2)) / np.mean(oc_err))
                if len(oc) > 0 and len(oc_err) == len(oc) and np.all(oc_err > 0)
                   and np.mean(oc_err) > 0 else None
            ),
        },
        'periodogram': periodogram_result,
        'ephemeris': {
            'preferred_model': preferred_model,
            'delta_bic': float(delta_bic) if delta_bic is not None else None,
            'linear': {
                'T0': float(linear_eph['T0']) if linear_eph.get('T0') else None,
                'T0_err': float(linear_eph['T0_err']) if linear_eph.get('T0_err') else None,
                'P': float(linear_eph['P']) if linear_eph.get('P') else None,
                'P_err': float(linear_eph['P_err']) if linear_eph.get('P_err') else None,
                'chi2': float(linear_eph['chi2']) if linear_eph.get('chi2') else None,
                'bic': float(linear_eph['bic']) if linear_eph.get('bic') else None,
                'dof': len(epochs_fit) - 2 if len(epochs_fit) >= 2 else None
            },
            'quadratic': {
                'T0': float(quadratic_eph['T0']) if quadratic_eph.get('T0') else None,
                'T0_err': float(quadratic_eph['T0_err']) if quadratic_eph.get('T0_err') else None,
                'P': float(quadratic_eph['P']) if quadratic_eph.get('P') else None,
                'P_err': float(quadratic_eph['P_err']) if quadratic_eph.get('P_err') else None,
                'Q': float(quadratic_eph['Q']) if quadratic_eph.get('Q') else None,
                'Q_err': float(quadratic_eph['Q_err']) if quadratic_eph.get('Q_err') else None,
                'dPdE': float(quadratic_eph['dPdE']) if quadratic_eph.get('dPdE') else None,
                'dPdE_err': float(quadratic_eph['dPdE_err']) if quadratic_eph.get('dPdE_err') else None,
                'chi2': float(quadratic_eph['chi2']) if quadratic_eph.get('chi2') else None,
                'bic': float(quadratic_eph['bic']) if quadratic_eph.get('bic') else None,
                'dof': len(epochs_fit) - 3 if len(epochs_fit) >= 3 else None
            }
        },
        'catalog': {
            'period': planet_params['period'],
            'depth_ppm': planet_params['depth_ppm'],
            'planet_radius': planet_params.get('planet_radius'),
            'stellar_radius': planet_params.get('stellar_radius'),
            'stellar_mass': planet_params.get('stellar_mass')
        }
    }

    # Add empty vetting report to results (eclipse detection removed)
    results_json['vetting_report'] = {}

    with open(OUTPUT_DIR / "results.json", 'w') as f:
        json.dump(results_json, f, indent=2, default=float)
    print(f"\n  Saved: results.json", flush=True)

    print(f"\n{'='*60}", flush=True)
    print("ANALYSIS COMPLETE", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"Results saved to: {OUTPUT_DIR}", flush=True)
    print(f"\nFiles generated:", flush=True)
    ext = config.PLOT_FORMAT
    print(f"  - chain_plot.{ext}", flush=True)
    print(f"  - corner_plot.{ext}", flush=True)
    print(f"  - phase_folded_lightcurve.{ext}", flush=True)
    print(f"  - full_phase_curve.{ext}", flush=True)
    # List individual transit plot files (may be split into parts)
    individual_transit_files = sorted(OUTPUT_DIR.glob(f"individual_transits*.{ext}"))
    for f in individual_transit_files:
        print(f"  - {f.name}", flush=True)
    print(f"  - oc_diagram.{ext}", flush=True)
    print(f"  - oc_linear_ephemeris.{ext}", flush=True)
    print(f"  - oc_quadratic_ephemeris.{ext}", flush=True)
    print(f"  - periodogram.{ext}", flush=True)
    print(f"  - production_chains.npy", flush=True)
    print(f"  - production_log_prob.npy", flush=True)
    print(f"  - burnin_chains.npy", flush=True)
    print(f"  - burnin_log_prob.npy", flush=True)
    print(f"  - results.json", flush=True)


if __name__ == "__main__":
    main()
