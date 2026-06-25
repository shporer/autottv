"""
Individual Transit Fitter (Step 2) for AutoTTV Pipeline v2.0

Fits individual transit events using Mandel & Agol (2002) model with MCMC.

Fitted Parameters (3):
    - Mid-transit time (T_mid)
    - Normalized flux level (baseline)
    - Linear slope (trend)

Fixed Parameters (from Step 1):
    - Period
    - Rp/Rs
    - a/Rs
    - Impact parameter
    - Limb darkening coefficients
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

import emcee

try:
    import batman
    BATMAN_AVAILABLE = True
except ImportError:
    BATMAN_AVAILABLE = False

from . import config
from .convergence import compute_rhat_split, compute_autocorr_time_simple
from .utils import compute_batman_model, setup_batman_params, create_cached_transit_model

logger = logging.getLogger(__name__)


@dataclass
class TransitResult:
    """Container for individual transit fitting results."""
    epoch: int
    t_mid: float
    t_mid_err_lower: float
    t_mid_err_upper: float
    baseline: float
    baseline_err: float
    slope: float
    slope_err: float
    n_points: int
    success: bool
    message: str = ""

    @property
    def t_mid_err(self) -> float:
        """Average error on mid-transit time."""
        return (self.t_mid_err_lower + self.t_mid_err_upper) / 2

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            'epoch': self.epoch,
            't_mid': self.t_mid,
            't_mid_err': self.t_mid_err,
            't_mid_err_lower': self.t_mid_err_lower,
            't_mid_err_upper': self.t_mid_err_upper,
            'baseline': self.baseline,
            'baseline_err': self.baseline_err,
            'slope': self.slope,
            'slope_err': self.slope_err,
            'n_points': self.n_points,
            'success': self.success,
            'message': self.message
        }
        # Add convergence diagnostics if available
        if hasattr(self, 'converged'):
            result['converged'] = self.converged
        if hasattr(self, 'n_steps'):
            result['n_steps'] = self.n_steps
        if hasattr(self, 'max_rhat'):
            result['max_rhat'] = self.max_rhat
        if hasattr(self, 'rhat'):
            result['rhat'] = self.rhat
        if hasattr(self, 'autocorr_time'):
            result['autocorr_time'] = self.autocorr_time
        if hasattr(self, 'ess'):
            result['ess'] = self.ess
        if hasattr(self, 'chi2'):
            result['chi2'] = self.chi2
        return result


class IndividualTransitFitter:
    """
    MCMC-based fitter for individual transit events.

    Uses fixed shape parameters from Step 1 phase-folded fit.
    Includes convergence checking with R-hat based adaptive stepping.
    """

    # Fitted parameter names
    PARAM_NAMES = ['t_mid', 'baseline', 'slope']
    N_PARAMS = 3

    def __init__(self, step1_results: Dict[str, Any], cadence: float = None):
        """
        Initialize with results from Step 1 phase-folded fitting.

        Parameters
        ----------
        step1_results : dict
            Results dictionary from PhaseFoldFitter.fit()
        cadence : float, optional
            Cadence in seconds for exposure time integration. If None, assumes
            short cadence (120s).
        """
        if not BATMAN_AVAILABLE:
            raise ImportError("batman-package required for Mandel-Agol fitting")

        # Extract fixed parameters from Step 1
        params = step1_results['parameters']

        self.period = params['period']['value']
        self.t0_ref = params['t0']['value']
        self.rp_rs = params['rp_rs']['value']
        self.a_rs = params['a_rs']['value']
        self.b = params['b']['value']

        # Limb darkening - check both 'limb_darkening' dict and 'parameters'
        if 'limb_darkening' in step1_results:
            ld = step1_results['limb_darkening']
            self.u1 = ld['u1']
            self.u2 = ld['u2']
        else:
            self.u1 = params['u1']['value']
            self.u2 = params['u2']['value']

        # Store cadence for exposure time integration
        self.cadence = cadence
        self.long_cadence_threshold = config.LONG_CADENCE_THRESHOLD

        # Transit duration for windowing
        duration_hr = step1_results['derived'].get('duration_hr', config.DEFAULT_TRANSIT_DURATION_HR)
        self.duration_days = duration_hr / 24.0

        # Compute inclination from impact parameter
        if self.a_rs > 0:
            cos_i = np.clip(self.b / self.a_rs, 0, 1)
            self.inc_deg = np.degrees(np.arccos(cos_i))
        else:
            self.inc_deg = config.DEFAULT_OMEGA

        # Create batman parameters template
        self._init_batman_params()

        logger.info(f"IndividualTransitFitter initialized:")
        logger.info(f"  Period: {self.period:.6f} days")
        logger.info(f"  Rp/Rs: {self.rp_rs:.4f}")
        logger.info(f"  a/Rs: {self.a_rs:.2f}")
        logger.info(f"  b: {self.b:.3f}")
        logger.info(f"  Duration: {self.duration_days*24:.2f} hours")

    def _init_batman_params(self):
        """Initialize batman TransitParams object using shared utility."""
        # Use t0=0 as placeholder; actual t0 is set per-transit in _compute_model
        self.batman_params = setup_batman_params(
            period=self.period,
            t0=0.0,  # Placeholder, updated per transit
            rp_rs=self.rp_rs,
            a_rs=self.a_rs,
            b=self.b,
            u1=self.u1,
            u2=self.u2,
            ecc=config.DEFAULT_ECCENTRICITY,
            omega=config.DEFAULT_OMEGA
        )

    def identify_transits(self, time: np.ndarray, t0: float = None,
                         period: float = None) -> List[Tuple[int, float]]:
        """
        Identify individual transit events in the time series.

        Parameters
        ----------
        time : np.ndarray
            Full time array
        t0 : float, optional
            Reference mid-transit time (uses Step 1 value if not provided)
        period : float, optional
            Orbital period (uses Step 1 value if not provided)

        Returns
        -------
        list of (epoch, predicted_t_mid)
            List of transit epochs and predicted mid-transit times
        """
        if t0 is None:
            t0 = self.t0_ref
        if period is None:
            period = self.period

        t_min, t_max = time.min(), time.max()

        # Find epoch range
        epoch_min = int(np.floor((t_min - t0) / period)) - 1
        epoch_max = int(np.ceil((t_max - t0) / period)) + 1

        transits = []
        for epoch in range(epoch_min, epoch_max + 1):
            t_mid_predicted = t0 + epoch * period

            # Check if transit is within data range
            if t_min <= t_mid_predicted <= t_max:
                transits.append((epoch, t_mid_predicted))

        logger.info(f"Identified {len(transits)} potential transit events")
        return transits

    def extract_transit_data(self, time: np.ndarray, flux: np.ndarray,
                            flux_err: np.ndarray, t_mid_predicted: float,
                            window_mult: float = config.TRANSIT_WINDOW_MULTIPLIER
                            ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Extract data window around a single transit.

        Parameters
        ----------
        time, flux, flux_err : np.ndarray
            Full light curve data
        t_mid_predicted : float
            Predicted mid-transit time
        window_mult : float
            Window extends window_mult * duration on each side

        Returns
        -------
        time_window, flux_window, flux_err_window : tuple or None
            Data within window, or None if insufficient data
        """
        half_window = window_mult * self.duration_days

        # Select data within window
        mask = (time >= t_mid_predicted - half_window) & \
               (time <= t_mid_predicted + half_window)

        n_points = np.sum(mask)

        if n_points < config.MIN_POINTS_TRANSIT:
            logger.debug(f"Insufficient points for transit at {t_mid_predicted:.4f}: {n_points}")
            return None

        # Check for complete coverage
        time_window = time[mask]
        t_min_window = time_window.min()
        t_max_window = time_window.max()

        # Require data on both sides of transit
        if t_min_window > t_mid_predicted - config.TRANSIT_COVERAGE_FRACTION * self.duration_days:
            logger.debug(f"Missing pre-transit data at {t_mid_predicted:.4f}")
            return None
        if t_max_window < t_mid_predicted + config.TRANSIT_COVERAGE_FRACTION * self.duration_days:
            logger.debug(f"Missing post-transit data at {t_mid_predicted:.4f}")
            return None

        return time_window, flux[mask], flux_err[mask]

    def transit_model_with_trend(self, time: np.ndarray, t_mid: float,
                                 baseline: float, slope: float,
                                 cadence: float = None,
                                 cached_model=None) -> np.ndarray:
        """
        Compute transit model with linear trend and optional exposure integration.

        Parameters
        ----------
        time : np.ndarray
            Time array
        t_mid : float
            Mid-transit time
        baseline : float
            Baseline flux level
        slope : float
            Linear slope (flux per day)
        cadence : float, optional
            Cadence in seconds. If None, uses self.cadence.
        cached_model : batman.TransitModel or list, optional
            Pre-created TransitModel for reuse during MCMC.

        Returns
        -------
        np.ndarray
            Model flux
        """
        # Update batman params
        self.batman_params.t0 = t_mid

        # Fast path: use cached model (bypass compute_batman_model wrapper)
        if cached_model is not None:
            if hasattr(cached_model, 'is_mixed'):
                if cached_model.is_mixed:
                    buf = cached_model.buffer
                    buf[:] = 0.0
                    for mask, m in cached_model.model:
                        buf[mask] = m.light_curve(self.batman_params)
                    transit_flux = buf
                else:
                    transit_flux = cached_model.model.light_curve(self.batman_params)
            elif isinstance(cached_model, list):
                transit_flux = np.zeros(len(time))
                for mask, m in cached_model:
                    transit_flux[mask] = m.light_curve(self.batman_params)
            else:
                transit_flux = cached_model.light_curve(self.batman_params)
        else:
            # Determine cadence
            if cadence is None:
                cadence = self.cadence

            # Compute transit model using shared utility function
            transit_flux = compute_batman_model(
                time, self.batman_params, cadence=cadence,
                long_cadence_threshold=self.long_cadence_threshold
            )

        # Add baseline and linear trend
        trend = baseline + slope * (time - t_mid)
        model = transit_flux * trend

        return model

    def fit_single_transit(self, time: np.ndarray, flux: np.ndarray,
                          flux_err: np.ndarray, t_mid_predicted: float,
                          epoch: int,
                          n_walkers: int = config.N_WALKERS_INDIVIDUAL,
                          n_burn: int = config.N_BURN_INDIVIDUAL,
                          n_steps_min: int = config.N_STEPS_MIN_INDIVIDUAL,
                          n_steps_max: int = config.N_STEPS_MAX_INDIVIDUAL,
                          rhat_threshold: float = config.CONVERGENCE_RHAT_INDIVIDUAL,
                          check_interval: int = config.CONVERGENCE_CHECK_INTERVAL_INDIVIDUAL,
                          cadence: float = None) -> TransitResult:
        """
        Fit a single transit event with convergence checking.

        Parameters
        ----------
        time, flux, flux_err : np.ndarray
            Transit data window
        t_mid_predicted : float
            Predicted mid-transit time (initial guess)
        epoch : int
            Transit epoch number
        n_walkers : int
            Number of MCMC walkers (default 16)
        n_burn : int
            Number of burn-in steps (default 200)
        n_steps_min : int
            Minimum number of production steps (default 500)
        n_steps_max : int
            Maximum number of production steps (default 5000)
        rhat_threshold : float
            R-hat threshold for convergence (default 1.02)
        check_interval : int
            Steps between convergence checks (default 100)
        cadence : float, optional
            Cadence in seconds for this transit window

        Returns
        -------
        TransitResult
            Fitting results for this transit
        """
        n_points = len(time)

        # Use provided cadence or fall back to stored cadence
        transit_cadence = cadence if cadence is not None else self.cadence

        # Create cached TransitModel for this transit's time array
        _cached = create_cached_transit_model(
            time, self.batman_params, cadence=transit_cadence,
            long_cadence_threshold=self.long_cadence_threshold
        )

        # Set up priors - allow t_mid to vary within configured window (±1% of period)
        t_mid_window = self.period * config.T_MID_WINDOW_FRACTION

        def log_prior(theta):
            t_mid, baseline, slope = theta

            if not (t_mid_predicted - t_mid_window < t_mid < t_mid_predicted + t_mid_window):
                return -np.inf
            if not (config.BASELINE_MIN_INDIVIDUAL < baseline < config.BASELINE_MAX_INDIVIDUAL):
                return -np.inf
            if not (-config.SLOPE_MAX < slope < config.SLOPE_MAX):
                return -np.inf

            return 0.0

        inv_var = 1.0 / flux_err**2

        def log_likelihood(theta):
            try:
                model = self.transit_model_with_trend(time, *theta, cached_model=_cached)
                residuals = flux - model
                return -0.5 * np.dot(residuals, residuals * inv_var)
            except Exception:
                return -np.inf

        def log_probability(theta):
            lp = log_prior(theta)
            if not np.isfinite(lp):
                return -np.inf
            return lp + log_likelihood(theta)

        # Initialize walkers - scatter from config
        p0 = np.zeros((n_walkers, self.N_PARAMS))
        for i in range(n_walkers):
            p0[i, 0] = t_mid_predicted + np.random.uniform(-config.WALKER_INIT_T_MID_DAYS, config.WALKER_INIT_T_MID_DAYS)
            p0[i, 1] = 1.0 + np.random.uniform(-config.WALKER_INIT_BASELINE_INDIVIDUAL, config.WALKER_INIT_BASELINE_INDIVIDUAL)
            p0[i, 2] = np.random.uniform(-config.WALKER_INIT_SLOPE, config.WALKER_INIT_SLOPE)

        # Run MCMC with convergence checking
        try:
            sampler = emcee.EnsembleSampler(n_walkers, self.N_PARAMS, log_probability)

            # Burn-in
            state = sampler.run_mcmc(p0, n_burn, progress=False)
            sampler.reset()

            # Production with convergence checking
            converged = False
            n_steps_total = 0

            while n_steps_total < n_steps_max:
                # Run for check_interval steps
                steps_to_run = min(check_interval, n_steps_max - n_steps_total)
                state = sampler.run_mcmc(state, steps_to_run, progress=False)
                n_steps_total += steps_to_run

                # Check convergence after minimum steps
                if n_steps_total >= n_steps_min:
                    chains = sampler.get_chain()
                    rhat = compute_rhat_split(chains)
                    max_rhat = np.max(rhat)

                    if max_rhat < rhat_threshold:
                        converged = True
                        break

            # Get final chains and compute diagnostics
            chains = sampler.get_chain()
            samples = sampler.get_chain(flat=True)

            # Compute final R-hat (using shared function from convergence module)
            rhat = compute_rhat_split(chains)
            max_rhat = float(np.max(rhat))

            # Compute autocorrelation time (using shared function from convergence module)
            try:
                tau = compute_autocorr_time_simple(samples)
            except Exception:
                tau = np.array([np.nan, np.nan, np.nan])

            # Compute effective sample size
            n_samples = len(samples)
            ess = n_samples / np.maximum(tau, 1)  # Avoid division by zero

            # Extract results
            percentiles = np.percentile(samples, list(config.MCMC_PERCENTILES), axis=0)

            t_mid_median = percentiles[1, 0]
            t_mid_err_lower = t_mid_median - percentiles[0, 0]
            t_mid_err_upper = percentiles[2, 0] - t_mid_median

            baseline_median = percentiles[1, 1]
            baseline_err = (percentiles[2, 1] - percentiles[0, 1]) / 2

            slope_median = percentiles[1, 2]
            slope_err = (percentiles[2, 2] - percentiles[0, 2]) / 2

            # Calculate chi2 at best-fit
            model = self.transit_model_with_trend(time, t_mid_median, baseline_median,
                                                   slope_median, cached_model=_cached)
            residuals = flux - model
            chi2 = float(np.dot(residuals, residuals * inv_var))

            result = TransitResult(
                epoch=epoch,
                t_mid=float(t_mid_median),
                t_mid_err_lower=float(t_mid_err_lower),
                t_mid_err_upper=float(t_mid_err_upper),
                baseline=float(baseline_median),
                baseline_err=float(baseline_err),
                slope=float(slope_median),
                slope_err=float(slope_err),
                n_points=n_points,
                success=True
            )

            # Add convergence diagnostics as attributes
            result.converged = converged
            result.n_steps = n_steps_total
            result.max_rhat = max_rhat
            result.rhat = {self.PARAM_NAMES[i]: float(rhat[i]) for i in range(self.N_PARAMS)}
            result.autocorr_time = {self.PARAM_NAMES[i]: float(tau[i]) for i in range(self.N_PARAMS)}
            result.ess = {self.PARAM_NAMES[i]: float(ess[i]) for i in range(self.N_PARAMS)}
            result.chi2 = chi2

            return result

        except Exception as e:
            logger.warning(f"Transit fit failed for epoch {epoch}: {e}")
            result = TransitResult(
                epoch=epoch,
                t_mid=t_mid_predicted,
                t_mid_err_lower=0.0,
                t_mid_err_upper=0.0,
                baseline=1.0,
                baseline_err=0.0,
                slope=0.0,
                slope_err=0.0,
                n_points=n_points,
                success=False,
                message=str(e)
            )
            result.converged = False
            result.n_steps = 0
            result.max_rhat = np.inf
            result.rhat = {}
            result.autocorr_time = {}
            result.ess = {}
            result.chi2 = np.inf
            return result

    def fit_all_transits(self, time: np.ndarray, flux: np.ndarray,
                        flux_err: np.ndarray, **kwargs) -> List[TransitResult]:
        """
        Fit all individual transit events.

        Parameters
        ----------
        time, flux, flux_err : np.ndarray
            Full light curve data
        **kwargs
            Additional arguments passed to fit_single_transit()

        Returns
        -------
        list of TransitResult
            Results for all successfully fitted transits
        """
        # Identify transits
        transits = self.identify_transits(time)

        if not transits:
            logger.warning("No transits identified in data")
            return []

        results = []

        for epoch, t_mid_predicted in transits:
            # Extract data window
            data = self.extract_transit_data(time, flux, flux_err, t_mid_predicted)

            if data is None:
                logger.debug(f"Skipping transit epoch {epoch}: insufficient data")
                continue

            time_window, flux_window, flux_err_window = data

            # Fit transit
            result = self.fit_single_transit(
                time_window, flux_window, flux_err_window,
                t_mid_predicted, epoch, **kwargs
            )

            if result.success:
                results.append(result)
                logger.debug(f"Epoch {epoch}: T_mid = {result.t_mid:.6f} +/- {result.t_mid_err:.6f}")
            else:
                logger.debug(f"Epoch {epoch}: fit failed - {result.message}")

        logger.info(f"Successfully fitted {len(results)} of {len(transits)} transits")
        return results


def fit_individual_transits(time: np.ndarray, flux: np.ndarray, flux_err: np.ndarray,
                           step1_results: Dict[str, Any], cadence: float = None,
                           **kwargs) -> List[Dict[str, Any]]:
    """
    Convenience function to fit all individual transits.

    Parameters
    ----------
    time, flux, flux_err : np.ndarray
        Full light curve data
    step1_results : dict
        Results from Step 1 phase-folded fitting
    cadence : float, optional
        Cadence in seconds for exposure time integration
    **kwargs
        Additional arguments for fitting

    Returns
    -------
    list of dict
        Results for each transit as dictionaries
    """
    fitter = IndividualTransitFitter(step1_results, cadence=cadence)
    results = fitter.fit_all_transits(time, flux, flux_err, **kwargs)

    return [r.to_dict() for r in results]


if __name__ == "__main__":
    # Test with synthetic data
    logging.basicConfig(level=logging.INFO)

    print("Testing IndividualTransitFitter with synthetic data...")

    # Create fake Step 1 results
    step1_results = {
        'parameters': {
            'period': {'value': 3.5},
            't0': {'value': 2458500.0},
            'rp_rs': {'value': 0.1},
            'a_rs': {'value': 10.0},
            'b': {'value': 0.3},
            'baseline': {'value': 1.0}
        },
        'limb_darkening': {
            'u1': 0.37,
            'u2': 0.25,
            'law': 'quadratic'
        },
        'derived': {
            'duration_hr': 3.0
        }
    }

    # Generate synthetic data
    np.random.seed(42)
    period = 3.5
    t0 = 2458500.0

    # Create time array spanning multiple transits
    times = []
    for i in range(5):
        t_mid = t0 + i * period
        t_transit = np.linspace(t_mid - 0.2, t_mid + 0.2, 150)
        times.extend(t_transit)

    time = np.array(times)

    # Generate flux using fitter model
    fitter = IndividualTransitFitter(step1_results)
    flux = fitter.transit_model_with_trend(time, t0, 1.0, 0.0)
    flux += np.random.normal(0, 0.001, len(flux))
    flux_err = np.ones_like(flux) * 0.001

    # Fit transits
    results = fitter.fit_all_transits(time, flux, flux_err)

    print(f"\nFitted {len(results)} transits:")
    for r in results:
        oc = (r.t_mid - (t0 + r.epoch * period)) * 24 * 60  # O-C in minutes
        print(f"  Epoch {r.epoch}: T_mid = {r.t_mid:.6f}, O-C = {oc:.2f} min, err = {r.t_mid_err*24*60:.2f} min")
