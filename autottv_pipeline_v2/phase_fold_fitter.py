"""
Phase-Folded Light Curve Fitter (Step 1) for AutoTTV Pipeline v2.0

Fits the phase-folded (stacked) light curve using Mandel & Agol (2002) model
with MCMC sampling via emcee.

Fitted Parameters:
    - Period (P)
    - Mid-transit time (T0)
    - Planet-to-star radius ratio (Rp/Rs)
    - Scaled semi-major axis (a/Rs)
    - Impact parameter (b)
    - Flux normalization (baseline)

Fixed Parameters:
    - Eccentricity: e = 0
    - Argument of periastron: omega = 90 degrees
    - Limb darkening coefficients: u1, u2 (from Claret 2017 PHOENIX r-method)
"""

import numpy as np
import logging
from typing import Dict, Tuple, Optional, Any
from pathlib import Path

import emcee
import corner

try:
    import batman
    BATMAN_AVAILABLE = True
except ImportError:
    BATMAN_AVAILABLE = False
    logging.warning("batman-package not installed!")

from . import config
from .limb_darkening import get_limb_darkening
from .convergence import check_convergence, run_until_converged, compute_rhat_split
from .utils import setup_batman_params, compute_batman_model, create_cached_transit_model, update_batman_params

logger = logging.getLogger(__name__)


class PhaseFoldFitter:
    """
    MCMC-based fitter for phase-folded transit light curves.

    Uses the Mandel & Agol (2002) transit model via batman.
    """

    # Parameter names and indices
    PARAM_NAMES_BASE = ['period', 't0', 'rp_rs', 'a_rs', 'b', 'baseline']
    PARAM_NAMES_LD = ['period', 't0', 'rp_rs', 'a_rs', 'b', 'baseline', 'u1', 'u2']

    def __init__(self, time: np.ndarray, flux: np.ndarray, flux_err: np.ndarray,
                 planet_params: Dict[str, Any], fix_ld: bool = True,
                 ld_prior_width_u1: float = config.LD_PRIOR_WIDTH_U1,
                 ld_prior_width_u2: float = config.LD_PRIOR_WIDTH_U2,
                 cadence: np.ndarray = None):
        """
        Initialize the fitter.

        Parameters
        ----------
        time : np.ndarray
            Time array in BJD
        flux : np.ndarray
            Normalized flux array
        flux_err : np.ndarray
            Flux error array
        planet_params : dict
            Dictionary containing planet parameters from TOI catalog:
            - period: Orbital period in days
            - t0: Initial mid-transit time in BJD
            - depth_ppm: Transit depth in ppm
            - duration_hr: Transit duration in hours
            - stellar_teff: Stellar effective temperature in K
            - stellar_logg: Stellar log(g) in cgs
        fix_ld : bool
            If True, fix limb darkening to theoretical values (default True).
            If False, fit u1 and u2 as free parameters with Gaussian priors.
        ld_prior_width_u1 : float
            Width of Gaussian prior on u1 (default 0.15)
        ld_prior_width_u2 : float
            Width of Gaussian prior on u2 (default 0.10)
        cadence : np.ndarray, optional
            Cadence in seconds for each data point. If provided, enables
            exposure time integration for long-cadence data (>200s).
        """
        if not BATMAN_AVAILABLE:
            raise ImportError("batman-package required for Mandel-Agol fitting")

        self.time = np.asarray(time)
        self.flux = np.asarray(flux)
        self.flux_err = np.asarray(flux_err)
        self.inv_var = 1.0 / np.asarray(flux_err)**2
        self.planet_params = planet_params

        # Limb darkening fitting options
        self.fix_ld = fix_ld
        self.ld_prior_width_u1 = ld_prior_width_u1
        self.ld_prior_width_u2 = ld_prior_width_u2

        # Set parameter names based on LD fitting mode
        if fix_ld:
            self.PARAM_NAMES = self.PARAM_NAMES_BASE
            self.N_PARAMS = 6
        else:
            self.PARAM_NAMES = self.PARAM_NAMES_LD
            self.N_PARAMS = 8

        # Store cadence information for exposure time integration
        if cadence is None:
            self.cadence = np.full(len(time), config.DEFAULT_CADENCE_SPOC)
        else:
            self.cadence = np.asarray(cadence)

        # Identify unique cadences and create masks for each
        self.unique_cadences = np.unique(self.cadence)
        self.cadence_masks = {cad: self.cadence == cad for cad in self.unique_cadences}

        # Long cadence threshold (seconds) - integrate for cadences > threshold
        self.long_cadence_threshold = config.LONG_CADENCE_THRESHOLD

        # Extract priors from planet params
        self.period_prior = planet_params.get('period', 1.0)
        self.t0_prior = planet_params.get('t0', self.time.mean())
        self.depth_prior = planet_params.get('depth_ppm', 10000) / 1e6
        self.duration_prior = planet_params.get('duration_hr', config.DEFAULT_TRANSIT_DURATION_HR) / 24.0

        # Derive Rp/Rs from planet and stellar radii if available
        planet_radius = planet_params.get('planet_radius')  # in R_Earth
        stellar_radius = planet_params.get('stellar_radius')  # in R_Sun

        # Check for valid values (not None, not NaN, and stellar_radius > 0)
        planet_radius_valid = (planet_radius is not None and
                               not (isinstance(planet_radius, float) and np.isnan(planet_radius)) and
                               planet_radius > 0)
        stellar_radius_valid = (stellar_radius is not None and
                                not (isinstance(stellar_radius, float) and np.isnan(stellar_radius)) and
                                stellar_radius > 0)

        if planet_radius_valid and stellar_radius_valid:
            # Rp/Rs = (Rp in R_Earth × R_Earth/R_Sun) / Rs in R_Sun
            self.rp_rs_prior = (planet_radius * config.R_EARTH_TO_R_SUN) / stellar_radius
            logger.info(f"  Rp/Rs derived from catalog radii: Rp={planet_radius:.2f} R_Earth, Rs={stellar_radius:.2f} R_Sun -> Rp/Rs={self.rp_rs_prior:.4f}")
        else:
            # Fallback to deriving from depth
            self.rp_rs_prior = np.sqrt(self.depth_prior)
            logger.info(f"  Rp/Rs derived from depth: {self.depth_prior*1e6:.0f} ppm -> Rp/Rs={self.rp_rs_prior:.4f}")

        # Derive a/Rs from Kepler's third law: a/Rs = 4.208 × (M_star)^(1/3) × (P_days)^(2/3) / Rs
        # This comes from: a³ = G × M_star × P² / (4π²), then dividing by Rs
        stellar_mass = planet_params.get('stellar_mass')  # in M_Sun

        # Check for valid stellar mass
        stellar_mass_valid = (stellar_mass is not None and
                             not (isinstance(stellar_mass, float) and np.isnan(stellar_mass)) and
                             stellar_mass > 0)

        # If stellar mass not in TOI catalog, try to get it from TIC
        if not stellar_mass_valid:
            tic_id = planet_params.get('tic_id')
            if tic_id is not None:
                stellar_mass = self._get_stellar_mass_from_tic(tic_id)
                stellar_mass_valid = stellar_mass is not None and stellar_mass > 0

        if stellar_mass_valid and stellar_radius_valid and self.period_prior > 0:
            # Kepler's third law: a/Rs = constant × (M_star/M_sun)^(1/3) × (P_days)^(2/3) / (Rs/R_sun)
            self.a_rs_prior = config.KEPLER_CONSTANT * (stellar_mass ** (1/3)) * (self.period_prior ** (2/3)) / stellar_radius
            logger.info(f"  a/Rs derived from Kepler's law: M_star={stellar_mass:.2f} M_Sun, Rs={stellar_radius:.2f} R_Sun, P={self.period_prior:.4f} days -> a/Rs={self.a_rs_prior:.2f}")
        elif self.duration_prior > 0 and self.period_prior > 0:
            # Fallback to duration-based estimate
            self.a_rs_prior = self.period_prior / (np.pi * self.duration_prior)
            logger.info(f"  a/Rs derived from duration: P={self.period_prior:.4f} days, dur={self.duration_prior*24:.2f} hr -> a/Rs={self.a_rs_prior:.2f}")
        else:
            self.a_rs_prior = config.DEFAULT_A_RS
            logger.info(f"  a/Rs set to default: {config.DEFAULT_A_RS}")

        # Clamp a/Rs to physical range
        self.a_rs_prior = np.clip(self.a_rs_prior, config.A_RS_CLIP_MIN, config.A_RS_CLIP_MAX)

        # Initial impact parameter (assume central transit)
        self.b_prior = config.DEFAULT_IMPACT_PARAMETER

        # Get limb darkening coefficients
        teff = planet_params.get('stellar_teff', config.SOLAR_TEFF)
        logg = planet_params.get('stellar_logg', config.SOLAR_LOGG)
        self.u1, self.u2 = get_limb_darkening(teff, logg)

        # Set T0 to be near the center of the time span
        self._adjust_t0_to_center()

        # Precompute prior sigmas and hard bounds for log_prior
        period_err = planet_params.get('period_err')
        if period_err and not np.isnan(period_err) and period_err > 0:
            self._period_sigma = config.CATALOG_ERROR_PRIOR_MULTIPLIER * period_err
        else:
            self._period_sigma = self.period_prior * config.PERIOD_PRIOR_WIDTH
        t0_err = planet_params.get('t0_err')
        if t0_err and not np.isnan(t0_err) and t0_err > 0:
            self._t0_sigma = config.CATALOG_ERROR_PRIOR_MULTIPLIER * t0_err
        else:
            self._t0_sigma = self.period_prior * config.T0_PRIOR_WIDTH
        self._rp_rs_sigma = self.rp_rs_prior * config.RP_RS_PRIOR_WIDTH
        self._a_rs_sigma = self.a_rs_prior * config.A_RS_PRIOR_WIDTH
        self._period_lo = self.period_prior * (1 - config.PERIOD_BOUND_FRACTION)
        self._period_hi = self.period_prior * (1 + config.PERIOD_BOUND_FRACTION)
        self._t0_lo = self.t0_prior - self.period_prior * config.T0_BOUND_FRACTION
        self._t0_hi = self.t0_prior + self.period_prior * config.T0_BOUND_FRACTION

        # Results storage
        self.sampler = None
        self.samples = None
        self.results = None
        self._cached_transit_model = None

        # Burn-in chain storage (for plotting)
        self.burnin_chain = None
        self.burnin_log_prob = None
        self.n_burnin_steps = 0

        logger.info(f"PhaseFoldFitter initialized:")
        logger.info(f"  Period prior: {self.period_prior:.6f} days")
        logger.info(f"  T0 prior: {self.t0_prior:.6f} BJD")
        logger.info(f"  Rp/Rs prior: {self.rp_rs_prior:.4f}")
        logger.info(f"  a/Rs prior: {self.a_rs_prior:.2f}")
        logger.info(f"  Limb darkening: u1={self.u1:.4f}, u2={self.u2:.4f}")

    def _get_stellar_mass_from_tic(self, tic_id: int) -> Optional[float]:
        """
        Query the TIC catalog for stellar mass.

        Parameters
        ----------
        tic_id : int
            TIC ID to query

        Returns
        -------
        float or None
            Stellar mass in solar masses, or None if not found
        """
        try:
            from astroquery.mast import Catalogs
            result = Catalogs.query_object(f"TIC {tic_id}", catalog="TIC")
            if len(result) > 0:
                mass = result[0].get('mass')
                if mass is not None and not np.isnan(mass):
                    logger.info(f"  Stellar mass from TIC: {mass:.2f} M_Sun")
                    return float(mass)
        except Exception as e:
            logger.warning(f"  Could not query TIC for stellar mass: {e}")
        return None

    def _adjust_t0_to_center(self):
        """Adjust T0 to be near the center of the observation time span."""
        if len(self.time) == 0:
            return

        t_center = (self.time.min() + self.time.max()) / 2.0

        # Find the epoch closest to t_center
        n_epochs = int((t_center - self.t0_prior) / self.period_prior)
        self.t0_prior = self.t0_prior + n_epochs * self.period_prior

        logger.debug(f"Adjusted T0 to {self.t0_prior:.6f} (center of data span)")

    def transit_model(self, time: np.ndarray, period: float, t0: float,
                     rp_rs: float, a_rs: float, b: float, baseline: float,
                     u1: float = None, u2: float = None,
                     cadence: np.ndarray = None) -> np.ndarray:
        """
        Compute Mandel-Agol transit model with optional exposure time integration.

        Parameters
        ----------
        time : np.ndarray
            Time array
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
        baseline : float
            Out-of-transit flux level
        u1 : float, optional
            Quadratic limb darkening coefficient u1 (uses stored value if None)
        u2 : float, optional
            Quadratic limb darkening coefficient u2 (uses stored value if None)
        cadence : np.ndarray, optional
            Cadence in seconds for each time point. If None, uses self.cadence.

        Returns
        -------
        np.ndarray
            Model flux
        """
        # Use stored limb darkening if not provided
        if u1 is None:
            u1 = self.u1
        if u2 is None:
            u2 = self.u2

        # Use cached model for MCMC (same time array), fresh model for plotting etc.
        if time is self.time and self._cached_transit_model is not None:
            # Update cached batman params in-place
            if not hasattr(self, '_cached_batman_params'):
                self._cached_batman_params = setup_batman_params(
                    period=period, t0=t0, rp_rs=rp_rs, a_rs=a_rs, b=b,
                    u1=u1, u2=u2,
                    ecc=config.DEFAULT_ECCENTRICITY, omega=config.DEFAULT_OMEGA
                )
            else:
                update_batman_params(self._cached_batman_params, period, t0, rp_rs, a_rs, b, u1, u2)

            cached = self._cached_transit_model
            if hasattr(cached, 'is_mixed'):
                if cached.is_mixed:
                    buf = cached.buffer
                    buf[:] = 0.0
                    for mask, m in cached.model:
                        buf[mask] = m.light_curve(self._cached_batman_params)
                    transit_flux = buf
                else:
                    transit_flux = cached.model.light_curve(self._cached_batman_params)
            elif isinstance(cached, list):
                transit_flux = np.zeros(len(time))
                for mask, m in cached:
                    transit_flux[mask] = m.light_curve(self._cached_batman_params)
            else:
                transit_flux = cached.light_curve(self._cached_batman_params)
            return transit_flux * baseline

        # Set up batman parameters using shared utility
        params = setup_batman_params(
            period=period, t0=t0, rp_rs=rp_rs, a_rs=a_rs, b=b,
            u1=u1, u2=u2,
            ecc=config.DEFAULT_ECCENTRICITY, omega=config.DEFAULT_OMEGA
        )

        # Determine cadence for this call
        if cadence is None:
            if len(time) == len(self.cadence):
                cadence = self.cadence
            else:
                # External call with different time array - assume short cadence
                cadence = np.full(len(time), config.DEFAULT_CADENCE_SPOC)

        # Compute transit model using shared utility (handles exposure time integration)
        transit_flux = compute_batman_model(
            time, params, cadence=cadence,
            long_cadence_threshold=self.long_cadence_threshold
        )

        return transit_flux * baseline

    def log_prior(self, theta: np.ndarray) -> float:
        """
        Log prior probability.

        Parameters
        ----------
        theta : np.ndarray
            Parameter vector [period, t0, rp_rs, a_rs, b, baseline] if fix_ld=True,
            or [period, t0, rp_rs, a_rs, b, baseline, u1, u2] if fix_ld=False

        Returns
        -------
        float
            Log prior probability
        """
        if self.fix_ld:
            period, t0, rp_rs, a_rs, b, baseline = theta
        else:
            period, t0, rp_rs, a_rs, b, baseline, u1, u2 = theta

        # Hard bounds using precomputed limits
        if not (self._period_lo < period < self._period_hi):
            return -np.inf
        if not (self._t0_lo < t0 < self._t0_hi):
            return -np.inf
        if not (config.RP_RS_MIN < rp_rs < config.RP_RS_MAX):
            return -np.inf
        if not (config.A_RS_MIN < a_rs < config.A_RS_MAX):
            return -np.inf
        if not (config.B_MIN <= b < config.B_MAX + rp_rs):  # Allow grazing transits
            return -np.inf
        if not (config.BASELINE_MIN < baseline < config.BASELINE_MAX):
            return -np.inf

        # Limb darkening bounds (only if not fixed)
        if not self.fix_ld:
            if not (config.LD_U1_MIN < u1 < config.LD_U1_MAX):
                return -np.inf
            if not (config.LD_U2_MIN < u2 < config.LD_U2_MAX):
                return -np.inf
            if not (u1 + u2 < config.LD_SUM_MAX):
                return -np.inf
            if not (u1 + 2.0 * u2 >= config.LD_U1_2U2_MIN):
                return -np.inf

        # Gaussian priors using precomputed sigmas
        log_p = 0.0
        log_p += -0.5 * ((period - self.period_prior) / self._period_sigma)**2
        log_p += -0.5 * ((t0 - self.t0_prior) / self._t0_sigma)**2
        log_p += -0.5 * ((rp_rs - self.rp_rs_prior) / self._rp_rs_sigma)**2
        log_p += -0.5 * ((a_rs - self.a_rs_prior) / self._a_rs_sigma)**2

        if not self.fix_ld:
            log_p += -0.5 * ((u1 - self.u1) / self.ld_prior_width_u1)**2
            log_p += -0.5 * ((u2 - self.u2) / self.ld_prior_width_u2)**2

        return log_p

    def log_likelihood(self, theta: np.ndarray) -> float:
        """
        Log likelihood (Gaussian errors).

        Parameters
        ----------
        theta : np.ndarray
            Parameter vector

        Returns
        -------
        float
            Log likelihood
        """
        try:
            if self.fix_ld:
                period, t0, rp_rs, a_rs, b, baseline = theta
                u1, u2 = self.u1, self.u2
            else:
                period, t0, rp_rs, a_rs, b, baseline, u1, u2 = theta

            model = self.transit_model(self.time, period, t0, rp_rs, a_rs, b, baseline, u1, u2)
            residuals = self.flux - model
            return -0.5 * np.dot(residuals, residuals * self.inv_var)
        except Exception:
            return -np.inf

    def log_probability(self, theta: np.ndarray) -> float:
        """Log posterior probability."""
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        ll = self.log_likelihood(theta)
        if not np.isfinite(ll):
            return -np.inf
        return lp + ll

    def fit(self, n_walkers: int = config.N_WALKERS,
            n_burn: int = config.N_BURN_MIN,
            n_steps: int = config.N_STEPS_MIN,
            check_convergence: bool = True) -> Dict[str, Any]:
        """
        Run MCMC fitting.

        Parameters
        ----------
        n_walkers : int
            Number of MCMC walkers
        n_burn : int
            Number of burn-in steps
        n_steps : int
            Number of production steps
        check_convergence : bool
            Whether to check and ensure convergence

        Returns
        -------
        dict
            Fitting results including parameter estimates and uncertainties
        """
        logger.info("Starting MCMC fitting for phase-folded light curve...")

        # Initialize walkers
        p0 = self._initialize_walkers(n_walkers)

        # Create cached TransitModel for MCMC (reused across all likelihood evaluations)
        _init_params = setup_batman_params(
            period=self.period_prior, t0=self.t0_prior,
            rp_rs=self.rp_rs_prior, a_rs=self.a_rs_prior,
            b=config.DEFAULT_IMPACT_PARAMETER, u1=self.u1, u2=self.u2,
            ecc=config.DEFAULT_ECCENTRICITY, omega=config.DEFAULT_OMEGA
        )
        self._cached_transit_model = create_cached_transit_model(
            self.time, _init_params, cadence=self.cadence,
            long_cadence_threshold=self.long_cadence_threshold
        )
        self._cached_batman_params = _init_params

        # Create sampler
        self.sampler = emcee.EnsembleSampler(
            n_walkers, self.N_PARAMS, self.log_probability
        )

        # Adaptive burn-in: run in chunks, stop early when R-hat converges
        burnin_check_interval = config.CONVERGENCE_CHECK_INTERVAL
        burnin_min = config.N_BURN_MIN
        burnin_max = n_burn
        logger.info(f"  Running adaptive burn-in (min={burnin_min}, max={burnin_max}, check every {burnin_check_interval})...")

        burnin_steps_done = 0
        state = p0
        while burnin_steps_done < burnin_max:
            chunk = min(burnin_check_interval, burnin_max - burnin_steps_done)
            state = self.sampler.run_mcmc(state, chunk, progress=False)
            burnin_steps_done += chunk

            if burnin_steps_done >= burnin_min:
                chains = self.sampler.get_chain()
                rhat = compute_rhat_split(chains)
                max_rhat = float(np.max(rhat))
                if max_rhat < config.CONVERGENCE_RHAT:
                    logger.info(f"    Burn-in converged at {burnin_steps_done} steps (R-hat={max_rhat:.4f})")
                    break
                else:
                    logger.info(f"    {burnin_steps_done} steps: R-hat={max_rhat:.4f} (not converged)")

        if burnin_steps_done >= burnin_max:
            logger.info(f"    Burn-in reached maximum {burnin_max} steps")

        self.burnin_chain = self.sampler.get_chain()
        self.burnin_log_prob = self.sampler.get_log_prob()
        self.n_burnin_steps = burnin_steps_done

        self.sampler.reset()

        # Production run
        logger.info(f"  Running production ({n_steps} steps)...")

        if check_convergence:
            # Run until converged or max steps
            diagnostics = run_until_converged(
                self.sampler, state,
                max_steps=config.N_STEPS_MAX,
                check_interval=config.CONVERGENCE_CHECK_INTERVAL,
                rhat_threshold=config.CONVERGENCE_RHAT,
                ess_threshold=config.CONVERGENCE_ESS,
                autocorr_threshold=config.CONVERGENCE_AUTOCORR
            )
        else:
            state = self.sampler.run_mcmc(state, n_steps, progress=False)
            diagnostics = {'converged': True, 'n_steps': n_steps}

        # Clean up MCMC cache (not needed for plotting)
        self._cached_transit_model = None
        self._cached_batman_params = None

        # Extract results
        self.samples = self.sampler.get_chain(flat=True)
        self.results = self._extract_results(diagnostics)

        logger.info(f"  MCMC complete: {len(self.samples)} samples")
        logger.info(f"  Converged: {diagnostics.get('converged', 'Unknown')}")

        return self.results

    def _initialize_walkers(self, n_walkers: int) -> np.ndarray:
        """Initialize walker positions near the prior estimates."""
        ndim = self.N_PARAMS
        p0 = np.zeros((n_walkers, ndim))

        for i in range(n_walkers):
            p0[i, 0] = self.period_prior + np.random.uniform(-self._period_sigma, self._period_sigma)
            p0[i, 1] = self.t0_prior + np.random.uniform(-self._t0_sigma, self._t0_sigma)
            p0[i, 2] = self.rp_rs_prior * (1 + np.random.uniform(-config.WALKER_INIT_RP_RS_FRAC, config.WALKER_INIT_RP_RS_FRAC))
            p0[i, 3] = self.a_rs_prior * (1 + np.random.uniform(-config.WALKER_INIT_A_RS_FRAC, config.WALKER_INIT_A_RS_FRAC))
            p0[i, 4] = np.random.uniform(config.B_MIN, config.B_MAX + self.rp_rs_prior)
            p0[i, 5] = 1.0 + np.random.uniform(-config.WALKER_INIT_BASELINE_FRAC, config.WALKER_INIT_BASELINE_FRAC)
            # Initialize u1, u2 with small scatter around prior (only if not fixed)
            if not self.fix_ld:
                p0[i, 6] = self.u1 + np.random.uniform(-config.WALKER_INIT_LD_WIDTH_U1, config.WALKER_INIT_LD_WIDTH_U1)
                p0[i, 7] = self.u2 + np.random.uniform(-config.WALKER_INIT_LD_WIDTH_U2, config.WALKER_INIT_LD_WIDTH_U2)

        return p0

    def _extract_results(self, diagnostics: Dict) -> Dict[str, Any]:
        """Extract parameter estimates and uncertainties from samples."""
        percentiles = np.percentile(self.samples, list(config.MCMC_PERCENTILES), axis=0)

        results = {
            'parameters': {},
            'derived': {},
            'diagnostics': diagnostics,
            'limb_darkening': {
                'u1': self.u1,
                'u2': self.u2,
                'law': 'quadratic',
                'source': 'Claret 2017 PHOENIX r-method',
                'fixed': self.fix_ld
            },
            'fixed': {
                'eccentricity': config.DEFAULT_ECCENTRICITY,
                'omega_deg': config.DEFAULT_OMEGA
            }
        }

        # Primary parameters
        for i, name in enumerate(self.PARAM_NAMES):
            median = percentiles[1, i]
            err_lower = median - percentiles[0, i]
            err_upper = percentiles[2, i] - median

            results['parameters'][name] = {
                'value': float(median),
                'err_lower': float(err_lower),
                'err_upper': float(err_upper),
                'err': float((err_lower + err_upper) / 2),
                'percentile_16': float(percentiles[0, i]),
                'percentile_84': float(percentiles[2, i])
            }

        # If limb darkening was fixed, add to parameters with zero uncertainty
        if self.fix_ld:
            results['parameters']['u1'] = {
                'value': self.u1,
                'err_lower': 0.0,
                'err_upper': 0.0,
                'err': 0.0,
                'percentile_16': self.u1,
                'percentile_84': self.u1,
                'fixed': True
            }
            results['parameters']['u2'] = {
                'value': self.u2,
                'err_lower': 0.0,
                'err_upper': 0.0,
                'err': 0.0,
                'percentile_16': self.u2,
                'percentile_84': self.u2,
                'fixed': True
            }

        # Derived parameters
        rp_rs = results['parameters']['rp_rs']['value']
        rp_rs_err = results['parameters']['rp_rs']['err']
        a_rs = results['parameters']['a_rs']['value']
        b = results['parameters']['b']['value']

        # Transit depth
        depth = rp_rs**2
        depth_err = 2 * rp_rs * rp_rs_err
        results['derived']['depth'] = {
            'value': float(depth),
            'err': float(depth_err),
            'depth_ppm': float(depth * 1e6),
            'depth_ppm_err': float(depth_err * 1e6)
        }

        # Orbital inclination
        if a_rs > 0:
            cos_i = np.clip(b / a_rs, 0, 1)
            inc_rad = np.arccos(cos_i)
            inc_deg = np.degrees(inc_rad)
        else:
            inc_deg = 90.0

        results['derived']['inclination_deg'] = float(inc_deg)

        # Transit duration (approximate for circular orbit)
        period = results['parameters']['period']['value']
        if a_rs > 0 and b < 1 + rp_rs:
            sin_i = np.sin(np.radians(inc_deg))
            duration_days = (period / np.pi) * np.arcsin(
                np.sqrt((1 + rp_rs)**2 - b**2) / (a_rs * sin_i)
            )
            results['derived']['duration_hr'] = float(duration_days * 24)
        else:
            results['derived']['duration_hr'] = float(self.duration_prior * 24)

        return results

    def get_best_fit_model(self, time: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Get the best-fit model evaluated at given times.

        Parameters
        ----------
        time : np.ndarray, optional
            Time array (uses fitting data if not provided)

        Returns
        -------
        np.ndarray
            Best-fit model flux
        """
        if self.results is None:
            raise ValueError("Must run fit() first")

        if time is None:
            time = self.time

        # Extract parameters
        period = self.results['parameters']['period']['value']
        t0 = self.results['parameters']['t0']['value']
        rp_rs = self.results['parameters']['rp_rs']['value']
        a_rs = self.results['parameters']['a_rs']['value']
        b = self.results['parameters']['b']['value']
        baseline = self.results['parameters']['baseline']['value']
        u1 = self.results['parameters']['u1']['value']
        u2 = self.results['parameters']['u2']['value']

        return self.transit_model(time, period, t0, rp_rs, a_rs, b, baseline, u1, u2)

    def get_phase_folded_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get phase-folded data using best-fit ephemeris.

        Returns
        -------
        phase, flux, flux_err : tuple of np.ndarray
            Phase-folded data
        """
        if self.results is None:
            # Use priors
            period = self.period_prior
            t0 = self.t0_prior
        else:
            period = self.results['parameters']['period']['value']
            t0 = self.results['parameters']['t0']['value']

        # Calculate phase
        phase = ((self.time - t0) / period) % 1.0
        # Center phase on transit (0.5 -> 0)
        phase = np.where(phase > 0.5, phase - 1.0, phase)

        return phase, self.flux, self.flux_err

    def get_binned_phase_folded(self, n_bins: int = config.N_PHASE_BINS
                                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get binned phase-folded data.

        Parameters
        ----------
        n_bins : int
            Number of phase bins

        Returns
        -------
        bin_centers, bin_flux, bin_err : tuple of np.ndarray
            Binned phase-folded data
        """
        phase, flux, flux_err = self.get_phase_folded_data()

        # Create bins
        bin_edges = np.linspace(-0.5, 0.5, n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        bin_flux = np.zeros(n_bins)
        bin_err = np.zeros(n_bins)

        for i in range(n_bins):
            mask = (phase >= bin_edges[i]) & (phase < bin_edges[i + 1])
            if np.sum(mask) > 0:
                # Weighted mean
                weights = 1.0 / flux_err[mask]**2
                bin_flux[i] = np.sum(flux[mask] * weights) / np.sum(weights)
                bin_err[i] = 1.0 / np.sqrt(np.sum(weights))
            else:
                bin_flux[i] = np.nan
                bin_err[i] = np.nan

        return bin_centers, bin_flux, bin_err


def fit_phase_folded(time: np.ndarray, flux: np.ndarray, flux_err: np.ndarray,
                     planet_params: Dict[str, Any], fix_ld: bool = True,
                     ld_prior_width_u1: float = config.LD_PRIOR_WIDTH_U1,
                     ld_prior_width_u2: float = config.LD_PRIOR_WIDTH_U2,
                     cadence: np.ndarray = None, **kwargs) -> Dict[str, Any]:
    """
    Convenience function to fit phase-folded light curve.

    Parameters
    ----------
    time : np.ndarray
        Time array in BJD
    flux : np.ndarray
        Normalized flux
    flux_err : np.ndarray
        Flux errors
    planet_params : dict
        Planet parameters from TOI catalog
    fix_ld : bool
        If True, fix limb darkening to theoretical values (default True)
    ld_prior_width_u1 : float
        Width of Gaussian prior on u1 (default 0.15)
    ld_prior_width_u2 : float
        Width of Gaussian prior on u2 (default 0.10)
    cadence : np.ndarray, optional
        Cadence in seconds for each data point
    **kwargs
        Additional arguments passed to PhaseFoldFitter.fit()

    Returns
    -------
    dict
        Fitting results
    """
    fitter = PhaseFoldFitter(time, flux, flux_err, planet_params,
                             fix_ld=fix_ld,
                             ld_prior_width_u1=ld_prior_width_u1,
                             ld_prior_width_u2=ld_prior_width_u2,
                             cadence=cadence)
    results = fitter.fit(**kwargs)

    # Add fitter reference for later use
    results['_fitter'] = fitter

    return results


if __name__ == "__main__":
    # Test with synthetic data
    logging.basicConfig(level=logging.INFO)

    print("Testing PhaseFoldFitter with synthetic data...")

    # Create synthetic transit data
    np.random.seed(42)

    # True parameters
    period_true = 3.5
    t0_true = 2458500.0
    rp_rs_true = 0.1
    a_rs_true = 10.0
    b_true = 0.3
    baseline_true = 1.0

    # Generate time array (multiple transits)
    n_transits = 10
    n_points_per_transit = 200
    times = []
    for i in range(n_transits):
        t_mid = t0_true + i * period_true
        t_transit = np.linspace(t_mid - 0.1, t_mid + 0.1, n_points_per_transit)
        times.extend(t_transit)

    time = np.array(times)

    # Generate model
    planet_params = {
        'period': period_true,
        't0': t0_true,
        'depth_ppm': (rp_rs_true**2) * 1e6,
        'duration_hr': 3.0,
        'stellar_teff': config.SOLAR_TEFF,
        'stellar_logg': config.SOLAR_LOGG
    }

    fitter = PhaseFoldFitter(time, np.ones_like(time), np.ones_like(time) * 0.001, planet_params)
    model_flux = fitter.transit_model(time, period_true, t0_true, rp_rs_true, a_rs_true, b_true, baseline_true)

    # Add noise
    flux_err = np.ones_like(time) * 0.001
    flux = model_flux + np.random.normal(0, 0.001, len(time))

    # Fit
    fitter = PhaseFoldFitter(time, flux, flux_err, planet_params)
    results = fitter.fit(n_burn=500, n_steps=1000, check_convergence=False)

    print("\nFitting Results:")
    print(f"  Period: {results['parameters']['period']['value']:.6f} +/- {results['parameters']['period']['err']:.6f}")
    print(f"  Rp/Rs: {results['parameters']['rp_rs']['value']:.4f} +/- {results['parameters']['rp_rs']['err']:.4f}")
    print(f"  a/Rs: {results['parameters']['a_rs']['value']:.2f} +/- {results['parameters']['a_rs']['err']:.2f}")
    print(f"  b: {results['parameters']['b']['value']:.3f} +/- {results['parameters']['b']['err']:.3f}")

    print("\nTrue values:")
    print(f"  Period: {period_true}")
    print(f"  Rp/Rs: {rp_rs_true}")
    print(f"  a/Rs: {a_rs_true}")
    print(f"  b: {b_true}")
