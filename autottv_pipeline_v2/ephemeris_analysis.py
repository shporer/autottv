"""
Ephemeris Analysis (Step 3) for AutoTTV Pipeline v2.0

Fits linear and quadratic ephemerides to measured mid-transit times
and uses BIC for model selection.

Linear ephemeris:
    T(E) = T0 + P * E

Quadratic ephemeris:
    T(E) = T0 + P * E + 0.5 * dP/dE * E^2

where dP/dE = P * (dP/dt) and dP/dt is the period derivative.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from scipy.optimize import curve_fit
from scipy import stats

from . import config
from .utils import bic as compute_bic_util

logger = logging.getLogger(__name__)

# Time conversion constants - import from centralized config
SECONDS_PER_YEAR = config.SECONDS_PER_YEAR
DAYS_PER_YEAR = config.DAYS_PER_YEAR
MILLISECONDS_PER_DAY = config.MILLISECONDS_PER_DAY


@dataclass
class EphemerisResult:
    """Container for ephemeris fitting results."""

    # Linear parameters
    t0_linear: float
    t0_linear_err: float
    period_linear: float
    period_linear_err: float

    # Quadratic parameters (if fitted)
    t0_quadratic: Optional[float] = None
    t0_quadratic_err: Optional[float] = None
    period_quadratic: Optional[float] = None
    period_quadratic_err: Optional[float] = None
    dP_dE: Optional[float] = None  # Period change per epoch
    dP_dE_err: Optional[float] = None

    # Model comparison
    bic_linear: float = 0.0
    bic_quadratic: float = 0.0
    aic_linear: float = 0.0
    aic_quadratic: float = 0.0
    selection_criterion: str = "BIC"  # "BIC" or "AIC" (AIC used when n_transits < 8)
    preferred_model: str = "linear"

    # O-C statistics
    oc_rms_linear: float = 0.0
    oc_rms_quadratic: float = 0.0
    chi2_linear: float = 0.0
    chi2_quadratic: float = 0.0
    n_transits: int = 0

    @property
    def delta_bic(self) -> float:
        """BIC difference (linear - quadratic). Positive favors quadratic."""
        return self.bic_linear - self.bic_quadratic

    @property
    def delta_aic(self) -> float:
        """AIC difference (linear - quadratic). Positive favors quadratic."""
        return self.aic_linear - self.aic_quadratic

    @property
    def delta_ic(self) -> float:
        """Information criterion difference used for model selection (BIC or AIC)."""
        if self.selection_criterion == "AIC":
            return self.delta_aic
        return self.delta_bic

    @property
    def dP_dt_ms_per_year(self) -> Optional[float]:
        """
        Period derivative in milliseconds per year.

        dP/dt = (dP/dE) / P  (dimensionless rate)
        Convert to ms/yr: dP/dt * (ms/day) * (days/year)
        """
        if self.dP_dE is None or self.period_quadratic is None:
            return None

        # dP/dE is in days per epoch
        # dP/dt = (dP/dE) / P where P is in days
        # To get ms/year: multiply by (ms/day) * (days/year) / (days/orbit)

        dP_dt = self.dP_dE / self.period_quadratic  # per orbit
        dP_dt_per_year = dP_dt * DAYS_PER_YEAR / self.period_quadratic  # per year
        dP_dt_ms_yr = dP_dt_per_year * MILLISECONDS_PER_DAY  # convert days to ms

        return dP_dt_ms_yr

    @property
    def dP_dt_ms_per_year_err(self) -> Optional[float]:
        """Error on period derivative in ms/year."""
        if self.dP_dE_err is None or self.period_quadratic is None:
            return None

        # Simplified error propagation (ignore period error)
        dP_dt_per_year = self.dP_dE_err / self.period_quadratic * DAYS_PER_YEAR / self.period_quadratic
        return dP_dt_per_year * MILLISECONDS_PER_DAY

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'linear': {
                't0': self.t0_linear,
                't0_err': self.t0_linear_err,
                'period': self.period_linear,
                'period_err': self.period_linear_err,
                'bic': self.bic_linear,
                'oc_rms_min': self.oc_rms_linear * 24 * 60,  # Convert to minutes
                'chi2': self.chi2_linear
            },
            'quadratic': {
                't0': self.t0_quadratic,
                't0_err': self.t0_quadratic_err,
                'period': self.period_quadratic,
                'period_err': self.period_quadratic_err,
                'dP_dE': self.dP_dE,
                'dP_dE_err': self.dP_dE_err,
                'dP_dt_ms_per_year': self.dP_dt_ms_per_year,
                'dP_dt_ms_per_year_err': self.dP_dt_ms_per_year_err,
                'bic': self.bic_quadratic,
                'oc_rms_min': self.oc_rms_quadratic * 24 * 60,
                'chi2': self.chi2_quadratic
            },
            'model_selection': {
                'preferred_model': self.preferred_model,
                'selection_criterion': self.selection_criterion,
                'delta_bic': self.delta_bic,
                'delta_aic': self.delta_aic,
                'n_transits': self.n_transits
            }
        }


def linear_ephemeris(epoch: np.ndarray, t0: float, period: float) -> np.ndarray:
    """
    Linear ephemeris model.

    T(E) = T0 + P * E
    """
    return t0 + period * epoch


def quadratic_ephemeris(epoch: np.ndarray, t0: float, period: float,
                        half_dP_dE: float) -> np.ndarray:
    """
    Quadratic ephemeris model.

    T(E) = T0 + P * E + 0.5 * dP/dE * E^2

    Note: We fit half_dP_dE = 0.5 * dP/dE directly for numerical stability.
    """
    return t0 + period * epoch + half_dP_dE * epoch**2


def compute_bic(n_data: int, n_params: int, chi2: float) -> float:
    """
    Compute Bayesian Information Criterion.

    Wrapper around utils.bic with parameter order matching legacy API.
    """
    return compute_bic_util(chi2, n_params, n_data)


def compute_aic(n_data: int, n_params: int, chi2: float) -> float:
    """
    Compute Akaike Information Criterion.

    AIC = chi2 + 2*k, where k is the number of parameters.
    For small samples (n/k < 40), uses corrected AICc:
    AICc = AIC + 2*k*(k+1) / (n - k - 1)
    """
    aic = chi2 + 2 * n_params
    # Apply small-sample correction (AICc) when n/k < 40
    if n_data > n_params + 1:
        aic += 2 * n_params * (n_params + 1) / (n_data - n_params - 1)
    return aic


class EphemerisAnalyzer:
    """
    Analyzes mid-transit times to fit ephemerides and detect period changes.
    """

    def __init__(self, epochs: np.ndarray, t_mids: np.ndarray,
                 t_mid_errs: np.ndarray):
        """
        Initialize ephemeris analyzer.

        Parameters
        ----------
        epochs : np.ndarray
            Transit epoch numbers
        t_mids : np.ndarray
            Measured mid-transit times (BJD)
        t_mid_errs : np.ndarray
            Mid-transit time errors (days)
        """
        self.epochs = np.asarray(epochs)
        self.t_mids = np.asarray(t_mids)
        self.t_mid_errs = np.asarray(t_mid_errs)

        # Remove any NaN values
        valid = (np.isfinite(self.epochs) & np.isfinite(self.t_mids) &
                 np.isfinite(self.t_mid_errs) & (self.t_mid_errs > 0))
        self.epochs = self.epochs[valid]
        self.t_mids = self.t_mids[valid]
        self.t_mid_errs = self.t_mid_errs[valid]

        self.n_transits = len(self.epochs)

        # Results storage
        self.result = None

        logger.info(f"EphemerisAnalyzer initialized with {self.n_transits} transits")

    def fit_linear(self) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """
        Fit linear ephemeris.

        Returns
        -------
        params, errors, chi2, bic : tuple
            Best-fit parameters, errors, chi-squared, and BIC
        """
        if self.n_transits < 2:
            raise ValueError("Need at least 2 transits for linear fit")

        # Initial guess
        p0 = [self.t_mids.mean(), np.median(np.diff(self.t_mids) / np.diff(self.epochs))]

        try:
            popt, pcov = curve_fit(
                linear_ephemeris,
                self.epochs, self.t_mids,
                p0=p0,
                sigma=self.t_mid_errs,
                absolute_sigma=True
            )
            perr = np.sqrt(np.diag(pcov))
        except Exception as e:
            logger.warning(f"Linear fit failed: {e}")
            # Fallback to simple linear regression
            slope, intercept, _, _, _ = stats.linregress(self.epochs, self.t_mids)
            popt = np.array([intercept, slope])
            perr = np.array([0.0, 0.0])

        # Compute chi-squared
        residuals = self.t_mids - linear_ephemeris(self.epochs, *popt)
        chi2 = np.sum((residuals / self.t_mid_errs)**2)

        # Compute BIC
        bic = compute_bic(self.n_transits, 2, chi2)

        return popt, perr, chi2, bic

    def fit_quadratic(self) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """
        Fit quadratic ephemeris.

        Returns
        -------
        params, errors, chi2, bic : tuple
            Best-fit parameters [t0, period, half_dP_dE], errors, chi2, BIC
        """
        if self.n_transits < 3:
            raise ValueError("Need at least 3 transits for quadratic fit")

        # Initial guess from linear fit
        linear_params, _, _, _ = self.fit_linear()
        p0 = [linear_params[0], linear_params[1], 0.0]

        try:
            popt, pcov = curve_fit(
                quadratic_ephemeris,
                self.epochs, self.t_mids,
                p0=p0,
                sigma=self.t_mid_errs,
                absolute_sigma=True
            )
            perr = np.sqrt(np.diag(pcov))
        except Exception as e:
            logger.warning(f"Quadratic fit failed: {e}")
            popt = np.array(p0)
            perr = np.array([0.0, 0.0, 0.0])

        # Compute chi-squared
        residuals = self.t_mids - quadratic_ephemeris(self.epochs, *popt)
        chi2 = np.sum((residuals / self.t_mid_errs)**2)

        # Compute BIC
        bic = compute_bic(self.n_transits, 3, chi2)

        return popt, perr, chi2, bic

    def analyze(self) -> EphemerisResult:
        """
        Perform complete ephemeris analysis.

        Fits both linear and quadratic models, computes BIC for model selection.

        Returns
        -------
        EphemerisResult
            Complete analysis results
        """
        if self.n_transits < 2:
            raise ValueError(f"Need at least 2 transits, got {self.n_transits}")

        # Fit linear ephemeris
        linear_params, linear_errs, chi2_linear, bic_linear = self.fit_linear()
        aic_linear = compute_aic(self.n_transits, 2, chi2_linear)

        # Compute O-C for linear
        oc_linear = self.t_mids - linear_ephemeris(self.epochs, *linear_params)
        oc_rms_linear = np.sqrt(np.mean(oc_linear**2))

        # Use AIC for model selection when n_transits < 8 (small sample)
        use_aic = self.n_transits < 8
        selection_criterion = "AIC" if use_aic else "BIC"

        # Initialize result with linear fit
        result = EphemerisResult(
            t0_linear=float(linear_params[0]),
            t0_linear_err=float(linear_errs[0]),
            period_linear=float(linear_params[1]),
            period_linear_err=float(linear_errs[1]),
            bic_linear=float(bic_linear),
            aic_linear=float(aic_linear),
            chi2_linear=float(chi2_linear),
            oc_rms_linear=float(oc_rms_linear),
            n_transits=self.n_transits,
            selection_criterion=selection_criterion,
            preferred_model="linear"
        )

        # Fit quadratic if we have enough transits
        if self.n_transits >= 3:
            try:
                quad_params, quad_errs, chi2_quad, bic_quad = self.fit_quadratic()

                # Compute O-C for quadratic
                oc_quad = self.t_mids - quadratic_ephemeris(self.epochs, *quad_params)
                oc_rms_quad = np.sqrt(np.mean(oc_quad**2))

                result.t0_quadratic = float(quad_params[0])
                result.t0_quadratic_err = float(quad_errs[0])
                result.period_quadratic = float(quad_params[1])
                result.period_quadratic_err = float(quad_errs[1])
                result.dP_dE = float(2 * quad_params[2])  # Convert from half_dP_dE
                result.dP_dE_err = float(2 * quad_errs[2])
                result.bic_quadratic = float(bic_quad)
                result.aic_quadratic = float(compute_aic(self.n_transits, 3, chi2_quad))
                result.chi2_quadratic = float(chi2_quad)
                result.oc_rms_quadratic = float(oc_rms_quad)

                # Model selection: use AIC for small samples (n < 8), BIC otherwise
                if use_aic:
                    if result.delta_aic > config.DELTA_BIC_MODEL_SELECTION:
                        result.preferred_model = "quadratic"
                    else:
                        result.preferred_model = "linear"
                else:
                    if result.delta_bic > config.DELTA_BIC_MODEL_SELECTION:
                        result.preferred_model = "quadratic"
                    else:
                        result.preferred_model = "linear"

            except Exception as e:
                logger.warning(f"Quadratic fit failed: {e}")

        self.result = result

        logger.info(f"Ephemeris analysis complete:")
        logger.info(f"  Linear: T0={result.t0_linear:.6f}, P={result.period_linear:.8f}")
        logger.info(f"  BIC linear: {result.bic_linear:.2f}")
        if result.t0_quadratic is not None:
            logger.info(f"  Quadratic: T0={result.t0_quadratic:.6f}, P={result.period_quadratic:.8f}")
            logger.info(f"  dP/dt = {result.dP_dt_ms_per_year:.3f} +/- {result.dP_dt_ms_per_year_err:.3f} ms/yr")
            logger.info(f"  BIC quadratic: {result.bic_quadratic:.2f}")
        if use_aic:
            logger.info(f"  Preferred model: {result.preferred_model} (delta_AIC = {result.delta_aic:.2f}, n={self.n_transits} < 8)")
        else:
            logger.info(f"  Preferred model: {result.preferred_model} (delta_BIC = {result.delta_bic:.2f})")

        return result

    def get_oc_residuals(self, model: str = "preferred") -> Tuple[np.ndarray, np.ndarray]:
        """
        Get O-C residuals for specified model.

        Parameters
        ----------
        model : str
            "linear", "quadratic", or "preferred"

        Returns
        -------
        epochs, oc_days : tuple of np.ndarray
            Epochs and O-C residuals in days
        """
        if self.result is None:
            self.analyze()

        if model == "preferred":
            model = self.result.preferred_model

        if model == "linear":
            predicted = linear_ephemeris(
                self.epochs,
                self.result.t0_linear,
                self.result.period_linear
            )
        else:
            predicted = quadratic_ephemeris(
                self.epochs,
                self.result.t0_quadratic,
                self.result.period_quadratic,
                self.result.dP_dE / 2  # Convert to half_dP_dE
            )

        oc = self.t_mids - predicted
        return self.epochs, oc

    def get_oc_minutes(self, model: str = "preferred") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get O-C residuals in minutes with errors.

        Returns
        -------
        epochs, oc_minutes, oc_err_minutes : tuple
        """
        epochs, oc_days = self.get_oc_residuals(model)
        oc_min = oc_days * 24 * 60
        oc_err_min = self.t_mid_errs * 24 * 60
        return epochs, oc_min, oc_err_min


def analyze_ephemeris(transit_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Convenience function to analyze ephemeris from transit results.

    Parameters
    ----------
    transit_results : list of dict
        Results from individual transit fitting (list of TransitResult.to_dict())

    Returns
    -------
    dict
        Ephemeris analysis results
    """
    # Extract data
    epochs = np.array([t['epoch'] for t in transit_results if t['success']])
    t_mids = np.array([t['t_mid'] for t in transit_results if t['success']])
    t_mid_errs = np.array([t['t_mid_err'] for t in transit_results if t['success']])

    if len(epochs) < 2:
        logger.warning("Not enough successful transits for ephemeris analysis")
        return {'error': 'Insufficient transits'}

    analyzer = EphemerisAnalyzer(epochs, t_mids, t_mid_errs)
    result = analyzer.analyze()

    output = result.to_dict()

    # Add O-C data
    epochs_oc, oc_linear = analyzer.get_oc_residuals("linear")
    output['oc_linear'] = {
        'epochs': epochs_oc.tolist(),
        'oc_days': oc_linear.tolist(),
        'oc_minutes': (oc_linear * 24 * 60).tolist(),
        't_mid_err_minutes': (t_mid_errs * 24 * 60).tolist()
    }

    if result.t0_quadratic is not None:
        _, oc_quad = analyzer.get_oc_residuals("quadratic")
        output['oc_quadratic'] = {
            'epochs': epochs_oc.tolist(),
            'oc_days': oc_quad.tolist(),
            'oc_minutes': (oc_quad * 24 * 60).tolist(),
            't_mid_err_minutes': (t_mid_errs * 24 * 60).tolist()
        }

    return output


if __name__ == "__main__":
    # Test with synthetic data
    logging.basicConfig(level=logging.INFO)

    print("Testing EphemerisAnalyzer...")

    # Generate synthetic transit times with a period change
    np.random.seed(42)

    t0_true = 2458500.0
    period_true = 3.5
    dP_dE_true = 1e-7  # Small period change

    epochs = np.arange(0, 100, 3)  # Every 3rd transit observed
    t_mids_true = t0_true + period_true * epochs + 0.5 * dP_dE_true * epochs**2

    # Add noise
    t_mid_errs = np.ones_like(epochs) * (1.0 / 24 / 60)  # 1 minute errors
    t_mids = t_mids_true + np.random.normal(0, t_mid_errs)

    # Analyze
    analyzer = EphemerisAnalyzer(epochs, t_mids, t_mid_errs)
    result = analyzer.analyze()

    print("\nResults:")
    print(f"  T0 (linear): {result.t0_linear:.6f}")
    print(f"  Period (linear): {result.period_linear:.8f}")
    print(f"  T0 (quadratic): {result.t0_quadratic:.6f}")
    print(f"  Period (quadratic): {result.period_quadratic:.8f}")
    print(f"  dP/dE: {result.dP_dE:.2e} (true: {dP_dE_true:.2e})")
    print(f"  dP/dt: {result.dP_dt_ms_per_year:.3f} ms/year")
    print(f"  Preferred model: {result.preferred_model}")
    print(f"  Delta BIC: {result.delta_bic:.2f}")
