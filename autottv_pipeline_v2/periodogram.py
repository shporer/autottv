"""
Lomb-Scargle Periodogram for TTV Detection - AutoTTV Pipeline v2.0

Searches for periodic signals in transit timing variations (TTVs) using
the Lomb-Scargle periodogram.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count

from astropy.timeseries import LombScargle

from . import config

logger = logging.getLogger(__name__)


def _bootstrap_fap_worker(args):
    """Worker for parallel bootstrap FAP computation (astropy LombScargle)."""
    worker_id, n_iter, time_days, oc_minutes, oc_err_minutes, frequencies, observed_peak = args
    rng = np.random.default_rng(seed=42 + worker_id)
    n_exceed = 0
    for _ in range(n_iter):
        perm = rng.permutation(len(oc_minutes))
        oc_shuf = oc_minutes[perm]
        err_shuf = oc_err_minutes[perm]
        ls_shuf = LombScargle(time_days, oc_shuf, dy=err_shuf)
        power_shuf = ls_shuf.power(frequencies)
        if power_shuf.max() >= observed_peak:
            n_exceed += 1
    return n_exceed


@dataclass
class PeriodogramResult:
    """Container for periodogram analysis results."""

    # Frequency grid and power
    frequencies: np.ndarray  # 1/day
    power: np.ndarray

    # Peak detection
    peak_frequency: float
    peak_period: float  # days
    peak_power: float

    # FAP levels
    fap_01: float  # 1% FAP level
    fap_05: float  # 5% FAP level
    fap_10: float  # 10% FAP level

    # Peak significance
    peak_fap: float  # FAP of the highest peak (analytic)

    # Peak width
    peak_period_error: float  # days, from FWHM of peak
    peak_fwhm_freq: float  # FWHM in frequency (1/day)

    # Analysis metadata
    n_transits: int
    time_span_days: float
    nyquist_frequency: float

    # Bootstrap FAP (defaults allow creation without bootstrap)
    bootstrap_fap: float = None  # Empirical FAP from bootstrap
    bootstrap_n_iter: int = 0  # Number of bootstrap iterations used

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excluding large arrays)."""
        return {
            'peak_frequency': self.peak_frequency,
            'peak_period_days': self.peak_period,
            'peak_period_error_days': self.peak_period_error,
            'peak_fwhm_freq': self.peak_fwhm_freq,
            'peak_power': self.peak_power,
            'peak_fap': self.peak_fap,
            'bootstrap_fap': self.bootstrap_fap,
            'bootstrap_n_iter': self.bootstrap_n_iter,
            'fap_levels': {
                '1%': self.fap_01,
                '5%': self.fap_05,
                '10%': self.fap_10
            },
            'is_significant_1pct': self.peak_power > self.fap_01,
            'is_significant_5pct': self.peak_power > self.fap_05,
            'is_significant_bootstrap_1pct': (self.bootstrap_fap is not None and self.bootstrap_fap < 0.01),
            'n_transits': self.n_transits,
            'time_span_days': self.time_span_days,
            'nyquist_frequency': self.nyquist_frequency,
            'frequency_range': {
                'min': float(self.frequencies.min()),
                'max': float(self.frequencies.max()),
                'n_frequencies': len(self.frequencies)
            }
        }

    def get_periodogram_data(self) -> Dict[str, List[float]]:
        """Get periodogram data for plotting/export."""
        return {
            'frequencies': self.frequencies.tolist(),
            'power': self.power.tolist()
        }


class TTVPeriodogram:
    """
    Lomb-Scargle periodogram analyzer for transit timing variations.
    """

    def __init__(self, epochs: np.ndarray, oc_days: np.ndarray,
                 oc_err_days: np.ndarray, period_days: float):
        """
        Initialize periodogram analyzer.

        Parameters
        ----------
        epochs : np.ndarray
            Transit epoch numbers
        oc_days : np.ndarray
            O-C residuals in days
        oc_err_days : np.ndarray
            O-C errors in days
        period_days : float
            Orbital period in days (for computing time span)
        """
        self.epochs = np.asarray(epochs)
        self.oc_days = np.asarray(oc_days)
        self.oc_err_days = np.asarray(oc_err_days)
        self.period_days = period_days

        # Remove NaN values
        valid = (np.isfinite(self.epochs) & np.isfinite(self.oc_days) &
                 np.isfinite(self.oc_err_days))
        self.epochs = self.epochs[valid]
        self.oc_days = self.oc_days[valid]
        self.oc_err_days = self.oc_err_days[valid]

        self.n_transits = len(self.epochs)

        # Compute time array (epochs * period for spacing info)
        self.time_days = self.epochs * period_days

        # Time span
        if self.n_transits > 1:
            self.time_span = self.time_days.max() - self.time_days.min()
        else:
            self.time_span = 0.0

        # Compute Nyquist frequency (based on median sampling)
        if self.n_transits > 1:
            median_dt = np.median(np.diff(np.sort(self.time_days)))
            self.nyquist_freq = 0.5 / median_dt
        else:
            self.nyquist_freq = config.DEFAULT_NYQUIST_FREQ

        # Minimum frequency (based on time span)
        if self.time_span > 0:
            self.min_freq = 2.0 / self.time_span  # Require at least 2 cycles
        else:
            self.min_freq = config.DEFAULT_MIN_FREQ

        self.result = None

        logger.info(f"TTVPeriodogram initialized:")
        logger.info(f"  N transits: {self.n_transits}")
        logger.info(f"  Time span: {self.time_span:.1f} days")
        logger.info(f"  Frequency range: {self.min_freq:.6f} - {self.nyquist_freq:.4f} 1/day")

    @staticmethod
    def _compute_peak_fwhm(frequencies: np.ndarray, power: np.ndarray,
                           peak_idx: int) -> float:
        """
        Compute FWHM of the periodogram peak in frequency space.

        Finds the half-maximum power level, then searches left and right
        from the peak for crossings, interpolating linearly.

        Parameters
        ----------
        frequencies : np.ndarray
            Frequency grid (1/day)
        power : np.ndarray
            Periodogram power values
        peak_idx : int
            Index of the peak in the arrays

        Returns
        -------
        float
            FWHM in frequency (1/day). Returns 0.0 if measurement fails.
        """
        half_power = power[peak_idx] / 2.0

        # Search left from peak
        f_left = frequencies[0]  # fallback: use grid edge
        for i in range(peak_idx, 0, -1):
            if power[i - 1] <= half_power:
                # Linear interpolation between i-1 and i
                frac = (half_power - power[i - 1]) / (power[i] - power[i - 1])
                f_left = frequencies[i - 1] + frac * (frequencies[i] - frequencies[i - 1])
                break

        # Search right from peak
        f_right = frequencies[-1]  # fallback: use grid edge
        for i in range(peak_idx, len(power) - 1):
            if power[i + 1] <= half_power:
                # Linear interpolation between i and i+1
                frac = (half_power - power[i]) / (power[i + 1] - power[i])
                f_right = frequencies[i] + frac * (frequencies[i + 1] - frequencies[i])
                break

        fwhm = f_right - f_left
        return max(fwhm, 0.0)

    def compute_periodogram(self, oversampling: int = config.PERIODOGRAM_OVERSAMPLING) -> PeriodogramResult:
        """
        Compute Lomb-Scargle periodogram.

        Parameters
        ----------
        oversampling : int
            Oversampling factor for frequency grid. The number of frequency
            samples is computed as: oversampling × time_span × (f_nyquist - f_min)

        Returns
        -------
        PeriodogramResult
            Periodogram results
        """
        if self.n_transits < config.MIN_TRANSITS_PERIODOGRAM:
            logger.warning(f"Too few transits ({self.n_transits}) for periodogram")
            return self._empty_result()

        # Create frequency grid
        # Number of frequencies based on oversampling factor
        # This ensures frequency resolution is fine enough to resolve peaks
        n_frequencies = int(oversampling * self.time_span * (self.nyquist_freq - self.min_freq))
        n_frequencies = max(n_frequencies, 100)  # Ensure minimum resolution

        frequencies = np.linspace(
            self.min_freq,
            self.nyquist_freq,
            n_frequencies
        )

        logger.info(f"  Periodogram: {n_frequencies} frequency samples (oversampling={oversampling})")

        # Compute Lomb-Scargle periodogram
        # Use O-C values converted to minutes for better numerical stability
        oc_minutes = self.oc_days * 24 * 60
        oc_err_minutes = self.oc_err_days * 24 * 60

        try:
            ls = LombScargle(self.time_days, oc_minutes, dy=oc_err_minutes)
            power = ls.power(frequencies)
        except Exception as e:
            logger.warning(f"Periodogram computation failed: {e}")
            return self._empty_result()

        # Find peak
        peak_idx = np.argmax(power)
        peak_freq = frequencies[peak_idx]
        peak_power = power[peak_idx]
        peak_period = 1.0 / peak_freq if peak_freq > 0 else np.inf

        # Compute FWHM-based period error
        fwhm_freq = self._compute_peak_fwhm(frequencies, power, peak_idx)
        # σ_f = FWHM/2 (half-width at half-max), σ_P = σ_f × P² (error propagation from P=1/f)
        if fwhm_freq > 0 and np.isfinite(peak_period):
            peak_period_error = (fwhm_freq / 2.0) * peak_period**2
        else:
            peak_period_error = 0.0

        # Compute FAP levels
        try:
            fap_01 = ls.false_alarm_level(0.01)
            fap_05 = ls.false_alarm_level(0.05)
            fap_10 = ls.false_alarm_level(0.10)
        except Exception:
            # Fallback estimates
            fap_01 = np.percentile(power, 99)
            fap_05 = np.percentile(power, 95)
            fap_10 = np.percentile(power, 90)

        # Compute FAP of peak
        try:
            peak_fap = ls.false_alarm_probability(peak_power)
        except Exception:
            peak_fap = 1.0

        self.result = PeriodogramResult(
            frequencies=frequencies,
            power=power,
            peak_frequency=float(peak_freq),
            peak_period=float(peak_period),
            peak_power=float(peak_power),
            fap_01=float(fap_01),
            fap_05=float(fap_05),
            fap_10=float(fap_10),
            peak_fap=float(peak_fap),
            peak_period_error=float(peak_period_error),
            peak_fwhm_freq=float(fwhm_freq),
            n_transits=self.n_transits,
            time_span_days=float(self.time_span),
            nyquist_frequency=float(self.nyquist_freq)
        )

        logger.info(f"Periodogram computed:")
        logger.info(f"  Peak frequency: {peak_freq:.6f} 1/day (period: {peak_period:.2f} ± {peak_period_error:.2f} days)")
        logger.info(f"  Peak FWHM: {fwhm_freq:.6f} 1/day")
        logger.info(f"  Peak power: {peak_power:.4f}")
        logger.info(f"  Peak FAP (analytic): {peak_fap:.4e}")
        logger.info(f"  Significant at 1%: {peak_power > fap_01}")

        return self.result

    def compute_bootstrap_fap(self, n_iterations: int = None,
                              parallel: bool = True) -> float:
        """
        Compute empirical FAP via bootstrap: shuffle O-C values among the
        fixed time stamps, recompute the periodogram, and measure how often
        the shuffled peak power exceeds the observed peak power.

        Parameters
        ----------
        n_iterations : int, optional
            Number of bootstrap iterations.  Defaults to config.BOOTSTRAP_FAP_N_ITERATIONS.

        Returns
        -------
        float
            Bootstrap FAP (fraction of shuffled periodograms with peak >= observed).
        """
        if n_iterations is None:
            n_iterations = config.BOOTSTRAP_FAP_N_ITERATIONS

        if self.result is None:
            self.compute_periodogram()

        if self.result.peak_power == 0 or len(self.result.frequencies) == 0:
            return 1.0

        observed_peak = self.result.peak_power
        frequencies = self.result.frequencies
        oc_minutes = self.oc_days * 24 * 60
        oc_err_minutes = self.oc_err_days * 24 * 60

        # Parallel bootstrap.  parallel=False forces the single-process path —
        # required when this is called from inside another multiprocessing.Pool
        # worker (daemon processes cannot spawn children).
        n_workers = max(1, min(cpu_count() - 1, 16)) if cpu_count() > 2 else 1
        if parallel and n_workers > 1 and n_iterations >= 1000:
            chunk_size = n_iterations // n_workers
            remainder = n_iterations % n_workers
            chunks = [(i, chunk_size + (1 if i < remainder else 0),
                        self.time_days, oc_minutes, oc_err_minutes,
                        frequencies, observed_peak)
                       for i in range(n_workers)]
            with Pool(processes=n_workers) as pool:
                results = pool.map(_bootstrap_fap_worker, chunks)
            n_exceed = sum(results)
        else:
            n_exceed = 0
            rng = np.random.default_rng(seed=42)
            check_interval = 1000

            for i in range(n_iterations):
                perm = rng.permutation(len(oc_minutes))
                oc_shuf = oc_minutes[perm]
                err_shuf = oc_err_minutes[perm]

                ls_shuf = LombScargle(self.time_days, oc_shuf, dy=err_shuf)
                power_shuf = ls_shuf.power(frequencies)
                if power_shuf.max() >= observed_peak:
                    n_exceed += 1

                if (i + 1) % check_interval == 0:
                    running_fap = (n_exceed + 1) / (i + 2)
                    if running_fap > 0.5:
                        n_iterations = i + 1
                        logger.info(f"  Bootstrap early stop at {n_iterations} iterations (running FAP={running_fap:.3f})")
                        break

        bootstrap_fap = (n_exceed + 1) / (n_iterations + 1)  # +1 to avoid FAP=0

        # Store in result
        self.result.bootstrap_fap = float(bootstrap_fap)
        self.result.bootstrap_n_iter = n_iterations

        logger.info(f"  Bootstrap FAP: {bootstrap_fap:.4f} ({n_exceed}/{n_iterations} exceed observed peak)")

        return bootstrap_fap

    def _empty_result(self) -> PeriodogramResult:
        """Return empty result for cases with insufficient data."""
        return PeriodogramResult(
            frequencies=np.array([]),
            power=np.array([]),
            peak_frequency=0.0,
            peak_period=0.0,
            peak_power=0.0,
            fap_01=1.0,
            fap_05=1.0,
            fap_10=1.0,
            peak_fap=1.0,
            peak_period_error=0.0,
            peak_fwhm_freq=0.0,
            n_transits=self.n_transits,
            time_span_days=self.time_span,
            nyquist_frequency=self.nyquist_freq
        )

    def find_significant_peaks(self, fap_threshold: float = config.PEAK_FAP_THRESHOLD,
                               n_peaks: int = config.N_SIGNIFICANT_PEAKS) -> List[Dict[str, float]]:
        """
        Find significant peaks in the periodogram.

        Parameters
        ----------
        fap_threshold : float
            Maximum FAP for a peak to be considered significant
        n_peaks : int
            Maximum number of peaks to return

        Returns
        -------
        list of dict
            List of significant peaks with frequency, period, power, FAP
        """
        if self.result is None:
            self.compute_periodogram()

        if len(self.result.frequencies) == 0:
            return []

        # Find local maxima
        power = self.result.power
        frequencies = self.result.frequencies

        peaks = []

        # Simple peak finding: points higher than both neighbors
        for i in range(1, len(power) - 1):
            if power[i] > power[i-1] and power[i] > power[i+1]:
                if power[i] > self.result.fap_01 * (1 - fap_threshold):
                    peaks.append({
                        'frequency': float(frequencies[i]),
                        'period_days': float(1.0 / frequencies[i]),
                        'power': float(power[i]),
                        'index': i
                    })

        # Sort by power and take top n_peaks
        peaks.sort(key=lambda x: x['power'], reverse=True)
        peaks = peaks[:n_peaks]

        # Compute FAP for each peak
        ls = LombScargle(self.time_days, self.oc_days * 24 * 60,
                        dy=self.oc_err_days * 24 * 60)
        for peak in peaks:
            try:
                peak['fap'] = float(ls.false_alarm_probability(peak['power']))
            except Exception:
                peak['fap'] = 1.0

        # Filter by FAP threshold
        peaks = [p for p in peaks if p['fap'] < fap_threshold]

        return peaks


def compute_ttv_periodogram(transit_results: List[Dict[str, Any]],
                           ephemeris_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convenience function to compute TTV periodogram.

    Parameters
    ----------
    transit_results : list of dict
        Results from individual transit fitting
    ephemeris_results : dict
        Results from ephemeris analysis

    Returns
    -------
    dict
        Periodogram results
    """
    # Get O-C residuals from preferred ephemeris
    model = ephemeris_results.get('model_selection', {}).get('preferred_model', 'linear')

    if model == 'quadratic' and 'oc_quadratic' in ephemeris_results:
        oc_data = ephemeris_results['oc_quadratic']
    else:
        oc_data = ephemeris_results.get('oc_linear', {})

    if not oc_data:
        return {'error': 'No O-C data available'}

    epochs = np.array(oc_data.get('epochs', []))
    oc_days = np.array(oc_data.get('oc_days', []))
    oc_err_days = np.array(oc_data.get('t_mid_err_minutes', [])) / (24 * 60)

    # Get period
    linear_params = ephemeris_results.get('linear', {})
    period = linear_params.get('period', 1.0)

    if len(epochs) < config.MIN_TRANSITS_PERIODOGRAM:
        return {
            'error': f'Insufficient transits ({len(epochs)}) for periodogram',
            'n_transits': len(epochs),
            'min_required': config.MIN_TRANSITS_PERIODOGRAM
        }

    # Compute periodogram
    analyzer = TTVPeriodogram(epochs, oc_days, oc_err_days, period)
    result = analyzer.compute_periodogram()

    output = result.to_dict()
    output['periodogram_data'] = result.get_periodogram_data()

    # Find significant peaks
    peaks = analyzer.find_significant_peaks()
    output['significant_peaks'] = peaks

    # Compute bootstrap FAP if analytic FAP is not already very significant
    if result.peak_power > 0:
        bootstrap_fap = analyzer.compute_bootstrap_fap()
        output['bootstrap_fap'] = bootstrap_fap
        output['bootstrap_n_iter'] = analyzer.result.bootstrap_n_iter
        output['is_significant_bootstrap_1pct'] = bootstrap_fap < 0.01

    return output


if __name__ == "__main__":
    # Test with synthetic data
    logging.basicConfig(level=logging.INFO)

    print("Testing TTVPeriodogram...")

    np.random.seed(42)

    # Create synthetic TTV signal
    period = 3.5  # days
    epochs = np.arange(0, 100, 2)

    # Inject periodic TTV
    ttv_period = 15.0  # days
    ttv_amplitude = 2.0 / (24 * 60)  # 2 minutes in days

    oc_signal = ttv_amplitude * np.sin(2 * np.pi * epochs * period / ttv_period)

    # Add noise
    oc_err = np.ones_like(epochs) * (0.5 / 24 / 60)  # 30 seconds
    oc = oc_signal + np.random.normal(0, oc_err)

    # Analyze
    analyzer = TTVPeriodogram(epochs, oc, oc_err, period)
    result = analyzer.compute_periodogram()

    print("\nResults:")
    print(f"  Injected TTV period: {ttv_period} days")
    print(f"  Detected peak period: {result.peak_period:.2f} ± {result.peak_period_error:.2f} days")
    print(f"  Peak FWHM (freq): {result.peak_fwhm_freq:.6f} 1/day")
    print(f"  Peak power: {result.peak_power:.4f}")
    print(f"  Peak FAP: {result.peak_fap:.4e}")
    print(f"  1% FAP level: {result.fap_01:.4f}")
    print(f"  Significant: {result.peak_power > result.fap_01}")

    # Find peaks
    peaks = analyzer.find_significant_peaks()
    print(f"\nSignificant peaks: {len(peaks)}")
    for p in peaks:
        print(f"  Period: {p['period_days']:.2f} days, FAP: {p['fap']:.4e}")
