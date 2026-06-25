"""
Configuration settings for the AutoTTV Pipeline v2.0

All configurable parameters and paths are defined here.
"""

from pathlib import Path

# =============================================================================
# Directory Configuration
# =============================================================================

# Base directory for the project (auto-detected from this file's location)
BASE_DIR = Path(__file__).resolve().parent.parent

# Input data directories
LIGHTCURVE_DIR = BASE_DIR / "transit-times"  # Light curves directory (reference planets)
CATALOG_FILE = BASE_DIR / "toi_catalog_240226.csv"

# Output directories
OUTPUT_DIR = BASE_DIR / "autottv_results_v2"
DATA_CACHE_DIR = OUTPUT_DIR / "data_cache"
WEBAPP_DIR = BASE_DIR / "webapp"
WEBAPP_DATA_DIR = WEBAPP_DIR / "data"
WEBAPP_PLANETS_DIR = WEBAPP_DATA_DIR / "planets"

# =============================================================================
# Catalog Filtering Configuration
# =============================================================================

# Output filtered catalog
CATALOG_FILE_FILTERED = BASE_DIR / "toi_catalog_240226_for_ttv.csv"

# Step 1: Dispositions to exclude
EXCLUDED_DISPOSITIONS = ['FP', 'FA']

# Step 2: Minimum period (days) — remove TOIs with period <= this value
MIN_PERIOD_DAYS = 0.0  # Excludes period == 0

# Step 3: Minimum expected single-transit SNR
MIN_TRANSIT_SNR = 5.0

# Step 4: Minimum number of full transits in TESS data
MIN_FULL_TRANSITS = 5

# SPOC CDPP data file for SNR computation
SPOC_CDPP_FILE = OUTPUT_DIR / "spoc_cdpp_sampled.npy"

# Magnitude half-window for CDPP fallback (mag)
CDPP_MAG_HALF_WINDOW = 0.25

# =============================================================================
# TESS Configuration
# =============================================================================

# BJD reference for TESS timestamps
BJDREF = 2457000.0

# Maximum sector to include (data up to and including this sector)
MAX_SECTOR = 96

# Cadence priority (in seconds): prefer shorter cadence
# Priority 1: SPOC 2-min = 120s
# Priority 2: QLP (each sector has only one cadence: 200s, 600s, or 1800s)
CADENCE_PRIORITY = [120, 200, 600, 1800]

CADENCE_NAMES = {
    120: "2min-SPOC",
    200: "200s-QLP",
    600: "10min-QLP",
    1800: "30min-QLP"
}

# Long cadence threshold for exposure time integration (seconds)
# Cadences above this use supersampling in batman model
LONG_CADENCE_THRESHOLD = 200.0

# Batman model supersampling parameters
BATMAN_MIN_SUPERSAMPLE = 5  # Minimum supersampling factor
BATMAN_SUPERSAMPLE_REF_CADENCE = 120  # Reference cadence (seconds) for computing supersample factor

# =============================================================================
# MCMC Configuration - Phase-Folded Fitting
# =============================================================================

# Number of walkers (should be at least 2 * ndim)
N_WALKERS = 64

# Burn-in steps
N_BURN = 4000  # Default burn-in for run_full_analysis.py
N_BURN_MIN = 2000

# Production steps
N_STEPS_MIN = 5000
N_STEPS_MAX = 25000

# Convergence criteria
CONVERGENCE_RHAT = 1.01  # Gelman-Rubin statistic threshold
CONVERGENCE_ESS = 1000  # Minimum effective sample size
CONVERGENCE_AUTOCORR = 50  # Minimum autocorrelation times

# Check convergence every N steps
CONVERGENCE_CHECK_INTERVAL = 500

# Minimum steps before computing R-hat (need enough samples to split chains)
CONVERGENCE_MIN_STEPS = 4

# Sokal window parameter for autocorrelation time estimation
SOKAL_WINDOW_PARAM = 5.0

# Bad walker detection: walkers with median log-prob < median - N*sigma are re-initialized
BAD_WALKER_SIGMA = 5

# MCMC percentiles for parameter estimation (16th, 50th, 84th = 1-sigma)
MCMC_PERCENTILES = (15.87, 50, 84.13)

# =============================================================================
# MCMC Configuration - Individual Transit Fitting
# =============================================================================

# Individual transit MCMC parameters (smaller than phase-folded due to 3 params)
N_WALKERS_INDIVIDUAL = 16
N_BURN_INDIVIDUAL = 500
N_STEPS_MIN_INDIVIDUAL = 500
N_STEPS_MAX_INDIVIDUAL = 10000
CONVERGENCE_RHAT_INDIVIDUAL = 1.01  # R-hat threshold for individual transit fitting
CONVERGENCE_CHECK_INTERVAL_INDIVIDUAL = 200

# =============================================================================
# MCMC Prior Bounds - Phase-Folded Fitting
# =============================================================================

# Period bounds (fraction of prior value)
PERIOD_BOUND_FRACTION = 0.025  # ±2.5% of prior period

# T0 bounds (fraction of period) for phase-folded fitting
T0_BOUND_FRACTION = 0.025  # ±2.5% of period

# Rp/Rs bounds
RP_RS_MIN = 0.001
RP_RS_MAX = 0.5

# a/Rs bounds
A_RS_MIN = 1.0
A_RS_MAX = 500.0

# Impact parameter bounds
B_MIN = 0.0
B_MAX = 1.0

# Baseline flux bounds
BASELINE_MIN = 0.9
BASELINE_MAX = 1.1

# =============================================================================
# MCMC Prior Widths - Phase-Folded Fitting
# =============================================================================

# Gaussian prior widths (as fraction of prior value)
PERIOD_PRIOR_WIDTH = 0.001  # 0.1% of period
T0_PRIOR_WIDTH = 0.01  # 1% of period (fallback when T0 error not in catalog)
RP_RS_PRIOR_WIDTH = 0.50  # 50% of Rp/Rs value
A_RS_PRIOR_WIDTH = 0.50  # 50% of a/Rs value

# =============================================================================
# MCMC Walker Initialization - Phase-Folded Fitting
# =============================================================================

# Walker initialization scatter (fraction or absolute)
WALKER_INIT_RP_RS_FRAC = 0.25  # ±25% of Rp/Rs
WALKER_INIT_A_RS_FRAC = 0.25  # ±25% of a/Rs
WALKER_INIT_BASELINE_FRAC = 0.001  # ±0.1%
WALKER_INIT_LD_WIDTH_U1 = 0.15  # ±0.15 for u1
WALKER_INIT_LD_WIDTH_U2 = 0.10  # ±0.10 for u2

# =============================================================================
# MCMC Configuration - Individual Transit Fitting Bounds
# =============================================================================

# T_mid search window (fraction of period)
T_MID_WINDOW_FRACTION = 0.05  # ±5% of orbital period

# Baseline bounds for individual transits
BASELINE_MIN_INDIVIDUAL = 0.9
BASELINE_MAX_INDIVIDUAL = 1.1

# Slope bounds (flux per day)
SLOPE_MAX = 0.1  # ±10%/day

# Walker initialization for individual transits
WALKER_INIT_T_MID_DAYS = 0.01  # ±0.01 days
WALKER_INIT_BASELINE_INDIVIDUAL = 0.01  # ±1%
WALKER_INIT_SLOPE = 0.001

# =============================================================================
# Limb Darkening Configuration
# =============================================================================

# Gaussian prior widths for limb darkening coefficients
LD_PRIOR_WIDTH_U1 = 0.15
LD_PRIOR_WIDTH_U2 = 0.10

# Whether to fix limb darkening to theoretical values (not fitted)
FIX_LD = False

# Limb darkening physical bounds (quadratic law constraints)
LD_U1_MIN = 0.0
LD_U1_MAX = 1.0
LD_U2_MIN = -0.5
LD_U2_MAX = 1.0
LD_SUM_MAX = 1.0  # u1 + u2 must be less than this
LD_U1_2U2_MIN = 0.0  # u1 + 2*u2 must be >= this (Kipping 2013: intensity must not increase toward limb)

# =============================================================================
# Transit Analysis Configuration
# =============================================================================

# Catalog error to prior sigma multiplier
# Prior sigma = CATALOG_ERROR_PRIOR_MULTIPLIER × catalog error
CATALOG_ERROR_PRIOR_MULTIPLIER = 2.0

# Transit/eclipse phase mask widths (fraction of orbital phase)
TRANSIT_MASK_PHASE_WIDTH = 0.15  # Half-width of transit mask in phase
ECLIPSE_MASK_PHASE_WIDTH = 0.15  # Half-width of eclipse mask in phase

# Transit identification coverage factor
# Requires coverage_factor × duration on each side of mid-transit
TRANSIT_COVERAGE_FACTOR = 1.5

# Outlier filtering thresholds (in units of sigma)
OUTLIER_SIGMA_T0_ERR = 10.0   # Filter transits with t0_err > N × median error
OUTLIER_SIGMA_OC = 10.0       # Filter O-C outliers > N sigma from median

# Fallback transit half-duration in phase units (when computation fails)
TRANSIT_HALF_DURATION_PHASE_FALLBACK = 0.05

# In-transit region clamp bounds (phase units) for residual statistics
IN_TRANSIT_HALF_WIDTH_MIN = 0.02
IN_TRANSIT_HALF_WIDTH_MAX = 0.10

# In-transit region multiplier (applied to transit half-duration)
IN_TRANSIT_DURATION_MULTIPLIER = 1.5

# Binning width for residual statistics and plots (minutes)
BIN_WIDTH_MINUTES = 10

# Data window for individual transit fitting (multiplier of transit duration)
# Total window = 4 * duration (2 before, 2 after mid-transit)
TRANSIT_WINDOW_MULTIPLIER = 2.0

# Transit window phase limits (half-window clamping for phase-folded fitting)
# These constrain the half-window size in phase units
TRANSIT_WINDOW_PHASE_MIN = 0.02  # Minimum half-window: 2% of orbital phase
TRANSIT_WINDOW_PHASE_MAX = 0.30  # Maximum half-window: 30% of orbital phase

# Phase bins for binned phase-folded light curve plot
N_PHASE_BINS = 200

# Minimum number of transits required for basic analysis
MIN_TRANSITS = 3

# Minimum number of transits for periodogram analysis
MIN_TRANSITS_PERIODOGRAM = 5

# =============================================================================
# TTV Detection Criteria
# =============================================================================

# Criterion 1: Quadratic ephemeris preference (ΔBIC > threshold)
TTV_DELTA_BIC_THRESHOLD = 6.0  # Strong evidence for quadratic ephemeris

# Criterion 2: Periodogram FAP threshold for significant periodicity
TTV_FAP_THRESHOLD = 0.01  # 1% false alarm probability

# Criterion 3: O-C scatter significance (RMS / mean_error > threshold)
TTV_OC_RMS_OVER_ERR_THRESHOLD = 2.0  # O-C RMS must exceed 2× mean timing error (relaxed from 3.0 on 2026-06-12 to match the C3>=2 LOO admission rule used for the canonical candidate list)

# Phase range for phase-folded model plotting
PHASE_PLOT_RANGE = 0.15  # -0.15 to +0.15 phase units

# Number of points for model plotting
N_MODEL_POINTS = 1000

# =============================================================================
# Periodogram Configuration
# =============================================================================

# Oversampling factor for frequency grid
# n_frequencies = oversampling × time_span × (f_nyquist - f_min)
PERIODOGRAM_OVERSAMPLING = 10
BOOTSTRAP_FAP_N_ITERATIONS = 100000  # Number of bootstrap iterations for empirical FAP

# Default Nyquist frequency when it cannot be computed
DEFAULT_NYQUIST_FREQ = 1.0

# Default minimum frequency for periodogram
DEFAULT_MIN_FREQ = 0.001

# Number of significant peaks to identify
N_SIGNIFICANT_PEAKS = 5

# FAP threshold for peak significance in peak finding
PEAK_FAP_THRESHOLD = 0.01

# =============================================================================
# Data Quality Thresholds
# =============================================================================

# Default flux error when not provided (as fraction of flux)
DEFAULT_FLUX_ERROR = 0.001  # 0.1%

# Minimum number of data points for a valid light curve
MIN_POINTS_LIGHTCURVE = 10

# Minimum number of data points for a single transit window
MIN_POINTS_TRANSIT = 10

# Cadence-dependent minimum points for individual transit fitting
MIN_POINTS_30MIN = 3   # 30-min cadence (≥1800s)
MIN_POINTS_10MIN = 6   # 10-min cadence (≥600s)
MIN_POINTS_2MIN = 9    # 200s or 2-min cadence

# Fraction of transit duration required on each side of mid-transit
TRANSIT_COVERAGE_FRACTION = 0.5

# Default cadences when not detected (seconds)
DEFAULT_CADENCE_SPOC = 120
DEFAULT_CADENCE_QLP = 1800

# =============================================================================
# Model Comparison Thresholds
# =============================================================================

# BIC threshold for ephemeris model selection
# ΔBIC > 2 suggests meaningful improvement
DELTA_BIC_MODEL_SELECTION = 2.0

# False Alarm Probability threshold for periodogram peaks
FAP_THRESHOLD = 0.01

# =============================================================================
# Physical Constants and Defaults
# =============================================================================

# Time conversion constants
SECONDS_PER_DAY = 86400
MILLISECONDS_PER_DAY = 86400000
SECONDS_PER_YEAR = 365.25 * SECONDS_PER_DAY
DAYS_PER_YEAR = 365.25

# Default FITS TIMEDEL value (120 seconds expressed in days)
DEFAULT_TIMEDEL_DAYS = 0.00138889

# Solar values for default fallback
SOLAR_TEFF = 5778.0  # K
SOLAR_LOGG = 4.44    # cgs
SOLAR_RADIUS = 1.0   # R_sun
SOLAR_MASS = 1.0     # M_sun

# Default eccentricity and argument of periastron (fixed in fitting)
DEFAULT_ECCENTRICITY = 0.0
DEFAULT_OMEGA = 90.0  # degrees

# Radius conversion
R_EARTH_TO_R_SUN = 0.00916794  # R_Earth / R_Sun

# Kepler's 3rd law constant for computing a/Rs
# a/Rs = (G * M * P^2 / (4 * pi^2))^(1/3) / Rs
# This constant = (G / (4 * pi^2))^(1/3) in appropriate units
KEPLER_CONSTANT = 4.208  # For P in days, M in M_sun, Rs in R_sun

# Default transit parameters when catalog values are missing
DEFAULT_A_RS = 10.0  # Semi-major axis in stellar radii
DEFAULT_IMPACT_PARAMETER = 0.3  # Impact parameter
DEFAULT_TRANSIT_DURATION_HR = 3.0  # Transit duration in hours

# a/Rs clipping bounds (for initial estimates)
A_RS_CLIP_MIN = 2.0
A_RS_CLIP_MAX = 200.0

# =============================================================================
# Webapp Export Binning Thresholds
# =============================================================================

# Period bins (days) for categorization
PERIOD_BIN_ULTRA_SHORT = 1.0  # < 1 day
PERIOD_BIN_SHORT = 10.0  # 1-10 days
PERIOD_BIN_MEDIUM = 100.0  # 10-100 days

# Transit count bins for categorization
TRANSIT_BIN_FEW = 10  # < 10 transits
TRANSIT_BIN_MODERATE = 50  # 10-50 transits

# =============================================================================
# Output Configuration
# =============================================================================

# Plot format and DPI
PLOT_FORMAT = "png"
PLOT_DPI = 150

# JSON indent for readability
JSON_INDENT = 2

