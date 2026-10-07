# Configuration

Everything tunable lives in `autottv_pipeline_v2/config.py`. Values below are the defaults
used for the published analysis. Changing them changes the science, so the notes matter as
much as the numbers.

---

## Catalog filtering

Applied when building the filtered TOI catalog.

| Key | Default | Meaning |
|---|---|---|
| `EXCLUDED_DISPOSITIONS` | `['FP', 'FA']` | Drop false positives and false alarms; keep PC, CP, KP, APC |
| `MIN_PERIOD_DAYS` | `0.0` | Excludes period-zero entries, i.e. single-transit detections |
| `MIN_TRANSIT_SNR` | `5.0` | Minimum expected single-transit SNR |
| `MIN_FULL_TRANSITS` | `5` | Minimum full transits in TESS data |
| `CDPP_MAG_HALF_WINDOW` | `0.25` | Magnitude half-window for the CDPP proxy used on QLP-only targets |

The SNR cut uses SPOC CDPP where available. For QLP-only targets there is no header CDPP,
so the pipeline substitutes the clipped mean CDPP of SPOC-observed TOIs within ±0.25 mag.

## Data retrieval

| Key | Default | Meaning |
|---|---|---|
| `MAX_SECTOR` | `96` | Highest sector included |
| `BJDREF` | `2457000.0` | TESS BJD reference offset |
| `CADENCE_PRIORITY` | `[120, 200, 600, 1800]` | Prefer shorter cadence, per sector |
| `LONG_CADENCE_THRESHOLD` | `200.0` | Cadences above this are supersampled in `batman` |
| `BATMAN_MIN_SUPERSAMPLE` | `5` | Minimum supersampling factor |
| `TRANSIT_MASK_PHASE_WIDTH` | `0.15` | Transit mask half-width for normalization |
| `ECLIPSE_MASK_PHASE_WIDTH` | `0.15` | Secondary-eclipse mask half-width |
| `TRANSIT_COVERAGE_FACTOR` | `1.5` | Durations of coverage required each side for a *full* transit |

!!! note "Supersampling"
    A 30-minute exposure smears the transit shape by integrating over a substantial
    fraction of ingress. Without supersampling the fit compensates by inflating $b$ — the
    same degeneracy discussed in [Physical background](physics.md#the-b-k-degeneracy).

## MCMC — phase fold

| Key | Default | Meaning |
|---|---|---|
| `N_WALKERS` | `64` | Ensemble size |
| `N_BURN` | `4000` | Burn-in steps |
| `N_STEPS_MAX` | `25000` | Production cap |
| `CONVERGENCE_RHAT` | `1.01` | $\hat{R}$ threshold, every parameter |
| `CONVERGENCE_ESS` | `1000` | Minimum effective sample size |
| `CONVERGENCE_AUTOCORR` | `50` | Minimum chain length in $\tau$ |
| `CONVERGENCE_CHECK_INTERVAL` | `500` | Steps between checks |
| `SOKAL_WINDOW_PARAM` | `5.0` | $c$ in the automated windowing |
| `BAD_WALKER_SIGMA` | `5` | Robust-sigma cut for re-initializing stranded walkers |
| `MCMC_PERCENTILES` | `(15.87, 50, 84.13)` | Reported quantiles |

## MCMC — individual transits

| Key | Default |
|---|---|
| `N_WALKERS_INDIVIDUAL` | `16` |
| `N_BURN_INDIVIDUAL` | `500` |
| `N_STEPS_MAX_INDIVIDUAL` | `10000` |
| `CONVERGENCE_RHAT_INDIVIDUAL` | `1.01` |
| `T_MID_WINDOW_FRACTION` | `0.05` |
| `SLOPE_MAX` | `0.1` |

## Priors and bounds

| Key | Default | Applies to |
|---|---|---|
| `CATALOG_ERROR_PRIOR_MULTIPLIER` | `2.0` | $\sigma$ = 2× catalog error for $P$ and $T_0$ |
| `PERIOD_BOUND_FRACTION` | `0.025` | Hard bound: ±2.5% of catalog period |
| `T0_BOUND_FRACTION` | `0.025` | Hard bound: ±2.5% of a period |
| `RP_RS_MIN` / `RP_RS_MAX` | `0.001` / `0.5` | $k$ bounds |
| `A_RS_MIN` / `A_RS_MAX` | `1.0` / `500.0` | $a/R_\star$ bounds |
| `BASELINE_MIN` / `BASELINE_MAX` | `0.9` / `1.1` | baseline bounds |
| `RP_RS_PRIOR_WIDTH` | `0.50` | Gaussian width, 50% of value |
| `A_RS_PRIOR_WIDTH` | `0.50` | Gaussian width, 50% of value |

The impact parameter is sampled as $b^2$ on $[0, (1+k)^2]$ — set in the log-probability,
not in `config.py`. The `B_MIN`/`B_MAX` keys are legacy.

### Limb darkening

| Key | Default |
|---|---|
| `LD_PRIOR_WIDTH_U1` | `0.15` |
| `LD_PRIOR_WIDTH_U2` | `0.10` |
| `FIX_LD` | `False` |
| `LD_U1_MIN` / `LD_U1_MAX` | `0.0` / `1.0` |
| `LD_U2_MIN` / `LD_U2_MAX` | `-0.5` / `1.0` |
| `LD_SUM_MAX` | `1.0` |
| `LD_U1_2U2_MIN` | `0.0` |

Prior centres come from the Claret (2017) TESS-band tables, interpolated in $T_{\rm eff}$
and $\log g$ at solar metallicity. The widths are set from the observed scatter between
theoretical and empirically measured coefficients, so metallicity-induced offsets are
absorbed rather than modelled. The last two constraints keep the intensity profile
physical — brightness must not increase toward the limb.

## TTV detection

| Key | Default | Criterion |
|---|---|---|
| `TTV_DELTA_BIC_THRESHOLD` | `6.0` | Quadratic |
| `TTV_FAP_THRESHOLD` | `0.01` | Periodic |
| `TTV_OC_RMS_OVER_ERR_THRESHOLD` | `2.0` | Scatter |
| `MIN_TRANSITS` | `3` | Minimum for basic analysis |
| `MIN_TRANSITS_PERIODOGRAM` | `5` | Minimum for Step 4 |
| `OUTLIER_SIGMA_T0_ERR` | `10.0` | Timing-uncertainty cut |
| `OUTLIER_SIGMA_OC` | `10.0` | O−C outlier cut |

!!! note
    `TTV_OC_RMS_OVER_ERR_THRESHOLD` was relaxed from 3.0 to 2.0 to match the admission rule
    used for the canonical candidate list. Raising it back would shrink the Scatter class.

## Periodogram

| Key | Default | Meaning |
|---|---|---|
| `PERIODOGRAM_OVERSAMPLING` | `10` | Frequency-grid oversampling |
| `BOOTSTRAP_FAP_N_ITERATIONS` | `100000` | Label permutations for the empirical FAP |
| `N_SIGNIFICANT_PEAKS` | `5` | Peaks reported |
| `PEAK_FAP_THRESHOLD` | `0.01` | Significance for peak finding |

Bootstrap iterations dominate Step 4's runtime. Lowering the count speeds things up at the
cost of the FAP floor: $10^5$ permutations can resolve a FAP of $10^{-5}$, and no lower.

## Data quality

| Key | Default | Meaning |
|---|---|---|
| `DEFAULT_FLUX_ERROR` | `0.001` | Fallback flux error when absent |
| `MIN_POINTS_LIGHTCURVE` | `10` | Minimum for a usable light curve |
| `MIN_POINTS_TRANSIT` | `10` | Minimum in a transit window |
| `MIN_POINTS_30MIN` | `3` | Cadence-dependent minima for per-transit fits |
| `MIN_POINTS_10MIN` | `6` | |
| `MIN_POINTS_2MIN` | `9` | |
| `TRANSIT_COVERAGE_FRACTION` | `0.5` | Duration fraction required each side |
| `BIN_WIDTH_MINUTES` | `10` | Binning for residual statistics and plots |

## Defaults and constants

| Key | Default | Notes |
|---|---|---|
| `SOLAR_TEFF` / `SOLAR_LOGG` | `5778.0` / `4.44` | Fallback when catalog stellar parameters are missing |
| `DEFAULT_ECCENTRICITY` | `0.0` | **Fixed, not fitted** |
| `DEFAULT_OMEGA` | `90.0` | Argument of periastron, degrees |
| `DEFAULT_A_RS` | `10.0` | Fallback when the catalog lacks it |
| `A_RS_CLIP_MIN` / `A_RS_CLIP_MAX` | `2.0` / `200.0` | Clipping on initial estimates |
| `KEPLER_CONSTANT` | `4.208` | For $a/R_\star$ from $P$, $M_\star$, $R_\star$ |

!!! warning "Circular orbits are assumed"
    Eccentricity is fixed at zero throughout. For an eccentric orbit the transit duration
    changes and $a/R_\star$ absorbs the difference, so a fitted $a/R_\star$ is really
    $a/R_\star$ times a factor involving $e$ and $\omega$. This does not affect $k$ much,
    but it does mean a stellar density derived from $a/R_\star$ will be wrong for an
    eccentric planet.

## Output

| Key | Default |
|---|---|
| `PLOT_FORMAT` | `"png"` |
| `PLOT_DPI` | `150` |
| `JSON_INDENT` | `2` |
| `N_PHASE_BINS` | `200` |

---

## Changing a threshold safely

Most keys are read at import, so editing `config.py` and re-running is enough. Two cautions:

- **Detection thresholds change the candidate list**, so the summary tables need rebuilding
  afterwards, and any previously computed leave-one-out verdicts no longer apply.
- **MCMC settings do not invalidate cached light curves** but do invalidate every fit.
  Delete the affected `autottv_results_v2/TOI_<X>/` directories rather than letting a
  partially updated set accumulate.
