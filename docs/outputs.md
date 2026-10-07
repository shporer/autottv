# Output files

## Per-TOI directory

Each run writes `autottv_results_v2/TOI_<X>/`, with the TOI's dot replaced by an
underscore — TOI 125.01 becomes `TOI_125_01`.

```
autottv_results_v2/TOI_125_01/
├── results.json                    # the output — everything else is derived
├── production_chains.npy           # (n_steps, n_walkers, n_params)
├── burnin_chains.npy
├── production_log_prob.npy
├── burnin_log_prob.npy
├── phase_folded_lightcurve.png
├── individual_transits.png
├── oc_diagram.png
├── oc_linear_ephemeris.png
├── oc_quadratic_ephemeris.png
├── periodogram.png
├── chain_plot.png
├── corner_plot.png
├── full_phase_curve.png
│
├── refined_transit/                # refined fit, iteration 0 (TTV candidates only)
├── refined_strict_iter1/           # later iterations of the refined fit
├── refined_strict_iter2/
├── iter_cascade_summary.json       # the iterations and the adopted one
├── joint/  joint_fixld/            # secondary fitters, if run
└── sinusoidal_ttv_joint/           # sinusoidal ephemeris fit
```

## The `results.json` schema

```
toi                     "125.01"
tic_id                  52368076
qlp_time_fix            with --qlp-time-fix only: version, and the QLP timestamp
                        corrections applied (sector, orbit, model, n_points, shifts)
sectors                 [1, 2, 28, 68, 69, 95, 96]
n_points_total          106367

normalization           per-sector dict, keyed by sector number
  └─ <sector>           oot_level, n_oot, n_transit, n_eclipse, n_total

parameters              per-parameter dict
  └─ period, t0, rp_rs, a_rs, b, baseline, u1, u2
       value            posterior median
       err_lower        median − 15.87th percentile
       err_upper        84.13th percentile − median
       err              symmetrized, (p84 − p16) / 2
       percentile_16    15.87th percentile
       percentile_84    84.13th percentile

convergence
  converged             1.0 if all three criteria passed
  message               which parameters failed, if any
  criteria              the thresholds in force
  summary               max_rhat, min_ess, max_autocorr_time, chain_size,
                        chain_over_autocorr, acceptance_rate
  per_parameter         rhat, autocorr_time, ess for each parameter

bad_walkers             n_bad_walkers, bad_walker_indices

mcmc_settings           n_walkers, n_burn, n_steps_max

binned_residuals        reduced_chi2, rms_ppm, median_bin_err_ppm,
                        rms_over_median_err, neumann_ratio, alarm_statistic,
                        in_transit_rms_ppm, out_of_transit_rms_ppm, ...

individual_transits
  n_transits_full       full transits identified
  n_transits_used       after filtering
  n_unconverged         unconverged fits among the used transits (0: they are dropped)
  n_unconverged_dropped transits dropped because their fit did not reach R-hat ≤ 1.01
  unconverged_transits_excluded[]
                        epoch, t_expected, t0_fit, t0_err, max_rhat, n_points of each
  transit_times[]       epoch, t_expected, t0_fit, t0_err, oc,
                        baseline_fit, slope_fit, n_points, mcmc_diagnostics
  oc_values[]           epoch, oc_minutes, oc_err_minutes
  oc_rms_minutes        weighted rms of the O−C about the weighted mean
  oc_median_err_minutes ← Scatter criterion: oc_rms_minutes / oc_median_err_minutes
  oc_rms_over_mean_err  legacy ratio over the mean error, not the criterion

ephemeris
  linear                T0, P and errors
  quadratic             T0, P, Q, dPdE and errors
  delta_bic             ← Quadratic criterion

periodogram             null with fewer than 5 transits or an empty frequency range
  peak_frequency, peak_period, peak_period_error, peak_power
  bootstrap_fap         ← Periodic criterion (also as peak_fap)
  bootstrap_n_iter, n_exceed
  n_freq, freq_min, freq_max   the frequency grid (from version 1.0.0)
```

!!! tip "The three numbers that matter"
    `ephemeris.delta_bic`, `periodogram.bootstrap_fap`, and the ratio
    `individual_transits.oc_rms_minutes / oc_median_err_minutes` are the
    [three detection criteria](statistics.md#the-three-detection-criteria).
    `find_ttv_candidates.py` recomputes the last one from `oc_values`.

### Reading the chains

```python
import numpy as np
chain = np.load("autottv_results_v2/TOI_125_01/production_chains.npy")
print(chain.shape)          # (n_steps, n_walkers, n_params)
flat = chain.reshape(-1, chain.shape[-1])
```

Parameter order matches `PARAM_NAMES` in `run_full_analysis.py`. Note that index 4 is
**$b^2$, not $b$** — take the square root before interpreting it.

---

## Summary tables

Built by `build_all_transit_times.py` and `build_fit_params_tables.py`.

!!! note
    These two scripts are not part of this repository; the tables are described here for
    reference. The README's "Reproducing the paper" section describes how the published
    tables were assembled.

### `tables/transit_times.csv`

The publishable transit-time catalog: `used == True` rows only.

| Column | Meaning |
|---|---|
| `TOI` | TOI number |
| `TIC_ID` | TESS Input Catalog identifier |
| `Epoch` | integer epoch relative to the reference ephemeris |
| `T_mid_BJD_TDB` | mid-transit time |
| `T_mid_err_days` | 1σ uncertainty |

For TTV candidates these times come from the **final iteration of the TTV-aware refit**, so
the shape template used to extract them is itself free of timing smearing.

### `tables/all_transit_times.csv`

The diagnostic version: 17 columns adding O−C, per-transit baseline and slope, $\hat{R}$,
the convergence flag, and a source tag recording which fit produced the time.

### `tables/fit_params_*.csv`

One table per detection class plus one for non-candidates. Columns:

| Column | Meaning |
|---|---|
| `TOI`, `TIC_ID` | identifiers |
| `period`, `period_err` | linear ephemeris from the shape fit |
| `T0`, `T0_err` | its epoch |
| `rp_rs`, `a_rs`, `b`, `u1`, `u2` and errors | transit shape |
| `fix_ld` | whether limb darkening was held fixed |
| `T0_quad`, `P_quad`, `dPdE` and errors | quadratic ephemeris (quadratic candidates) |

!!! warning "Two different periods"
    For quadratic candidates, `period`/`T0` are the shape fit's own linear ephemeris while
    `P_quad`/`T0_quad` are the quadratic one. They differ — for TOI 125.01, by 14 seconds
    in period and 19 minutes in epoch. The published tables use the **quadratic** values
    for quadratic candidates. Check which you are reading.

### Source priority

For TTV candidates, values are taken from the highest available
`refined_strict_iter{N}`, falling back to `refined_transit_strict` (iteration 0) and then
to the standard fit. For non-candidates the source is always the standard pipeline.


---

## Rebuilding

```bash
python build_all_transit_times.py
python build_fit_params_tables.py
```

Both are idempotent and re-read every per-TOI `results.json`, so they can be run after any
partial re-analysis. (Neither is part of this repository; see the note under
[Summary tables](#summary-tables).)
