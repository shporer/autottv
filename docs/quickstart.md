# Quick start

This page runs one TOI end to end and explains what came out. Budget a few minutes for a
short-period target with a handful of sectors; a target with dozens of sectors and
100,000+ cadences takes considerably longer.

## Run one TOI

```bash
python run_full_analysis.py 105.01
```

That is the whole invocation. The TOI number is the only required argument. Everything
else — the catalog period and epoch, the stellar parameters, which sectors exist — is
looked up automatically.

To use fewer cores than the default:

```bash
python run_full_analysis.py 105.01 --cpus=4
```

The default is `cpu_count() - 1` on machines with 16 or fewer cores, and 16 on anything
larger. It never grabs every core.

## What happens

```
Step 0   Query MAST, download and cache every sector's light curve
         Prefer SPOC 2-min; fall back to QLP FFI (200 s / 600 s / 1800 s)
         Mask sibling planets, normalize each sector, identify full transits

Step 1   Phase-fold MCMC — 64 walkers, up to 4,000 burn-in and 25,000 production steps,
         convergence checked every 500 steps

Step 2   One MCMC per transit — 16 walkers, 3 free parameters (T_mid, baseline, slope),
         transit shape held fixed at the Step 1 posterior median

Step 3   Weighted least squares of a linear and a quadratic ephemeris; ΔBIC decides

Step 4   Lomb–Scargle periodogram of the O−C residuals, with a bootstrap FAP from
         100,000 label permutations
```

Progress is printed as it goes, including convergence diagnostics for Step 1 and a
per-transit tally for Step 2.

## What you get

Everything lands in `autottv_results_v2/TOI_105_01/`:

| File | Contents |
|---|---|
| `results.json` | **The output.** Fitted parameters, uncertainties, convergence diagnostics, every transit time, the ephemeris comparison, and the periodogram result. |
| `production_chains.npy` | Post-burn-in MCMC chains, shape `(n_steps, n_walkers, n_params)` |
| `burnin_chains.npy` | Burn-in chains, for diagnosing initialization problems |
| `phase_folded_lightcurve.png` | The stacked transit with the best-fit model |
| `individual_transits.png` | Every transit with its own fit |
| `oc_diagram.png` | O−C residuals against the linear ephemeris |
| `oc_linear_ephemeris.png`, `oc_quadratic_ephemeris.png` | The two ephemeris fits |
| `periodogram.png` | Lomb–Scargle power spectrum of the O−C series |
| `chain_plot.png`, `corner_plot.png` | MCMC diagnostics |
| `full_phase_curve.png` | Full orbital phase, useful for spotting secondary eclipses |

## Reading `results.json`

The first three things worth checking:

```python
import json
r = json.load(open("autottv_results_v2/TOI_105_01/results.json"))

# 1. Did it converge?
print(r["convergence"]["converged"], r["convergence"]["summary"]["max_rhat"])

# 2. The transit shape
p = r["parameters"]
print(p["rp_rs"]["value"], "+", p["rp_rs"]["err_upper"], "-", p["rp_rs"]["err_lower"])

# 3. Is there a TTV signal?
print("delta_BIC :", r["ephemeris"]["delta_bic"])          # > 6  → quadratic favoured
print("FAP       :", r["periodogram"]["bootstrap_fap"])     # < 0.01 → periodic signal
print("O-C rms/err:", r["individual_transits"]["oc_rms_over_mean_err"])  # > 2 → excess scatter
```

Those three numbers are the [three detection criteria](statistics.md#the-three-detection-criteria).
A TOI that trips any of them is a TTV candidate, subject to the leave-one-out validation
described there.

!!! warning "Convergence is not automatic"
    The MCMC stopping condition requires $\hat{R} < 1.01$, and when that threshold is not 
    reached the convergence is officially considered as failed (`converged: false` in 
    `results.json` file). Although, larger values of $\hat{R}$ can be considered 
    satisfactory for convergence. The published analysis retains fits with $\hat{R}$ up to 1.1.
    

## Useful flags

```bash
# Hold limb darkening at the theoretical Claret values instead of fitting it
python run_full_analysis.py 105.01 --fix-ld

# Override the catalog ephemeris (e.g. after a BLS re-derivation)
python run_full_analysis.py 105.01 --period=3.5 --t0=2458325.5

# Restrict to specific sectors
python run_full_analysis.py 105.01 --sectors=1,28

# Ignore the light-curve cache and re-download
python run_full_analysis.py 105.01 --no-cache
```

The full flag list is in [Running the pipeline](running.md#command-line-flags).

## Next

- [Running the pipeline](running.md) — batch mode, the secondary fitters, resource planning
- [What the code does](pipeline.md) — each step in detail
- [Output files](outputs.md) — the complete `results.json` schema and table columns
