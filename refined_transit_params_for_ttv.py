#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Refined transit parameters for TTV systems.

For systems with detected TTVs, the standard phase-folded fit uses a linear
ephemeris which smears the transit due to timing variations. This script
fixes the individual transit times (from Step 2) and fits for the transit
shape parameters: Rp/Rs, a/Rs, b, u1, u2.

Algorithm:
  1. Load individual transit times from results.json
  2. For each transit, extract data window and normalize by OOT baseline
  3. Phase-fold all transits using individual T0s (not linear ephemeris)
  4. Fit the stacked data for: Rp/Rs, a/Rs, b^2, baseline, u1, u2
  5. Re-time every transit with the new shape, and repeat 3-4 on the re-timed
     transits until the shape settles (run_refined_iterations)

Usage:
    python refined_transit_params_for_ttv.py <TOI> [--cpus=N] [--fix-ld] [--max-iters=N]
        [--results-root=DIR]
    python refined_transit_params_for_ttv.py 924.01 --cpus=15

--results-root reads and writes TOI_*/ under DIR instead of autottv_results_v2
(use it with run_full_analysis.py --results-root=DIR for test runs).

The command line runs run_refined_iterations: the refined fit, then up to
five iterations of re-timing and refitting, stopping once Rp/Rs, a/Rs and b
each move by less than 1 sigma. --max-iters=0 runs the refined fit and one
re-timing only.
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import sys
import batman
import emcee
from pathlib import Path
from multiprocessing import Pool, cpu_count

sys.path.insert(0, str(Path(__file__).parent))

from autottv_pipeline_v2.data_loader import DataLoader, load_toi_catalog, get_toi_parameters
from autottv_pipeline_v2.convergence import compute_rhat_split
from autottv_pipeline_v2.individual_transit_fitter import IndividualTransitFitter as ModularIndividualTransitFitter
from autottv_pipeline_v2 import config
from run_full_analysis import normalize_to_oot_no_eclipse, identify_transits

RESULTS_ROOT_DEFAULT = 'autottv_results_v2'

# Module-level shared data for multiprocessing
_SHARED = {}
_CACHED = {}


def _init_worker(shared):
    global _SHARED, _CACHED
    _SHARED = shared
    _CACHED = {}


def _log_probability(theta):
    """Log probability for transit shape fit on TTV-corrected stacked data.

    Parameters: Rp/Rs, a/Rs, b^2, baseline, [u1, u2]
    """
    if _SHARED['fix_ld']:
        rp_rs, a_rs, b_sq, baseline = theta
        u1 = _SHARED['u1_fixed']
        u2 = _SHARED['u2_fixed']
    else:
        rp_rs, a_rs, b_sq, baseline, u1, u2 = theta

    # Hard bounds
    if not (config.RP_RS_MIN < rp_rs < config.RP_RS_MAX): return -np.inf
    if not (config.A_RS_MIN < a_rs < config.A_RS_MAX): return -np.inf
    if not (config.B_MIN <= b_sq < (config.B_MAX + rp_rs)**2): return -np.inf
    if not (config.BASELINE_MIN < baseline < config.BASELINE_MAX): return -np.inf
    if not _SHARED['fix_ld']:
        if not (config.LD_U1_MIN < u1 < config.LD_U1_MAX): return -np.inf
        if not (config.LD_U2_MIN < u2 < config.LD_U2_MAX): return -np.inf
        if not (u1 + u2 < config.LD_SUM_MAX): return -np.inf
        if not (u1 + 2.0 * u2 >= config.LD_U1_2U2_MIN): return -np.inf

    b = np.sqrt(b_sq)

    # Gaussian priors on LD coefficients
    log_prior = 0.0
    if not _SHARED['fix_ld']:
        log_prior += -0.5 * ((u1 - _SHARED['u1_prior']) / _SHARED['u1_sigma'])**2
        log_prior += -0.5 * ((u2 - _SHARED['u2_prior']) / _SHARED['u2_sigma'])**2

    # Optional Gaussian priors on Rp/Rs and a/Rs (centered on iter0 values)
    if _SHARED.get('rp_rs_prior_mean') is not None:
        log_prior += -0.5 * ((rp_rs - _SHARED['rp_rs_prior_mean']) /
                              _SHARED['rp_rs_prior_sigma'])**2
    if _SHARED.get('a_rs_prior_mean') is not None:
        log_prior += -0.5 * ((a_rs - _SHARED['a_rs_prior_mean']) /
                              _SHARED['a_rs_prior_sigma'])**2

    # Transit model on the phase-folded time grid
    try:
        if 'bp' not in _CACHED:
            bp = batman.TransitParams()
            bp.per = _SHARED['period']
            bp.t0 = 0.0  # Phase-folded, transit at phase 0
            bp.rp = rp_rs
            bp.a = a_rs
            bp.inc = np.degrees(np.arccos(b / a_rs)) if a_rs > 0 and b / a_rs < 1 else 90.0
            bp.ecc = 0.0
            bp.w = 90.0
            bp.limb_dark = "quadratic"
            bp.u = [u1, u2]
            _CACHED['bp'] = bp
            _CACHED['model'] = batman.TransitModel(bp, _SHARED['phase_time'])
        else:
            bp = _CACHED['bp']
            bp.rp = rp_rs
            bp.a = a_rs
            bp.inc = np.degrees(np.arccos(b / a_rs)) if a_rs > 0 and b / a_rs < 1 else 90.0
            bp.u = [u1, u2]
        model_flux = _CACHED['model'].light_curve(bp) * baseline
    except:
        return -np.inf

    residuals = _SHARED['flux'] - model_flux
    chi2 = np.sum(residuals**2 * _SHARED['inv_var'])

    if not np.isfinite(chi2): return -np.inf
    return log_prior - 0.5 * chi2


def refined_transit_params(toi, n_cpus=15, fix_ld=False, include_rejected=False,
                            transit_times_override=None, output_subdir=None,
                            rp_rs_prior=None, a_rs_prior=None,
                            init_rp_rs=None, init_a_rs=None, init_b=None,
                            results_root=RESULTS_ROOT_DEFAULT):
    """
    Fit refined transit parameters using individual transit times.

    Parameters
    ----------
    toi : str
        TOI to analyze
    n_cpus : int
        CPUs for MCMC
    fix_ld : bool
        If True, fix limb darkening to theoretical values
    include_rejected : bool
        If True, include transits rejected by t0_err or O-C filters
    transit_times_override : dict, optional
        {epoch: t0_fit_BJD} mapping that replaces the standard transit_times
        from results.json. Used for iterated runs where the iterated T_mids
        from individual_refit_results.json should drive the stacking.
    output_subdir : str, optional
        Sub-folder name under TOI dir for outputs (default 'refined_transit').
        Use 'refined_transit_iter1' (etc.) for iteration runs.
    results_root : str
        Results tree that holds TOI_*/results.json and receives the outputs.
    """
    output_dir = Path(results_root) / f'TOI_{toi.replace(".", "_")}'
    results_path = output_dir / 'results.json'
    refined_dir = output_dir / (output_subdir or 'refined_transit')
    refined_dir.mkdir(exist_ok=True)

    with open(results_path) as f:
        results = json.load(f)

    tic_id = results['tic_id']
    sectors = results['sectors']
    params_std = results['parameters']

    period = params_std['period']['value']
    t0_ref = params_std['t0']['value']
    rp_rs_std = params_std['rp_rs']['value']
    a_rs_std = params_std['a_rs']['value']
    b_std = params_std['b']['value']
    u1_std = params_std['u1']['value']
    u2_std = params_std['u2']['value']

    print(f"  Standard fit: Rp/Rs={rp_rs_std:.4f}, a/Rs={a_rs_std:.2f}, "
          f"b={b_std:.3f}, u1={u1_std:.3f}, u2={u2_std:.3f}")

    # Load individual transit times
    transit_times = results.get('individual_transits', {}).get('transit_times', [])
    if not transit_times:
        print(f"  ERROR: No individual transit times found"); return None

    # Use only transits that were actually used (not excluded)
    used_epochs = set()
    oc_values = results.get('individual_transits', {}).get('oc_values', [])
    if oc_values:
        used_epochs = {oc['epoch'] for oc in oc_values}
    else:
        used_epochs = {tt['epoch'] for tt in transit_times}

    transit_info = {}
    for tt in transit_times:
        if tt['epoch'] in used_epochs:
            transit_info[tt['epoch']] = {
                't0': tt['t0_fit'],
                'baseline': tt.get('baseline_fit', 1.0),
                'slope': tt.get('slope_fit', 0.0),
                'rejected': False,
                'reject_reason': None,
            }

    # Include rejected transits if requested
    n_rejected_added = 0
    if include_rejected:
        indiv = results.get('individual_transits', {})
        for source, label in [('excluded_by_t0err', 't0_err'), ('excluded_by_oc', 'oc_outlier')]:
            for exc in indiv.get(source, []):
                ep = exc['epoch']
                if ep in transit_info:
                    continue
                # Use fitted t0 if available, otherwise compute from linear ephemeris
                if 't0_fit' in exc:
                    t0_val = exc['t0_fit']
                else:
                    t0_val = t0_ref + ep * period
                    if source == 'excluded_by_oc' and 'oc_minutes' in exc:
                        t0_val += exc['oc_minutes'] / (24 * 60)
                transit_info[ep] = {
                    't0': t0_val,
                    'baseline': exc.get('baseline_fit', 1.0),
                    'slope': exc.get('slope_fit', 0.0),
                    'rejected': True,
                    'reject_reason': label,
                }
                n_rejected_added += 1

    # Override t0 values from iterated individual_refit_results.json if provided
    if transit_times_override:
        n_overridden = 0
        for ep, info in transit_info.items():
            if ep in transit_times_override:
                info['t0'] = float(transit_times_override[ep])
                n_overridden += 1
        print(f"  Overrode {n_overridden} t0 values from transit_times_override")

    n_transits = len(transit_info)
    print(f"  {n_transits} transit times loaded")
    if n_rejected_added > 0:
        print(f"    (including {n_rejected_added} previously rejected transits)")

    # Load light curve data
    catalog = load_toi_catalog()
    loader = DataLoader(tic_id)
    if not loader.load_from_npz_cache():
        if not loader.download_from_mast():
            print(f"  Failed to load data"); return None

    combined_time, combined_flux, combined_flux_err, combined_cadence = [], [], [], []
    for lc in loader.lightcurves:
        if lc.sector in sectors:
            fn, en, _ = normalize_to_oot_no_eclipse(lc.time, lc.flux, lc.flux_err, period, t0_ref)
            combined_time.append(lc.time)
            combined_flux.append(fn)
            combined_flux_err.append(en)
            combined_cadence.append(np.full(len(lc.time), lc.cadence))

    time = np.concatenate(combined_time)
    flux = np.concatenate(combined_flux)
    flux_err = np.concatenate(combined_flux_err)
    cadence = np.concatenate(combined_cadence)
    sort_idx = np.argsort(time)
    time, flux, flux_err, cadence = time[sort_idx], flux[sort_idx], flux_err[sort_idx], cadence[sort_idx]

    # Auto-detect and mask sibling TOIs in multi-planet systems
    catalog = load_toi_catalog()
    toi_number = toi.split('.')[0]
    sibling_rows = catalog[catalog['TOI'].apply(lambda x: str(x).split('.')[0] == toi_number)]
    sibling_tois = [str(row['TOI']) for _, row in sibling_rows.iterrows()
                    if str(row['TOI']) != toi and not np.isnan(row.get('Period (days)', float('nan')))]
    if sibling_tois:
        print(f"  Multi-planet system: masking {len(sibling_tois)} sibling TOIs: {', '.join(sibling_tois)}")
        sibling_mask = np.ones(len(time), dtype=bool)
        for sib_toi in sibling_tois:
            sib_pp = get_toi_parameters(catalog, toi=sib_toi)
            if sib_pp is None: continue
            sib_period = sib_pp['period']
            sib_t0 = sib_pp['t0']
            sib_dur = sib_pp.get('duration_hr', 3.0) / 24.0
            sib_hw = 1.5 * sib_dur
            n_min = int(np.floor((time.min() - sib_t0) / sib_period))
            n_max = int(np.ceil((time.max() - sib_t0) / sib_period))
            n_masked = 0
            for n in range(n_min, n_max + 1):
                t_transit = sib_t0 + n * sib_period
                in_transit = np.abs(time - t_transit) < sib_hw
                sibling_mask &= ~in_transit
                n_masked += np.sum(in_transit)
            print(f"    Masked TOI {sib_toi}: {n_masked} points")
        time = time[sibling_mask]
        flux = flux[sibling_mask]
        flux_err = flux_err[sibling_mask]
        cadence = cadence[sibling_mask]
        print(f"    After masking: {len(time)} points")

    # Compute transit duration
    if b_std < (1 + rp_rs_std) and a_rs_std > 0:
        sin_arg = np.sqrt((1 + rp_rs_std)**2 - b_std**2) / a_rs_std
        duration_days = period / np.pi * np.arcsin(min(sin_arg, 1.0)) if sin_arg <= 1 else period / (np.pi * a_rs_std) * rp_rs_std
    else:
        duration_days = period / (np.pi * a_rs_std) * rp_rs_std if a_rs_std > 0 else 0.01

    coverage_hw = 1.5 * duration_days  # 1.5x duration on each side

    # Extract and phase-fold each transit using individual T0s
    # Normalize each transit by its fitted baseline + slope before stacking
    phase_time_all = []  # Time relative to T0 for each transit
    flux_all = []
    flux_err_all = []

    transit_data_for_plot = []  # For individual transit plot

    for epoch, info in sorted(transit_info.items()):
        t0_ind = info['t0']
        bl = info['baseline']
        sl = info['slope']

        mask = np.abs(time - t0_ind) < coverage_hw
        if np.sum(mask) < 5:
            continue

        t_local = time[mask]
        f_local = flux[mask]
        e_local = flux_err[mask]

        # Phase time: time relative to individual T0
        dt = t_local - t0_ind  # in days, centered on transit

        # Normalize by fitted baseline + slope: trend = baseline + slope * dt
        trend = bl + sl * dt
        f_normalized = f_local / trend
        e_normalized = e_local / trend

        phase_time_all.append(dt)
        flux_all.append(f_normalized)
        flux_err_all.append(e_normalized)

        transit_data_for_plot.append({
            'epoch': epoch, 't0': t0_ind,
            'baseline': bl, 'slope': sl,
            'time': t_local, 'flux': f_local, 'flux_err': e_local,
            'flux_normalized': f_normalized, 'flux_err_normalized': e_normalized,
            'dt_hours': dt * 24
        })

    phase_time = np.concatenate(phase_time_all)
    stacked_flux = np.concatenate(flux_all)
    stacked_err = np.concatenate(flux_err_all)

    # Sort by phase time
    sort_idx = np.argsort(phase_time)
    phase_time = phase_time[sort_idx]
    stacked_flux = stacked_flux[sort_idx]
    stacked_err = stacked_err[sort_idx]

    print(f"  Stacked data: {len(phase_time)} points from {n_transits} transits")
    print(f"  Transit duration: {duration_days*24:.2f} hr, window: +/- {coverage_hw*24:.2f} hr")

    # =====================================================================
    # MCMC fit for transit shape
    # =====================================================================
    print(f"\n{'='*60}")
    print(f"TRANSIT SHAPE FIT (individual T0s fixed)")
    print(f"{'='*60}")

    if fix_ld:
        ndim = 4  # Rp/Rs, a/Rs, b^2, baseline
        param_names = ['rp_rs', 'a_rs', 'b_sq', 'baseline']
    else:
        ndim = 6  # Rp/Rs, a/Rs, b^2, baseline, u1, u2
        param_names = ['rp_rs', 'a_rs', 'b_sq', 'baseline', 'u1', 'u2']

    # LD priors
    u1_sigma = config.LD_WIDTH_U1 if hasattr(config, 'LD_WIDTH_U1') else 0.15
    u2_sigma = config.LD_WIDTH_U2 if hasattr(config, 'LD_WIDTH_U2') else 0.10

    shared = {
        'phase_time': phase_time,
        'flux': stacked_flux,
        'inv_var': 1.0 / stacked_err**2,
        'period': period,
        'fix_ld': fix_ld,
        'u1_fixed': u1_std, 'u2_fixed': u2_std,
        'u1_prior': u1_std, 'u2_prior': u2_std,
        'u1_sigma': u1_sigma, 'u2_sigma': u2_sigma,
        'rp_rs_prior_mean': rp_rs_prior[0] if rp_rs_prior else None,
        'rp_rs_prior_sigma': rp_rs_prior[1] if rp_rs_prior else None,
        'a_rs_prior_mean': a_rs_prior[0] if a_rs_prior else None,
        'a_rs_prior_sigma': a_rs_prior[1] if a_rs_prior else None,
    }
    if rp_rs_prior:
        print(f"  Gaussian prior Rp/Rs: N({rp_rs_prior[0]:.4f}, σ={rp_rs_prior[1]:.4f})")
    if a_rs_prior:
        print(f"  Gaussian prior a/Rs:  N({a_rs_prior[0]:.2f}, σ={a_rs_prior[1]:.2f})")

    n_walkers = config.N_WALKERS  # 64

    # Initialize walkers around init values (default: standard fit)
    rp_rs_init = init_rp_rs if init_rp_rs is not None else rp_rs_std
    a_rs_init = init_a_rs if init_a_rs is not None else a_rs_std
    b_init = init_b if init_b is not None else b_std
    if fix_ld:
        p0 = np.array([rp_rs_init, a_rs_init, b_init**2, 1.0])
        scatter = np.array([0.1*rp_rs_init + 1e-4, 0.1*a_rs_init + 0.1,
                            0.1*b_init**2 + 0.01, 0.001])
    else:
        p0 = np.array([rp_rs_init, a_rs_init, b_init**2, 1.0, u1_std, u2_std])
        scatter = np.array([0.1*rp_rs_init + 1e-4, 0.1*a_rs_init + 0.1,
                            0.1*b_init**2 + 0.01, 0.001, 0.05, 0.05])

    pos = p0 + scatter * np.random.randn(n_walkers, ndim)
    for i in range(n_walkers):
        pos[i, 0] = max(config.RP_RS_MIN + 1e-4, pos[i, 0])
        pos[i, 1] = max(config.A_RS_MIN + 0.1, pos[i, 1])
        pos[i, 2] = max(0.0, pos[i, 2])
        if not fix_ld:
            pos[i, 4] = np.clip(pos[i, 4], config.LD_U1_MIN + 0.01, config.LD_U1_MAX - 0.01)
            pos[i, 5] = np.clip(pos[i, 5], config.LD_U2_MIN + 0.01, config.LD_U2_MAX - 0.01)

    ld_str = "FIXED" if fix_ld else "FREE"
    print(f"  {n_walkers} walkers, {ndim} params (LD {ld_str})")
    print(f"  Init: Rp/Rs={rp_rs_init:.4f}, a/Rs={a_rs_init:.2f}, b={b_init:.3f}")

    # Burn-in
    n_burn = config.N_BURN
    print(f"  Burn-in ({n_burn} steps)...", flush=True)

    with Pool(processes=n_cpus, initializer=_init_worker, initargs=(shared,)) as pool:
        sampler = emcee.EnsembleSampler(n_walkers, ndim, _log_probability, pool=pool)
        state = sampler.run_mcmc(pos, n_burn, progress=False)

        # Bad walker rejection
        lp = sampler.get_log_prob()
        burnin_chains = sampler.get_chain()
        burnin_lp = lp

        tail = max(100, lp.shape[0] // 10)
        walker_meds = np.median(lp[-tail:, :], axis=0)
        med_of_meds = np.median(walker_meds)
        mad = np.median(np.abs(walker_meds - med_of_meds))
        sigma = 1.48 * mad
        if sigma > 0:
            bad = walker_meds < med_of_meds - config.BAD_WALKER_SIGMA * sigma
            n_bad = np.sum(bad)
            if n_bad > 0:
                print(f"  Rejected {n_bad} bad walkers", flush=True)
                good_idx = np.where(~bad)[0]
                new_pos = state.coords.copy()
                for i in np.where(bad)[0]:
                    new_pos[i] = state.coords[np.random.choice(good_idx)]
                    new_pos[i] += 1e-5 * np.random.randn(ndim)
                sampler.reset()
                print(f"  Second burn-in (2000 steps)...", flush=True)
                state = sampler.run_mcmc(new_pos, 2000, progress=False)
                burnin_chains = np.concatenate([burnin_chains, sampler.get_chain()], axis=0)
                burnin_lp = np.concatenate([burnin_lp, sampler.get_log_prob()], axis=0)

        # Production with convergence checking
        n_prod = config.N_STEPS_MAX
        print(f"  Production (up to {n_prod} steps)...", flush=True)
        sampler.reset()
        check_interval = 2000
        for step in range(0, n_prod, check_interval):
            n_run = min(check_interval, n_prod - step)
            state = sampler.run_mcmc(state, n_run, progress=False)
            current = step + n_run
            if current >= 4000:
                ch = sampler.get_chain()
                rhats = []
                for i in range(ndim):
                    rh = compute_rhat_split(ch[:, :, i:i+1])
                    rhats.append(float(rh[0]) if hasattr(rh, '__len__') else float(rh))
                mr = max(rhats)
                print(f"    {current} steps: R-hat={mr:.4f}", flush=True)
                if mr <= config.CONVERGENCE_RHAT:
                    print(f"    Converged!", flush=True)
                    break

    # Extract results
    chains = sampler.get_chain().copy()
    # Convert b^2 -> b
    chains[:, :, 2] = np.sqrt(np.abs(chains[:, :, 2]))
    param_names_display = list(param_names)
    param_names_display[2] = 'b'

    n_discard = chains.shape[0] // 5
    samples = chains[n_discard:, :, :].reshape(-1, ndim)
    medians = np.median(samples, axis=0)
    errs_lo = medians - np.percentile(samples, 16, axis=0)
    errs_hi = np.percentile(samples, 84, axis=0) - medians

    rhats = []
    for i in range(ndim):
        rh = compute_rhat_split(chains[:, :, i:i+1])
        rhats.append(float(rh[0]) if hasattr(rh, '__len__') else float(rh))
    max_rhat = max(rhats)

    print(f"\n  Results (max R-hat={max_rhat:.4f}):")
    for i, name in enumerate(param_names_display):
        print(f"    {name:<10}: {medians[i]:.6f} +{errs_hi[i]:.6f} -{errs_lo[i]:.6f}  R-hat={rhats[i]:.4f}")

    # Comparison with standard fit
    rp_rs_ref = medians[0]; a_rs_ref = medians[1]; b_ref = medians[2]
    if not fix_ld:
        u1_ref = medians[4]; u2_ref = medians[5]
    else:
        u1_ref = u1_std; u2_ref = u2_std

    print(f"\n  Comparison with standard fit:")
    for name, std_val, ref_val, err in [
        ('rp_rs', rp_rs_std, medians[0], (errs_lo[0]+errs_hi[0])/2),
        ('a_rs', a_rs_std, medians[1], (errs_lo[1]+errs_hi[1])/2),
        ('b', b_std, medians[2], (errs_lo[2]+errs_hi[2])/2),
    ]:
        diff = abs(ref_val - std_val) / err if err > 0 else 0
        print(f"    {name:<10}: std={std_val:.4f}, refined={ref_val:.4f}, diff={diff:.1f}σ")

    # Save chains
    np.save(refined_dir / 'production_chains.npy', chains)

    # =====================================================================
    # PLOTS
    # =====================================================================
    print(f"\n{'='*60}")
    print("PLOTS")
    print(f"{'='*60}")

    baseline_ref = medians[3]

    # --- 1. Chain plot ---
    prod_lp = sampler.get_log_prob()
    n_burnin = burnin_chains.shape[0]
    n_prod_actual = chains.shape[0]
    n_panels = ndim + 1

    fig, axes = plt.subplots(n_panels, 1, figsize=(12, 2 * n_panels), sharex=True)
    for i, (ax, name) in enumerate(zip(axes[:ndim], param_names_display)):
        for w in range(burnin_chains.shape[1]):
            # Convert b^2 -> b for burnin too
            vals = burnin_chains[:, w, i] if i != 2 else np.sqrt(np.abs(burnin_chains[:, w, i]))
            ax.plot(np.arange(n_burnin), vals, alpha=0.15, lw=0.5, color='gray')
        for w in range(chains.shape[1]):
            ax.plot(np.arange(n_burnin, n_burnin + n_prod_actual), chains[:, w, i], alpha=0.2, lw=0.5)
        ax.axvline(n_burnin, color='black', ls='--', lw=1, alpha=0.7)
        ax.set_ylabel(name, fontsize=8)
        ax.axhline(medians[i], color='red', lw=1)

    ax_lp = axes[-1]
    for w in range(burnin_lp.shape[1]):
        ax_lp.plot(np.arange(n_burnin), burnin_lp[:, w], alpha=0.15, lw=0.5, color='gray')
    for w in range(prod_lp.shape[1]):
        ax_lp.plot(np.arange(n_burnin, n_burnin + n_prod_actual), prod_lp[:, w], alpha=0.2, lw=0.5)
    ax_lp.axvline(n_burnin, color='black', ls='--', lw=1, alpha=0.7)
    ax_lp.set_ylabel('log(p)')
    ax_lp.axhline(np.median(prod_lp), color='red', lw=1)
    ax_lp.set_xlabel('Step')
    fig.suptitle(f'TOI {toi} — Refined Transit Parameters (TTV-corrected)', y=1.01)
    fig.tight_layout()
    fig.savefig(refined_dir / f'chain_plot.{config.PLOT_FORMAT}', dpi=config.PLOT_DPI, bbox_inches='tight')
    plt.close(fig)
    print("  Saved chain_plot")

    # --- 2. Corner plot ---
    try:
        import corner
        ranges = []
        for i in range(ndim):
            r_lo, r_hi = np.percentile(samples[:, i], [1, 99])
            if r_hi - r_lo < 1e-10:
                ranges.append((r_lo - 0.01, r_hi + 0.01))
            else:
                ranges.append((r_lo, r_hi))
        fig = corner.corner(samples, labels=param_names_display, quantiles=[0.16, 0.5, 0.84],
                            show_titles=True, title_kwargs={"fontsize": 10}, range=ranges)
        fig.suptitle(f'TOI {toi} — Refined Transit Parameters', y=1.02)
        fig.savefig(refined_dir / f'corner_plot.{config.PLOT_FORMAT}', dpi=config.PLOT_DPI, bbox_inches='tight')
        plt.close(fig)
        print("  Saved corner_plot")
    except ImportError:
        print("  corner not installed, skipping")

    # --- 3. Phase-folded transit (zoomed) ---
    bat_params = batman.TransitParams()
    bat_params.per = period; bat_params.t0 = 0.0; bat_params.rp = rp_rs_ref
    bat_params.a = a_rs_ref
    bat_params.inc = np.degrees(np.arccos(b_ref / a_rs_ref)) if a_rs_ref > 0 and b_ref / a_rs_ref < 1 else 90.0
    bat_params.ecc = 0.0; bat_params.w = 90.0
    bat_params.limb_dark = "quadratic"; bat_params.u = [u1_ref, u2_ref]

    # Phase as fraction of period
    phase = phase_time / period

    # Zoomed phase-folded
    t14_phase = duration_days / period / 2
    pw = max(0.02, min(2.0 * t14_phase, 0.15))
    in_win = np.abs(phase) <= pw
    n_bins = max(20, int(2 * pw * period * 24 * 60 / config.BIN_WIDTH_MINUTES))
    bins = np.linspace(-pw, pw, n_bins + 1)
    bc = 0.5 * (bins[:-1] + bins[1:])
    bf = np.full(n_bins, np.nan); be = np.full(n_bins, np.nan)
    for i in range(n_bins):
        m = in_win & (phase >= bins[i]) & (phase < bins[i+1])
        if np.sum(m) > 2:
            w = 1.0 / stacked_err[m]**2
            bf[i] = np.sum(stacked_flux[m] * w) / np.sum(w)
            be[i] = 1.0 / np.sqrt(np.sum(w))
    v = ~np.isnan(bf)

    # High-resolution model for plotting
    mp = np.linspace(-pw, pw, 2000)
    mt = mp * period  # Convert phase to time for batman
    mf = batman.TransitModel(bat_params, mt).light_curve(bat_params) * baseline_ref

    # Bin-averaged model for residuals (supersample model across each bin)
    mf_binned = np.full(n_bins, np.nan)
    for i in range(n_bins):
        in_bin = (mp >= bins[i]) & (mp < bins[i+1])
        if np.sum(in_bin) > 0:
            mf_binned[i] = np.mean(mf[in_bin])

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True,
                                  gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.05})
    a1.scatter(phase[in_win], stacked_flux[in_win], s=1, alpha=0.1, color='gray')
    a1.errorbar(bc[v], bf[v], yerr=be[v], fmt='o', ms=4, color='blue', capsize=2)
    a1.plot(mp, mf, 'r-', lw=2, label='Refined model')

    # Overlay standard model for comparison
    bat_std = batman.TransitParams()
    bat_std.per = period; bat_std.t0 = 0.0; bat_std.rp = rp_rs_std
    bat_std.a = a_rs_std
    bat_std.inc = np.degrees(np.arccos(b_std / a_rs_std)) if a_rs_std > 0 and b_std / a_rs_std < 1 else 90.0
    bat_std.ecc = 0.0; bat_std.w = 90.0; bat_std.limb_dark = "quadratic"; bat_std.u = [u1_std, u2_std]
    # Apply the REFINED baseline to the standard model too — the data being plotted
    # are stacked on the refined per-transit t_mids, so baseline_ref is the actual
    # OOT level here. Using the standard pipeline's own (different) baseline would
    # put the two model lines at different OOT levels for no good reason; this gives
    # a pure-shape comparison. (Plot-consistency fix, 2026-06-09.)
    mf_std = batman.TransitModel(bat_std, mt).light_curve(bat_std) * baseline_ref
    a1.plot(mp, mf_std, 'g--', lw=1.5, alpha=0.7, label='Standard fit')

    a1.set_ylabel('Flux')
    a1.set_title(f'TOI {toi} — Refined Transit (Rp/Rs={rp_rs_ref:.4f} vs std {rp_rs_std:.4f})')
    a1.legend()

    rb = bf[v] - mf_binned[v]
    a2.errorbar(bc[v], rb * 1e6, yerr=be[v] * 1e6, fmt='o', ms=4, color='blue', capsize=2)
    a2.axhline(0, color='gray', ls='--')
    a2.set_xlabel('Phase')
    a2.set_ylabel('Residuals (ppm)')
    a2.text(0.02, 0.95, f'RMS = {np.sqrt(np.nanmean(rb**2))*1e6:.0f} ppm', transform=a2.transAxes, va='top')
    fig.tight_layout()
    fig.savefig(refined_dir / f'phase_folded_transit.{config.PLOT_FORMAT}', dpi=config.PLOT_DPI, bbox_inches='tight')
    plt.close(fig)
    print("  Saved phase_folded_transit")

    # =====================================================================
    # Save results
    # =====================================================================
    refined_results = {
        'toi': toi,
        'model': 'refined_transit_params_ttv_corrected',
        'n_transits_used': n_transits,
        'n_data_points': len(phase_time),
        'fix_ld': fix_ld,
        'parameters': {
            name: {
                'value': float(medians[i]),
                'err_lower': float(errs_lo[i]),
                'err_upper': float(errs_hi[i]),
                'err': float((errs_lo[i] + errs_hi[i]) / 2),
                'rhat': float(rhats[i]),
            }
            for i, name in enumerate(param_names_display)
        },
        'convergence': {
            'max_rhat': float(max_rhat),
            'converged': max_rhat <= config.CONVERGENCE_RHAT,
        },
        'comparison_with_standard': {
            name: {
                'standard': float(std_val),
                'refined': float(medians[param_names_display.index(name)]),
                'diff_sigma': float(abs(medians[param_names_display.index(name)] - std_val)
                                    / ((errs_lo[param_names_display.index(name)] + errs_hi[param_names_display.index(name)]) / 2))
                if (errs_lo[param_names_display.index(name)] + errs_hi[param_names_display.index(name)]) > 0 else 0,
            }
            for name, std_val in [('rp_rs', rp_rs_std), ('a_rs', a_rs_std), ('b', b_std)]
        },
        'transit_t0s': {str(e): float(info['t0']) for e, info in sorted(transit_info.items())},
        'rejected_transits': {str(e): info['reject_reason'] for e, info in sorted(transit_info.items()) if info.get('rejected')},
    }

    with open(refined_dir / 'results.json', 'w') as f:
        json.dump(refined_results, f, indent=2, default=float)

    print(f"\n  All results saved to {refined_dir}")
    return refined_results


def refit_individual_transits(toi, extra_epochs=None, n_cpus=15, sibling_mask_factor=None,
                              subdir=None, results_root=RESULTS_ROOT_DEFAULT):
    """
    Refit individual transit times using refined shape parameters.

    Uses the refined Rp/Rs, a/Rs, b, u1, u2 from the refined transit fit
    and refits T0, baseline, slope for each transit — including any extra
    epochs that were previously rejected.

    Parameters
    ----------
    toi : str
        TOI identifier (e.g. '1130.02')
    extra_epochs : list of int, optional
        Additional epochs to attempt fitting (e.g. rejected transits)
    n_cpus : int
        CPUs for parallel fitting
    sibling_mask_factor : float, optional
        Multiplier for sibling transit masking width (default 1.5x duration).
        Use larger values (e.g. 5.0) for deep sibling transits.
    subdir : str, optional
        Folder under the TOI directory that holds the refined shape to use and
        receives the output (default 'refined_transit').
    results_root : str
        Results tree that holds TOI_*/ (default 'autottv_results_v2').
    """
    from scipy.ndimage import median_filter

    output_dir = Path(results_root) / f'TOI_{toi.replace(".", "_")}'
    results_path = output_dir / 'results.json'
    refined_dir = output_dir / (subdir or 'refined_transit')

    with open(results_path) as f:
        results = json.load(f)

    # Load refined shape parameters (fall back to standard if no refined fit)
    refined_results_path = refined_dir / 'results.json'
    if refined_results_path.exists():
        with open(refined_results_path) as f:
            refined = json.load(f)
        ref_params = refined['parameters']
        rp_rs = ref_params['rp_rs']['value']
        a_rs = ref_params['a_rs']['value']
        b = ref_params['b']['value']
        u1 = ref_params.get('u1', ref_params.get('u1', {})).get('value', results['parameters']['u1']['value'])
        u2 = ref_params.get('u2', ref_params.get('u2', {})).get('value', results['parameters']['u2']['value'])
        print(f"  Using refined shape: Rp/Rs={rp_rs:.4f}, a/Rs={a_rs:.2f}, b={b:.3f}")
    else:
        params_std = results['parameters']
        rp_rs = params_std['rp_rs']['value']
        a_rs = params_std['a_rs']['value']
        b = params_std['b']['value']
        u1 = params_std['u1']['value']
        u2 = params_std['u2']['value']
        print(f"  No refined fit found, using standard shape: Rp/Rs={rp_rs:.4f}, a/Rs={a_rs:.2f}, b={b:.3f}")

    tic_id = results['tic_id']
    sectors = results['sectors']
    period = results['parameters']['period']['value']
    t0_ref = results['parameters']['t0']['value']

    # Load light curve data
    catalog = load_toi_catalog()
    loader = DataLoader(tic_id)
    if not loader.load_from_npz_cache():
        if not loader.download_from_mast():
            print(f"  Failed to load data"); return None

    combined_time, combined_flux, combined_flux_err, combined_cadence = [], [], [], []
    for lc in loader.lightcurves:
        if lc.sector in sectors:
            fn, en, _ = normalize_to_oot_no_eclipse(lc.time, lc.flux, lc.flux_err, period, t0_ref)
            combined_time.append(lc.time)
            combined_flux.append(fn)
            combined_flux_err.append(en)
            combined_cadence.append(np.full(len(lc.time), lc.cadence))

    time = np.concatenate(combined_time)
    flux = np.concatenate(combined_flux)
    flux_err = np.concatenate(combined_flux_err)
    cadence = np.concatenate(combined_cadence)
    sort_idx = np.argsort(time)
    time, flux, flux_err, cadence = time[sort_idx], flux[sort_idx], flux_err[sort_idx], cadence[sort_idx]

    # Mask sibling TOIs
    toi_number = toi.split('.')[0]
    sibling_rows = catalog[catalog['TOI'].apply(lambda x: str(x).split('.')[0] == toi_number)]
    sibling_tois = [str(row['TOI']) for _, row in sibling_rows.iterrows()
                    if str(row['TOI']) != toi and not np.isnan(row.get('Period (days)', float('nan')))]
    if sibling_tois:
        mask_factor = sibling_mask_factor if sibling_mask_factor is not None else 1.5
        print(f"  Masking {len(sibling_tois)} sibling TOIs (mask factor={mask_factor}x duration)")
        sibling_mask = np.ones(len(time), dtype=bool)
        n_masked_total = 0
        for sib_toi in sibling_tois:
            sib_pp = get_toi_parameters(catalog, toi=sib_toi)
            if sib_pp is None: continue
            sib_period = sib_pp['period']
            sib_t0 = sib_pp['t0']
            sib_dur = sib_pp.get('duration_hr', 3.0) / 24.0
            sib_hw = mask_factor * sib_dur
            n_min = int(np.floor((time.min() - sib_t0) / sib_period))
            n_max = int(np.ceil((time.max() - sib_t0) / sib_period))
            n_masked = 0
            for n in range(n_min, n_max + 1):
                t_transit = sib_t0 + n * sib_period
                in_mask = np.abs(time - t_transit) < sib_hw
                sibling_mask &= ~in_mask
                n_masked += np.sum(in_mask)
            print(f"    TOI {sib_toi}: ±{sib_hw*24:.1f} hr per transit, {n_masked} pts masked")
            n_masked_total += n_masked
        time = time[sibling_mask]
        flux = flux[sibling_mask]
        flux_err = flux_err[sibling_mask]
        cadence = cadence[sibling_mask]
        print(f"    {n_masked_total} points masked total, {len(time)} remaining")

    # Compute transit duration from refined params
    if b < (1 + rp_rs) and a_rs > 0:
        sin_arg = np.sqrt((1 + rp_rs)**2 - b**2) / a_rs
        duration_days = period / np.pi * np.arcsin(min(sin_arg, 1.0)) if sin_arg <= 1 else period / (np.pi * a_rs) * rp_rs
    else:
        duration_days = period / (np.pi * a_rs) * rp_rs if a_rs > 0 else 0.01

    # Data extraction half-width: must cover the full T_mid prior window
    # so the transit model can evaluate at any allowed T_mid position
    t_mid_hw = config.T_MID_WINDOW_FRACTION * period
    # Use the larger of coverage_factor * duration or t_mid prior window,
    # plus extra baseline padding (1.5 * duration beyond the prior edge)
    extraction_hw = max(config.TRANSIT_COVERAGE_FACTOR * duration_days,
                        t_mid_hw + 1.5 * duration_days)
    print(f"  T_mid prior: ±{t_mid_hw*24:.1f} hr, data extraction: ±{extraction_hw*24:.1f} hr")

    # Identify all transits using the wider extraction window
    transits_full, transits_partial = identify_transits(
        time, period, t0_ref, duration_days=duration_days,
        require_full_coverage=False, cadence=cadence
    )

    # Build set of epochs to fit: all transits with data in the extraction window
    all_epochs = {}
    for t_info in transits_full + transits_partial:
        ep = t_info['epoch']
        t_expected = t_info['t_expected']
        mask = np.abs(time - t_expected) < extraction_hw
        n_pts = np.sum(mask)
        if n_pts >= 3:
            all_epochs[ep] = {
                'epoch': ep,
                't_expected': t_expected,
                'mask': mask,
                'n_points': n_pts,
                'coverage': t_info.get('coverage', 'full'),
            }

    # Add extra requested epochs not already found
    extra_epochs = extra_epochs or []
    for ep in extra_epochs:
        if ep not in all_epochs:
            t_expected = t0_ref + ep * period
            mask = np.abs(time - t_expected) < extraction_hw
            n_pts = np.sum(mask)
            if n_pts >= 3:
                all_epochs[ep] = {
                    'epoch': ep,
                    't_expected': t_expected,
                    'mask': mask,
                    'n_points': n_pts,
                    'coverage': 'forced',
                }
            else:
                print(f"  WARNING: Epoch {ep} has only {n_pts} points, skipping")

    print(f"  {len(all_epochs)} transits to fit ({len(transits_full)} full + "
          f"{len(all_epochs) - len(transits_full)} extra)")

    # Build step1_results with refined parameters
    step1_results = {
        'parameters': {
            'period': {'value': period},
            't0': {'value': t0_ref},
            'rp_rs': {'value': rp_rs},
            'a_rs': {'value': a_rs},
            'b': {'value': b},
            'baseline': {'value': 1.0},
            'u1': {'value': u1},
            'u2': {'value': u2},
        },
        'derived': {
            'duration_hr': duration_days * 24,
        }
    }

    # Fit each transit
    fit_results = []
    for epoch in sorted(all_epochs.keys()):
        td = all_epochs[epoch]
        t_expected = td.get('t_expected', t0_ref + epoch * period)
        mask = td['mask']
        t_data = time[mask]
        f_data = flux[mask]
        f_err = flux_err[mask]
        cad_data = cadence[mask]
        med_cadence = float(np.median(cad_data))

        ind_fitter = ModularIndividualTransitFitter(step1_results, cadence=med_cadence)

        # Smart T0 initialization: use flux minimum if dip is significant
        t0_guess = t_expected
        try:
            n_smooth = max(5, len(f_data) // 20)
            f_smooth = median_filter(f_data, size=n_smooth)
            f_min = np.min(f_smooth)
            f_med = np.median(f_smooth)
            f_std = 1.48 * np.median(np.abs(f_smooth - f_med))
            if f_std > 0 and (f_med - f_min) > 3 * f_std:
                t0_guess = t_data[np.argmin(f_smooth)]
        except:
            pass

        result = ind_fitter.fit_single_transit(
            t_data, f_data, f_err, t0_guess, epoch, cadence=med_cadence
        )

        is_extra = epoch in extra_epochs
        oc_minutes = (result.t_mid - t_expected) * 24 * 60
        conv_str = "" if getattr(result, 'converged', True) else " [NOT CONVERGED]"
        extra_str = " [EXTRA]" if is_extra else ""
        print(f"    Epoch {epoch:5d}: T0={result.t_mid:.6f} ± {result.t_mid_err:.6f} "
              f"(O-C={oc_minutes:+.2f} min) R-hat={getattr(result, 'max_rhat', 0):.4f}"
              f"{conv_str}{extra_str}")

        fit_results.append({
            'epoch': epoch,
            't_expected': t_expected,
            't0_fit': result.t_mid,
            't0_err': result.t_mid_err,
            't0_err_minutes': result.t_mid_err * 24 * 60,
            'oc_minutes': oc_minutes,
            'baseline_fit': result.baseline,
            'baseline_err': result.baseline_err,
            'slope_fit': result.slope,
            'slope_err': result.slope_err,
            'n_points': td['n_points'],
            'converged': getattr(result, 'converged', True),
            'max_rhat': getattr(result, 'max_rhat', 1.0),
            'is_extra_epoch': is_extra,
            'coverage': td.get('coverage', 'full'),
        })

    # Save results
    refit_results = {
        'toi': toi,
        'model': 'refined_individual_transit_refit',
        'shape_parameters': {
            'rp_rs': rp_rs, 'a_rs': a_rs, 'b': b, 'u1': u1, 'u2': u2,
        },
        'n_transits_fit': len(fit_results),
        'n_extra_epochs': len([f for f in fit_results if f['is_extra_epoch']]),
        'transit_fits': fit_results,
    }

    refit_path = refined_dir / 'individual_refit_results.json'
    with open(refit_path, 'w') as f:
        json.dump(refit_results, f, indent=2, default=float)

    # Plot O-C diagram with extra epochs highlighted
    fig, ax = plt.subplots(figsize=(12, 5))
    for fr in fit_results:
        color = 'red' if fr['is_extra_epoch'] else 'blue'
        marker = 's' if fr['is_extra_epoch'] else 'o'
        label = None
        if fr['is_extra_epoch'] and not any(f['is_extra_epoch'] and f is not fr for f in fit_results[:fit_results.index(fr)]):
            label = 'Extra (previously rejected)'
        elif not fr['is_extra_epoch'] and fr == [f for f in fit_results if not f['is_extra_epoch']][0]:
            label = 'Used transits'
        ax.errorbar(fr['epoch'], fr['oc_minutes'], yerr=fr['t0_err_minutes'],
                    fmt=marker, color=color, capsize=3, ms=6, label=label)
    ax.axhline(0, color='gray', ls='--', alpha=0.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('O-C (minutes)')
    ax.set_title(f'TOI {toi} — Individual Transit Refit (Refined Shape)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(refined_dir / f'oc_refit.{config.PLOT_FORMAT}', dpi=config.PLOT_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved oc_refit plot")

    # Plot individual transit light curves for extra epochs
    if extra_epochs:
        n_extra = len([e for e in extra_epochs if e in all_epochs])
        if n_extra > 0:
            fig, axes = plt.subplots(1, n_extra, figsize=(6*n_extra, 5), squeeze=False)
            for idx, ep in enumerate(sorted(e for e in extra_epochs if e in all_epochs)):
                ax = axes[0, idx]
                td = all_epochs[ep]
                mask = td['mask']
                t_data = time[mask]
                f_data = flux[mask]
                f_err = flux_err[mask]
                t_expected = td.get('t_expected', t0_ref + ep * period)

                # Find the fit result for this epoch
                fr = [f for f in fit_results if f['epoch'] == ep][0]

                # Plot data
                dt_hours = (t_data - fr['t0_fit']) * 24
                ax.errorbar(dt_hours, f_data, yerr=f_err, fmt='.', color='gray',
                           alpha=0.5, ms=3, capsize=0)

                # Plot model
                t_model = np.linspace(t_data.min(), t_data.max(), 500)
                dt_model = t_model - fr['t0_fit']
                ind_fitter = ModularIndividualTransitFitter(step1_results,
                    cadence=float(np.median(cadence[td['mask']])))
                model_flux = ind_fitter.transit_model_with_trend(
                    t_model, fr['t0_fit'], fr['baseline_fit'], fr['slope_fit'])
                ax.plot((t_model - fr['t0_fit']) * 24, model_flux, 'r-', lw=2)

                conv_str = "" if fr['converged'] else " [NC]"
                ax.set_title(f"Epoch {ep} ({fr['n_points']} pts)\n"
                           f"O-C={fr['oc_minutes']:+.1f} min, "
                           f"R-hat={fr['max_rhat']:.3f}{conv_str}")
                ax.set_xlabel('Hours from T0')
                ax.set_ylabel('Normalized Flux')

            fig.suptitle(f'TOI {toi} — Extra Epoch Transits (Refined Shape)', y=1.02)
            fig.tight_layout()
            fig.savefig(refined_dir / f'extra_epoch_transits.{config.PLOT_FORMAT}',
                       dpi=config.PLOT_DPI, bbox_inches='tight')
            plt.close(fig)
            print(f"  Saved extra_epoch_transits plot")

    print(f"\n  All refit results saved to {refit_path}")
    return refit_results


REFINED_MAX_ITERS = 5    # iterations after the first refined fit
REFINED_SHAPE_TOL = 1.0  # stop when each shape parameter moves by less than this many sigma


def _irr_t_mids(irr_path):
    """{epoch: t0_fit} of the converged, non-extra transits in an individual_refit_results.json."""
    with open(irr_path) as f:
        fits = json.load(f).get('transit_fits') or []
    return {int(f['epoch']): float(f['t0_fit']) for f in fits
            if not f.get('is_extra_epoch', False) and f.get('converged', True)}


def _refined_shape(results_path):
    """{'rp_rs': (value, sigma), 'a_rs': ..., 'b': ...} from a refined-fit results.json."""
    with open(results_path) as f:
        params = json.load(f)['parameters']
    return {k: (float(params[k]['value']), float(params[k].get('err', 0)) or 1e-9)
            for k in ('rp_rs', 'a_rs', 'b')}


def run_refined_iterations(toi, max_iters=REFINED_MAX_ITERS, shape_tol=REFINED_SHAPE_TOL,
                           n_cpus=15, fix_ld=False, include_rejected=False,
                           extra_epochs=None, sibling_mask_factor=None,
                           results_root=RESULTS_ROOT_DEFAULT):
    """
    Iterate the refined fit until the transit shape stops moving.

    Iteration 0 stacks the transits at their Step-2 times, refits the shape
    (refined_transit/) and re-times every transit with it. Iteration
    n = 1..max_iters refits the shape on iteration n-1's re-timed transits and
    re-times them again (refined_strict_iter{n}/). The loop stops once Rp/Rs,
    a/Rs and b each move by less than shape_tol times their previous
    posterior sigma (paper, Section 6.1). This is the pattern of the per-TOI
    cascades used for the paper, which also stopped after five iterations.

    Returns a dict with each iteration's shape and largest shift, whether it
    converged, and 'adopted_subdir', the last iteration written, which is the
    result. The dict is also written to iter_cascade_summary.json.
    """
    toi_dir = Path(results_root) / f'TOI_{toi.replace(".", "_")}'

    def fitted(subdir, override):
        """Refit the shape in subdir and re-time the transits; None if either step failed."""
        if refined_transit_params(toi, n_cpus=n_cpus, fix_ld=fix_ld,
                                  include_rejected=include_rejected,
                                  transit_times_override=override,
                                  output_subdir=subdir,
                                  results_root=results_root) is None:
            print(f"  Shape refit failed in {subdir}/; stopping")
            return None
        if refit_individual_transits(toi, extra_epochs=extra_epochs, n_cpus=n_cpus,
                                     sibling_mask_factor=sibling_mask_factor,
                                     subdir=subdir, results_root=results_root) is None:
            print(f"  Re-timing failed in {subdir}/; stopping")
            return None
        return _refined_shape(toi_dir / subdir / 'results.json')

    history, converged = [], False
    print(f"\n{'='*60}\nRefined fit, iteration 0 (refined_transit/)\n{'='*60}")
    shape = fitted('refined_transit', None)
    if shape is not None:
        history.append({'iter': 0, 'subdir': 'refined_transit', 'shape': shape,
                        'max_shift_sigma': None})
        for n in range(1, max_iters + 1):
            override = _irr_t_mids(toi_dir / history[-1]['subdir'] / 'individual_refit_results.json')
            if len(override) < 5:
                print(f"  Only {len(override)} re-timed transits; stopping")
                break
            subdir = f'refined_strict_iter{n}'
            print(f"\n{'='*60}\nRefined fit, iteration {n} ({subdir}/)\n{'='*60}")
            new = fitted(subdir, override)
            if new is None:
                break
            shift = max(abs(new[k][0] - shape[k][0]) / shape[k][1] for k in new)
            history.append({'iter': n, 'subdir': subdir, 'shape': new, 'max_shift_sigma': shift})
            print(f"  Largest shape change since iteration {n - 1}: {shift:.2f} sigma")
            shape = new
            if shift < shape_tol:
                converged = True
                break

    adopted = history[-1]['subdir'] if history else None
    summary = {'toi': toi, 'max_iters': max_iters, 'shape_tol': shape_tol,
               'converged': converged, 'adopted_subdir': adopted, 'history': history}
    if history:
        with open(toi_dir / 'iter_cascade_summary.json', 'w') as f:
            json.dump(summary, f, indent=2, default=float)
        last = history[-1]['iter']
        stale = sorted(d.name for d in toi_dir.glob('refined_strict_iter*')
                       if d.name[len('refined_strict_iter'):].isdigit()
                       and int(d.name[len('refined_strict_iter'):]) > last)
        status = ('iteration 0 only' if last == 0 else
                  f"{'converged' if converged else 'not converged'} after {last} iteration(s)")
        print(f"\n  Adopted: {adopted}/ ({status})")
        if stale:
            print(f"  Note: {', '.join(stale)} {'is' if len(stale) == 1 else 'are'} not part of this "
                  f"result (left from an earlier run, or an iteration that failed)")
    return summary


if __name__ == '__main__':
    toi = sys.argv[1] if len(sys.argv) > 1 else '924.01'
    n_cpus = 15
    fix_ld = False
    include_rejected = False
    refit_epochs = []
    max_iters = REFINED_MAX_ITERS
    results_root = RESULTS_ROOT_DEFAULT
    for arg in sys.argv[2:]:
        if arg.startswith('--cpus='):
            n_cpus = int(arg.split('=')[1])
        elif arg == '--fix-ld':
            fix_ld = True
        elif arg == '--include-rejected':
            include_rejected = True
        elif arg.startswith('--refit-epochs='):
            refit_epochs = [int(e) for e in arg.split('=')[1].split(',')]
        elif arg.startswith('--max-iters='):
            max_iters = int(arg.split('=')[1])
        elif arg.startswith('--results-root='):
            results_root = arg.split('=', 1)[1].strip().rstrip('/')

    print(f"{'='*60}")
    print(f"Refined Transit Parameters for TOI {toi}")
    print(f"{'='*60}")

    run_refined_iterations(toi, max_iters=max_iters, n_cpus=n_cpus, fix_ld=fix_ld,
                           include_rejected=include_rejected,
                           extra_epochs=refit_epochs or None,
                           results_root=results_root)
