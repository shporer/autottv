#!/usr/bin/env python3
"""
Search for TTV candidates among all analyzed TOIs using three detection criteria:

  C1: Delta BIC (linear - quadratic) > TTV_DELTA_BIC_THRESHOLD (quadratic preferred)
  C2: Periodogram bootstrap FAP < TTV_FAP_THRESHOLD
  C3: weighted O-C RMS / median error > TTV_OC_RMS_OVER_ERR_THRESHOLD

A TOI is a candidate if at least one criterion it meets survives the
per-criterion leave-one-out test (C4, leave_one_out_test). Its type is the
surviving criterion with the highest priority: Periodic (C2), then Quadratic
(C1), then Scatter (C3).

Outputs:
  - ttv_candidates.csv: the candidates
  - ttv_candidates_summary.txt: summary statistics
"""

import json
import glob
import os
import sys
import csv
from multiprocessing import Pool
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from autottv_pipeline_v2 import config
from autottv_pipeline_v2.ephemeris_analysis import EphemerisAnalyzer, linear_ephemeris
from astropy.timeseries import LombScargle


def weighted_oc_rms(oc, oc_err):
    """
    Weighted RMS of O-C residuals about the inverse-variance-weighted mean.

    weights w_i = 1 / oc_err_i²
    weighted_mean = Σ w_i oc_i / Σ w_i
    weighted_rms  = sqrt( Σ w_i (oc_i - weighted_mean)² / Σ w_i )

    This is the canonical "scatter" used by C3 — small-error transits dominate
    the scatter, large-error transits are downweighted, and any non-zero
    unweighted bias is removed.

    Returns None if inputs are degenerate.
    """
    if oc is None or oc_err is None:
        return None
    oc = np.asarray(oc, dtype=float)
    oc_err = np.asarray(oc_err, dtype=float)
    if oc.size == 0 or oc.size != oc_err.size or not np.all(oc_err > 0):
        return None
    w = 1.0 / (oc_err * oc_err)
    sw = w.sum()
    if sw <= 0:
        return None
    mean_w = (w * oc).sum() / sw
    var_w = (w * (oc - mean_w) ** 2).sum() / sw
    return float(np.sqrt(var_w))


# Permutations per removed transit when the leave-one-out test re-checks C2,
# as in run_per_criterion_loo_v2.py, the test applied to the paper's candidates.
N_PERMUTATIONS_LOO = 10_000


def _loo_frequency_grid(time_epochs):
    """Frequency grid for the C2 re-checks: 2/span up to 0.5/(median transit spacing)."""
    time_span = time_epochs.max() - time_epochs.min()
    if time_span <= 0 or len(time_epochs) < 2:
        return None
    dt_med = float(np.median(np.diff(np.sort(time_epochs))))
    f_min = 2.0 / time_span
    f_nyq = 0.5 / max(dt_med, 1e-9)
    if f_nyq <= f_min:
        return None
    n_freq = max(200, int(config.PERIODOGRAM_OVERSAMPLING * time_span * (f_nyq - f_min)))
    return np.linspace(f_min, f_nyq, n_freq)


def _permutation_batch(args):
    """Count permuted periodograms whose peak reaches obs_peak (one worker's share)."""
    time_ep, oc, oc_err, freqs, obs_peak, seed, n_perms = args
    rng = np.random.default_rng(seed)
    n_exceed = 0
    for _ in range(n_perms):
        perm = rng.permutation(len(oc))
        power = LombScargle(time_ep, oc[perm], dy=oc_err[perm]).power(freqs, normalization='psd')
        if float(power.max()) >= obs_peak:
            n_exceed += 1
    return n_exceed


def _permutation_fap(time_ep, oc, oc_err, freqs, obs_peak, seed_base, pool, n_workers):
    """Permutation FAP from N_PERMUTATIONS_LOO shuffles, split across n_workers."""
    per_worker = -(-N_PERMUTATIONS_LOO // n_workers)  # ceiling division
    args = [(time_ep, oc, oc_err, freqs, obs_peak, seed_base + w, per_worker)
            for w in range(n_workers)]
    counts = pool.map(_permutation_batch, args) if pool is not None else map(_permutation_batch, args)
    return (sum(counts) + 1) / (per_worker * n_workers + 1)


def leave_one_out_test(epochs, oc_minutes, oc_err_minutes, period,
                       delta_bic_threshold, fap_threshold, oc_rms_over_err_threshold,
                       c1_pass, c2_pass, c3_pass, pool=None, n_workers=1):
    """
    Per-criterion leave-one-out (LOO) test (paper, Section 5.1).

    Each criterion the TOI originally passed is recomputed with each transit
    removed in turn, and it survives only if it still passes after every
    removal. The TOI passes C4 if at least one criterion survives. This is
    the test run_per_criterion_loo_v2.py applies in the analysis repository.

    C2 is re-checked with N_PERMUTATIONS_LOO permutations per removal, split
    across the workers of `pool` (serial when pool is None).

    Returns
    -------
    survives : dict
        {'C1': bool or None, 'C2': ..., 'C3': ...}; None for a criterion the
        TOI did not originally pass.
    fail_at : dict
        Per criterion: the 1-based index of the first removal that broke it,
        0 if it survived, None if it was not tested.
    """
    passed = {'C1': c1_pass, 'C2': c2_pass, 'C3': c3_pass}
    survives = {k: (True if v else None) for k, v in passed.items()}
    fail_at = {k: (0 if v else None) for k, v in passed.items()}
    n = len(epochs)
    if n < 4:  # need at least 3 transits after removing one
        for k in survives:
            if passed[k]:
                survives[k], fail_at[k] = False, 1
        return survives, fail_at

    epochs = np.asarray(epochs, dtype=float)
    oc = np.asarray(oc_minutes, dtype=float)
    oc_err = np.asarray(oc_err_minutes, dtype=float)
    time_epochs = epochs * period  # days

    freqs = _loo_frequency_grid(time_epochs) if survives['C2'] else None  # grid of the full series

    def fail(key, i):
        survives[key], fail_at[key] = False, i + 1

    for i in range(n):
        m = np.ones(n, dtype=bool)
        m[i] = False

        # C1: delta BIC of the quadratic over the linear ephemeris. Times are
        # rebuilt from the O-C; the offset from T0 does not change delta BIC.
        if survives['C1']:
            try:
                t_loo = epochs[m] * period + oc[m] / (24 * 60)
                r = EphemerisAnalyzer(epochs[m], t_loo, oc_err[m] / (24 * 60)).analyze()
                if not (r.delta_bic is not None and r.delta_bic > delta_bic_threshold):
                    fail('C1', i)
            except Exception:
                fail('C1', i)

        # C3: weighted O-C rms over the median error
        if survives['C3']:
            rms = weighted_oc_rms(oc[m], oc_err[m])
            med = float(np.median(oc_err[m]))
            if rms is None or med <= 0 or rms / med <= oc_rms_over_err_threshold:
                fail('C3', i)

        # C2: permutation FAP of the Lomb-Scargle peak
        if survives['C2']:
            if freqs is None or m.sum() < 5:
                fail('C2', i)
            else:
                t_loo, oc_loo, err_loo = time_epochs[m], oc[m], oc_err[m]
                obs_peak = float(LombScargle(t_loo, oc_loo, dy=err_loo)
                                 .power(freqs, normalization='psd').max())
                fap = _permutation_fap(t_loo, oc_loo, err_loo, freqs, obs_peak,
                                       42 + i, pool, n_workers)
                if fap >= fap_threshold:
                    fail('C2', i)

        if not any(survives.values()):
            break
    return survives, fail_at


def find_ttv_candidates(
    results_dir='autottv_results_v2',
    delta_bic_threshold=config.TTV_DELTA_BIC_THRESHOLD,
    fap_threshold=config.TTV_FAP_THRESHOLD,
    oc_rms_over_err_threshold=config.TTV_OC_RMS_OVER_ERR_THRESHOLD,
    require_converged=False,
    exclude_tois=None,
    output_csv='ttv_candidates.csv',
    verbose=True,
    n_workers=None
):
    """
    Scan all results.json files and find TTV candidates.

    Parameters
    ----------
    results_dir : str
        Directory containing TOI_*/results.json files.
    delta_bic_threshold : float
        Minimum delta BIC (linear - quadratic).
        Default: config.TTV_DELTA_BIC_THRESHOLD (6.0).
    fap_threshold : float
        Maximum bootstrap FAP. Default: 0.01.
    oc_rms_over_err_threshold : float
        Minimum weighted O-C RMS / median error ratio.
        Default: config.TTV_OC_RMS_OVER_ERR_THRESHOLD (2.0).
    require_converged : bool
        If True, only include converged TOIs. Default: False.
    output_csv : str
        Output CSV file path.
    verbose : bool
        Print progress and results.
    n_workers : int, optional
        Processes for the C2 leave-one-out permutations. Default: all CPUs.

    Returns
    -------
    list of dict
        TTV candidates.
    """
    all_tois = sorted(glob.glob(os.path.join(results_dir, 'TOI_*/results.json')))

    candidates = []
    n_total = 0
    n_c1 = 0
    n_c2 = 0
    n_c3 = 0
    n_c1c2 = 0
    n_c1c3 = 0
    n_c2c3 = 0

    n_workers = n_workers or os.cpu_count() or 1
    pool = None  # started at the first C2 leave-one-out test

    for f in all_tois:
        try:
            with open(f) as fh:
                r = json.load(fh)
        except Exception:
            continue

        toi = r.get('toi', os.path.basename(os.path.dirname(f)).replace('TOI_', '').replace('_', '.'))
        tic_id = r.get('tic_id', '')
        n_total += 1

        # Skip excluded TOIs
        if exclude_tois and toi in exclude_tois:
            continue

        # Check convergence
        converged = r.get('convergence', {}).get('converged', False)
        if require_converged and not converged:
            continue

        rhat = r.get('convergence', {}).get('summary', {}).get('max_rhat', None)
        period = r.get('parameters', {}).get('period', {}).get('value', None)

        # C1: Delta BIC
        eph = r.get('ephemeris', {})
        delta_bic = eph.get('delta_bic', None)
        c1 = delta_bic is not None and delta_bic > delta_bic_threshold

        # C2: Periodogram FAP
        peri = r.get('periodogram', {})
        fap = peri.get('bootstrap_fap', None) if peri else None
        c2 = fap is not None and fap < fap_threshold

        # C3: O-C scatter / median error.
        # Scatter is the inverse-variance-weighted RMS about the weighted
        # mean — computed on the fly from oc_values so we don't depend on
        # the legacy stored oc_rms_minutes (which used unweighted RMS-about-zero
        # in older catalog snapshots).
        ind = r.get('individual_transits', {})
        oc_values = ind.get('oc_values', [])
        if len(oc_values) > 0:
            import numpy as np
            oc_arr  = np.array([v['oc_minutes']     for v in oc_values
                                if 'oc_minutes' in v and 'oc_err_minutes' in v], dtype=float)
            err_arr = np.array([v['oc_err_minutes'] for v in oc_values
                                if 'oc_minutes' in v and 'oc_err_minutes' in v], dtype=float)
            oc_rms = weighted_oc_rms(oc_arr, err_arr)
            oc_median_err = float(np.median(err_arr)) if err_arr.size else None
            oc_ratio = (oc_rms / oc_median_err
                        if oc_rms is not None and oc_median_err and oc_median_err > 0
                        else None)
        else:
            oc_rms = None
            oc_ratio = None
            oc_median_err = None
        c3 = oc_ratio is not None and oc_ratio > oc_rms_over_err_threshold

        # Count individual criteria
        if c1: n_c1 += 1
        if c2: n_c2 += 1
        if c3: n_c3 += 1
        if c1 and c2: n_c1c2 += 1
        if c1 and c3: n_c1c3 += 1
        if c2 and c3: n_c2c3 += 1

        # Any one criterion is sufficient, but must also pass LOO (computed below)
        if c1 or c2 or c3:
            # Get additional info
            n_transits = ind.get('n_transits_used', 0)
            oc_rms_out = ind.get('oc_rms_minutes', None)

            # C4: per-criterion leave-one-out test
            if len(oc_values) >= 4:
                loo_epochs = np.array([v['epoch'] for v in oc_values])
                loo_oc = np.array([v['oc_minutes'] for v in oc_values])
                loo_err = np.array([v['oc_err_minutes'] for v in oc_values])
                if c2 and pool is None and n_workers > 1:
                    pool = Pool(n_workers)
                loo, _ = leave_one_out_test(
                    loo_epochs, loo_oc, loo_err, period,
                    delta_bic_threshold, fap_threshold, oc_rms_over_err_threshold,
                    c1, c2, c3, pool=pool, n_workers=n_workers)
            else:
                loo = {'C1': False if c1 else None, 'C2': False if c2 else None,
                       'C3': False if c3 else None}
            c4 = any(v is True for v in loo.values())
            loo_n_fail = sum(1 for v in loo.values() if v is False)  # criteria that failed LOO
            ttv_type = ('Periodic' if loo['C2'] else 'Quadratic' if loo['C1']
                        else 'Scatter' if loo['C3'] else None)

            # dP/dE from quadratic ephemeris
            quad = eph.get('quadratic', {})
            dPdE = quad.get('dPdE', None)
            dPdE_err = quad.get('dPdE_err', None)

            # TTV period from periodogram
            ttv_period = peri.get('peak_period', None) if peri else None

            # Disposition from catalog
            disposition = r.get('catalog', {}).get('disposition', None)
            if disposition is None:
                # Try to get from catalog
                try:
                    from autottv_pipeline_v2.data_loader import load_toi_catalog, get_toi_parameters
                    cat = load_toi_catalog()
                    pp = get_toi_parameters(cat, toi=toi)
                    disposition = pp.get('disposition', None) if pp else None
                except:
                    pass

            candidates.append({
                'TOI': toi,
                'TIC_ID': tic_id,
                'Disposition': disposition,
                'Period': period,
                'Rhat': rhat,
                'N_tr': n_transits,
                'dBIC': delta_bic,
                'FAP': fap,
                'OC_ratio': oc_ratio,
                'TTV_period': ttv_period,
                'OC_rms_min': oc_rms_out,
                'OC_median_err_min': oc_median_err,
                'dPdE': dPdE,
                'dPdE_err': dPdE_err,
                'C1': c1,
                'C2': c2,
                'C3': c3,
                'C1_LOO': loo['C1'],
                'C2_LOO': loo['C2'],
                'C3_LOO': loo['C3'],
                'TTV type': ttv_type,
                'C4_LOO': c4,
                'LOO_n_fail': loo_n_fail,
                'converged': converged,
            })

    if pool is not None:
        pool.close()
        pool.join()

    # Sort by delta BIC (strongest first)
    candidates.sort(key=lambda x: -(x['dBIC'] or 0))

    # Filter to only LOO-passing candidates for the output file
    ttv_candidates = [c for c in candidates if c.get('C4_LOO')]

    # Write CSV (only LOO-passing)
    if output_csv:
        fieldnames = ['TOI', 'TIC_ID', 'Disposition', 'Period', 'Rhat', 'N_tr', 'dBIC', 'FAP',
                      'OC_ratio', 'TTV_period', 'dPdE', 'dPdE_err', 'C1', 'C2', 'C3',
                      'C1_LOO', 'C2_LOO', 'C3_LOO', 'TTV type', 'C4_LOO', 'LOO_n_fail']
        with open(output_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(ttv_candidates)

    if verbose:
        print(f"TTV Candidate Search Results")
        print(f"{'='*50}")
        print(f"Total TOIs scanned: {n_total}")
        print(f"Thresholds: dBIC > {delta_bic_threshold}, FAP < {fap_threshold}, OC_ratio > {oc_rms_over_err_threshold}")
        print(f"")
        print(f"Individual criteria:")
        print(f"  C1 (dBIC > {delta_bic_threshold}):        {n_c1}")
        print(f"  C2 (FAP < {fap_threshold}):          {n_c2}")
        print(f"  C3 (OC_ratio > {oc_rms_over_err_threshold}):    {n_c3}")
        print(f"")
        print(f"Pairwise:")
        print(f"  C1 & C2:                    {n_c1c2}")
        print(f"  C1 & C3:                    {n_c1c3}")
        print(f"  C2 & C3:                    {n_c2c3}")
        print(f"")
        n_any = len(candidates)
        n_c4 = sum(1 for c in candidates if c.get('C4_LOO'))
        print(f"Any one (C1 | C2 | C3):       {n_any}")
        print(f"C4 (LOO survives):            {n_c4}")
        for key, name in (("C2", "Periodic"), ("C1", "Quadratic"), ("C3", "Scatter")):
            n_surv = sum(1 for c in candidates if c.get(f"{key}_LOO") is True)
            n_type = sum(1 for c in candidates if c.get("TTV type") == name)
            print(f"  {key} survives LOO: {n_surv:5d}   type {name}: {n_type}")
        print(f"")
        if candidates:
            print(f"Saved: {output_csv}")
            print(f"")
            print(f"{'TOI':<12} {'dBIC':>10} {'FAP':>12} {'OC_ratio':>10} {'N_tr':>6} {'Rhat':>8} {'LOO':>5}")
            print(f"{'-'*67}")
            for c in candidates:
                fap_str = f"{c['FAP']:.1e}" if c['FAP'] else '?'
                loo_str = c['TTV type'][:4] if c.get('C4_LOO') else 'FAIL'
                print(f"{c['TOI']:<12} {c['dBIC']:>10.2f} {fap_str:>12} {c['OC_ratio']:>10.2f} {c['N_tr']:>6} {c['Rhat']:>8.4f} {loo_str:>5}")

    return candidates


def load_rejected_tois(xlsx_path='rejected_TOIs_list.xlsx', csv_path='rejected_TOIs_list.csv'):
    """Load the rejected-TOI list.

    Reads rejected_TOIs_list.xlsx (TOI and TIC ID in the first two columns,
    no header row) when it exists, as in the private repo, and otherwise
    rejected_TOIs_list.csv, its export in the public repo. Re-export the CSV
    after editing the xlsx.
    """
    if not os.path.exists(xlsx_path) and os.path.exists(csv_path):
        with open(csv_path, newline='') as f:
            return {row['TOI'].strip() for row in csv.DictReader(f) if '.' in row['TOI']}
    import openpyxl
    wb = openpyxl.load_workbook(xlsx_path)
    ws = wb.active
    rejected = set()
    for row in ws.iter_rows(min_row=1, values_only=True):  # no header row; text cells are skipped below
        toi = row[0]
        if toi and isinstance(toi, (int, float, str)):
            s = str(toi)
            if '.' in s:
                rejected.add(s)
    return rejected


if __name__ == '__main__':
    rejected = load_rejected_tois()
    print(f"Excluding {len(rejected)} rejected TOIs\n")
    find_ttv_candidates(exclude_tois=rejected)
