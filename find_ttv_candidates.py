#!/usr/bin/env python3
"""
Search for TTV candidates among all analyzed TOIs using three detection criteria:

  C1: Delta BIC (linear - quadratic) > TTV_DELTA_BIC_THRESHOLD (quadratic preferred)
  C2: Periodogram bootstrap FAP < TTV_FAP_THRESHOLD
  C3: O-C RMS / mean error > TTV_OC_RMS_OVER_ERR_THRESHOLD

Outputs:
  - ttv_candidates.csv: TOIs meeting ALL three criteria
  - ttv_candidates_summary.txt: summary statistics
"""

import json
import glob
import os
import sys
import csv
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


def leave_one_out_test(epochs, oc_minutes, oc_err_minutes, period,
                       delta_bic_threshold, fap_threshold, oc_rms_over_err_threshold,
                       c1_pass, c2_pass, c3_pass):
    """
    Leave-one-out (LOO) jackknife test for TTV detection robustness.

    For each transit, remove it and recheck whichever criteria (C1/C2/C3)
    the TOI originally passed. The TOI passes C4 if all LOO iterations
    still satisfy at least one of the originally-passing criteria.

    Returns
    -------
    passes_loo : bool
        True if the detection survives removal of any single transit.
    n_fail : int
        Number of LOO iterations where no criterion was satisfied.
    """
    n = len(epochs)
    if n < 4:  # Need at least 3 transits after removing one
        return False, n

    epochs = np.asarray(epochs, dtype=float)
    oc = np.asarray(oc_minutes)
    oc_err = np.asarray(oc_err_minutes)
    time_epochs = epochs * period  # days

    # Precompute frequency grid for periodogram. Matches the primary C2
    # test (astropy LombScargle, weighted, median-cadence Nyquist with
    # PERIODOGRAM_OVERSAMPLING). See docs/notes/2026-04-23-c4-loo-periodogram-inconsistency.md.
    time_span = time_epochs.max() - time_epochs.min()
    if time_span > 0 and len(time_epochs) >= 2:
        dt_med = float(np.median(np.diff(np.sort(time_epochs))))
        f_min = 2.0 / time_span
        f_nyq = 0.5 / max(dt_med, 1e-9)
        if f_nyq > f_min:
            n_freq = max(200, int(config.PERIODOGRAM_OVERSAMPLING
                                  * time_span * (f_nyq - f_min)))
            frequencies = np.linspace(f_min, f_nyq, n_freq)
        else:
            frequencies = None
    else:
        frequencies = None

    for i in range(n):
        # Remove transit i
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        ep_loo = epochs[mask]
        oc_loo = oc[mask]
        oc_err_loo = oc_err[mask]
        t_ep_loo = time_epochs[mask]

        any_pass = False

        # Re-check C1 (delta BIC) if it originally passed
        if c1_pass and len(ep_loo) >= 3:
            try:
                eph = EphemerisAnalyzer(ep_loo, ep_loo * period + oc_loo / (24 * 60),
                                        oc_err_loo / (24 * 60))
                result = eph.analyze()
                if result.delta_bic > delta_bic_threshold:
                    any_pass = True
            except Exception:
                pass

        # Re-check C2 (periodogram with bootstrap FAP) if it originally
        # passed. Uses astropy error-weighted LombScargle (PSD normalization)
        # to match the primary C2 test that produces
        # results.json.bootstrap_fap.
        # See docs/notes/2026-04-23-c4-loo-periodogram-inconsistency.md.
        if c2_pass and not any_pass and frequencies is not None and len(oc_loo) >= 5:
            ls_loo = LombScargle(t_ep_loo, oc_loo, dy=oc_err_loo)
            obs_peak = float(ls_loo.power(frequencies, normalization='psd').max())

            # Bootstrap FAP with early stopping once running FAP > threshold
            n_bootstrap = 100_000
            check_interval = 500
            n_exceed = 0
            rng = np.random.default_rng(seed=42 + i)

            for j in range(n_bootstrap):
                perm = rng.permutation(len(oc_loo))
                p_shuf = LombScargle(t_ep_loo, oc_loo[perm], dy=oc_err_loo[perm]).power(
                    frequencies, normalization='psd')
                if p_shuf.max() >= obs_peak:
                    n_exceed += 1
                if (j + 1) % check_interval == 0:
                    running_fap = (n_exceed + 1) / (j + 2)
                    if running_fap > fap_threshold:
                        break

            fap_loo = (n_exceed + 1) / (j + 2)
            if fap_loo < fap_threshold:
                any_pass = True

        # Re-check C3 (O-C ratio) if it originally passed.
        # Scatter is the inverse-variance-weighted RMS about the weighted mean.
        if c3_pass and not any_pass:
            rms_loo = weighted_oc_rms(oc_loo, oc_err_loo)
            median_err_loo = np.median(oc_err_loo)
            if rms_loo is not None and median_err_loo > 0:
                ratio_loo = rms_loo / median_err_loo
                if ratio_loo > oc_rms_over_err_threshold:
                    any_pass = True

        # Early termination: one failure means C4 fails
        if not any_pass:
            return False, i + 1

    return True, 0


def find_ttv_candidates(
    results_dir='autottv_results_v2',
    delta_bic_threshold=config.TTV_DELTA_BIC_THRESHOLD,
    fap_threshold=config.TTV_FAP_THRESHOLD,
    oc_rms_over_err_threshold=config.TTV_OC_RMS_OVER_ERR_THRESHOLD,
    require_converged=False,
    exclude_tois=None,
    output_csv='ttv_candidates.csv',
    verbose=True
):
    """
    Scan all results.json files and find TTV candidates.

    Parameters
    ----------
    results_dir : str
        Directory containing TOI_*/results.json files.
    delta_bic_threshold : float
        Minimum delta BIC (linear - quadratic). Default: 0.0.
    fap_threshold : float
        Maximum bootstrap FAP. Default: 0.01.
    oc_rms_over_err_threshold : float
        Minimum O-C RMS / mean error ratio. Default: 3.0.
    require_converged : bool
        If True, only include converged TOIs. Default: False.
    output_csv : str
        Output CSV file path.
    verbose : bool
        Print progress and results.

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

            # C4: Leave-one-out test
            if len(oc_values) >= 4:
                loo_epochs = np.array([v['epoch'] for v in oc_values])
                loo_oc = np.array([v['oc_minutes'] for v in oc_values])
                loo_err = np.array([v['oc_err_minutes'] for v in oc_values])
                c4, loo_n_fail = leave_one_out_test(
                    loo_epochs, loo_oc, loo_err, period,
                    delta_bic_threshold, fap_threshold, oc_rms_over_err_threshold,
                    c1, c2, c3)
            else:
                c4 = False
                loo_n_fail = len(oc_values)

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
                'C4_LOO': c4,
                'LOO_n_fail': loo_n_fail,
                'converged': converged,
            })

    # Sort by delta BIC (strongest first)
    candidates.sort(key=lambda x: -(x['dBIC'] or 0))

    # Filter to only LOO-passing candidates for the output file
    ttv_candidates = [c for c in candidates if c.get('C4_LOO')]

    # Write CSV (only LOO-passing)
    if output_csv:
        fieldnames = ['TOI', 'TIC_ID', 'Disposition', 'Period', 'Rhat', 'N_tr', 'dBIC', 'FAP',
                      'OC_ratio', 'TTV_period', 'dPdE', 'dPdE_err', 'C1', 'C2', 'C3', 'C4_LOO', 'LOO_n_fail']
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
        print(f"")
        if candidates:
            print(f"Saved: {output_csv}")
            print(f"")
            print(f"{'TOI':<12} {'dBIC':>10} {'FAP':>12} {'OC_ratio':>10} {'N_tr':>6} {'Rhat':>8} {'LOO':>5}")
            print(f"{'-'*67}")
            for c in candidates:
                fap_str = f"{c['FAP']:.1e}" if c['FAP'] else '?'
                loo_str = 'PASS' if c.get('C4_LOO') else f"F{c.get('LOO_n_fail', '?')}"
                print(f"{c['TOI']:<12} {c['dBIC']:>10.2f} {fap_str:>12} {c['OC_ratio']:>10.2f} {c['N_tr']:>6} {c['Rhat']:>8.4f} {loo_str:>5}")

    return candidates


def load_rejected_tois(xlsx_path='rejected_TOIs.xlsx'):
    """Load rejected TOI list from xlsx."""
    import openpyxl
    wb = openpyxl.load_workbook(xlsx_path)
    ws = wb.active
    rejected = set()
    for row in ws.iter_rows(min_row=2, values_only=True):
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
