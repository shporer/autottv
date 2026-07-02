"""
Utility to calculate the expected signal-to-noise ratio (SNR) of individual
transits for each TOI.

Steps:
  1. Determine 1-hour CDPP for each TOI:
     - If SPOC FITS CDPP is available for any sector, use the mean across
       those sectors with 3-sigma outlier rejection.
     - Otherwise, use the mean SPOC CDPP of all TOIs within +/-0.25 mag
       of this TOI's TESS magnitude (i.e., a 0.5-mag bin), with 3-sigma
       outlier rejection.
  2. Scale to transit-duration CDPP:
       CDPP_transit = CDPP_1hr / sqrt(duration_hours)
  3. Calculate SNR:
       SNR = depth_ppm / CDPP_transit
"""

import numpy as np
import pandas as pd
from collections import defaultdict
from pathlib import Path


def sigma_clip_mean(data, sigma=3, max_iter=10):
    """Iterative sigma clipping, returns mean of survivors."""
    cleaned = np.asarray(data, dtype=float).copy()
    for _ in range(max_iter):
        n = len(cleaned)
        if n < 3:
            break
        mean = np.mean(cleaned)
        std = np.std(cleaned, ddof=1)
        if std == 0:
            break
        keep = np.abs(cleaned - mean) <= sigma * std
        if keep.all():
            break
        cleaned = cleaned[keep]
    return float(np.mean(cleaned))


def compute_transit_snr(
    catalog_path='toi_catalog_for_ttv.csv',
    catalog_df=None,
    spoc_cdpp_path='autottv_results_v2/spoc_cdpp_sampled.npy',
    mag_half_window=0.25,
):
    """
    Compute expected individual-transit SNR for each TOI.

    Parameters
    ----------
    catalog_path : str
        Path to the TOI catalog CSV.
    catalog_df : pd.DataFrame, optional
        If provided, use this DataFrame directly instead of reading from
        catalog_path. The DataFrame must have the same columns as the CSV.
    spoc_cdpp_path : str
        Path to the SPOC CDPP structured numpy array (.npy) with fields
        'tic', 'tmag', 'sector', 'cdpp'.
    mag_half_window : float
        Half-width of the magnitude bin for the fallback CDPP lookup (default
        0.25 mag, giving a 0.5 mag bin).

    Returns
    -------
    result : pd.DataFrame
        One row per TOI with columns: TOI, TIC_ID, TESS_Mag, Depth_ppm,
        Duration_hrs, CDPP_1hr, CDPP_transit, SNR, CDPP_source.
    """
    base = Path(catalog_path).parent

    # --- Load catalog ---
    if catalog_df is not None:
        catalog = catalog_df.copy()
    else:
        catalog = pd.read_csv(catalog_path)
    catalog['TIC ID'] = catalog['TIC ID'].astype(int)

    # --- Load SPOC FITS CDPP ---
    spoc_data = np.load(spoc_cdpp_path)
    spoc_by_tic = defaultdict(list)
    for row in spoc_data:
        spoc_by_tic[int(row['tic'])].append(float(row['cdpp']))

    # --- Per-TOI SPOC mean CDPP (3-sigma clipping) ---
    spoc_per_toi = {}
    for tic_id, vals in spoc_by_tic.items():
        spoc_per_toi[tic_id] = sigma_clip_mean(vals)

    # --- Build the fallback: mean SPOC CDPP in 0.5-mag bins ---
    # Collect all per-TOI SPOC CDPP values with their magnitudes
    spoc_tmags = []
    spoc_cdpps = []
    for tic_id, cdpp in spoc_per_toi.items():
        row = catalog.loc[catalog['TIC ID'] == tic_id]
        if len(row) > 0:
            tmag = float(row['TESS Mag'].iloc[0])
            if np.isfinite(tmag):
                spoc_tmags.append(tmag)
                spoc_cdpps.append(cdpp)
    spoc_tmags = np.array(spoc_tmags)
    spoc_cdpps = np.array(spoc_cdpps)

    def fallback_cdpp(tmag):
        """Mean SPOC CDPP of TOIs within +/-0.25 mag, with 3-sigma clipping."""
        mask = np.abs(spoc_tmags - tmag) <= mag_half_window
        if mask.sum() < 3:
            return np.nan
        return sigma_clip_mean(spoc_cdpps[mask])

    # --- Compute SNR for each TOI ---
    results = []
    for _, row in catalog.iterrows():
        tic_id = int(row['TIC ID'])
        toi = row['TOI']
        tmag = row['TESS Mag']
        depth_ppm = row['Depth (ppm)']
        dur_hrs = row['Duration (hours)']

        if not (np.isfinite(depth_ppm) and np.isfinite(dur_hrs) and dur_hrs > 0):
            results.append({
                'TOI': toi, 'TIC_ID': tic_id, 'TESS_Mag': tmag,
                'Depth_ppm': depth_ppm, 'Duration_hrs': dur_hrs,
                'CDPP_1hr': np.nan, 'CDPP_transit': np.nan,
                'SNR': np.nan, 'CDPP_source': 'missing_data',
            })
            continue

        # Step 1: 1-hour CDPP
        if tic_id in spoc_per_toi:
            cdpp_1hr = spoc_per_toi[tic_id]
            source = 'SPOC_FITS'
        else:
            cdpp_1hr = fallback_cdpp(tmag)
            source = 'mag_bin'

        # Step 2: scale to transit duration
        cdpp_transit = cdpp_1hr / np.sqrt(dur_hrs)

        # Step 3: SNR
        snr = depth_ppm / cdpp_transit if cdpp_transit > 0 else np.nan

        results.append({
            'TOI': toi, 'TIC_ID': tic_id, 'TESS_Mag': tmag,
            'Depth_ppm': depth_ppm, 'Duration_hrs': dur_hrs,
            'CDPP_1hr': cdpp_1hr, 'CDPP_transit': cdpp_transit,
            'SNR': snr, 'CDPP_source': source,
        })

    return pd.DataFrame(results)


if __name__ == '__main__':
    base = Path(__file__).resolve().parent
    df = compute_transit_snr(
        catalog_path=str(base / 'toi_catalog_240226_for_ttv.csv'),
        spoc_cdpp_path=str(base / 'autottv_results_v2' / 'spoc_cdpp_sampled.npy'),
    )

    # Save
    out_path = base / 'autottv_results_v2' / 'transit_snr.csv'
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} TOIs to {out_path}")

    # Summary
    valid = df.dropna(subset=['SNR'])
    print(f"\nValid SNR: {len(valid)} / {len(df)} TOIs")
    print(f"CDPP source: {df['CDPP_source'].value_counts().to_dict()}")
    print(f"\nSNR statistics:")
    print(f"  Median: {valid['SNR'].median():.1f}")
    print(f"  Mean:   {valid['SNR'].mean():.1f}")
    print(f"  Min:    {valid['SNR'].min():.1f}")
    print(f"  Max:    {valid['SNR'].max():.1f}")
    print(f"  SNR > 10: {(valid['SNR'] > 10).sum()} ({100*(valid['SNR'] > 10).mean():.1f}%)")
    print(f"  SNR > 5:  {(valid['SNR'] > 5).sum()} ({100*(valid['SNR'] > 5).mean():.1f}%)")
    print(f"  SNR > 3:  {(valid['SNR'] > 3).sum()} ({100*(valid['SNR'] > 3).mean():.1f}%)")
