"""
Filter the raw TOI catalog through four sequential steps to produce the
analysis catalog for TTV analysis.

Steps:
  1. Remove TOIs with TFOPWG disposition FP or FA
  2. Remove TOIs with period <= 0 or NaN
  3. Remove TOIs with expected single-transit SNR <= 5
  4. Remove TOIs with <= 5 full transits in TESS data

Usage:
  python filter_toi_catalog.py              # Default settings from config
  python filter_toi_catalog.py --cpus=16    # Override worker count for Step 4
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from multiprocessing import Pool

from autottv_pipeline_v2 import config
from autottv_pipeline_v2.data_loader import DataLoader
from compute_transit_snr import compute_transit_snr
from run_full_analysis import identify_transits, get_min_points_for_cadence


def _count_transits_for_toi(args):
    """
    Worker function to count full transits for a single TOI.

    Parameters
    ----------
    args : tuple
        (toi, tic_id, period, t0, duration_hr, cache_dir, max_sector)

    Returns
    -------
    tuple
        (toi, transit_count)
    """
    toi, tic_id, period, t0, duration_hr, cache_dir, max_sector = args

    if not (np.isfinite(period) and period > 0 and
            np.isfinite(t0) and np.isfinite(duration_hr) and duration_hr > 0):
        return (toi, 0)

    try:
        loader = DataLoader(tic_id)
        if not loader.load_from_npz_cache(cache_dir):
            if not loader.download_from_mast(max_sector):
                return (toi, 0)
            loader.save_to_npz_cache(cache_dir)

        if not loader.lightcurves:
            return (toi, 0)

        # Combine time and cadence arrays from all sectors
        time_all = np.concatenate([lc.time for lc in loader.lightcurves])
        cadence_all = np.concatenate(
            [np.full(len(lc.time), lc.cadence) for lc in loader.lightcurves]
        )

        # Count full transits
        duration_days = duration_hr / 24.0
        transits, _ = identify_transits(
            time_all, period, t0, duration_days,
            require_full_coverage=True, cadence=cadence_all
        )
        return (toi, len(transits))

    except Exception:
        return (toi, 0)


def filter_toi_catalog(
    catalog_path=config.CATALOG_FILE,
    output_path=config.CATALOG_FILE_FILTERED,
    spoc_cdpp_path=config.SPOC_CDPP_FILE,
    excluded_dispositions=config.EXCLUDED_DISPOSITIONS,
    min_period_days=config.MIN_PERIOD_DAYS,
    min_transit_snr=config.MIN_TRANSIT_SNR,
    min_full_transits=config.MIN_FULL_TRANSITS,
    cache_dir=config.DATA_CACHE_DIR,
    max_sector=config.MAX_SECTOR,
    n_workers=16,
    verbose=True,
):
    """
    Filter the raw TOI catalog through four sequential steps.

    Parameters
    ----------
    catalog_path : Path
        Path to the raw TOI catalog CSV.
    output_path : Path
        Path to write the filtered catalog CSV.
    spoc_cdpp_path : Path
        Path to the SPOC CDPP numpy file for SNR computation.
    excluded_dispositions : list of str
        TFOPWG dispositions to exclude (default: ['FP', 'FA']).
    min_period_days : float
        Remove TOIs with period <= this value (default: 0.0).
    min_transit_snr : float
        Minimum expected single-transit SNR (default: 5.0).
    min_full_transits : int
        Minimum number of full transits in TESS data (default: 5).
    cache_dir : Path
        Directory for light curve npz cache.
    max_sector : int
        Maximum TESS sector to include.
    n_workers : int
        Number of parallel workers for Step 4.
    verbose : bool
        Print progress messages.

    Returns
    -------
    pd.DataFrame
        Filtered catalog DataFrame.
    """
    catalog_path = Path(catalog_path)
    output_path = Path(output_path)
    spoc_cdpp_path = Path(spoc_cdpp_path)
    cache_dir = Path(cache_dir)

    # Load raw catalog
    df = pd.read_csv(catalog_path)
    n_initial = len(df)
    if verbose:
        print(f"Loaded {n_initial} TOIs from {catalog_path.name}")

    # Track counts for summary
    counts = [n_initial]

    # =========================================================================
    # Step 1: Disposition filter
    # =========================================================================
    disp_col = 'TFOPWG Disposition'
    mask_disp = ~df[disp_col].isin(excluded_dispositions)
    n_removed_1 = (~mask_disp).sum()
    df = df[mask_disp].reset_index(drop=True)
    counts.append(len(df))
    if verbose:
        print(f"Step 1: Removed {n_removed_1} {'/'.join(excluded_dispositions)} "
              f"-> {len(df)} remaining")

    # =========================================================================
    # Step 2: Period filter
    # =========================================================================
    period_col = 'Period (days)'
    mask_period = df[period_col].notna() & (df[period_col] > min_period_days)
    n_removed_2 = (~mask_period).sum()
    df = df[mask_period].reset_index(drop=True)
    counts.append(len(df))
    if verbose:
        print(f"Step 2: Removed {n_removed_2} with period<=0 or NaN "
              f"-> {len(df)} remaining")

    # =========================================================================
    # Step 3: SNR filter
    # =========================================================================
    snr_df = compute_transit_snr(
        catalog_path=str(catalog_path),
        catalog_df=df,
        spoc_cdpp_path=str(spoc_cdpp_path),
        mag_half_window=config.CDPP_MAG_HALF_WINDOW,
    )

    # Merge SNR column back by TOI
    df = df.merge(
        snr_df[['TOI', 'SNR']],
        on='TOI',
        how='left',
    )
    mask_snr = df['SNR'].notna() & (df['SNR'] > min_transit_snr)
    n_removed_3 = (~mask_snr).sum()
    df = df[mask_snr].reset_index(drop=True)
    df = df.drop(columns=['SNR'])
    counts.append(len(df))
    if verbose:
        print(f"Step 3: Removed {n_removed_3} with SNR<={min_transit_snr} "
              f"-> {len(df)} remaining")

    # =========================================================================
    # Step 4: Transit count filter (parallel)
    # =========================================================================
    if verbose:
        print(f"Step 4: Counting full transits for {len(df)} TOIs "
              f"({n_workers} workers)...")

    # Prepare worker arguments — keyed by TOI (not TIC) so multi-planet systems
    # each get their own transit count with their own period/epoch/duration
    df['TIC ID'] = df['TIC ID'].astype(int)
    worker_args = []
    for _, row in df.iterrows():
        toi = row['TOI']
        tic_id = int(row['TIC ID'])
        period = row['Period (days)']
        t0 = row['Epoch (BJD)']
        duration_hr = row['Duration (hours)']
        worker_args.append((toi, tic_id, period, t0, duration_hr, cache_dir, max_sector))

    # Run transit counting in parallel
    transit_counts = {}
    try:
        from tqdm import tqdm
        has_tqdm = True
    except ImportError:
        has_tqdm = False

    with Pool(n_workers) as pool:
        if has_tqdm:
            results = list(tqdm(
                pool.imap_unordered(_count_transits_for_toi, worker_args),
                total=len(worker_args),
                desc="Counting transits",
            ))
        else:
            results = list(pool.imap_unordered(_count_transits_for_toi, worker_args))
            if verbose:
                print(f"  Processed {len(results)} TOIs")

    for toi, count in results:
        transit_counts[toi] = count

    # Map transit counts back to DataFrame
    df['_transit_count'] = df['TOI'].map(transit_counts).fillna(0).astype(int)
    mask_transits = df['_transit_count'] > min_full_transits
    n_removed_4 = (~mask_transits).sum()
    df = df[mask_transits].reset_index(drop=True)
    df = df.drop(columns=['_transit_count'])
    counts.append(len(df))
    if verbose:
        print(f"Step 4: Removed {n_removed_4} with <={min_full_transits} transits "
              f"-> {len(df)} remaining")

    # =========================================================================
    # Save and summarize
    # =========================================================================
    df.to_csv(output_path, index=False)
    if verbose:
        print(f"\nSaved {len(df)} TOIs to {output_path.name}")
        print(f"\n{'='*60}")
        print(f"{'Step':<30} {'From':>6} -> {'To':>6}  ({'Removed':>7})")
        print(f"{'='*60}")
        labels = [
            f"Step 1 (Disposition)",
            f"Step 2 (Period > 0)",
            f"Step 3 (SNR > {min_transit_snr})",
            f"Step 4 (Transits > {min_full_transits})",
        ]
        for i, label in enumerate(labels):
            print(f"{label:<30} {counts[i]:>6} -> {counts[i+1]:>6}  "
                  f"({counts[i] - counts[i+1]:>7})")
        print(f"{'='*60}")

    return df


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Filter TOI catalog for TTV analysis'
    )
    parser.add_argument('--cpus', type=int, default=16,
                        help='Workers for Step 4 downloads (default: 16)')
    args = parser.parse_args()
    filter_toi_catalog(n_workers=args.cpus)
