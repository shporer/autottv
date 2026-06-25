#!/usr/bin/env python3
"""
AutoTTV Pipeline v2.0 - Main Entry Point

Complete TESS transit timing analysis pipeline using:
- Mandel & Agol (2002) transit model
- Claret 2017 limb darkening coefficients (PHOENIX r-method)
- MCMC fitting with convergence checking
- Linear/quadratic ephemeris analysis
- Lomb-Scargle periodogram for TTV detection

Usage:
    python3 -m autottv_pipeline_v2.main --tic 402026209
    python3 -m autottv_pipeline_v2.main --toi 232.01
    python3 -m autottv_pipeline_v2.main --batch --limit 10
    python3 -m autottv_pipeline_v2.main --batch --parallel 8
"""

import os
import sys
import argparse
import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List
import traceback

import numpy as np

from . import config
from .data_loader import DataLoader, load_toi_catalog, get_toi_parameters
from .phase_fold_fitter import PhaseFoldFitter
from .individual_transit_fitter import IndividualTransitFitter
from .ephemeris_analysis import EphemerisAnalyzer, analyze_ephemeris
from .utils import json_serializer
from .periodogram import compute_ttv_periodogram
from .plotting import (
    plot_mcmc_chains, plot_corner, plot_phase_folded_lightcurve,
    plot_oc_diagram, plot_oc_comparison, plot_periodogram,
    create_summary_figure
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(config.OUTPUT_DIR / 'pipeline.log')
    ]
)
logger = logging.getLogger(__name__)


class AutoTTVPipeline:
    """
    Main pipeline class for TESS transit timing analysis.
    """

    def __init__(self, tic_id: int = None, toi: str = None,
                 output_dir: Path = None):
        """
        Initialize pipeline.

        Parameters
        ----------
        tic_id : int, optional
            TIC ID to analyze
        toi : str, optional
            TOI designation to analyze
        output_dir : Path, optional
            Output directory for results
        """
        self.tic_id = tic_id
        self.toi = toi
        self.output_dir = Path(output_dir) if output_dir else config.OUTPUT_DIR

        # Will be populated during analysis
        self.catalog = None
        self.planet_params = None
        self.data_loader = None
        self.time = None
        self.flux = None
        self.flux_err = None

        # Results
        self.step1_results = None
        self.transit_results = None
        self.ephemeris_results = None
        self.periodogram_results = None

        # Create output directories
        self.tic_output_dir = None

    def load_catalog(self) -> bool:
        """Load TOI catalog and get target parameters."""
        logger.info("=" * 70)
        logger.info("STEP 0: Loading TOI Catalog")
        logger.info("=" * 70)

        try:
            self.catalog = load_toi_catalog()
            logger.info(f"Loaded {len(self.catalog)} TOI entries")

            self.planet_params = get_toi_parameters(
                self.catalog,
                tic_id=self.tic_id,
                toi=self.toi
            )

            if self.planet_params is None:
                logger.error(f"Target not found in catalog: TIC={self.tic_id}, TOI={self.toi}")
                return False

            self.tic_id = self.planet_params['tic_id']
            self.toi = self.planet_params['toi']

            logger.info(f"Target: TIC {self.tic_id} (TOI {self.toi})")
            logger.info(f"  Period: {self.planet_params['period']:.6f} days")
            logger.info(f"  Epoch: {self.planet_params['t0']:.6f} BJD")
            logger.info(f"  Depth: {self.planet_params['depth_ppm']:.1f} ppm")
            logger.info(f"  Disposition: {self.planet_params['disposition']}")

            # Set up output directory using TOI name
            toi_name = str(self.toi).replace('.', '_')  # e.g., "232.01" -> "232_01"
            self.tic_output_dir = self.output_dir / f"TOI_{toi_name}"
            self.tic_output_dir.mkdir(parents=True, exist_ok=True)

            return True

        except Exception as e:
            logger.error(f"Failed to load catalog: {e}")
            return False

    def load_data(self, download: bool = True, force_download: bool = True) -> bool:
        """Load light curve data.

        Parameters
        ----------
        download : bool
            Whether to allow downloading from MAST
        force_download : bool
            If True, always download from MAST to ensure all sectors are included.
            If False, try cache first (may have incomplete data).
        """
        logger.info("=" * 70)
        logger.info("STEP 0.5: Loading Light Curve Data")
        logger.info("=" * 70)

        self.data_loader = DataLoader(self.tic_id)

        # Always download from MAST to ensure complete sector coverage
        if force_download and download:
            logger.info("Downloading ALL available sectors from MAST...")
            if not self.data_loader.download_from_mast(max_sector=config.MAX_SECTOR):
                # Fall back to cache if MAST fails
                logger.warning("MAST download failed, trying cache...")
                if not self.data_loader.load_from_cache():
                    logger.error("Failed to load data from MAST or cache")
                    return False
        elif self.data_loader.load_from_cache():
            logger.info("Loaded data from cache")
        elif download:
            logger.info("Downloading data from MAST...")
            if not self.data_loader.download_from_mast(max_sector=config.MAX_SECTOR):
                logger.error("Failed to download data")
                return False
        else:
            logger.error("No cached data and download disabled")
            return False

        # Get combined data
        self.time, self.flux, self.flux_err = self.data_loader.get_combined_lightcurve()

        if len(self.time) == 0:
            logger.error("No data available")
            return False

        summary = self.data_loader.get_data_summary()
        logger.info(f"Data loaded:")
        logger.info(f"  Sectors: {summary['sectors']}")
        logger.info(f"  Total points: {summary['total_n_points']}")
        logger.info(f"  Time span: {summary['time_span_days']:.1f} days")

        return True

    def run_step1_phase_fold(self) -> bool:
        """Step 1: Fit phase-folded light curve."""
        logger.info("=" * 70)
        logger.info("STEP 1: Phase-Folded Light Curve Fitting")
        logger.info("=" * 70)

        try:
            fitter = PhaseFoldFitter(
                self.time, self.flux, self.flux_err,
                self.planet_params
            )

            self.step1_results = fitter.fit(check_convergence=True)

            # Save results
            step1_dir = self.tic_output_dir / "step1_phase_fold"
            step1_dir.mkdir(exist_ok=True)

            with open(step1_dir / "step1_results.json", 'w') as f:
                json.dump(self.step1_results, f, indent=2, default=json_serializer)

            # Generate plots
            if fitter.sampler is not None:
                plot_mcmc_chains(
                    fitter.sampler,
                    fitter.PARAM_NAMES,
                    step1_dir / "chain_plot.png",
                    title=f"TIC {self.tic_id} - MCMC Chains",
                    burnin_chain=fitter.burnin_chain,
                    burnin_log_prob=fitter.burnin_log_prob
                )

                if fitter.samples is not None:
                    plot_corner(
                        fitter.samples,
                        fitter.PARAM_NAMES,
                        step1_dir / "corner_plot.png",
                        title=f"TIC {self.tic_id} - Parameter Covariance"
                    )

            # Phase-folded plot
            phase, flux, flux_err = fitter.get_phase_folded_data()
            bin_phase, bin_flux, bin_err = fitter.get_binned_phase_folded()

            # Generate model for plot
            model_phase = np.linspace(-config.PHASE_PLOT_RANGE, config.PHASE_PLOT_RANGE, config.N_MODEL_POINTS)
            period = self.step1_results['parameters']['period']['value']
            t0 = self.step1_results['parameters']['t0']['value']
            model_time = t0 + model_phase * period
            model_flux = fitter.get_best_fit_model(model_time)

            plot_phase_folded_lightcurve(
                phase, flux, flux_err,
                bin_phase, bin_flux, bin_err,
                model_phase, model_flux,
                step1_dir / "phase_folded_lightcurve.png",
                title=f"TIC {self.tic_id} - Phase-Folded Transit"
            )

            # Store fitter for Step 2
            self._phase_fold_fitter = fitter

            logger.info("Step 1 complete")
            logger.info(f"  Period: {self.step1_results['parameters']['period']['value']:.8f} days")
            logger.info(f"  Rp/Rs: {self.step1_results['parameters']['rp_rs']['value']:.4f}")

            return True

        except Exception as e:
            logger.error(f"Step 1 failed: {e}")
            traceback.print_exc()
            return False

    def run_step2_individual_transits(self) -> bool:
        """Step 2: Fit individual transit events."""
        logger.info("=" * 70)
        logger.info("STEP 2: Individual Transit Fitting")
        logger.info("=" * 70)

        try:
            fitter = IndividualTransitFitter(self.step1_results)
            results = fitter.fit_all_transits(self.time, self.flux, self.flux_err)

            self.transit_results = [r.to_dict() for r in results]

            # Save results
            step2_dir = self.tic_output_dir / "step2_individual_transits"
            step2_dir.mkdir(exist_ok=True)

            with open(step2_dir / "transit_times.json", 'w') as f:
                json.dump(self.transit_results, f, indent=2, default=json_serializer)

            # Save as CSV too
            self._save_transit_csv(step2_dir / "transit_times.csv")

            logger.info(f"Step 2 complete: fitted {len(self.transit_results)} transits")

            return len(self.transit_results) >= config.MIN_TRANSITS

        except Exception as e:
            logger.error(f"Step 2 failed: {e}")
            traceback.print_exc()
            return False

    def _save_transit_csv(self, path: Path):
        """Save transit times as CSV."""
        with open(path, 'w') as f:
            f.write("epoch,t_mid_bjd,t_mid_err_sec,baseline,slope\n")
            for t in self.transit_results:
                if t['success']:
                    f.write(f"{t['epoch']},{t['t_mid']:.8f},"
                           f"{t['t_mid_err']*config.SECONDS_PER_DAY:.2f},"
                           f"{t['baseline']:.6f},{t['slope']:.8f}\n")

    def run_step3_ephemeris(self) -> bool:
        """Step 3: Ephemeris analysis."""
        logger.info("=" * 70)
        logger.info("STEP 3: Ephemeris Analysis")
        logger.info("=" * 70)

        try:
            self.ephemeris_results = analyze_ephemeris(self.transit_results)

            if 'error' in self.ephemeris_results:
                logger.warning(f"Ephemeris analysis: {self.ephemeris_results['error']}")
                return False

            # Save results
            step3_dir = self.tic_output_dir / "step3_ephemeris"
            step3_dir.mkdir(exist_ok=True)

            with open(step3_dir / "ephemeris_results.json", 'w') as f:
                json.dump(self.ephemeris_results, f, indent=2, default=json_serializer)

            # Generate O-C plots
            oc_linear = self.ephemeris_results.get('oc_linear', {})
            if oc_linear:
                epochs = np.array(oc_linear['epochs'])
                oc_min = np.array(oc_linear['oc_minutes'])
                oc_err = np.array(oc_linear['t_mid_err_minutes'])

                plot_oc_diagram(
                    epochs, oc_min, oc_err,
                    step3_dir / "oc_diagram.png",
                    model="linear",
                    title=f"TIC {self.tic_id} - O-C Diagram"
                )

            # If quadratic preferred, also plot comparison
            model_sel = self.ephemeris_results.get('model_selection', {})
            if model_sel.get('preferred_model') == 'quadratic':
                oc_quad = self.ephemeris_results.get('oc_quadratic', {})
                if oc_quad:
                    plot_oc_comparison(
                        epochs,
                        oc_min,
                        np.array(oc_quad['oc_minutes']),
                        oc_err,
                        step3_dir / "oc_linear_comparison.png",
                        title=f"TIC {self.tic_id} - Ephemeris Comparison"
                    )

                # Report dP/dt
                quad_params = self.ephemeris_results.get('quadratic', {})
                dP_dt = quad_params.get('dP_dt_ms_per_year')
                if dP_dt is not None:
                    logger.info(f"  dP/dt = {dP_dt:.3f} ms/year")

            logger.info(f"Step 3 complete")
            logger.info(f"  Preferred model: {model_sel.get('preferred_model', 'linear')}")
            logger.info(f"  Delta BIC: {model_sel.get('delta_bic', 0):.2f}")

            return True

        except Exception as e:
            logger.error(f"Step 3 failed: {e}")
            traceback.print_exc()
            return False

    def run_step4_periodogram(self) -> bool:
        """Step 4: Lomb-Scargle periodogram."""
        logger.info("=" * 70)
        logger.info("STEP 4: TTV Periodogram Analysis")
        logger.info("=" * 70)

        try:
            self.periodogram_results = compute_ttv_periodogram(
                self.transit_results,
                self.ephemeris_results
            )

            if 'error' in self.periodogram_results:
                logger.warning(f"Periodogram: {self.periodogram_results['error']}")
                return False

            # Save results
            step4_dir = self.tic_output_dir / "step4_periodogram"
            step4_dir.mkdir(exist_ok=True)

            # Save without full periodogram data (too large)
            results_to_save = {k: v for k, v in self.periodogram_results.items()
                              if k != 'periodogram_data'}
            with open(step4_dir / "periodogram_results.json", 'w') as f:
                json.dump(results_to_save, f, indent=2, default=json_serializer)

            # Generate plot
            pg_data = self.periodogram_results.get('periodogram_data', {})
            if pg_data:
                freqs = np.array(pg_data['frequencies'])
                power = np.array(pg_data['power'])

                if len(freqs) > 0:
                    fap_levels = self.periodogram_results.get('fap_levels', {})
                    plot_periodogram(
                        freqs, power,
                        step4_dir / "periodogram.png",
                        fap_01=fap_levels.get('1%'),
                        fap_05=fap_levels.get('5%'),
                        peak_freq=self.periodogram_results.get('peak_frequency'),
                        title=f"TIC {self.tic_id} - TTV Periodogram"
                    )

            logger.info("Step 4 complete")
            logger.info(f"  Peak FAP: {self.periodogram_results.get('peak_fap', 1.0):.2e}")
            logger.info(f"  Significant (1%): {self.periodogram_results.get('is_significant_1pct', False)}")

            return True

        except Exception as e:
            logger.error(f"Step 4 failed: {e}")
            traceback.print_exc()
            return False

    def run(self, download: bool = True) -> Dict[str, Any]:
        """
        Run complete analysis pipeline.

        Parameters
        ----------
        download : bool
            Whether to download data if not cached

        Returns
        -------
        dict
            Summary of analysis results
        """
        start_time = datetime.now()
        logger.info(f"Starting AutoTTV Pipeline v2.0 for TIC {self.tic_id or self.toi}")

        results = {
            'tic_id': self.tic_id,
            'toi': self.toi,
            'success': False,
            'steps_completed': [],
            'errors': []
        }

        try:
            # Load catalog
            if not self.load_catalog():
                results['errors'].append('Failed to load catalog')
                return results
            results['tic_id'] = self.tic_id
            results['toi'] = self.toi

            # Load data
            if not self.load_data(download=download):
                results['errors'].append('Failed to load data')
                return results
            results['steps_completed'].append('data_loading')

            # Step 1: Phase-folded fitting
            if not self.run_step1_phase_fold():
                results['errors'].append('Step 1 (phase fold) failed')
                return results
            results['steps_completed'].append('step1_phase_fold')

            # Step 2: Individual transits
            if not self.run_step2_individual_transits():
                results['errors'].append('Step 2 (individual transits) failed or insufficient transits')
                # Continue anyway for partial results

            if self.transit_results and len(self.transit_results) >= config.MIN_TRANSITS:
                results['steps_completed'].append('step2_individual_transits')

                # Step 3: Ephemeris
                if self.run_step3_ephemeris():
                    results['steps_completed'].append('step3_ephemeris')

                # Step 4: Periodogram
                if self.run_step4_periodogram():
                    results['steps_completed'].append('step4_periodogram')

            results['success'] = len(results['steps_completed']) >= 3

            # Add summary
            results['n_transits'] = len(self.transit_results) if self.transit_results else 0
            if self.ephemeris_results:
                results['preferred_model'] = self.ephemeris_results.get(
                    'model_selection', {}).get('preferred_model', 'linear')
            if self.periodogram_results:
                results['ttv_significant'] = self.periodogram_results.get(
                    'is_significant_1pct', False)

        except Exception as e:
            results['errors'].append(str(e))
            logger.error(f"Pipeline failed: {e}")
            traceback.print_exc()

        elapsed = (datetime.now() - start_time).total_seconds()
        results['elapsed_seconds'] = elapsed

        logger.info(f"Pipeline complete in {elapsed:.1f}s")
        logger.info(f"Success: {results['success']}")

        return results


def run_single(tic_id: int = None, toi: str = None, **kwargs) -> Dict[str, Any]:
    """Run pipeline for a single target."""
    pipeline = AutoTTVPipeline(tic_id=tic_id, toi=toi)
    return pipeline.run(**kwargs)


def run_batch(limit: int = None, start_index: int = 0,
              parallel: int = 1, **kwargs) -> List[Dict[str, Any]]:
    """Run pipeline for multiple targets."""
    catalog = load_toi_catalog()

    # Get list of TIC IDs
    tic_ids = catalog['TIC ID'].unique().tolist()

    if limit:
        tic_ids = tic_ids[start_index:start_index + limit]
    else:
        tic_ids = tic_ids[start_index:]

    logger.info(f"Running batch analysis for {len(tic_ids)} targets")

    results = []

    if parallel > 1:
        from multiprocessing import Pool

        def worker(tic_id):
            try:
                return run_single(tic_id=tic_id, **kwargs)
            except Exception as e:
                return {'tic_id': tic_id, 'success': False, 'errors': [str(e)]}

        with Pool(parallel) as pool:
            results = pool.map(worker, tic_ids)
    else:
        for tic_id in tic_ids:
            result = run_single(tic_id=tic_id, **kwargs)
            results.append(result)

    # Summary
    n_success = sum(1 for r in results if r.get('success', False))
    logger.info(f"Batch complete: {n_success}/{len(results)} successful")

    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='AutoTTV Pipeline v2.0 - TESS Transit Timing Analysis'
    )

    # Target selection
    target_group = parser.add_mutually_exclusive_group()
    target_group.add_argument('--tic', type=int, help='TIC ID to analyze')
    target_group.add_argument('--toi', type=str, help='TOI designation to analyze')
    target_group.add_argument('--batch', action='store_true',
                             help='Run batch analysis on all TOIs')

    # Batch options
    parser.add_argument('--limit', type=int, help='Limit number of targets in batch')
    parser.add_argument('--start', type=int, default=0,
                       help='Start index for batch processing')
    parser.add_argument('--parallel', type=int, default=1,
                       help='Number of parallel workers')

    # Options
    parser.add_argument('--no-download', action='store_true',
                       help='Do not download data (use cache only)')
    parser.add_argument('--output-dir', type=str,
                       help='Output directory for results')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Create output directory
    output_dir = Path(args.output_dir) if args.output_dir else config.OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    kwargs = {
        'download': not args.no_download
    }

    if args.batch:
        results = run_batch(
            limit=args.limit,
            start_index=args.start,
            parallel=args.parallel,
            **kwargs
        )
        # Save batch results
        with open(output_dir / 'batch_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=json_serializer)

    elif args.tic or args.toi:
        result = run_single(tic_id=args.tic, toi=args.toi, **kwargs)
        print(json.dumps(result, indent=2, default=json_serializer))

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
