"""
Webapp Export Module for AutoTTV Pipeline v2.0

Generates JSON files compatible with the existing webapp structure.
Updates both individual planet files and the catalog summary.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

from . import config
from .utils import json_serializer

logger = logging.getLogger(__name__)


def create_planet_json(tic_id: int, toi: str, planet_params: Dict[str, Any],
                       step1_results: Dict[str, Any],
                       transit_results: List[Dict[str, Any]],
                       ephemeris_results: Dict[str, Any],
                       periodogram_results: Dict[str, Any],
                       data_summary: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create planet JSON structure for webapp.

    Parameters
    ----------
    tic_id : int
        TIC ID
    toi : str
        TOI designation
    planet_params : dict
        Original planet parameters from catalog
    step1_results : dict
        Phase-folded fitting results
    transit_results : list
        Individual transit results
    ephemeris_results : dict
        Ephemeris analysis results
    periodogram_results : dict
        Periodogram analysis results
    data_summary : dict
        Data summary from loader

    Returns
    -------
    dict
        Planet data structure for webapp JSON
    """
    # Build catalog section
    catalog = {
        'tic_id': str(tic_id),
        'toi': str(toi),
        'disposition': planet_params.get('disposition', 'Unknown'),
        'period_days': planet_params.get('period'),
        'epoch_bjd': planet_params.get('t0'),
        'depth_ppm': planet_params.get('depth_ppm'),
        'duration_hr': planet_params.get('duration_hr'),
        'stellar_teff': planet_params.get('stellar_teff'),
        'stellar_logg': planet_params.get('stellar_logg'),
        'stellar_radius': planet_params.get('stellar_radius'),
        'stellar_mass': planet_params.get('stellar_mass'),
        'ra': planet_params.get('ra'),
        'dec': planet_params.get('dec'),
        'tess_mag': planet_params.get('tess_mag')
    }

    # Build fitted section from Step 1
    fitted = {}
    if 'parameters' in step1_results:
        params = step1_results['parameters']
        fitted = {
            'period': params.get('period', {}),
            't0': params.get('t0', {}),
            'rp_rs': params.get('rp_rs', {}),
            'a_rs': params.get('a_rs', {}),
            'impact_parameter': params.get('b', {}),
            'baseline': params.get('baseline', {})
        }

    if 'derived' in step1_results:
        derived = step1_results['derived']
        fitted['depth_ppm'] = derived.get('depth_ppm')
        fitted['depth_ppm_err'] = derived.get('depth_ppm_err')
        fitted['inclination_deg'] = derived.get('inclination_deg')
        fitted['duration_hr'] = derived.get('duration_hr')

    if 'limb_darkening' in step1_results:
        fitted['limb_darkening'] = step1_results['limb_darkening']

    # Build transit list
    transits = []
    for t in transit_results:
        if t.get('success', False):
            transits.append({
                'epoch': t['epoch'],
                't_mid': t['t_mid'],
                't_mid_err_sec': t['t_mid_err'] * config.SECONDS_PER_DAY,  # Convert days to seconds
                'oc_min': None,  # Will be filled from ephemeris
                'baseline': t['baseline'],
                'slope': t['slope']
            })

    # Add O-C values from ephemeris
    if 'oc_linear' in ephemeris_results:
        oc_data = ephemeris_results['oc_linear']
        epochs = oc_data.get('epochs', [])
        oc_min = oc_data.get('oc_minutes', [])

        for transit in transits:
            try:
                idx = epochs.index(transit['epoch'])
                transit['oc_min'] = oc_min[idx]
            except (ValueError, IndexError):
                pass

    # Build O-C analysis section
    oc_analysis = {
        'linear': ephemeris_results.get('linear', {}),
        'quadratic': ephemeris_results.get('quadratic', {}),
        'model_selection': ephemeris_results.get('model_selection', {})
    }

    # Build periodic analysis section
    periodic_analysis = {
        'peak_frequency': periodogram_results.get('peak_frequency'),
        'peak_period_days': periodogram_results.get('peak_period_days'),
        'peak_period_error_days': periodogram_results.get('peak_period_error_days'),
        'peak_fwhm_freq': periodogram_results.get('peak_fwhm_freq'),
        'peak_power': periodogram_results.get('peak_power'),
        'peak_fap': periodogram_results.get('peak_fap'),
        'fap_levels': periodogram_results.get('fap_levels', {}),
        'is_significant_1pct': periodogram_results.get('is_significant_1pct', False),
        'is_significant_5pct': periodogram_results.get('is_significant_5pct', False),
        'significant_peaks': periodogram_results.get('significant_peaks', [])
    }

    # Build complete planet structure
    planet_data = {
        'tic_id': str(tic_id),
        'toi': str(toi),
        'name': f'TOI {toi}',
        'catalog': catalog,
        'fitted': fitted,
        'data_description': {
            'source': 'TESS SPOC/QLP Pipeline',
            'sectors': data_summary.get('sectors', []),
            'n_sectors': data_summary.get('n_sectors', 0),
            'total_points': data_summary.get('total_n_points', 0),
            'time_span_days': data_summary.get('time_span_days', 0),
            'cadences_used': data_summary.get('cadences_used', [])
        },
        'transits': transits,
        'n_transits': len(transits),
        'oc_analysis': oc_analysis,
        'periodic_analysis': periodic_analysis,
        'analysis_timestamp': datetime.utcnow().isoformat() + 'Z',
        'pipeline_version': '2.0.0'
    }

    return planet_data


def save_planet_json(planet_data: Dict[str, Any],
                     output_dir: Path = config.WEBAPP_PLANETS_DIR) -> Path:
    """
    Save planet JSON file to webapp data directory.

    Parameters
    ----------
    planet_data : dict
        Planet data structure
    output_dir : Path
        Output directory

    Returns
    -------
    Path
        Path to saved file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    toi = planet_data.get('toi', planet_data.get('tic_id'))
    toi_name = str(toi).replace('.', '_')  # e.g., "232.01" -> "232_01"
    output_file = output_dir / f"TOI_{toi_name}.json"

    with open(output_file, 'w') as f:
        json.dump(planet_data, f, indent=config.JSON_INDENT, default=json_serializer)

    logger.info(f"Saved planet JSON to {output_file}")
    return output_file


def create_catalog_entry(planet_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a catalog entry for a planet (for catalog.json).

    Parameters
    ----------
    planet_data : dict
        Full planet data structure

    Returns
    -------
    dict
        Catalog entry
    """
    catalog = planet_data.get('catalog', {})
    fitted = planet_data.get('fitted', {})
    oc_analysis = planet_data.get('oc_analysis', {})
    periodic = planet_data.get('periodic_analysis', {})
    model_sel = oc_analysis.get('model_selection', {})

    # Determine period bin
    period = catalog.get('period_days')
    if period is not None:
        if period < config.PERIOD_BIN_ULTRA_SHORT:
            period_bin = 'ultra-short'
        elif period < config.PERIOD_BIN_SHORT:
            period_bin = 'short'
        elif period < config.PERIOD_BIN_MEDIUM:
            period_bin = 'medium'
        else:
            period_bin = 'long'
    else:
        period_bin = 'unknown'

    # Determine transit bin
    n_transits = planet_data.get('n_transits', 0)
    if n_transits < config.TRANSIT_BIN_FEW:
        transit_bin = 'few'
    elif n_transits < config.TRANSIT_BIN_MODERATE:
        transit_bin = 'moderate'
    else:
        transit_bin = 'many'

    # Determine TTV status
    if periodic.get('is_significant_1pct', False):
        ttv_status = 'Significant'
    elif periodic.get('is_significant_5pct', False):
        ttv_status = 'Marginal'
    elif n_transits >= config.MIN_TRANSITS_PERIODOGRAM:
        ttv_status = 'None'
    else:
        ttv_status = 'Insufficient'

    # Get O-C RMS
    linear = oc_analysis.get('linear', {})
    oc_rms_min = linear.get('oc_rms_min', 0)

    entry = {
        'tic_id': planet_data['tic_id'],
        'toi': planet_data['toi'],
        'name': planet_data['name'],
        'disposition': catalog.get('disposition', 'Unknown'),
        'period_days': catalog.get('period_days'),
        'depth_ppm': catalog.get('depth_ppm'),
        'period_bin': period_bin,
        'transit_bin': transit_bin,
        'n_transits': n_transits,
        'preferred_model': model_sel.get('preferred_model', 'linear'),
        'ttv_status': ttv_status,
        'oc_rms_min': oc_rms_min,
        'delta_bic': model_sel.get('delta_bic', 0),
        'peak_ttv_fap': periodic.get('peak_fap', 1.0),
        'is_oc_excess': False,  # Computed later
        'is_period_change': model_sel.get('preferred_model') == 'quadratic',
        'is_periodic_ttv': periodic.get('is_significant_1pct', False),
        'is_depth_variable': False,  # Computed later
        'is_inclination_variable': False,  # Computed later
        'stellar_teff': catalog.get('stellar_teff'),
        'stellar_radius': catalog.get('stellar_radius')
    }

    return entry


def update_catalog_json(new_entries: List[Dict[str, Any]],
                        catalog_file: Path = config.WEBAPP_DATA_DIR / "catalog.json",
                        mode: str = 'update') -> Path:
    """
    Update the main catalog.json file.

    Parameters
    ----------
    new_entries : list
        List of new catalog entries
    catalog_file : Path
        Path to catalog.json
    mode : str
        'update' to merge with existing, 'replace' to overwrite

    Returns
    -------
    Path
        Path to updated catalog file
    """
    catalog_file = Path(catalog_file)

    # Load existing catalog if updating
    if mode == 'update' and catalog_file.exists():
        with open(catalog_file, 'r') as f:
            catalog = json.load(f)
        existing_planets = {p['tic_id']: p for p in catalog.get('planets', [])}
    else:
        catalog = {'stats': {}, 'planets': []}
        existing_planets = {}

    # Update with new entries
    for entry in new_entries:
        existing_planets[entry['tic_id']] = entry

    # Rebuild planets list
    planets = list(existing_planets.values())

    # Compute statistics
    n_total = len(planets)
    n_with_analysis = sum(1 for p in planets if (p.get('n_transits') or 0) > 0)
    total_transits = sum((p.get('n_transits') or 0) for p in planets)

    oc_rms_values = [(p.get('oc_rms_min') or 0) for p in planets if (p.get('oc_rms_min') or 0) > 0]
    oc_rms_median = float(np.median(oc_rms_values)) if oc_rms_values else 0

    # Count interesting systems
    interesting = {
        'oc_excess': sum(1 for p in planets if p.get('is_oc_excess', False)),
        'period_change': sum(1 for p in planets if p.get('is_period_change', False)),
        'periodic_ttv': sum(1 for p in planets if p.get('is_periodic_ttv', False)),
        'depth_variable': sum(1 for p in planets if p.get('is_depth_variable', False)),
        'inclination_variable': sum(1 for p in planets if p.get('is_inclination_variable', False))
    }

    catalog['stats'] = {
        'total': n_total,
        'with_analysis': n_with_analysis,
        'total_transits': total_transits,
        'oc_rms_median': oc_rms_median,
        'interesting_systems': interesting,
        'last_updated': datetime.utcnow().isoformat() + 'Z',
        'pipeline_version': '2.0.0'
    }
    catalog['planets'] = planets

    # Save
    catalog_file.parent.mkdir(parents=True, exist_ok=True)
    with open(catalog_file, 'w') as f:
        json.dump(catalog, f, indent=config.JSON_INDENT, default=json_serializer)

    logger.info(f"Updated catalog with {len(new_entries)} entries -> {catalog_file}")
    return catalog_file


class WebappExporter:
    """
    Class to manage webapp data export.
    """

    def __init__(self, output_dir: Path = config.WEBAPP_DATA_DIR):
        """
        Initialize exporter.

        Parameters
        ----------
        output_dir : Path
            Base output directory for webapp data
        """
        self.output_dir = Path(output_dir)
        self.planets_dir = self.output_dir / "planets"
        self.catalog_file = self.output_dir / "catalog.json"

        # Track exported entries for batch catalog update
        self.pending_entries = []

    def export_planet(self, tic_id: int, toi: str, planet_params: Dict,
                      step1_results: Dict, transit_results: List[Dict],
                      ephemeris_results: Dict, periodogram_results: Dict,
                      data_summary: Dict) -> Path:
        """
        Export a single planet's results.

        Returns path to saved JSON file.
        """
        # Create planet JSON
        planet_data = create_planet_json(
            tic_id, toi, planet_params,
            step1_results, transit_results,
            ephemeris_results, periodogram_results,
            data_summary
        )

        # Save planet file
        planet_file = save_planet_json(planet_data, self.planets_dir)

        # Create catalog entry and add to pending
        entry = create_catalog_entry(planet_data)
        self.pending_entries.append(entry)

        return planet_file

    def flush_catalog(self) -> Path:
        """
        Update catalog.json with all pending entries.

        Returns path to catalog file.
        """
        if self.pending_entries:
            catalog_path = update_catalog_json(self.pending_entries, self.catalog_file)
            self.pending_entries = []
            return catalog_path
        return self.catalog_file


# Need numpy for median calculation
import numpy as np


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Testing webapp export...")

    # Create test data
    test_planet_data = {
        'tic_id': '123456789',
        'toi': '999.01',
        'name': 'TOI 999.01',
        'catalog': {
            'period_days': 3.5,
            'depth_ppm': 10000,
            'disposition': 'PC'
        },
        'fitted': {
            'period': {'value': 3.500001, 'err': 0.000001}
        },
        'n_transits': 25,
        'oc_analysis': {
            'linear': {'oc_rms_min': 2.5},
            'model_selection': {'preferred_model': 'linear', 'delta_bic': 1.5}
        },
        'periodic_analysis': {
            'is_significant_1pct': False,
            'is_significant_5pct': True,
            'peak_fap': 0.03
        }
    }

    entry = create_catalog_entry(test_planet_data)
    print(f"\nCatalog entry:")
    print(json.dumps(entry, indent=2))
