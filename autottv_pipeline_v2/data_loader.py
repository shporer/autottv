"""
Data Loader Module for AutoTTV Pipeline v2.0

Handles downloading and loading TESS light curves from MAST via lightkurve.
Supports SPOC PDCSAP (2-min) with fallback to QLP detrended light curves.

Priority order:
1. SPOC PDCSAP 2-minute cadence
2. QLP (cadence varies by sector: 200s, 10-min, or 30-min)

Note: QLP data for each sector is available at only one cadence, depending on
when the sector was observed. The pipeline uses whichever QLP cadence is available.
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from astropy.io import fits

try:
    import lightkurve as lk
    LIGHTKURVE_AVAILABLE = True
except ImportError:
    LIGHTKURVE_AVAILABLE = False
    logging.warning("lightkurve not installed. Data download will not work.")

from . import config
from .utils import normalize_flux

logger = logging.getLogger(__name__)


class LightCurveData:
    """Container for processed light curve data."""

    def __init__(self, time: np.ndarray, flux: np.ndarray, flux_err: np.ndarray,
                 sector: int, cadence: float, source: str):
        """
        Initialize light curve data container.

        Parameters
        ----------
        time : np.ndarray
            Time array in BJD
        flux : np.ndarray
            Normalized flux array
        flux_err : np.ndarray
            Flux error array (normalized)
        sector : int
            TESS sector number
        cadence : float
            Cadence in seconds
        source : str
            Data source (e.g., "SPOC", "QLP")
        """
        self.time = time
        self.flux = flux
        self.flux_err = flux_err
        self.sector = sector
        self.cadence = cadence
        self.source = source

    @property
    def n_points(self) -> int:
        """Number of data points."""
        return len(self.time)

    @property
    def time_span(self) -> float:
        """Time span in days."""
        if len(self.time) > 1:
            return float(self.time.max() - self.time.min())
        return 0.0

    def __repr__(self):
        return (f"LightCurveData(sector={self.sector}, source={self.source}, "
                f"cadence={self.cadence}s, n_points={self.n_points})")


class DataLoader:
    """
    Load and manage TESS light curves from MAST.

    Handles both SPOC and QLP data with automatic fallback.
    """

    def __init__(self, tic_id: int, cache_dir: Optional[Path] = None):
        """
        Initialize the data loader.

        Parameters
        ----------
        tic_id : int
            TIC ID of the target
        cache_dir : Path, optional
            Directory to cache downloaded files
        """
        self.tic_id = tic_id
        self.cache_dir = cache_dir or config.LIGHTCURVE_DIR
        self.lightcurves: List[LightCurveData] = []
        self._download_log: List[Dict] = []

    def download_from_mast(self, max_sector: int = config.MAX_SECTOR) -> bool:
        """
        Download light curves from MAST for all available sectors.

        Parameters
        ----------
        max_sector : int
            Maximum sector number to include

        Returns
        -------
        bool
            True if at least one sector was successfully downloaded
        """
        if not LIGHTKURVE_AVAILABLE:
            logger.error("lightkurve not available. Cannot download data.")
            return False

        logger.info(f"Searching for TIC {self.tic_id} light curves on MAST...")

        try:
            # Single MAST search for all TESS light curves
            all_search_results = lk.search_lightcurve(
                f"TIC {self.tic_id}",
                mission='TESS'
            )

            # Organize by sector, splitting SPOC/QLP by author
            all_results = self._organize_by_sector(all_search_results, max_sector)

            if not all_results:
                logger.warning(f"No light curves found for TIC {self.tic_id}")
                return False

            # Download best available data for each sector
            success_count = 0
            for sector, candidates in all_results.items():
                try:
                    lc_data = self._download_best_cadence(sector, candidates)
                    if lc_data is not None:
                        self.lightcurves.append(lc_data)
                        success_count += 1
                        logger.info(f"  Sector {sector}: {lc_data.source} "
                                   f"({lc_data.cadence}s, {lc_data.n_points} points)")
                except Exception as e:
                    logger.warning(f"  Sector {sector}: Failed - {e}")

            logger.info(f"Downloaded {success_count} sectors for TIC {self.tic_id}")
            return success_count > 0

        except Exception as e:
            logger.error(f"Error searching MAST for TIC {self.tic_id}: {e}")
            return False

    def _organize_by_sector(self, search_results, max_sector: int) -> Dict:
        """
        Organize search results by sector with cadence information.

        Accepts a single unified search result and splits by author (SPOC/QLP).

        Returns dict: {sector: [(search_result_entry, cadence, source), ...]}
        """
        sectors = {}

        if search_results is None or len(search_results) == 0:
            return sectors

        for i, entry in enumerate(search_results):
            sector = self._extract_sector(entry)
            cadence = self._extract_cadence(entry)
            source = self._extract_source(entry)
            if sector is not None and sector <= max_sector and source is not None:
                if sector not in sectors:
                    sectors[sector] = []
                # Use slice notation to get SearchResult object (not Row)
                sectors[sector].append((search_results[i:i+1], cadence, source))

        return sectors

    def _extract_sector(self, entry) -> Optional[int]:
        """Extract sector number from search result entry."""
        import re
        try:
            if hasattr(entry, 'mission'):
                mission = entry.mission
                # Handle numpy arrays
                if hasattr(mission, '__iter__') and not isinstance(mission, str):
                    mission = mission[0] if len(mission) > 0 else ''
                mission_str = str(mission)
                # Extract sector number using regex (handles "TESS Sector 01", "Sector 28", etc.)
                match = re.search(r'[Ss]ector\s*(\d+)', mission_str)
                if match:
                    return int(match.group(1))
        except Exception:
            pass
        return None

    def _extract_cadence(self, entry) -> Optional[float]:
        """Extract exposure time/cadence from search result entry."""
        try:
            if hasattr(entry, 'exptime'):
                exp = entry.exptime
                if hasattr(exp, 'value'):
                    return float(exp.value)
                return float(exp)
        except Exception:
            pass
        return None

    def _extract_source(self, entry) -> Optional[str]:
        """Extract data source (SPOC/QLP) from search result entry's author field."""
        try:
            if hasattr(entry, 'author'):
                author = entry.author
                if hasattr(author, '__iter__') and not isinstance(author, str):
                    author = author[0] if len(author) > 0 else ''
                author_str = str(author).upper()
                if 'SPOC' in author_str:
                    return 'SPOC'
                elif 'QLP' in author_str:
                    return 'QLP'
        except Exception:
            pass
        return None

    def _download_best_cadence(self, sector: int, candidates: List) -> Optional[LightCurveData]:
        """
        Download the best available cadence for a sector.

        Prioritizes shorter cadences (SPOC 2-min > QLP 200s > QLP 10-min > QLP 30-min).
        """
        # Sort candidates by cadence priority
        def cadence_priority(item):
            _, cadence, source = item
            if cadence is None:
                return 9999
            # Lower cadence = higher priority
            try:
                idx = config.CADENCE_PRIORITY.index(int(cadence))
                return idx
            except ValueError:
                return 9999

        candidates.sort(key=cadence_priority)

        # Try each candidate in priority order
        for search_result, cadence, source in candidates:
            try:
                # Download the light curve
                lc_collection = search_result.download()

                if lc_collection is None:
                    continue

                # Get the light curve object
                # Check if it's already a single LightCurve (has 'time' attribute)
                # vs a LightCurveCollection (where [0] gives a LightCurve)
                if hasattr(lc_collection, 'time') and hasattr(lc_collection, 'flux'):
                    # It's already a single LightCurve object
                    lc = lc_collection
                elif hasattr(lc_collection, '__len__') and len(lc_collection) > 0:
                    # It's a collection, get the first LightCurve
                    lc = lc_collection[0]
                else:
                    lc = lc_collection

                # Extract data based on source
                if source == "SPOC":
                    lc_data = self._process_spoc_lightcurve(lc, sector, cadence)
                else:
                    lc_data = self._process_qlp_lightcurve(lc, sector, cadence)

                if lc_data is not None and lc_data.n_points > 0:
                    return lc_data

            except Exception as e:
                logger.debug(f"Failed to download {source} sector {sector}: {e}")
                continue

        return None

    def _process_spoc_lightcurve(self, lc, sector: int, cadence: float) -> Optional[LightCurveData]:
        """Process SPOC light curve to extract PDCSAP flux."""
        try:
            # Get time and flux
            time = lc.time.btjd + config.BJDREF
            if hasattr(time, 'value'):
                time = time.value

            # Prefer PDCSAP_FLUX if available
            if hasattr(lc, 'pdcsap_flux') and lc.pdcsap_flux is not None:
                flux = np.array(lc.pdcsap_flux)
                flux_err = np.array(lc.pdcsap_flux_err) if hasattr(lc, 'pdcsap_flux_err') else None
            elif hasattr(lc, 'flux'):
                flux = np.array(lc.flux)
                flux_err = np.array(lc.flux_err) if hasattr(lc, 'flux_err') else None
            else:
                return None

            if flux_err is None:
                flux_err = np.abs(flux) * config.DEFAULT_FLUX_ERROR

            # Handle astropy units
            if hasattr(flux, 'value'):
                flux = flux.value
            if hasattr(flux_err, 'value'):
                flux_err = flux_err.value

            time = np.array(time)
            flux = np.array(flux)
            flux_err = np.array(flux_err)

            # Filter good data
            good = (np.isfinite(time) & np.isfinite(flux) &
                    np.isfinite(flux_err) & (flux > 0))

            if hasattr(lc, 'quality'):
                quality = np.array(lc.quality)
                good = good & (quality == 0)

            time = time[good]
            flux = flux[good]
            flux_err = flux_err[good]

            if len(time) < config.MIN_POINTS_LIGHTCURVE:
                return None

            # Normalize flux
            flux, flux_err, _ = normalize_flux(flux, flux_err)

            return LightCurveData(
                time=time,
                flux=flux,
                flux_err=flux_err,
                sector=sector,
                cadence=cadence or config.DEFAULT_CADENCE_SPOC,
                source="SPOC"
            )

        except Exception as e:
            logger.debug(f"Error processing SPOC light curve: {e}")
            return None

    def _process_qlp_lightcurve(self, lc, sector: int, cadence: float) -> Optional[LightCurveData]:
        """Process QLP light curve to extract detrended flux."""
        try:
            # Get time
            time = lc.time.btjd + config.BJDREF
            if hasattr(time, 'value'):
                time = time.value

            # QLP detrended flux: kspsap_flux for sectors 1-55, det_flux for sectors 56+
            if sector >= 56 and hasattr(lc, 'det_flux') and lc.det_flux is not None:
                flux = np.array(lc.det_flux)
                flux_err = np.array(lc.det_flux_err) if hasattr(lc, 'det_flux_err') else None
            elif hasattr(lc, 'kspsap_flux') and lc.kspsap_flux is not None:
                flux = np.array(lc.kspsap_flux)
                flux_err = np.array(lc.kspsap_flux_err) if hasattr(lc, 'kspsap_flux_err') else None
            else:
                return None

            if flux_err is None:
                flux_err = np.abs(flux) * config.DEFAULT_FLUX_ERROR

            # Handle astropy units
            if hasattr(flux, 'value'):
                flux = flux.value
            if hasattr(flux_err, 'value'):
                flux_err = flux_err.value

            time = np.array(time)
            flux = np.array(flux)
            flux_err = np.array(flux_err)

            # Filter good data
            good = (np.isfinite(time) & np.isfinite(flux) &
                    np.isfinite(flux_err) & (flux > 0))

            if hasattr(lc, 'quality'):
                quality = np.array(lc.quality)
                good = good & (quality == 0)

            time = time[good]
            flux = flux[good]
            flux_err = flux_err[good]

            if len(time) < config.MIN_POINTS_LIGHTCURVE:
                return None

            # Normalize flux
            flux, flux_err, _ = normalize_flux(flux, flux_err)

            return LightCurveData(
                time=time,
                flux=flux,
                flux_err=flux_err,
                sector=sector,
                cadence=cadence or config.DEFAULT_CADENCE_QLP,
                source="QLP"
            )

        except Exception as e:
            logger.debug(f"Error processing QLP light curve: {e}")
            return None

    def save_to_npz_cache(self, cache_dir: Path = None) -> None:
        """Save processed light curves to .npz cache files."""
        cache_dir = Path(cache_dir or config.DATA_CACHE_DIR)
        tic_dir = cache_dir / f"TIC_{self.tic_id}"
        tic_dir.mkdir(parents=True, exist_ok=True)

        for lc in self.lightcurves:
            path = tic_dir / f"sector_{lc.sector:03d}.npz"
            np.savez(path,
                     time=lc.time, flux=lc.flux, flux_err=lc.flux_err,
                     sector=lc.sector, cadence=lc.cadence,
                     source=np.array(lc.source))
        logger.info(f"Cached {len(self.lightcurves)} sectors to {tic_dir}")

    def load_from_npz_cache(self, cache_dir: Path = None) -> bool:
        """Load light curves from .npz cache files."""
        cache_dir = Path(cache_dir or config.DATA_CACHE_DIR)
        tic_dir = cache_dir / f"TIC_{self.tic_id}"

        if not tic_dir.exists():
            return False

        npz_files = sorted(tic_dir.glob("sector_*.npz"))
        if not npz_files:
            return False

        # Check for stale QLP sectors 56+ cached with SAP instead of DET_FLUX.
        # If found, delete them and force full re-download from MAST.
        for npz_file in npz_files:
            data = np.load(npz_file, allow_pickle=True)
            sector = int(data['sector'])
            source = str(data['source'])
            if source == "QLP" and sector >= 56:
                logger.info(f"Stale QLP cache for sector {sector}, deleting {npz_file}")
                npz_file.unlink()
                # Delete all cached files and force re-download
                for f in tic_dir.glob("sector_*.npz"):
                    f.unlink()
                return False

        for npz_file in sorted(tic_dir.glob("sector_*.npz")):
            data = np.load(npz_file, allow_pickle=True)
            lc = LightCurveData(
                time=data['time'],
                flux=data['flux'],
                flux_err=data['flux_err'],
                sector=int(data['sector']),
                cadence=float(data['cadence']),
                source=str(data['source'])
            )
            self.lightcurves.append(lc)

        logger.info(f"Loaded {len(self.lightcurves)} sectors from cache: {tic_dir}")
        return len(self.lightcurves) > 0

    def load_from_cache(self, cache_dir: Optional[Path] = None) -> bool:
        """
        Load light curves from cached FITS files.

        Parameters
        ----------
        cache_dir : Path, optional
            Directory containing cached FITS files

        Returns
        -------
        bool
            True if at least one file was successfully loaded
        """
        cache_dir = cache_dir or self.cache_dir
        tic_dir = cache_dir / f"TIC_{self.tic_id}"

        if not tic_dir.exists():
            logger.warning(f"Cache directory not found: {tic_dir}")
            return False

        fits_files = sorted(tic_dir.glob("*.fits"))
        if not fits_files:
            logger.warning(f"No FITS files found in {tic_dir}")
            return False

        logger.info(f"Loading {len(fits_files)} FITS files from cache...")

        for fits_file in fits_files:
            try:
                lc_data = self._load_fits_file(fits_file)
                if lc_data is not None:
                    self.lightcurves.append(lc_data)
                    logger.debug(f"  Loaded {fits_file.name}: {lc_data}")
            except Exception as e:
                logger.warning(f"  Failed to load {fits_file.name}: {e}")

        logger.info(f"Loaded {len(self.lightcurves)} light curves from cache")
        return len(self.lightcurves) > 0

    def _load_fits_file(self, fits_file: Path) -> Optional[LightCurveData]:
        """Load a single FITS file and extract light curve data."""
        try:
            with fits.open(fits_file) as hdul:
                data = hdul[1].data
                header = hdul[0].header

                time = data['TIME'] + config.BJDREF

                # Try different flux columns
                flux = None
                flux_err = None

                if 'PDCSAP_FLUX' in data.names:
                    flux = data['PDCSAP_FLUX']
                    flux_err = data['PDCSAP_FLUX_ERR'] if 'PDCSAP_FLUX_ERR' in data.names else None
                    source = "SPOC"
                elif 'DET_FLUX' in data.names:
                    flux = data['DET_FLUX']
                    flux_err = data['DET_FLUX_ERR'] if 'DET_FLUX_ERR' in data.names else None
                    source = "QLP"
                elif 'KSPSAP_FLUX' in data.names:
                    flux = data['KSPSAP_FLUX']
                    flux_err = data['KSPSAP_FLUX_ERR'] if 'KSPSAP_FLUX_ERR' in data.names else None
                    source = "QLP"
                elif 'SAP_FLUX' in data.names:
                    flux = data['SAP_FLUX']
                    flux_err = data['SAP_FLUX_ERR'] if 'SAP_FLUX_ERR' in data.names else None
                    source = "SAP"
                else:
                    return None

                if flux_err is None:
                    flux_err = np.abs(flux) * config.DEFAULT_FLUX_ERROR

                # Get quality flags
                if 'QUALITY' in data.names:
                    quality = data['QUALITY']
                else:
                    quality = np.zeros(len(time))

                # Filter good data
                good = (np.isfinite(time) & np.isfinite(flux) &
                        np.isfinite(flux_err) & (flux > 0) & (quality == 0))

                time = time[good]
                flux = flux[good]
                flux_err = flux_err[good]

                if len(time) < config.MIN_POINTS_LIGHTCURVE:
                    return None

                # Normalize
                flux, flux_err, _ = normalize_flux(flux, flux_err)

                # Extract sector and cadence from header
                sector = header.get('SECTOR', 0)
                cadence = header.get('TIMEDEL', config.DEFAULT_TIMEDEL_DAYS) * config.SECONDS_PER_DAY

                return LightCurveData(
                    time=time,
                    flux=flux,
                    flux_err=flux_err,
                    sector=sector,
                    cadence=cadence,
                    source=source
                )

        except Exception as e:
            logger.debug(f"Error loading FITS file {fits_file}: {e}")
            return None

    def get_combined_lightcurve(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get all light curves combined into single arrays.

        Returns
        -------
        time, flux, flux_err : tuple of np.ndarray
            Combined time, flux, and flux_err arrays sorted by time
        """
        if not self.lightcurves:
            return np.array([]), np.array([]), np.array([])

        all_time = np.concatenate([lc.time for lc in self.lightcurves])
        all_flux = np.concatenate([lc.flux for lc in self.lightcurves])
        all_flux_err = np.concatenate([lc.flux_err for lc in self.lightcurves])

        # Sort by time
        sort_idx = np.argsort(all_time)
        return all_time[sort_idx], all_flux[sort_idx], all_flux_err[sort_idx]

    def get_data_summary(self) -> Dict[str, Any]:
        """Get summary statistics of loaded data."""
        if not self.lightcurves:
            return {"n_sectors": 0, "n_points": 0, "time_span_days": 0}

        time, _, _ = self.get_combined_lightcurve()

        sectors_info = []
        for lc in self.lightcurves:
            sectors_info.append({
                "sector": lc.sector,
                "source": lc.source,
                "cadence_sec": lc.cadence,
                "n_points": lc.n_points,
                "time_span_days": lc.time_span
            })

        return {
            "tic_id": self.tic_id,
            "n_sectors": len(self.lightcurves),
            "sectors": sorted([lc.sector for lc in self.lightcurves]),
            "sectors_info": sectors_info,
            "total_n_points": len(time),
            "time_span_days": float(time.max() - time.min()) if len(time) > 1 else 0,
            "start_bjd": float(time.min()) if len(time) > 0 else 0,
            "end_bjd": float(time.max()) if len(time) > 0 else 0,
            "cadences_used": list(set(lc.source for lc in self.lightcurves))
        }


def load_toi_catalog(catalog_file: Path = config.CATALOG_FILE) -> pd.DataFrame:
    """
    Load the TOI catalog.

    Parameters
    ----------
    catalog_file : Path
        Path to the TOI catalog CSV file

    Returns
    -------
    pd.DataFrame
        TOI catalog dataframe
    """
    if not catalog_file.exists():
        raise FileNotFoundError(f"TOI catalog not found: {catalog_file}")

    return pd.read_csv(catalog_file)


def get_toi_parameters(catalog: pd.DataFrame, tic_id: int = None,
                       toi: str = None) -> Optional[Dict[str, Any]]:
    """
    Get parameters for a specific TOI from the catalog.

    Parameters
    ----------
    catalog : pd.DataFrame
        TOI catalog dataframe
    tic_id : int, optional
        TIC ID to look up
    toi : str, optional
        TOI designation to look up (e.g., "101.01")

    Returns
    -------
    dict or None
        Dictionary of planet parameters, or None if not found
    """
    if tic_id is not None:
        match = catalog[catalog['TIC ID'] == tic_id]
    elif toi is not None:
        match = catalog[catalog['TOI'] == float(toi)]
    else:
        return None

    if len(match) == 0:
        return None

    row = match.iloc[0]

    return {
        'tic_id': int(row['TIC ID']),
        'toi': row['TOI'],
        'period': row.get('Period (days)', None),
        'period_err': row.get('Period (days) err', None),
        't0': row.get('Epoch (BJD)', None),
        't0_err': row.get('Epoch (BJD) err', None),
        'depth_ppm': row.get('Depth (ppm)', None),
        'duration_hr': row.get('Duration (hours)', config.DEFAULT_TRANSIT_DURATION_HR),
        'disposition': row.get('TFOPWG Disposition', 'Unknown'),
        'tess_disposition': row.get('TESS Disposition', 'Unknown'),
        'planet_radius': row.get('Planet Radius (R_Earth)', None),
        'stellar_teff': row.get('Stellar Eff Temp (K)', None),
        'stellar_logg': row.get('Stellar log(g) (cm/s^2)', None),
        'stellar_radius': row.get('Stellar Radius (R_Sun)', None),
        'stellar_mass': row.get('Stellar Mass (M_Sun)', None),
        'ra': row.get('RA', None),
        'dec': row.get('Dec', None),
        'tess_mag': row.get('TESS Mag', None),
        'sectors': row.get('Sectors', None),
    }


if __name__ == "__main__":
    # Test the data loader
    logging.basicConfig(level=logging.INFO)

    # Test with a known TOI
    loader = DataLoader(tic_id=231663901)

    # Try loading from cache first
    if not loader.load_from_cache():
        print("Cache not available, would download from MAST")

    print("\nData Summary:")
    print(loader.get_data_summary())
