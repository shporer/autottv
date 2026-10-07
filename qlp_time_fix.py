"""Known QLP timestamp errors and their correction (Table 2 update, 2026-10).

QLP light curves carry TIME stamps converted to BJD_TDB by QLP's own barycentric
correction. For some sector-orbits that correction used a wrong observer position:

  s1415  Sectors 14 and 15 (found 2026-10-04): the observer position was
         (Y_ecl, Z_ecl, 0), i.e. the spacecraft's ecliptic y and z coordinates put
         in the equatorial x and y slots, instead of the equatorial (x, y, z).
  s7479  the ecliptic-frame ephemeris used as equatorial, the error QLP documents
         for Sectors 74-79 (fixed there in v02). It is also seen in the cached S80
         light curves and in orbit 2 of S85 for TIC 272829240 (TOI-7125.01).

error_s() returns QLP time minus true BJD_TDB, in seconds; the true time is
T_QLP - error_s/86400. The Earth's barycentric position comes from astropy; the
correction for an S14/S15 transit reproduces the QLP time stamps of 48 independent
stars to 0.8 s rms. Over a transit the error changes by less than a second, so
shifting a fitted mid-time is equivalent to refitting a corrected light curve.

Which transits are corrected is data, not code:
  qlp_time_errors.csv  one row per affected sector-orbit: TIC (or 'all'), sector,
                       orbit (1, 2 or 'all'), model, note
  qlp_time_spans.csv   the QLP data span of each affected star and sector-orbit
                       (from the pipeline's light-curve cache and the audit downloads)
A transit is corrected when its star's QLP data span for an affected sector-orbit
contains its mid-time (within 0.5 d for whole-sector spans, as for the S14/S15 rows).
"""
import csv
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent
ERRORS_CSV = REPO / "qlp_time_errors.csv"
SPANS_CSV = REPO / "qlp_time_spans.csv"
CATALOG_CSV = REPO / "toi_catalog_240226.csv"
SECTOR_MARGIN_D = 0.5


def _astropy():
    from astropy.time import Time
    from astropy.coordinates import get_body_barycentric
    import astropy.units as u
    import astropy.constants as const
    return Time, get_body_barycentric, u, const


def earth_bary(jd):
    Time, get_body_barycentric, u, _ = _astropy()
    p = get_body_barycentric('earth', Time(np.atleast_1d(np.asarray(jd, float)), format='jd', scale='tdb'))
    return np.stack([p.x.to_value(u.AU), p.y.to_value(u.AU), p.z.to_value(u.AU)], -1)


EPS = np.radians(84381.406 / 3600)                       # J2000 obliquity (IAU 2006)


def to_ecl(v):
    return np.stack([v[..., 0], v[..., 1] * np.cos(EPS) + v[..., 2] * np.sin(EPS),
                     -v[..., 1] * np.sin(EPS) + v[..., 2] * np.cos(EPS)], -1)


def unit(ra, dec):
    ra, de = np.radians(ra), np.radians(dec)
    return np.stack([np.cos(de) * np.cos(ra), np.cos(de) * np.sin(ra), np.sin(de)], -1)


def error_s(model, jd, ra, dec):
    """QLP time minus true BJD_TDB (seconds) for error `model` at times `jd` (BJD), star at (ra, dec) in degrees."""
    _, _, u, const = _astropy()
    au_s = float((1 * u.AU / const.c).to(u.s).value)   # 499.004784 s
    r = earth_bary(jd)
    re = to_ecl(r)
    if model == 's1415':
        ru = np.stack([re[..., 1], re[..., 2], np.zeros(re.shape[:-1])], -1)
    elif model == 's7479':
        ru = re
    else:
        raise ValueError(f'unknown QLP error model {model!r}')
    return np.sum((ru - r) * unit(ra, dec), -1) * au_s


class QLPTimeFix:
    """Look up and compute the correction of one transit mid-time."""

    def __init__(self, errors_csv=ERRORS_CSV, spans_csv=SPANS_CSV, catalog_csv=CATALOG_CSV):
        self.rules = []                                  # (tic or None, sector, orbit or None, model)
        for r in csv.DictReader(open(errors_csv)):
            self.rules.append((None if r['tic'] == 'all' else int(r['tic']), int(r['sector']),
                               None if r['orbit'] == 'all' else int(r['orbit']), r['model']))
        self.spans = {}                                  # tic -> [(sector, orbit or None, lo, hi)]
        for r in csv.DictReader(open(spans_csv)):
            orbit = None if r['orbit'] == 'all' else int(r['orbit'])
            m = SECTOR_MARGIN_D if orbit is None else 0.0
            self.spans.setdefault(int(r['TIC_ID']), []).append(
                (int(r['sector']), orbit, float(r['t_first']) - m, float(r['t_last']) + m))
        import pandas as pd
        cat = pd.read_csv(catalog_csv, encoding='latin-1', usecols=['TOI', 'TIC ID', 'RA', 'Dec'])
        self.pos_toi = {f'{t:.2f}': (ra, de) for t, ra, de in zip(cat.TOI, cat.RA, cat.Dec)}
        self.pos_tic = {}
        for tic, ra, de in zip(cat['TIC ID'], cat.RA, cat.Dec):
            self.pos_tic.setdefault(int(tic), (ra, de))

    def model_for(self, tic, t):
        """(model, sector, orbit) of the affected QLP span of star `tic` that contains time t, else None."""
        for sector, orbit, lo, hi in sorted(self.spans.get(int(tic), []), key=lambda s: s[0]):
            if lo <= t <= hi:
                for rtic, rsec, rorb, model in self.rules:
                    if rsec == sector and rtic in (None, int(tic)) and (rorb is None or rorb == orbit):
                        return model, sector, orbit
        return None

    def correction(self, tic, toi, t):
        """(seconds to ADD to the QLP-based time, tag); (0.0, '') when the time is not in an affected span."""
        hit = self.model_for(tic, t)
        if hit is None:
            return 0.0, ''
        model, sector, orbit = hit
        ra, de = self.pos_toi.get(toi) or self.pos_tic[int(tic)]
        err = float(error_s(model, t, ra, de)[0])
        return -err, f'{model}:S{sector}' + ('' if orbit is None else f'o{orbit}')

    def correct_lightcurve(self, tic, toi, sector, t):
        """Corrected copy of the QLP time stamps t (BJD) of star `tic` in `sector`, and a list describing what
        was applied ([] when the sector-orbit has no known error). Used by run_full_analysis.py --qlp-time-fix."""
        t = np.asarray(t, float)
        rules = [(rorb, model) for rtic, rsec, rorb, model in self.rules
                 if rsec == int(sector) and rtic in (None, int(tic))]
        if not rules or t.size == 0:
            return t, []
        ts = np.sort(t)
        g = np.diff(ts)
        cut = ts[int(np.argmax(g))] + g.max() / 2 if g.size and g.max() > 0.5 else ts[-1] + 1
        ra, de = self.pos_toi.get(toi) or self.pos_tic[int(tic)]
        out, applied = t.copy(), []
        for orbit, model in rules:
            sel = np.ones(t.size, bool) if orbit is None else ((t < cut) if orbit == 1 else (t >= cut))
            if not sel.any():
                continue
            err = error_s(model, t[sel], ra, de)
            out[sel] = t[sel] - err / 86400.0
            applied.append({'sector': int(sector), 'orbit': 'all' if orbit is None else int(orbit), 'model': model,
                            'n_points': int(sel.sum()), 'shift_s_min': float(-err.max()), 'shift_s_max': float(-err.min())})
        return out, applied
