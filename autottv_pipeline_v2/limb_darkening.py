"""
Limb Darkening Coefficients from Claret 2017 (A&A 600, A30)

This module provides quadratic limb darkening coefficients for the TESS bandpass
using bilinear interpolation in (Teff, log(g)) space.

Reference:
    Claret, A. 2017, A&A, 600, A30
    "Limb and gravity-darkening coefficients for the TESS satellite at several
    metallicities, surface gravities, and microturbulent velocities"
    https://ui.adsabs.harvard.edu/abs/2017A%26A...600A..30C/abstract

Coefficients are from the PHOENIX spherical models using the r-method
(radial intensity method) with the Least Squares Method (LSM) fitting.

The quadratic limb darkening law is:
    I(mu)/I(1) = 1 - u1*(1-mu) - u2*(1-mu)^2

where mu = cos(theta) and theta is the angle from the line of sight.

Note: The Claret 2017 PHOENIX r-method tables are only available at solar
metallicity ([M/H] = 0.0). Metallicity parameter is accepted for API
compatibility but is not used in interpolation.
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from . import config

# =============================================================================
# Claret 2017 Table for TESS bandpass - Quadratic Limb Darkening Law
# PHOENIX spherical models, r-method, LSM fitting
# Format: 2D arrays indexed by [Teff, logg]
# Solar metallicity only ([M/H] = 0.0)
# =============================================================================

# Grid of Teff values (K) - selected range for typical exoplanet host stars
TEFF_GRID = np.array([
    3500, 3600, 3700, 3800, 3900, 4000, 4100, 4200, 4300, 4400,
    4500, 4600, 4700, 4800, 4900, 5100, 5200, 5300, 5400, 5500,
    5600, 5700, 5800, 5900, 6000, 6100, 6200, 6300, 6400, 6500,
    6600, 6700, 6800, 6900, 7000, 7200, 7400, 7600, 7800, 8000,
    8200, 8400, 8600, 8800, 9000, 9200, 9400, 9600, 9800, 10000
])

# Grid of log(g) values (cgs)
LOGG_GRID = np.array([3.0, 3.5, 4.0, 4.5, 5.0])

# =============================================================================
# u1 coefficients 2D array [Teff, logg]
# From Claret 2017 PHOENIX r-method, column 'aLSM'
# =============================================================================
U1_TABLE = np.array([
    [0.4443, 0.3279, 0.2316, 0.1716, 0.1383],  # Teff = 3500
    [0.5081, 0.3815, 0.2680, 0.1868, 0.1439],  # Teff = 3600
    [0.5482, 0.4448, 0.3115, 0.2081, 0.1524],  # Teff = 3700
    [0.5592, 0.5032, 0.3653, 0.2354, 0.1630],  # Teff = 3800
    [0.5707, 0.5500, 0.4137, 0.2780, 0.1795],  # Teff = 3900
    [0.5542, 0.5520, 0.4608, 0.3232, 0.2073],  # Teff = 4000
    [0.5294, 0.5322, 0.5010, 0.3812, 0.2495],  # Teff = 4100
    [0.5099, 0.5241, 0.5251, 0.4291, 0.2997],  # Teff = 4200
    [0.4927, 0.5084, 0.5115, 0.4702, 0.3561],  # Teff = 4300
    [0.4772, 0.4934, 0.4992, 0.4949, 0.4130],  # Teff = 4400
    [0.4632, 0.4790, 0.4859, 0.4874, 0.4503],  # Teff = 4500
    [0.4470, 0.4648, 0.4717, 0.4809, 0.4742],  # Teff = 4600
    [0.4339, 0.4508, 0.4583, 0.4669, 0.4702],  # Teff = 4700
    [0.4212, 0.4381, 0.4450, 0.4534, 0.4604],  # Teff = 4800
    [0.4095, 0.4260, 0.4335, 0.4400, 0.4470],  # Teff = 4900
    [0.3912, 0.4089, 0.4042, 0.4136, 0.4187],  # Teff = 5100
    [0.3837, 0.4014, 0.3941, 0.4027, 0.4069],  # Teff = 5200
    [0.3784, 0.3850, 0.3853, 0.3927, 0.3964],  # Teff = 5300
    [0.3700, 0.3793, 0.3777, 0.3839, 0.3877],  # Teff = 5400
    [0.3584, 0.3709, 0.3694, 0.3763, 0.3787],  # Teff = 5500
    [0.3703, 0.3639, 0.3622, 0.3688, 0.3543],  # Teff = 5600
    [0.3444, 0.3580, 0.3565, 0.3612, 0.3448],  # Teff = 5700
    [0.3563, 0.3545, 0.3500, 0.3398, 0.3358],  # Teff = 5800
    [0.3497, 0.3441, 0.3436, 0.3361, 0.3274],  # Teff = 5900
    [0.3438, 0.3389, 0.3376, 0.3249, 0.3199],  # Teff = 6000
    [0.3373, 0.3345, 0.3345, 0.3179, 0.3156],  # Teff = 6100
    [0.3306, 0.3274, 0.3288, 0.3120, 0.3086],  # Teff = 6200
    [0.3243, 0.3232, 0.3218, 0.3080, 0.3013],  # Teff = 6300
    [0.2966, 0.3167, 0.3168, 0.3021, 0.2952],  # Teff = 6400
    [0.2891, 0.3098, 0.3112, 0.2986, 0.2905],  # Teff = 6500
    [0.2814, 0.3037, 0.3053, 0.2921, 0.2863],  # Teff = 6600
    [0.2741, 0.2952, 0.2996, 0.2878, 0.2821],  # Teff = 6700
    [0.2664, 0.2871, 0.2934, 0.2839, 0.2783],  # Teff = 6800
    [0.2583, 0.2824, 0.2870, 0.2802, 0.2743],  # Teff = 6900
    [0.2508, 0.2751, 0.2804, 0.2757, 0.2707],  # Teff = 7000
    [0.2336, 0.2604, 0.2641, 0.2666, 0.2631],  # Teff = 7200
    [0.2179, 0.2444, 0.2518, 0.2344, 0.2548],  # Teff = 7400
    [0.2010, 0.2287, 0.2183, 0.2378, 0.2467],  # Teff = 7600
    [0.1857, 0.2042, 0.2221, 0.2260, 0.2373],  # Teff = 7800
    [0.1723, 0.1976, 0.2090, 0.2138, 0.2276],  # Teff = 8000
    [0.1596, 0.1653, 0.1956, 0.2011, 0.2165],  # Teff = 8200
    [0.1506, 0.1545, 0.1842, 0.1890, 0.2056],  # Teff = 8400
    [0.1440, 0.1461, 0.1710, 0.1771, 0.1943],  # Teff = 8600
    [0.1388, 0.1398, 0.1630, 0.1670, 0.1761],  # Teff = 8800
    [0.1330, 0.1354, 0.1579, 0.1578, 0.1676],  # Teff = 9000
    [0.1277, 0.1309, 0.1531, 0.1519, 0.1587],  # Teff = 9200
    [0.1212, 0.1260, 0.1496, 0.1477, 0.1512],  # Teff = 9400
    [0.1136, 0.1212, 0.1464, 0.1443, 0.1459],  # Teff = 9600
    [0.1049, 0.1150, 0.1426, 0.1399, 0.1416],  # Teff = 9800
    [0.0955, 0.1345, 0.1381, 0.1364, 0.1375],  # Teff = 10000
])

# =============================================================================
# u2 coefficients 2D array [Teff, logg]
# From Claret 2017 PHOENIX r-method, column 'bLSM'
# =============================================================================
U2_TABLE = np.array([
    [0.2555, 0.3492, 0.4156, 0.4496, 0.4644],  # Teff = 3500
    [0.1807, 0.2908, 0.3749, 0.4329, 0.4540],  # Teff = 3600
    [0.1337, 0.2306, 0.3356, 0.4113, 0.4425],  # Teff = 3700
    [0.1308, 0.1766, 0.2941, 0.3904, 0.4298],  # Teff = 3800
    [0.1139, 0.1311, 0.2509, 0.3508, 0.4182],  # Teff = 3900
    [0.1374, 0.1378, 0.2150, 0.3173, 0.3943],  # Teff = 4000
    [0.1674, 0.1607, 0.1827, 0.2772, 0.3623],  # Teff = 4100
    [0.1845, 0.1592, 0.1557, 0.2350, 0.3257],  # Teff = 4200
    [0.1975, 0.1704, 0.1641, 0.2035, 0.2897],  # Teff = 4300
    [0.2078, 0.1800, 0.1721, 0.1763, 0.2327],  # Teff = 4400
    [0.2139, 0.1889, 0.1794, 0.1779, 0.2089],  # Teff = 4500
    [0.2273, 0.1972, 0.1882, 0.1743, 0.1826],  # Teff = 4600
    [0.2336, 0.2049, 0.1955, 0.1832, 0.1812],  # Teff = 4700
    [0.2388, 0.2106, 0.2025, 0.1909, 0.1856],  # Teff = 4800
    [0.2429, 0.2153, 0.2055, 0.1989, 0.1939],  # Teff = 4900
    [0.2410, 0.2099, 0.2169, 0.2054, 0.2034],  # Teff = 5100
    [0.2421, 0.2102, 0.2201, 0.2093, 0.2085],  # Teff = 5200
    [0.2394, 0.2220, 0.2220, 0.2122, 0.2116],  # Teff = 5300
    [0.2423, 0.2204, 0.2224, 0.2139, 0.2113],  # Teff = 5400
    [0.2473, 0.2237, 0.2247, 0.2136, 0.2134],  # Teff = 5500
    [0.2185, 0.2256, 0.2260, 0.2147, 0.2308],  # Teff = 5600
    [0.2538, 0.2257, 0.2252, 0.2166, 0.2335],  # Teff = 5700
    [0.2229, 0.2266, 0.2263, 0.2316, 0.2359],  # Teff = 5800
    [0.2246, 0.2301, 0.2274, 0.2316, 0.2379],  # Teff = 5900
    [0.2251, 0.2299, 0.2282, 0.2350, 0.2388],  # Teff = 6000
    [0.2269, 0.2303, 0.2278, 0.2365, 0.2390],  # Teff = 6100
    [0.2285, 0.2312, 0.2282, 0.2368, 0.2400],  # Teff = 6200
    [0.2294, 0.2299, 0.2288, 0.2368, 0.2388],  # Teff = 6300
    [0.2673, 0.2310, 0.2288, 0.2360, 0.2402],  # Teff = 6400
    [0.2701, 0.2325, 0.2299, 0.2354, 0.2397],  # Teff = 6500
    [0.2731, 0.2344, 0.2305, 0.2367, 0.2390],  # Teff = 6600
    [0.2759, 0.2370, 0.2316, 0.2366, 0.2385],  # Teff = 6700
    [0.2786, 0.2402, 0.2332, 0.2360, 0.2378],  # Teff = 6800
    [0.2820, 0.2420, 0.2350, 0.2353, 0.2376],  # Teff = 6900
    [0.2842, 0.2444, 0.2371, 0.2357, 0.2371],  # Teff = 7000
    [0.2908, 0.2490, 0.2424, 0.2357, 0.2357],  # Teff = 7200
    [0.2976, 0.2551, 0.2460, 0.2511, 0.2363],  # Teff = 7400
    [0.3042, 0.2603, 0.2660, 0.2524, 0.2358],  # Teff = 7600
    [0.3100, 0.2709, 0.2573, 0.2542, 0.2364],  # Teff = 7800
    [0.3142, 0.2703, 0.2590, 0.2564, 0.2373],  # Teff = 8000
    [0.3176, 0.3051, 0.2600, 0.2574, 0.2368],  # Teff = 8200
    [0.3183, 0.3059, 0.2602, 0.2575, 0.2374],  # Teff = 8400
    [0.3189, 0.3057, 0.2611, 0.2583, 0.2379],  # Teff = 8600
    [0.3203, 0.3054, 0.2605, 0.2576, 0.2491],  # Teff = 8800
    [0.3259, 0.3052, 0.2582, 0.2560, 0.2462],  # Teff = 9000
    [0.3318, 0.3075, 0.2582, 0.2551, 0.2464],  # Teff = 9200
    [0.3405, 0.3122, 0.2584, 0.2540, 0.2452],  # Teff = 9400
    [0.3505, 0.3174, 0.2596, 0.2553, 0.2454],  # Teff = 9600
    [0.3627, 0.3252, 0.2628, 0.2563, 0.2458],  # Teff = 9800
    [0.3752, 0.2885, 0.2677, 0.2587, 0.2473],  # Teff = 10000
])

# Solar values - import from centralized config
SOLAR_TEFF = config.SOLAR_TEFF
SOLAR_LOGG = config.SOLAR_LOGG
# Pre-computed solar LD coefficients for TESS bandpass (Teff=5778, logg=4.44)
# Interpolated from the table above
SOLAR_U1 = 0.3516
SOLAR_U2 = 0.2274


class LimbDarkeningInterpolator:
    """
    Bilinear interpolator for Claret 2017 limb darkening coefficients.

    Uses the quadratic limb darkening law for the TESS bandpass,
    interpolating over effective temperature and surface gravity.

    Note: Metallicity parameter is accepted for API compatibility but
    is not used (only solar metallicity data available in PHOENIX r-method).
    """

    def __init__(self):
        """Initialize the interpolators for u1 and u2."""
        # Create RegularGridInterpolator for u1 and u2 (2D)
        self._u1_interp = RegularGridInterpolator(
            (TEFF_GRID, LOGG_GRID),
            U1_TABLE,
            method='linear',
            bounds_error=False,
            fill_value=None  # Use nearest value for out-of-bounds
        )

        self._u2_interp = RegularGridInterpolator(
            (TEFF_GRID, LOGG_GRID),
            U2_TABLE,
            method='linear',
            bounds_error=False,
            fill_value=None
        )

        # Store grid bounds for clamping
        self._teff_min = TEFF_GRID.min()
        self._teff_max = TEFF_GRID.max()
        self._logg_min = LOGG_GRID.min()
        self._logg_max = LOGG_GRID.max()

    def get_coefficients(self, teff, logg, mh=None):
        """
        Get quadratic limb darkening coefficients for given stellar parameters.

        Parameters
        ----------
        teff : float
            Stellar effective temperature in Kelvin.
        logg : float
            Stellar surface gravity log(g) in cgs units.
        mh : float, optional
            Stellar metallicity [M/H] in dex.
            Accepted for API compatibility but not used (solar only).

        Returns
        -------
        u1, u2 : tuple of float
            Quadratic limb darkening coefficients.

        Notes
        -----
        If teff or logg is None or NaN, solar values are used as default.
        Values outside the grid are clamped to the nearest edge.
        """
        # Handle missing or invalid values
        if teff is None or not np.isfinite(teff):
            teff = SOLAR_TEFF
        if logg is None or not np.isfinite(logg):
            logg = SOLAR_LOGG

        # Clamp to grid bounds (extrapolation uses edge values)
        teff_clamped = np.clip(teff, self._teff_min, self._teff_max)
        logg_clamped = np.clip(logg, self._logg_min, self._logg_max)

        # Interpolate
        point = np.array([[teff_clamped, logg_clamped]])
        u1 = float(self._u1_interp(point)[0])
        u2 = float(self._u2_interp(point)[0])

        return u1, u2


# Module-level interpolator instance (singleton pattern)
_interpolator = None


def get_limb_darkening(teff, logg=None, mh=None):
    """
    Get quadratic limb darkening coefficients for the TESS bandpass.

    Uses Claret 2017 (A&A 600, A30) tables with bilinear interpolation
    in (Teff, log(g)) space. Coefficients are from PHOENIX spherical
    models using the r-method (radial intensity method).

    Parameters
    ----------
    teff : float
        Stellar effective temperature in Kelvin.
    logg : float, optional
        Stellar surface gravity log(g) in cgs units.
        If None or invalid, solar value (4.44) is used.
    mh : float, optional
        Stellar metallicity [M/H] in dex.
        Accepted for API compatibility but not used (solar only).

    Returns
    -------
    u1, u2 : tuple of float
        Quadratic limb darkening coefficients for use in the law:
        I(mu)/I(1) = 1 - u1*(1-mu) - u2*(1-mu)^2

    Examples
    --------
    >>> u1, u2 = get_limb_darkening(5778, 4.44)  # Solar
    >>> print(f"u1={u1:.4f}, u2={u2:.4f}")
    u1=0.3516, u2=0.2274

    >>> u1, u2 = get_limb_darkening(4000, 4.5)  # K dwarf
    >>> print(f"u1={u1:.4f}, u2={u2:.4f}")
    u1=0.3232, u2=0.3173
    """
    global _interpolator
    if _interpolator is None:
        _interpolator = LimbDarkeningInterpolator()

    return _interpolator.get_coefficients(teff, logg, mh)


def get_solar_limb_darkening():
    """
    Get solar limb darkening coefficients.

    Returns
    -------
    u1, u2 : tuple of float
        Solar quadratic limb darkening coefficients for TESS bandpass.
    """
    return SOLAR_U1, SOLAR_U2


if __name__ == "__main__":
    # Test the interpolation
    print("Testing Claret 2017 Limb Darkening Interpolation")
    print("PHOENIX r-method, quadratic law, TESS bandpass")
    print("=" * 60)

    test_cases = [
        (5778, 4.44, "Sun"),
        (4000, 4.5, "K dwarf"),
        (4500, 4.5, "K dwarf (cooler)"),
        (5500, 4.5, "G dwarf"),
        (6000, 4.0, "F star"),
        (6500, 4.0, "F star (hotter)"),
        (8000, 4.0, "A star"),
        (3500, 5.0, "M dwarf"),
        (None, None, "Default (solar)"),
    ]

    print(f"{'Star Type':<20s} {'Teff':>6s} {'logg':>5s}   {'u1':>6s} {'u2':>6s}")
    print("-" * 50)
    for teff, logg, name in test_cases:
        u1, u2 = get_limb_darkening(teff, logg)
        teff_str = f"{teff}" if teff else "None"
        logg_str = f"{logg}" if logg else "None"
        print(f"{name:<20s} {teff_str:>6s} {logg_str:>5s}   {u1:>6.4f} {u2:>6.4f}")
