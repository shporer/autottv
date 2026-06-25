"""Joint MCMC over shape parameters + per-transit Tmids.

Parameter vector (length 6 + N_tr):
    [Rp/Rs, a/Rs, b², u1, u2, baseline, T_mid_1, ..., T_mid_N]

P and T0 (linear ephemeris) are FIXED at the cascade-converged values to
avoid degeneracy with the average of the T_mid samples. This focuses the
joint MCMC on shape + per-transit timing — the regime where the
TTV–shape degeneracy actually bites.

Supersampling per cadence:
    ≤ 200 s  (2-min)  → factor 1 (no supersampling)
    600 s    (10-min) → factor 5
    1800 s   (30-min) → factor 15
"""

import logging
from typing import Dict, Tuple, Optional, Any
import numpy as np
import emcee

from . import config

try:
    import batman
except ImportError:
    raise ImportError("batman-package is required: pip install batman-package")

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------
# Module-level state for multiprocessing.Pool pickleability.
# -----------------------------------------------------------------------
_JOINT_DATA: Dict[str, Any] = {}


def supersample_factor(cadence_seconds: float) -> int:
    """Cadence-aware supersampling factor."""
    if cadence_seconds <= 200.0:
        return 1
    elif cadence_seconds <= 700.0:
        return 5
    else:
        return 15


def compute_window_half(window_min: float, period_d: float) -> float:
    """Hard T_mid search half-width in days.

    The requested half-width (window_min minutes) is auto-capped at 0.25*P so
    that a transit's ±window plus its data pad can never overlap an adjacent
    transit, even for short-period planets.
    """
    requested_d = float(window_min) / 1440.0
    cap_d = 0.25 * float(period_d)
    return min(requested_d, cap_d)


# -----------------------------------------------------------------------
# Phase template construction.  One template per (current shape, cadence).
# Built fresh each MCMC step.
# -----------------------------------------------------------------------
def _build_templates(
    rp_rs: float, a_rs: float, b: float, u1: float, u2: float,
    period_d: float, t14_d: float, cadence_groups: Dict[float, int],
    n_grid: int = 2001,
) -> Optional[Dict[float, Tuple[np.ndarray, np.ndarray]]]:
    """Build one (phase, flux) template per cadence group.

    cadence_groups: {cadence_s: n_super}.  Returns None on invalid geometry.
    """
    if b >= a_rs or not (0 < rp_rs < 1):
        return None
    half = 1.5 * t14_d
    phase = np.linspace(-half, half, n_grid)
    inc = float(np.degrees(np.arccos(b / a_rs)))
    p = batman.TransitParams()
    p.t0 = 0.0
    p.per = float(period_d)
    p.rp = float(rp_rs)
    p.a = float(a_rs)
    p.inc = inc
    p.ecc = 0.0
    p.w = 90.0
    p.u = [float(u1), float(u2)]
    p.limb_dark = "quadratic"
    templates: Dict[float, Tuple[np.ndarray, np.ndarray]] = {}
    for cad_s, n_super in cadence_groups.items():
        exp_time_d = cad_s / 86400.0
        try:
            if n_super > 1:
                m = batman.TransitModel(
                    p, phase, supersample_factor=int(n_super), exp_time=float(exp_time_d))
            else:
                m = batman.TransitModel(p, phase)
            flux = m.light_curve(p)
        except Exception:
            return None
        templates[float(cad_s)] = (phase, flux)
    return templates


# -----------------------------------------------------------------------
# Joint log-posterior (top-level for multiprocessing).
# -----------------------------------------------------------------------
def _joint_log_probability(theta: np.ndarray) -> float:
    n_tr = _JOINT_DATA["n_tr"]
    fix_ld = _JOINT_DATA.get("fix_ld", False)
    if fix_ld:
        rp_rs, a_rs, bsq, baseline = theta[:4]
        u1 = _JOINT_DATA["u1_fixed"]
        u2 = _JOINT_DATA["u2_fixed"]
        t_mids = theta[4:4 + n_tr]
    else:
        rp_rs, a_rs, bsq, u1, u2, baseline = theta[:6]
        t_mids = theta[6:6 + n_tr]

    pri = _JOINT_DATA["priors"]

    # Hard bounds on shape
    if not (pri["rp_lo"] < rp_rs < pri["rp_hi"]):  return -np.inf
    if not (pri["ar_lo"] < a_rs < pri["ar_hi"]):    return -np.inf
    if bsq < 0.0:                                    return -np.inf
    b_max_sq = (1.0 + rp_rs) ** 2
    if bsq > b_max_sq:                               return -np.inf
    if abs(u1) > 1 or abs(u2) > 1:                   return -np.inf
    if not (-1.0 < u1 + u2 < 1.0):                   return -np.inf
    if u1 + 2.0 * u2 < 0.0:                          return -np.inf  # Kipping 2013: no limb brightening
    if not (0.95 < baseline < 1.05):                 return -np.inf
    # Hard bounds on each Tmid: within wide window of linear-ephemeris expectation
    expected_tmid = _JOINT_DATA["expected_tmid"]
    tmid_window_d = pri["tmid_window_d"]
    if np.any(np.abs(t_mids - expected_tmid) > tmid_window_d):
        return -np.inf

    b = float(np.sqrt(max(bsq, 0.0)))

    # Gaussian priors
    lp = 0.0
    lp += -0.5 * ((rp_rs - pri["rp_mu"]) / pri["rp_sigma"]) ** 2
    lp += -0.5 * ((a_rs - pri["ar_mu"]) / pri["ar_sigma"]) ** 2
    if not fix_ld:
        lp += -0.5 * ((u1 - pri["u1_mu"]) / pri["u1_sigma"]) ** 2
        lp += -0.5 * ((u2 - pri["u2_mu"]) / pri["u2_sigma"]) ** 2
    # Per-transit Tmid Gaussian prior around linear ephemeris (skipped if relaxed)
    tmid_sigma = pri["tmid_sigma_d"]   # per-transit prior width in days
    if np.isfinite(tmid_sigma):
        lp += np.sum(-0.5 * ((t_mids - expected_tmid) / tmid_sigma) ** 2)

    # Build templates for current shape
    templates = _build_templates(
        rp_rs, a_rs, b, u1, u2,
        _JOINT_DATA["period_d"], _JOINT_DATA["t14_d"],
        _JOINT_DATA["cadence_groups"],
    )
    if templates is None:
        return -np.inf

    # Vectorized model evaluation by transit
    time = _JOINT_DATA["time"]
    transit_idx_of_pt = _JOINT_DATA["transit_idx_of_pt"]
    cadence_of_pt = _JOINT_DATA["cadence_of_pt"]
    model = np.full_like(time, baseline)
    # Loop per transit (cheap — n_tr small, vectorized inside)
    for j in range(n_tr):
        mask = transit_idx_of_pt == j
        if not mask.any():
            continue
        t_local = time[mask] - t_mids[j]
        cad_local = cadence_of_pt[mask]
        # Most transits have a single cadence; loop only over unique
        for cad_s, (ph_grid, fl_grid) in templates.items():
            cm = cad_local == cad_s
            if cm.any():
                # Compose mask back into the global array
                global_idx = np.where(mask)[0][cm]
                model[global_idx] = baseline * np.interp(
                    t_local[cm], ph_grid, fl_grid, left=1.0, right=1.0)

    flux = _JOINT_DATA["flux"]
    inv_var = _JOINT_DATA["inv_var"]
    chi2 = float(np.sum((flux - model) ** 2 * inv_var))
    return float(lp - 0.5 * chi2)


# -----------------------------------------------------------------------
# Bad-walker rejection (cascade-style; ported from run_full_analysis.py).
# -----------------------------------------------------------------------
def identify_bad_walkers(log_prob: np.ndarray, sigma_mult: float = 10.0):
    """Cascade-style bad-walker detection over the last 10% of burn-in.

    log_prob shape: (n_steps, n_walkers).  Returns (bad_indices, sigma,
    median_of_medians).  Mirrors run_full_analysis.py::_identify_bad_walkers.
    """
    n_steps, n_walkers = log_prob.shape
    tail_steps = max(100, n_steps // 10)
    log_prob_tail = log_prob[-tail_steps:, :]
    walker_medians = np.median(log_prob_tail, axis=0)
    median_of_medians = float(np.median(walker_medians))
    mad = float(np.median(np.abs(walker_medians - median_of_medians)))
    sigma = 1.48 * mad
    if sigma > 0:
        threshold = median_of_medians - sigma_mult * sigma
        bad_indices = np.where(walker_medians < threshold)[0].tolist()
    else:
        bad_indices = []
    return bad_indices, sigma, median_of_medians


def reinitialize_bad_walkers(chains: np.ndarray, bad_indices) -> np.ndarray:
    """Cascade-style re-init: draw each bad walker per-parameter from
    median +/- U(-1,1)*(1.48*MAD) of the good walkers' chains.

    chains shape: (n_steps, n_walkers, n_params).  Returns new positions
    (n_walkers, n_params).  Mirrors run_full_analysis.py::_reinitialize_bad_walkers.
    """
    n_steps, n_walkers, n_params = chains.shape
    final_positions = chains[-1, :, :].copy()
    if not bad_indices:
        return final_positions
    good_indices = [i for i in range(n_walkers) if i not in bad_indices]
    good_chains = chains[:, good_indices, :]
    param_medians = np.zeros(n_params)
    param_sigmas = np.zeros(n_params)
    for p in range(n_params):
        vals = good_chains[:, :, p].ravel()
        param_medians[p] = np.median(vals)
        param_sigmas[p] = 1.48 * np.median(np.abs(vals - param_medians[p]))
    rng = np.random.default_rng(123)
    for idx in bad_indices:
        final_positions[idx, :] = param_medians + rng.uniform(-1, 1, n_params) * param_sigmas
    return final_positions


def compute_window_diagnostics(tmid_chain, expected_tmid, window_half_d, epochs,
                               edge_frac=0.02, p99_frac_thresh=0.95,
                               edge_count_thresh=0.05):
    """Flag transits whose T_mid posterior pushes the hard window bound.

    tmid_chain : (n_samples, n_tr) flattened posterior of each T_mid (days).
    expected_tmid : (n_tr,) cascade individual times (window centers, days).
    window_half_d : hard half-width (days).
    epochs : (n_tr,) epoch numbers for reporting.

    A transit is flagged if dev_p99 >= p99_frac_thresh*window_half OR the
    fraction of samples within edge_frac of the bound exceeds edge_count_thresh.
    Returns (flagged_list, n_flagged).  Each dict: epoch, dev_max_min,
    dev_p99_min, window_min, frac_at_edge.
    """
    tmid_chain = np.asarray(tmid_chain)
    expected_tmid = np.asarray(expected_tmid, dtype=float)
    n_tr = tmid_chain.shape[1]
    window_min = window_half_d * 1440.0
    flagged = []
    for j in range(n_tr):
        dev = np.abs(tmid_chain[:, j] - expected_tmid[j])
        dev_p99 = float(np.percentile(dev, 99))
        dev_max = float(dev.max())
        frac_at_edge = float(np.mean(dev >= (1.0 - edge_frac) * window_half_d))
        if dev_p99 >= p99_frac_thresh * window_half_d or frac_at_edge > edge_count_thresh:
            flagged.append({
                "epoch": int(epochs[j]),
                "dev_max_min": dev_max * 1440.0,
                "dev_p99_min": dev_p99 * 1440.0,
                "window_min": float(window_min),
                "frac_at_edge": frac_at_edge,
            })
    return flagged, len(flagged)


# -----------------------------------------------------------------------
# Driver class.
# -----------------------------------------------------------------------
class JointTransitFitter:
    """Joint MCMC for shape + per-transit Tmids on real TESS data.

    Designed to be called once per TOI, after the cascade has run.
    Holds (P, T0) fixed at the cascade's linear ephemeris and samples the
    remaining 6 + N_tr parameters.
    """

    def __init__(
        self,
        n_walkers_min: int = 256,
        n_walkers_cap: int = 1024,
        n_burn: int = 4000,
        n_steps_max: int = 9000,
        progress: bool = False,
        n_walkers_per_dim: int = 2,
    ):
        self.n_walkers_min = n_walkers_min
        self.n_walkers_cap = n_walkers_cap
        self.n_burn = n_burn
        self.n_steps_max = n_steps_max
        self.progress = progress
        # Target walkers = max(n_walkers_min, n_walkers_per_dim * n_dim), capped.
        # emcee's hard floor is 2*n_dim; >=4 gives better ensemble mixing for the
        # high-N_tr (high-dimensional) joint fits.
        self.n_walkers_per_dim = n_walkers_per_dim

    def fit(
        self,
        time: np.ndarray,
        flux: np.ndarray,
        flux_err: np.ndarray,
        cadence_of_pt: np.ndarray,         # in seconds, same length as time
        tmid_init: np.ndarray,             # per-transit initial guesses (days)
        tmid_init_err: np.ndarray,         # per-transit cascade Tmid errors (days)
        period_d: float,                   # cascade-converged P
        t0_d: float,                       # cascade-converged T0
        t14_d: float,                      # cascade-converged transit duration
        shape_init: Dict[str, float],      # rp_rs, a_rs, b, u1, u2 from cascade
        shape_init_err: Dict[str, float],  # cascade uncertainties for walker spread
        priors: Optional[Dict[str, float]] = None,
        fix_ld: bool = False,
        u1_fixed: Optional[float] = None,
        u2_fixed: Optional[float] = None,
        tmid_prior_mode: str = "gaussian",
        tmid_window_min: float = 30.0,
    ) -> Dict[str, Any]:
        """Run the joint MCMC.  Returns posteriors + chains."""
        n_tr = len(tmid_init)
        half = 1.5 * t14_d

        # Hard search window (auto-capped to avoid adjacent-transit overlap)
        tmid_window_d = compute_window_half(tmid_window_min, period_d)
        requested_d = tmid_window_min / 1440.0
        if tmid_window_d < requested_d - 1e-12:
            logger.warning(
                f"  T_mid window capped {tmid_window_min:.0f}min -> "
                f"{tmid_window_d*1440:.1f}min (0.25*P={0.25*period_d*1440:.1f}min)")
        padded_half = half + tmid_window_d

        in_window = np.zeros_like(time, dtype=bool)
        transit_idx_of_pt = np.full(len(time), -1, dtype=np.int32)
        for j, tm in enumerate(tmid_init):
            mj = (time >= tm - padded_half) & (time <= tm + padded_half)
            if mj.any():
                in_window |= mj
                transit_idx_of_pt[mj] = j

        t_w = time[in_window]; f_w = flux[in_window]
        fe_w = flux_err[in_window]; cad_w = cadence_of_pt[in_window]
        tidx_w = transit_idx_of_pt[in_window]

        keep_transit = np.zeros(n_tr, dtype=bool)
        for j in range(n_tr):
            if (tidx_w == j).sum() >= 5:
                keep_transit[j] = True
        kept_indices = np.where(keep_transit)[0]  # indices into the ORIGINAL tmid_init/epochs
        if not keep_transit.all():
            dropped = (~keep_transit).sum()
            logger.warning(f"  Dropping {dropped} transits with < 5 in-window points")
            old_to_new = -np.ones(n_tr, dtype=np.int32)
            old_to_new[keep_transit] = np.arange(keep_transit.sum())
            tidx_w = np.where(tidx_w >= 0, old_to_new[tidx_w], -1)
            keep_mask = tidx_w >= 0
            t_w = t_w[keep_mask]; f_w = f_w[keep_mask]; fe_w = fe_w[keep_mask]
            cad_w = cad_w[keep_mask]; tidx_w = tidx_w[keep_mask]
            tmid_init = tmid_init[keep_transit]
            tmid_init_err = tmid_init_err[keep_transit]
            n_tr = int(keep_transit.sum())
        if n_tr < 3:
            raise ValueError(f"Only {n_tr} usable transits after filtering — aborting")

        unique_cads = np.unique(cad_w)
        cadence_groups = {float(c): supersample_factor(float(c)) for c in unique_cads}
        logger.info(f"  Cadence groups: {cadence_groups}  (n_tr={n_tr}, n_pts={len(t_w)})")

        if priors is None:
            priors = {}
        if tmid_prior_mode == "relaxed":
            tmid_sigma_d = np.inf
        else:
            tmid_sigma_d = float(priors.get("tmid_sigma_d",
                                            min(0.05 * period_d, 0.5 * t14_d)))
        if fix_ld and (u1_fixed is None or u2_fixed is None):
            raise ValueError("fix_ld=True requires u1_fixed and u2_fixed")
        priors_full = {
            "rp_lo": 0.001, "rp_hi": 0.5, "ar_lo": 1.0, "ar_hi": 500.0,
            "rp_mu": shape_init["rp_rs"],
            "rp_sigma": max(shape_init_err.get("rp_rs", 0.01), 0.001),
            "ar_mu": shape_init["a_rs"],
            "ar_sigma": max(shape_init_err.get("a_rs", 0.5), 0.05),
            "u1_mu": shape_init["u1"], "u1_sigma": priors.get("u1_sigma", 0.15),
            "u2_mu": shape_init["u2"], "u2_sigma": priors.get("u2_sigma", 0.10),
            "tmid_window_d": tmid_window_d,
            "tmid_sigma_d": tmid_sigma_d,
        }
        priors_full.update({k: v for k, v in priors.items() if k not in priors_full})

        global _JOINT_DATA
        _JOINT_DATA.clear()
        _JOINT_DATA.update({
            "time": t_w, "flux": f_w,
            "inv_var": 1.0 / (fe_w.astype(np.float64) ** 2),
            "cadence_of_pt": cad_w.astype(np.float64),
            "transit_idx_of_pt": tidx_w.astype(np.int32),
            "expected_tmid": tmid_init.astype(np.float64),
            "n_tr": int(n_tr), "period_d": float(period_d),
            "t14_d": float(t14_d), "cadence_groups": cadence_groups,
            "priors": priors_full,
            "fix_ld": bool(fix_ld),
            "u1_fixed": (float(u1_fixed) if fix_ld else None),
            "u2_fixed": (float(u2_fixed) if fix_ld else None),
        })
        shape_dim = 4 if fix_ld else 6

        # MCMC dimensions
        n_dim = shape_dim + n_tr
        n_walkers = max(self.n_walkers_min, self.n_walkers_per_dim * n_dim)
        n_walkers = min(n_walkers, self.n_walkers_cap)
        if n_walkers < 2 * n_dim:  # emcee hard requirement
            n_walkers = 2 * n_dim + 4
        if n_walkers % 2:          # emcee requires an even walker count
            n_walkers += 1

        b_init = float(shape_init.get("b", 0.5))
        bsq_init = b_init ** 2
        if fix_ld:
            center_shape = np.array([shape_init["rp_rs"], shape_init["a_rs"],
                                     bsq_init, 1.0])
            sigma_shape = np.array([
                max(0.25 * shape_init_err.get("rp_rs", 0.01), 1e-4),
                max(0.25 * shape_init_err.get("a_rs", 0.5), 1e-3),
                max(0.5 * (2 * abs(b_init) * shape_init_err.get("b", 0.1)), 1e-4),
                1e-3,
            ])
        else:
            center_shape = np.array([shape_init["rp_rs"], shape_init["a_rs"],
                                     bsq_init, shape_init["u1"], shape_init["u2"], 1.0])
            sigma_shape = np.array([
                max(0.25 * shape_init_err.get("rp_rs", 0.01), 1e-4),
                max(0.25 * shape_init_err.get("a_rs", 0.5), 1e-3),
                max(0.5 * (2 * abs(b_init) * shape_init_err.get("b", 0.1)), 1e-4),
                0.05, 0.05, 1e-3,
            ])
        center = np.concatenate([center_shape, tmid_init.astype(np.float64)])
        sigma = np.concatenate([sigma_shape,
                                np.maximum(tmid_init_err.astype(np.float64), 1e-4)])
        rng = np.random.default_rng(42)
        p0 = center + sigma * rng.standard_normal((n_walkers, n_dim))
        p0[:, 2] = np.clip(p0[:, 2], 1e-8, (1 + center[0]) ** 2 - 1e-8)

        logger.info(f"  Joint MCMC: {n_walkers} walkers x <= {self.n_steps_max} "
                    f"steps x {n_dim} dim (fix_ld={fix_ld})")
        sampler = emcee.EnsembleSampler(n_walkers, n_dim, _joint_log_probability)

        # First burn-in
        state = sampler.run_mcmc(p0, self.n_burn, progress=self.progress,
                                 skip_initial_state_check=True)
        # Cascade-style bad-walker rejection (last 10% detection)
        burnin_log_prob = sampler.get_log_prob()
        burnin_chains = sampler.get_chain()
        bad_indices, sigma_bw, _ = identify_bad_walkers(
            burnin_log_prob, sigma_mult=getattr(config, "BAD_WALKER_SIGMA", 10.0))
        n_bad = len(bad_indices)
        if n_bad > 0:
            new_positions = reinitialize_bad_walkers(burnin_chains, bad_indices)
            logger.info(f"  Re-initialized {n_bad} bad walkers; second burn-in")
            sampler.reset()
            second_burn = getattr(config, "N_BURN_MIN", 2000)
            state = sampler.run_mcmc(new_positions, second_burn,
                                     progress=self.progress,
                                     skip_initial_state_check=True)
        sampler.reset()
        # Production
        sampler.run_mcmc(state, self.n_steps_max, progress=self.progress,
                         skip_initial_state_check=True)

        flat = sampler.get_chain(discard=0, flat=True)
        med_post = np.median(flat, axis=0)
        std_post = np.std(flat, axis=0)
        b_post = np.sqrt(np.clip(flat[:, 2], 0, None))
        b_med = float(np.median(b_post)); b_err = float(np.std(b_post))
        acc = float(np.mean(sampler.acceptance_fraction))
        chains = sampler.get_chain()
        rhat = _split_rhat(chains)

        # Window-limit diagnostics from the flat T_mid samples
        tmid_flat = flat[:, shape_dim:shape_dim + n_tr]
        epochs_arr = np.arange(n_tr)  # placeholder; driver overrides with true epochs
        window_limited, n_window_limited = compute_window_diagnostics(
            tmid_flat, tmid_init.astype(np.float64), tmid_window_d, epochs_arr)

        # Shape-only production chain (n_steps, n_walkers, shape_dim)
        shape_chain = chains[:, :, :shape_dim].astype(np.float32)
        shape_param_names = (["rp_rs", "a_rs", "b_sq", "baseline"] if fix_ld
                             else ["rp_rs", "a_rs", "b_sq", "u1", "u2", "baseline"])

        result = {
            "rp_rs": float(med_post[0]), "rp_rs_err": float(std_post[0]),
            "a_rs": float(med_post[1]), "a_rs_err": float(std_post[1]),
            "b_sq": float(med_post[2]), "b_sq_err": float(std_post[2]),
            "b": b_med, "b_err": b_err,
            "baseline": float(med_post[shape_dim - 1]),
            "baseline_err": float(std_post[shape_dim - 1]),
            "t_mids": med_post[shape_dim:shape_dim + n_tr].astype(float).tolist(),
            "t_mids_err": std_post[shape_dim:shape_dim + n_tr].astype(float).tolist(),
            "n_walkers": int(n_walkers), "n_steps": int(self.n_steps_max),
            "n_burn": int(self.n_burn), "n_dim": int(n_dim),
            "n_transits_used": int(n_tr), "n_points_in_window": int(len(t_w)),
            "kept_transit_indices": kept_indices.tolist(),
            "acceptance_fraction": acc,
            "rhat_shape_max": float(np.max(rhat[:shape_dim])),
            "rhat_tmid_max": float(np.max(rhat[shape_dim:])),
            "rhat_global_max": float(np.max(rhat)),
            "n_bad_walkers": int(n_bad),
            "cadence_groups": {f"{int(k)}s": int(v) for k, v in cadence_groups.items()},
            "shape_init": shape_init,
            "ld_fixed": bool(fix_ld),
            "u1_fixed": (float(u1_fixed) if fix_ld else None),
            "u2_fixed": (float(u2_fixed) if fix_ld else None),
            "window_min": float(tmid_window_d * 1440.0),
            "window_limited_transits": window_limited,
            "n_window_limited": int(n_window_limited),
            "_shape_chain": shape_chain,            # popped + saved by driver
            "_shape_param_names": shape_param_names, # popped by driver
        }
        if not fix_ld:
            result["u1"] = float(med_post[3]); result["u1_err"] = float(std_post[3])
            result["u2"] = float(med_post[4]); result["u2_err"] = float(std_post[4])
        _JOINT_DATA.clear()
        return result


# -----------------------------------------------------------------------
# Convergence diagnostics
# -----------------------------------------------------------------------
def _split_rhat(chains: np.ndarray) -> np.ndarray:
    """Split-Rhat per parameter.  chains shape: (n_steps, n_walkers, n_dim)."""
    n_steps, n_walkers, n_dim = chains.shape
    if n_steps < 4:
        return np.full(n_dim, np.nan)
    half = n_steps // 2
    first = chains[:half]
    second = chains[half:2 * half]
    combined = np.concatenate([first, second], axis=1)  # (half, 2*n_walkers, n_dim)
    m = combined.shape[1]
    n = combined.shape[0]
    chain_means = np.mean(combined, axis=0)
    chain_vars = np.var(combined, axis=0, ddof=1)
    grand_mean = np.mean(chain_means, axis=0)
    B = n * np.var(chain_means, axis=0, ddof=1)
    W = np.mean(chain_vars, axis=0)
    var_hat = ((n - 1) / n) * W + (1 / n) * B
    with np.errstate(invalid='ignore', divide='ignore'):
        rhat = np.sqrt(var_hat / W)
    rhat[~np.isfinite(rhat)] = np.nan
    return rhat
