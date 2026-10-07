#!/usr/bin/env python3
"""
Joint sinusoidal-TTV fit for TTV candidates that pass C2 (FAP < 0.01) AND
the C2 LOO test.

Model:  T_obs(e) = T0 + P * e + A * sin(2π · e · P / P_TTV + φ)

5 free parameters (T0, P, A, P_TTV, φ) fit jointly via emcee MCMC. Results
written to a dedicated per-TOI sub-folder so the top-level results.json is
never touched.

Output per TOI:
  sinusoidal_ttv_joint/results.json
  sinusoidal_ttv_joint/corner_plot.png
  sinusoidal_ttv_joint/chain_plot.png
  sinusoidal_ttv_joint/oc_jointfit.png
  sinusoidal_ttv_joint/production_chains.npy
  sinusoidal_ttv_joint/production_log_prob.npy
  sinusoidal_ttv_joint/burnin_chains.npy
  sinusoidal_ttv_joint/burnin_log_prob.npy

Resume-friendly: skips TOIs where sinusoidal_ttv_joint/results.json already exists.
"""
import json
import sys
import time
import warnings
from pathlib import Path
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import emcee
from scipy.optimize import curve_fit

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parent
RESULTS = REPO / "autottv_results_v2"
SURVIVORS = REPO / "c2_loo_survivors.csv"

N_WALKERS = 64
N_BURNIN_MAX = 4000
N_BURNIN_MIN = 1000
N_PRODUCTION_MAX = 25000
N_PRODUCTION_MIN = 2000
RHAT_THRESH = 1.01

# Multiplier on the periodogram peak FWHM-based σ used to set the P_TTV
# prior half-width (P_TTV ∈ [P_init − mult·σ, P_init + mult·σ], clipped at
# 2·P_orb on the low side). The original default is 5.0; set to 1.0 to
# confine the chain tightly to the detected periodogram peak.
P_TTV_PRIOR_SIGMA_MULT = 5.0

# φ-prior range. "0_2pi" (default) bounds φ ∈ [0, 2π); "minus_pi_pi" uses
# φ ∈ [-π, π]. Mathematically equivalent (uniform over a 2π interval) but
# move the seam — useful when the chain's circular mean falls near 0 and
# walkers risk straddling the 0/2π edge.
PHI_PRIOR_RANGE = "0_2pi"
CHECK_INTERVAL = 500
PARAM_NAMES = ["T0", "P", "A_minutes", "P_TTV_days", "phi_radians"]
N_PARAMS = 5


def split_rhat(chain, n_split=2, wrap_aware=True):
    """Gelman-Rubin R-hat with split-chain comparison.

    If wrap_aware=True, the phi column (PARAM_NAMES index 4) is evaluated
    in a circular-statistic-safe way: phi samples are rotated so the
    circular mean lands at 0, the result is unwrapped to [-π, π], and
    standard split-Rhat is computed on that 1D angular distance. This
    avoids the false-bimodality artifact when phi samples straddle the
    0/2π seam OR when phi sits near 0 or π (where one of {cos, sin}
    is near zero and noise-driven sign flips inflate the linear Rhat).
    """
    n_walkers, n_steps, n_dim = chain.shape
    if n_steps < n_split * 2:
        return np.full(n_dim, np.inf)
    if wrap_aware and n_dim >= 5:
        phi = chain[:, :, 4]
        # Circular mean of the entire flat sample
        mean_c = float(np.arctan2(np.sin(phi).mean(), np.cos(phi).mean()))
        d = ((phi - mean_c + np.pi) % (2 * np.pi)) - np.pi  # ∈ [-π, π]
        chain_lin = chain.copy()
        chain_lin[:, :, 4] = d
        return _split_rhat_linear(chain_lin, n_split)
    return _split_rhat_linear(chain, n_split)


def _split_rhat_linear(chain, n_split=2):
    n_walkers, n_steps, n_dim = chain.shape
    half = n_steps // n_split
    splits = chain[:, -half * n_split:, :].reshape(n_walkers * n_split, half, n_dim)
    means = splits.mean(axis=1)
    vars_ = splits.var(axis=1, ddof=1)
    B = half * means.var(axis=0, ddof=1)
    W = vars_.mean(axis=0)
    var_hat = ((half - 1) / half) * W + B / half
    rhat = np.sqrt(np.where(W > 0, var_hat / W, np.inf))
    return rhat


def make_log_prob(epochs, t_obs, t_err,
                   T0_prior, T0_sigma, P_prior, P_sigma,
                   T0_lo, T0_hi, P_lo, P_hi,
                   A_lo, A_hi, P_TTV_lo, P_TTV_hi):
    inv_var = 1.0 / (t_err ** 2)
    epochs_arr = np.asarray(epochs, dtype=float)
    t_obs_arr = np.asarray(t_obs, dtype=float)
    two_pi = 2.0 * np.pi

    def log_prob(theta):
        T0, P, A, P_TTV, phi = theta
        # Hard bounds
        if not (T0_lo < T0 < T0_hi): return -np.inf
        if not (P_lo  < P  < P_hi):  return -np.inf
        if not (A_lo  < A  < A_hi):  return -np.inf
        if not (P_TTV_lo < P_TTV < P_TTV_hi): return -np.inf
        if PHI_PRIOR_RANGE == "minus_pi_pi":
            if not (-np.pi <= phi < np.pi): return -np.inf
        else:
            if not (0 <= phi < two_pi): return -np.inf
        # Gaussian priors on T0 and P
        log_prior = (
            -0.5 * ((T0 - T0_prior) / T0_sigma) ** 2
            -0.5 * ((P  - P_prior)  / P_sigma)  ** 2
        )
        # Likelihood
        model_days = (
            T0 + P * epochs_arr
            + (A / (24 * 60)) * np.sin(two_pi * epochs_arr * P / P_TTV + phi)
        )
        log_lik = -0.5 * np.sum(((t_obs_arr - model_days) ** 2) * inv_var)
        return float(log_prior + log_lik)

    return log_prob


def _load_canonical_times(toi_str):
    """Canonical strict-cascade transit times from tables/all_transit_times.csv
    (used==True rows) — the O-C basis the TTV classification rests on. Returns
    (epochs, t_obs_BJD, t_err_days) sorted by epoch, or None if TOI is absent."""
    f = REPO / "tables" / "all_transit_times.csv"
    if not f.exists():
        return None
    try:
        df = pd.read_csv(f, dtype={"TOI": str})
    except Exception:
        return None
    sub = df[df["TOI"].astype(str).str.strip() == toi_str]
    used = sub["used"].astype(str).str.strip().str.lower().isin(("true", "1", "1.0"))
    sub = sub[used]
    if len(sub) < 1:
        return None
    eps = sub["epoch"].to_numpy(dtype=float)
    t0  = sub["t0_fit_BJD"].to_numpy(dtype=float)
    er  = sub["t0_err_days"].to_numpy(dtype=float)
    si = np.argsort(eps)
    return eps[si], t0[si], er[si]


def make_initial_guess(toi_dir, data):
    """Build (initial_theta, prior_setup) from the TOI's stored values."""
    eph_lin = (data.get("ephemeris") or {}).get("linear") or {}
    toi_str = toi_dir.name.replace("TOI_", "").replace("_", ".")

    # Canonical (strict-cascade) transit times — the O-C basis the Periodic
    # classification was built on. Fall back to the top-level standard-pipeline
    # individual_transits only if the TOI isn't tabulated in all_transit_times.csv.
    canon = _load_canonical_times(toi_str)
    if canon is not None:
        epochs_arr, t_obs_arr, t_err_arr = canon
    else:
        tt = [t for t in (data.get("individual_transits") or {}).get("transit_times", [])
              if t.get("used", True)]
        epochs_arr = np.array([t["epoch"]   for t in tt], dtype=float)
        t_obs_arr  = np.array([t["t0_fit"]  for t in tt], dtype=float)
        t_err_arr  = np.array([t["t0_err"]  for t in tt], dtype=float)

    # Always compute a fresh weighted LSQ linear ephemeris on the actual
    # transit_times, so T0_lin / P_lin are guaranteed to be near the data
    # (the stored Stage-3 ephemeris.linear is sometimes None or stale).
    w = 1.0 / (t_err_arr ** 2)
    A_mat = np.array([
        [w.sum(),                    (w * epochs_arr).sum()],
        [(w * epochs_arr).sum(),     (w * epochs_arr ** 2).sum()],
    ])
    b_vec = np.array([(w * t_obs_arr).sum(), (w * epochs_arr * t_obs_arr).sum()])
    sol = np.linalg.solve(A_mat, b_vec)
    T0_lin, P_lin = float(sol[0]), float(sol[1])

    # Estimate T0/P uncertainties from the fit covariance
    cov = np.linalg.inv(A_mat)
    T0_lin_err = float(np.sqrt(cov[0, 0]))
    P_lin_err  = float(np.sqrt(cov[1, 1]))
    # Use the larger of stored vs computed (just in case)
    T0_lin_err = max(T0_lin_err, float(eph_lin.get("T0_err") or 0))
    P_lin_err  = max(P_lin_err,  float(eph_lin.get("P_err")  or 0))

    # Prefer the strict-cascade periodogram (TTV_period / TTV_period_err in
    # ttv_candidates_canonical_strict_full.csv) — these reflect the iter1
    # T_mids and are what the canonical Periodic-candidate status was
    # built on. Fall back to data["periodogram"] only if the TOI isn't in
    # the canonical CSV.
    peri = data.get("periodogram") or {}
    toi_str = toi_dir.name.replace("TOI_", "").replace("_", ".")
    pk = None; pk_err = None
    canon_csv = REPO / "ttv_candidates_canonical_strict_full.csv"
    if canon_csv.exists():
        try:
            import pandas as _pd
            _df = _pd.read_csv(canon_csv, dtype={"TOI": str})
            _row = _df[_df["TOI"].astype(str).str.strip() == toi_str]
            if len(_row):
                _v = _row.iloc[0].get("TTV_period")
                _e = _row.iloc[0].get("TTV_period_err")
                if _v is not None and not _pd.isna(_v) and _v > 0:
                    pk = float(_v)
                if _e is not None and not _pd.isna(_e) and _e > 0:
                    pk_err = float(_e)
        except Exception:
            pk = None
    if pk is None:
        pk = peri.get("peak_period")
    if pk is None:
        raise RuntimeError("no periodogram peak_period")
    P_TTV_init = float(pk)
    if pk_err is None:
        pk_err = peri.get("peak_period_error")
    P_TTV_err = float(pk_err) if pk_err is not None and pk_err > 0 else P_TTV_init * 0.1

    epochs = epochs_arr
    t_obs  = t_obs_arr
    t_err  = t_err_arr

    # Initial A, phi from a curve_fit to O-C against the freshly-fitted linear
    oc_min = (t_obs - (T0_lin + P_lin * epochs)) * 24 * 60
    err_min = t_err * 24 * 60
    phase = (epochs * P_lin / P_TTV_init) % 1.0

    def _sine(ph, A, phi, C):
        return A * np.sin(2 * np.pi * ph + phi) + C

    try:
        popt, _ = curve_fit(_sine, phase, oc_min, p0=[float(np.std(oc_min)), 0.0, 0.0],
                             sigma=err_min, absolute_sigma=True, maxfev=5000)
        A_init   = abs(float(popt[0]))
        phi_init = float(popt[1] % (2 * np.pi))
    except Exception:
        A_init   = float(np.std(oc_min))
        phi_init = 0.0

    A_init = max(A_init, 0.1)  # avoid zero start

    initial = np.array([T0_lin, P_lin, A_init, P_TTV_init, phi_init])
    priors = dict(
        T0_prior=T0_lin, T0_sigma=max(5 * T0_lin_err, 1e-3),
        P_prior=P_lin,   P_sigma=max(5 * P_lin_err,  1e-7),
        T0_lo=T0_lin - 0.5 * P_lin,  T0_hi=T0_lin + 0.5 * P_lin,
        P_lo=P_lin * 0.99,           P_hi=P_lin * 1.01,
        A_lo=0.0,                    A_hi=max(3 * float(np.max(np.abs(oc_min))), 10.0),
        P_TTV_lo=max(P_TTV_init - P_TTV_PRIOR_SIGMA_MULT * P_TTV_err, 2 * P_lin),
        P_TTV_hi=P_TTV_init + P_TTV_PRIOR_SIGMA_MULT * P_TTV_err,
    )
    return epochs, t_obs, t_err, initial, priors, T0_lin, P_lin, P_TTV_init


def fit_one_toi(toi_str):
    toi_dir = RESULTS / f"TOI_{toi_str.replace('.', '_')}"
    out_dir = toi_dir / "sinusoidal_ttv_joint"
    if (out_dir / "results.json").exists():
        return toi_str, "already done"
    rj = toi_dir / "results.json"
    if not rj.exists():
        return toi_str, "no results.json"
    data = json.loads(rj.read_text())
    if not data.get("individual_transits"):
        return toi_str, "no individual_transits"

    try:
        epochs, t_obs, t_err, initial, priors, T0_lin, P_lin, P_TTV_init = \
            make_initial_guess(toi_dir, data)
    except Exception as e:
        return toi_str, f"initial guess failed: {type(e).__name__}: {e}"

    if len(epochs) < 10:
        return toi_str, f"too few transits ({len(epochs)})"

    log_prob = make_log_prob(epochs, t_obs, t_err, **priors)

    # Walker initialization — small ball around initial
    init_scale = np.array([
        max(priors["T0_sigma"] * 0.5, 1e-4),
        max(priors["P_sigma"]  * 0.5, 1e-8),
        max(0.05 * initial[2], 0.1),
        max(0.01 * initial[3], 0.5),
        0.1,
    ])
    p0 = initial + init_scale * np.random.standard_normal((N_WALKERS, N_PARAMS))
    # Keep within bounds
    p0[:, 0] = np.clip(p0[:, 0], priors["T0_lo"]+1e-6, priors["T0_hi"]-1e-6)
    p0[:, 1] = np.clip(p0[:, 1], priors["P_lo"]+1e-9,  priors["P_hi"]-1e-9)
    p0[:, 2] = np.clip(p0[:, 2], priors["A_lo"]+1e-3,  priors["A_hi"]-1e-3)
    p0[:, 3] = np.clip(p0[:, 3], priors["P_TTV_lo"]+1e-3, priors["P_TTV_hi"]-1e-3)
    if priors["P_TTV_hi"] - priors["P_TTV_lo"] <= 2e-3:
        # A P_TTV prior no wider than the 1e-3 d clip margins (2026-10; none of the published fits)
        # would clip every walker to the same P_TTV, which the stretch move can never leave:
        # start the walkers uniformly within the prior instead.
        p0[:, 3] = np.random.uniform(priors["P_TTV_lo"], priors["P_TTV_hi"], N_WALKERS)
    if PHI_PRIOR_RANGE == "minus_pi_pi":
        p0[:, 4] = ((p0[:, 4] + np.pi) % (2 * np.pi)) - np.pi
    else:
        p0[:, 4] = p0[:, 4] % (2 * np.pi)

    sampler = emcee.EnsembleSampler(N_WALKERS, N_PARAMS, log_prob)

    # Burn-in with adaptive R-hat
    n_burn_done = 0
    while n_burn_done < N_BURNIN_MAX:
        chunk = min(CHECK_INTERVAL, N_BURNIN_MAX - n_burn_done)
        state = sampler.run_mcmc(p0 if n_burn_done == 0 else None, chunk, progress=False)
        n_burn_done += chunk
        if n_burn_done >= N_BURNIN_MIN:
            chain = sampler.get_chain()  # (n_steps, n_walkers, n_dim)
            chain = chain.transpose(1, 0, 2)  # (n_walkers, n_steps, n_dim)
            rhat = split_rhat(chain)
            if np.all(rhat <= RHAT_THRESH):
                break
    burnin_chains = sampler.get_chain()  # (n_steps, n_walkers, n_dim)
    burnin_log_prob = sampler.get_log_prob()

    # Bad-walker rejection (matches the standard-pipeline phase-fold pattern):
    # walkers whose median log_prob over the last 400 burn-in steps is more
    # than 10×MAD-sigma below the population median are re-initialized at
    # the position of a randomly-chosen good walker (with small jitter),
    # then a short second burn-in (2000 steps) is run before production.
    tail = burnin_log_prob[-min(400, burnin_log_prob.shape[0]):, :]
    walker_med = np.median(tail, axis=0)
    pop_med = float(np.median(walker_med))
    mad = float(np.median(np.abs(walker_med - pop_med)))
    sigma_robust = 1.4826 * mad if mad > 0 else float(np.std(walker_med))
    bad_mask = (walker_med < (pop_med - 10.0 * sigma_robust)) if sigma_robust > 0 else np.zeros(N_WALKERS, bool)
    n_bad_walkers_reinit = int(bad_mask.sum())
    if n_bad_walkers_reinit > 0:
        good_idx = np.where(~bad_mask)[0]
        new_state = state.coords.copy()
        for bi in np.where(bad_mask)[0]:
            gi = good_idx[np.random.randint(len(good_idx))]
            new_state[bi] = new_state[gi] + np.random.standard_normal(N_PARAMS) * (init_scale * 0.1)
        # Clip to bounds
        new_state[:, 0] = np.clip(new_state[:, 0], priors["T0_lo"]+1e-6, priors["T0_hi"]-1e-6)
        new_state[:, 1] = np.clip(new_state[:, 1], priors["P_lo"]+1e-9,  priors["P_hi"]-1e-9)
        new_state[:, 2] = np.clip(new_state[:, 2], priors["A_lo"]+1e-3,  priors["A_hi"]-1e-3)
        if priors["P_TTV_hi"] - priors["P_TTV_lo"] > 2e-3:
            new_state[:, 3] = np.clip(new_state[:, 3], priors["P_TTV_lo"]+1e-3, priors["P_TTV_hi"]-1e-3)
        else:  # narrow P_TTV prior, as above: keep the jittered walkers inside it without collapsing them
            eps = 1e-3 * (priors["P_TTV_hi"] - priors["P_TTV_lo"])
            new_state[:, 3] = np.clip(new_state[:, 3], priors["P_TTV_lo"]+eps, priors["P_TTV_hi"]-eps)
        if PHI_PRIOR_RANGE == "minus_pi_pi":
            new_state[:, 4] = ((new_state[:, 4] + np.pi) % (2 * np.pi)) - np.pi
        else:
            new_state[:, 4] = new_state[:, 4] % (2 * np.pi)
        state.coords = new_state
        sampler.reset()
        state = sampler.run_mcmc(new_state, 2000, progress=False)
    else:
        sampler.reset()

    # Production
    n_prod_done = 0
    while n_prod_done < N_PRODUCTION_MAX:
        chunk = min(CHECK_INTERVAL, N_PRODUCTION_MAX - n_prod_done)
        state = sampler.run_mcmc(state, chunk, progress=False)
        n_prod_done += chunk
        if n_prod_done >= N_PRODUCTION_MIN:
            chain = sampler.get_chain().transpose(1, 0, 2)
            rhat = split_rhat(chain)
            if np.all(rhat <= RHAT_THRESH):
                break
    production_chains = sampler.get_chain()
    production_log_prob = sampler.get_log_prob()
    flat = sampler.get_chain(flat=True)
    flat_lp = sampler.get_log_prob(flat=True)
    final_rhat = split_rhat(production_chains.transpose(1, 0, 2))
    acceptance = float(np.mean(sampler.acceptance_fraction))

    # Parameter summaries
    pcts = np.percentile(flat, [15.87, 50, 84.13], axis=0)
    params_out = {}
    for i, name in enumerate(PARAM_NAMES):
        med = float(pcts[1, i])
        params_out[name] = {
            "value":         med,
            "err_lower":     med - float(pcts[0, i]),
            "err_upper":     float(pcts[2, i]) - med,
            "percentile_16": float(pcts[0, i]),
            "percentile_84": float(pcts[2, i]),
        }

    # Best-fit parameters at posterior median
    T0_med, P_med, A_med, PT_med, phi_med = pcts[1]

    # chi² for the joint fit and for linear-only / quadratic-only
    model_days = (T0_med + P_med * epochs
                  + (A_med / (24*60)) * np.sin(2*np.pi*epochs*P_med/PT_med + phi_med))
    chi2_sin = float(np.sum(((t_obs - model_days) / t_err) ** 2))
    chi2_lin = float(np.sum(((t_obs - (T0_lin + P_lin * epochs)) / t_err) ** 2))

    quad = (data.get("ephemeris") or {}).get("quadratic") or {}
    if quad.get("T0") is not None and quad.get("P") is not None and quad.get("Q") is not None:
        T0_q, P_q, Q_q = float(quad["T0"]), float(quad["P"]), float(quad["Q"])
        chi2_quad = float(np.sum(((t_obs - (T0_q + P_q*epochs + Q_q*epochs**2)) / t_err) ** 2))
        bic_quad = chi2_quad + 3 * np.log(len(epochs))
    else:
        chi2_quad = None
        bic_quad = None

    n_obs = len(epochs)
    bic_lin = chi2_lin + 2 * np.log(n_obs)
    bic_sin = chi2_sin + 5 * np.log(n_obs)
    aic_sin = chi2_sin + 2 * 5

    # Write output sub-folder
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "production_chains.npy", production_chains)
    np.save(out_dir / "production_log_prob.npy", production_log_prob)
    np.save(out_dir / "burnin_chains.npy", burnin_chains)
    np.save(out_dir / "burnin_log_prob.npy", burnin_log_prob)

    out_json = {
        "toi": toi_str,
        "tic_id": data.get("tic_id"),
        "model": "T_obs = T0 + P*epoch + A*sin(2*pi*epoch*P/P_TTV + phi)",
        "fit_method": "emcee, 64 walkers",
        "n_transits": int(n_obs),
        "transit_times_used":
            "from individual_transits.transit_times where used=True (post-Stage-2 filter)",
        "priors": {
            "T0":    {"type": "Gaussian", "center": float(priors["T0_prior"]),
                       "sigma": float(priors["T0_sigma"]),
                       "hard_lo": float(priors["T0_lo"]), "hard_hi": float(priors["T0_hi"])},
            "P":     {"type": "Gaussian", "center": float(priors["P_prior"]),
                       "sigma": float(priors["P_sigma"]),
                       "hard_lo": float(priors["P_lo"]),  "hard_hi": float(priors["P_hi"])},
            "A_minutes":   {"type": "Uniform", "lo": float(priors["A_lo"]),     "hi": float(priors["A_hi"])},
            "P_TTV_days":  {"type": "Uniform", "lo": float(priors["P_TTV_lo"]), "hi": float(priors["P_TTV_hi"])},
            "phi_radians": ({"type": "Uniform", "lo": float(-np.pi), "hi": float(np.pi)}
                              if PHI_PRIOR_RANGE == "minus_pi_pi"
                              else {"type": "Uniform", "lo": 0.0, "hi": float(2*np.pi)}),
        },
        "parameters": params_out,
        "convergence": {
            "max_rhat":             float(np.max(final_rhat)),
            "rhat_per_param":       {n: float(r) for n, r in zip(PARAM_NAMES, final_rhat)},
            "n_steps_used":         int(n_prod_done),
            "burnin_steps_used":    int(n_burn_done),
            "acceptance_rate":      acceptance,
            "converged":            bool(np.all(final_rhat <= RHAT_THRESH)),
            "n_bad_walkers_reinit": int(n_bad_walkers_reinit),
        },
        "fit_quality": {
            "chi2":          chi2_sin,
            "dof":           int(n_obs - 5),
            "reduced_chi2":  chi2_sin / max(n_obs - 5, 1),
            "bic":           bic_sin,
            "aic":           aic_sin,
        },
        "comparison_with_linear": {
            "chi2_linear":   chi2_lin,
            "bic_linear":    bic_lin,
            "delta_chi2":    chi2_lin - chi2_sin,
            "delta_bic":     bic_lin  - bic_sin,
        },
        "comparison_with_quadratic": {
            "chi2_quadratic": chi2_quad,
            "bic_quadratic":  bic_quad,
            "delta_chi2":     (chi2_quad - chi2_sin) if chi2_quad is not None else None,
            "delta_bic":      (bic_quad  - bic_sin)  if bic_quad  is not None else None,
        },
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(out_json, f, indent=2, default=float)

    # ---- Plots ----
    # Corner plot
    try:
        import corner
        fig = corner.corner(flat, labels=PARAM_NAMES, quantiles=[0.16, 0.5, 0.84],
                            show_titles=True, title_kwargs={"fontsize": 9})
        fig.savefig(out_dir / "corner_plot.png", dpi=120, bbox_inches="tight")
        plt.close(fig)
    except Exception:
        pass

    # Chain plot
    fig, axes = plt.subplots(N_PARAMS, 1, figsize=(10, 2 * N_PARAMS), sharex=True)
    for i, name in enumerate(PARAM_NAMES):
        for w in range(N_WALKERS):
            axes[i].plot(production_chains[:, w, i], color="black", alpha=0.06, lw=0.4)
        axes[i].set_ylabel(name)
    axes[-1].set_xlabel("Step")
    fig.tight_layout()
    fig.savefig(out_dir / "chain_plot.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # O-C diagram with joint-fit sinusoidal model overlaid
    oc_lin_min = (t_obs - (T0_lin + P_lin * epochs)) * 24 * 60
    err_min = t_err * 24 * 60
    oc_jointfit_lin = (t_obs - (T0_med + P_med * epochs)) * 24 * 60  # OC against fit's linear part
    e_dense = np.linspace(epochs.min(), epochs.max(), 1000)
    sin_curve = (A_med) * np.sin(2*np.pi*e_dense*P_med/PT_med + phi_med)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7),
                                    gridspec_kw={"height_ratios": [2.0, 1.0]}, sharex=True)
    ax1.errorbar(epochs, oc_jointfit_lin, yerr=err_min, fmt="o", ms=5, color="navy",
                 ecolor="navy", capsize=3, label=f"O-C vs joint-fit linear")
    ax1.plot(e_dense, sin_curve, "r-", lw=2,
             label=(f"joint-fit sinusoid: A={A_med:.2f}±{(params_out['A_minutes']['err_upper']+params_out['A_minutes']['err_lower'])/2:.2f} min, "
                    f"P_TTV={PT_med:.2f}±{(params_out['P_TTV_days']['err_upper']+params_out['P_TTV_days']['err_lower'])/2:.2f} d, "
                    f"φ={phi_med:.2f} rad"))
    ax1.axhline(0, color="gray", ls="--", lw=0.6)
    ax1.set_ylabel("O-C (minutes)")
    ax1.set_title(f"TOI {toi_str} — joint sinusoidal-TTV fit "
                  f"(N={n_obs}, ΔBIC vs linear = {bic_lin - bic_sin:+.1f})")
    ax1.legend(fontsize=8, loc="best")

    # Residuals (data − full sinusoidal model)
    full_model = (T0_med + P_med * epochs
                   + (A_med / (24*60)) * np.sin(2*np.pi*epochs*P_med/PT_med + phi_med))
    resid_min = (t_obs - full_model) * 24 * 60
    rms = float(np.sqrt(np.mean(resid_min**2)))
    ax2.errorbar(epochs, resid_min, yerr=err_min, fmt="o", ms=4, color="firebrick",
                 ecolor="firebrick", capsize=3)
    ax2.axhline(0, color="gray", ls="--", lw=0.6)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Residual (min)")
    ax2.set_title(f"Residuals (data − full sinusoidal model), RMS = {rms:.2f} min")

    fig.tight_layout()
    fig.savefig(out_dir / "oc_jointfit.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    return toi_str, (
        f"OK — A={A_med:.2f}, P_TTV={PT_med:.2f}, "
        f"max_rhat={float(np.max(final_rhat)):.4f}, "
        f"ΔBIC_vs_linear={bic_lin-bic_sin:+.1f}"
    )


def main():
    n_workers = 10
    for arg in sys.argv[1:]:
        if arg.startswith("--cpus="):
            n_workers = int(arg.split("=")[1])
    n_workers = max(1, min(n_workers, cpu_count() - 1))

    surv = pd.read_csv(SURVIVORS, dtype={"TOI": str})
    surv["TOI"] = surv["TOI"].astype(str).str.strip()
    surv = surv[surv["LOO_survives"].astype(bool)]
    print(f"survivors in c2_loo_survivors.csv (LOO_survives=True): {len(surv)}")

    # Cross-check current FAP < 0.01 in results.json
    todo = []
    skipped_fap = 0
    for toi in surv["TOI"]:
        rj = RESULTS / f"TOI_{toi.replace('.','_')}" / "results.json"
        if not rj.exists():
            continue
        try:
            d = json.loads(rj.read_text())
        except Exception:
            continue
        fap = (d.get("periodogram") or {}).get("bootstrap_fap")
        if fap is None or fap >= 0.01:
            skipped_fap += 1
            continue
        # Skip if already done
        out = (RESULTS / f"TOI_{toi.replace('.','_')}" / "sinusoidal_ttv_joint" / "results.json")
        if out.exists():
            continue
        todo.append(toi)

    print(f"  excluded for current FAP >= 0.01: {skipped_fap}")
    print(f"  to fit now (resume-friendly):     {len(todo)}", flush=True)
    if not todo:
        print("nothing to do.")
        return

    print(f"running with {n_workers} workers ...", flush=True)
    t0 = time.time()
    n_done = 0
    n_failed = 0
    with Pool(processes=n_workers) as pool:
        for k, (toi, msg) in enumerate(pool.imap_unordered(fit_one_toi, todo), 1):
            elapsed = time.time() - t0
            if msg.startswith("OK"):
                n_done += 1
            else:
                n_failed += 1
            print(f"[{k}/{len(todo)}] TOI {toi:<10} {msg}  "
                  f"(elapsed {elapsed/60:.1f} min, ok {n_done}, fail {n_failed})",
                  flush=True)
    print(f"\nDone in {(time.time()-t0)/60:.1f} min")
    print(f"  fitted: {n_done}/{len(todo)}")
    print(f"  failed: {n_failed}/{len(todo)}")


if __name__ == "__main__":
    main()
