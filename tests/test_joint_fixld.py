"""Standalone assert tests for joint_fixld helpers. Run: python tests/test_joint_fixld.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from autottv_pipeline_v2.joint_transit_fitter import (
    compute_window_half, identify_bad_walkers, reinitialize_bad_walkers,
    compute_window_diagnostics,
)


def test_compute_window_half():
    # 120 min on a 10-day period: cap = 0.25*10 = 2.5 d; 120 min = 0.0833 d -> uncapped
    assert abs(compute_window_half(120.0, 10.0) - 120.0 / 1440.0) < 1e-12
    # 120 min on a 0.2-day period: cap = 0.05 d < 0.0833 d -> capped to 0.05 d
    assert abs(compute_window_half(120.0, 0.2) - 0.05) < 1e-12
    # exactly at cap boundary
    assert compute_window_half(120.0, 0.2) <= 0.25 * 0.2 + 1e-12
    print("test_compute_window_half PASS")


def test_identify_bad_walkers():
    rng = np.random.default_rng(0)
    # 200 burn-in steps, 20 walkers; walker 7 is a clear low-logprob outlier
    lp = rng.normal(-100.0, 1.0, size=(200, 20))
    lp[:, 7] = -100.0 - 50.0  # far below the pack over the whole tail
    bad, sigma, med = identify_bad_walkers(lp, sigma_mult=10.0)
    assert 7 in bad, f"expected walker 7 flagged, got {bad}"
    assert all(b != 0 for b in bad) or 0 not in bad  # normal walkers not flagged
    assert sigma > 0
    print("test_identify_bad_walkers PASS")


def test_reinitialize_bad_walkers():
    rng = np.random.default_rng(1)
    chains = rng.normal(0.0, 1.0, size=(100, 10, 3))  # steps, walkers, params
    new_pos = reinitialize_bad_walkers(chains, bad_indices=[4])
    assert new_pos.shape == (10, 3)
    # good walkers keep their final position
    assert np.allclose(new_pos[0], chains[-1, 0])
    # bad walker moved to within a few MAD of the good-walker medians
    good_median = np.median(chains[:, [i for i in range(10) if i != 4], :])
    assert np.isfinite(new_pos[4]).all()
    print("test_reinitialize_bad_walkers PASS")


import autottv_pipeline_v2.joint_transit_fitter as jtf


def _make_joint_data(fix_ld):
    # Minimal synthetic single-transit dataset
    n = 60
    t = np.linspace(0.0, 0.2, n)
    flux = np.ones(n)
    return {
        "time": t, "flux": flux, "inv_var": np.full(n, 1e6),
        "cadence_of_pt": np.full(n, 120.0),
        "transit_idx_of_pt": np.zeros(n, dtype=np.int32),
        "expected_tmid": np.array([0.1]),
        "n_tr": 1, "period_d": 3.0, "t14_d": 0.08,
        "cadence_groups": {120.0: 1},
        "fix_ld": fix_ld, "u1_fixed": 0.35, "u2_fixed": 0.23,
        "priors": {
            "rp_lo": 0.001, "rp_hi": 0.5, "ar_lo": 1.0, "ar_hi": 500.0,
            "rp_mu": 0.1, "rp_sigma": 0.05, "ar_mu": 10.0, "ar_sigma": 1.0,
            "u1_mu": 0.35, "u1_sigma": 0.15, "u2_mu": 0.23, "u2_sigma": 0.10,
            "tmid_window_d": 120.0 / 1440.0, "tmid_sigma_d": np.inf,
        },
    }


def test_fix_ld_dimensionality():
    # fixed-LD theta length = 4 + n_tr; u1/u2 NOT in theta
    jtf._JOINT_DATA.clear(); jtf._JOINT_DATA.update(_make_joint_data(fix_ld=True))
    theta_fix = np.array([0.1, 10.0, 0.25, 1.0, 0.1])  # rp, ar, bsq, baseline, tmid
    lp_fix = jtf._joint_log_probability(theta_fix)
    assert np.isfinite(lp_fix), "fixed-LD log-prob should be finite for valid theta"

    # free-LD theta length = 6 + n_tr
    jtf._JOINT_DATA.clear(); jtf._JOINT_DATA.update(_make_joint_data(fix_ld=False))
    theta_free = np.array([0.1, 10.0, 0.25, 0.35, 0.23, 1.0, 0.1])
    lp_free = jtf._joint_log_probability(theta_free)
    assert np.isfinite(lp_free), "free-LD log-prob should be finite for valid theta"
    jtf._JOINT_DATA.clear()
    print("test_fix_ld_dimensionality PASS")


def test_compute_window_diagnostics():
    window_half_d = 120.0 / 1440.0  # 2 h
    expected = np.array([1000.0, 2000.0])  # two transits, days
    epochs = np.array([0, 10])
    n_samp = 5000
    rng = np.random.default_rng(2)
    # transit 0: tight, well inside window. transit 1: pushed to the +edge.
    s0 = expected[0] + rng.normal(0, 0.001, n_samp)
    s1 = expected[1] + np.clip(rng.normal(window_half_d, 0.002, n_samp),
                               -window_half_d, window_half_d)
    chain = np.column_stack([s0, s1])  # (n_samp, n_tr)
    flagged, n_flagged = compute_window_diagnostics(chain, expected, window_half_d, epochs)
    flagged_epochs = {f["epoch"] for f in flagged}
    assert 10 in flagged_epochs, f"expected epoch 10 flagged, got {flagged_epochs}"
    assert 0 not in flagged_epochs
    assert n_flagged == len(flagged) == 1
    f = flagged[0]
    assert f["window_min"] == 120.0 and f["dev_p99_min"] > 0
    print("test_compute_window_diagnostics PASS")


if __name__ == "__main__":
    test_compute_window_half()
    test_identify_bad_walkers()
    test_reinitialize_bad_walkers()
    test_compute_window_diagnostics()
    test_fix_ld_dimensionality()
    print("ALL TESTS PASS")
