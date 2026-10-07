# AutoTTV

**Automated MCMC-based transit-timing analysis of TESS planet candidates.**

A pipeline that downloads TESS light curves for each TOI, fits the global transit shape
and every individual transit time with MCMC, then searches for linear/quadratic
ephemerides and Lomb–Scargle TTV signals. A candidate-detection step flags TTV systems,
and a refined fitter re-stacks transits at their individually-fitted times to de-smear
systems that a linear-ephemeris stack would bias.

- **Catalog scope:** 3,775 TOIs in the filtered catalog (from the 7,890-TOI ExoFOP TOI catalog), of which 3,650 remain after the 125 removals listed in `rejected_TOIs_list.csv`
- **Version 1.0.0** is the code used for Shporer & Drori, *AutoTTV: Homogeneous Transit Timing of 3,650 TESS TOIs and the Radius Bias of Strict-Period Phase Folding* (submitted to ApJS). [Reproducing the paper](#reproducing-the-paper) lists which parts of that analysis this repository contains.

---

## Quick Start

### Clone

```bash
git clone git@github.com:shporer/autottv.git
cd autottv
```

### Install

```bash
pip install -r requirements.txt
```

Dependencies: `lightkurve`, `emcee`, `batman-package`, `astropy`, `numpy`, `scipy`,
`pandas`, `matplotlib`, `corner` (optional). There is no lockfile; `requirements.txt`
is the source of truth.

### Run the full analysis on one TOI

```bash
python run_full_analysis.py 109.01                # default
python run_full_analysis.py 109.01 --cpus=15      # override worker count
python run_full_analysis.py 109.01 --fix-ld       # hold u1/u2 at theoretical values
python run_full_analysis.py 109.01 --period=3.5 --t0=2458325.5   # override ephemeris
python run_full_analysis.py 109.01 --qlp-time-fix # correct the known QLP timestamp errors first
python run_full_analysis.py 109.01 --results-root=test_runs      # write to test_runs/TOI_109_01/
```

Outputs land in `autottv_results_v2/TOI_<X>/` and include `results.json`, MCMC chains,
and a stack of diagnostic plots. `--step1-only` stops after the phase-folded fit (Step 1),
keeping its chains and writing `step1_results.json`.

### Batch processing

```bash
./run_batch_lines.sh 2 100        # processes lines 2-100 of toi_catalog_240226_for_ttv.csv
```

(`run_batch_lines.sh` hardcodes `--cpus=64`; edit it down for smaller machines.)

---

## The 4-step analysis pipeline

For every TOI, `run_full_analysis.py` executes:

| Step | What | Module |
|------|------|--------|
| **1. Phase-fold MCMC** | Fit the global transit shape (P, T₀, Rp/Rs, a/Rs, b, u1, u2, baseline) by stacking all transits on a linear ephemeris. | `FullAnalysisFitter` in `run_full_analysis.py` |
| **2. Individual-transit MCMC** | Fix the shape from Step 1; fit each transit's T_mid, baseline, slope. | `individual_transit_fitter.py` |
| **3. Ephemeris analysis** | Compare linear vs quadratic ephemerides (ΔBIC ≥ 6 favours quadratic), compute O–C residuals. | `ephemeris_analysis.py` |
| **4. TTV periodogram** | Lomb–Scargle of O–C on ten frequencies per 1/T_baseline (at least 200) from 2/T_baseline to 0.5/(P ΔE), ΔE the median spacing of the observed epochs, with an empirical FAP from 10⁵ permutations of the (O–C, error) pairs among the epochs (requires ≥ 5 transits). | `plot_periodogram` in `run_full_analysis.py` |

The package's `phase_fold_fitter.py` and `periodogram.py` are separate implementations, the
first with a flat prior on b, the second with a grid of at least 100 rather than 200
frequencies. Only `autottv_pipeline_v2/main.py` uses them.

### Priors (Step 1, free LD)

- Period: σ = 2 × catalog error
- T₀: σ = 2 × catalog error
- Rp/Rs, a/Rs: 50 % Gaussian width around catalog-derived value (sampled uniformly within hard bounds)
- b² ∈ [0, (1 + Rp/Rs)²], uniform in b², so p(b) ∝ b: more weight at high b than isotropic orbits, which give a uniform prior on b
- u1: σ = 0.15, centered on theoretical Claret value
- u2: σ = 0.10, centered on theoretical Claret value

---

## QLP timestamp errors

The QLP light curves of Sectors 14 and 15 carry barycentric-correction errors of up to
10.6 min. Those of Sector 80, and of the second orbit of Sector 85 for one star, carry a
smaller error of the kind the QLP team documented for Sectors 74–79 (paper, Section 3.1).
`qlp_time_fix.py` computes both errors from the Earth's barycentric position.
`qlp_time_errors.csv` lists the affected sectors (and the orbit and star, where the error
is limited to them), and `qlp_time_spans.csv` the QLP data span of each affected star and
sector.

- `run_full_analysis.py <TOI> --qlp-time-fix` corrects the affected QLP timestamps in
  memory before fitting (the light-curve cache is not changed) and records what it applied
  under `qlp_time_fix` in `results.json`.
- For transit times already measured from uncorrected data,
  `qlp_time_fix.QLPTimeFix().correction(tic, toi, t_mid)` returns the shift in seconds to
  add, and a tag naming the error and sector. The error changes by less than 1 s over a
  transit, so this is equivalent to refitting a corrected light curve.

---

## Pipeline implementation (`autottv_pipeline_v2/`)

- Sampler: `emcee` Affine-Invariant Ensemble Sampler with `vectorize=True` and a `multiprocessing.Pool`
- Transit model: `batman` (C, cadence-aware supersampling)
- Convergence: split R-hat ≤ 1.01, ESS > 1000, chain > 50 × τ
- Post-MCMC: Chauvenet outlier rejection + refit
- All configuration (bounds, priors, thresholds, paths) lives in `autottv_pipeline_v2/config.py`

---

## Post-pipeline analysis

These re-use the per-TOI `autottv_results_v2/TOI_<X>/results.json` produced by the main pipeline:

| Script | Purpose |
|--------|---------|
| `find_ttv_candidates.py` | Apply the detection criteria — quadratic ΔBIC (C1), periodogram FAP (C2), O–C-RMS / median-error ratio (C3) — to flag TTV candidates. Each criterion must survive a per-criterion leave-one-out test, and the type follows from the surviving criteria. |
| `fit_joint_sinusoidal_ttv.py` | Fit a linear ephemeris plus a sinusoid to the transit times of the TOIs listed in `c2_loo_survivors.csv` (emcee). |
| `refined_transit_params_for_ttv.py` | Re-stack transits at their individually-fitted T_mids and re-fit the shape (de-smears TTV systems that the linear-ephemeris stack biases). Iterates, up to 5 times, until the shape moves by less than 1σ. |

---

## Reproducing the paper

This repository holds the per-TOI pipeline and the fitters that produced the paper's
measurements. The steps that combined the per-TOI results into the paper's tables, and the
validation experiments, were run with scripts from the authors' working repository that are
not part of this release. They were done as follows (section numbers refer to the paper):

1. **Sample (Section 2).** The 3,775 TOIs of `toi_catalog_240226_for_ttv.csv` (made by
   `filter_toi_catalog.py`), minus the 125 listed in `rejected_TOIs_list.csv`.
2. **Per-TOI analysis (Section 3).** `run_full_analysis.py` on each of the 3,650 TOIs. The
   228 TOIs whose period or reference time the QLP timestamp errors would otherwise change
   by more than 0.3σ were run with `--qlp-time-fix`; for the others, the correction of
   `QLPTimeFix.correction` was applied to the measured transit times.
3. **Detection (Section 5.1).** The three criteria, each with its leave-one-out test, as in
   `find_ttv_candidates.py`, evaluated on the Step-2 transit times. The periodogram FAPs
   come from 10⁵ permutations (10⁴ for each leave-one-out subset) on the frequency grid of
   Section 3.4: ten frequencies per 1/T_baseline, and at least 200, from 2/T_baseline to
   0.5/(P ΔE), with ΔE the median spacing of the observed epochs. The Step-4 periodogram of
   `run_full_analysis.py` and the leave-one-out re-checks of `find_ttv_candidates.py` use
   this grid. (The per-TOI runs made for the paper used 1,000 frequencies from 2/T_baseline
   to 0.5/P in Step 4, so the FAPs in their `results.json` differ from the paper's, which
   were computed separately on this grid.)
4. **TTV-corrected refit (Sections 5.1 and 6.1).** `refined_transit_params_for_ttv.py` for
   each TOI that passed a criterion. Its re-timing centers each transit's fit on the
   linear-ephemeris prediction, which for large TTVs can lock onto another feature of the
   light curve. For the paper's final times the transits were therefore re-measured with
   the same per-transit fitter, with each fit's data window and starting point centered on
   the transit itself (located by sliding the transit shape over the light curve), and the
   shape was refit on exactly those times, iterating until it moved by less than 1σ. A TOI
   was kept as a candidate only if it also passed on these re-measured times.
5. **Sinusoidal fits (Table 4).** `fit_one_toi` of `fit_joint_sinusoidal_ttv.py` for each
   of the 45 Periodic candidates, on its Table 2 transit times, with fits that did not
   reach R-hat ≤ 1.01 repeated with longer chains and more walkers. Run as a script, it
   fits the TOIs marked `LOO_survives` in `c2_loo_survivors.csv`, a list that predates the
   paper's final candidate list.
6. **Transit-time catalog (Table 2).** The Step-2 times, except for the candidates, whose
   times are the final re-measured times of item 4, and for the 29 TOIs that passed a
   criterion on their Step-2 times but are not candidates, whose times are those
   re-measured with a TTV-corrected template, on which they were evaluated.
7. **Not included:** the comparisons with published transit times (Sections 4.2 and 4.3),
   the control sample (Section 6.2) and the injection–recovery experiments (Section 7).

---

## Catalog preparation

The two catalogs needed to run are included (`toi_catalog_240226.csv` = full,
`toi_catalog_240226_for_ttv.csv` = filtered). To regenerate the filtered catalog from the
full one (counts transits, computes per-transit SNR):

```bash
python filter_toi_catalog.py
```

(`filter_toi_catalog.py` uses `compute_transit_snr.py`.)

---

## Project structure

```
autottv/
├── run_full_analysis.py              # main 4-step pipeline
├── run_batch_lines.sh                # batch driver (lines from filtered catalog)
├── find_ttv_candidates.py            # TTV detection criteria (C1/C2/C3)
├── refined_transit_params_for_ttv.py # iterative refined shift-and-stack re-fit
├── filter_toi_catalog.py             # catalog filtering / transit counting
├── compute_transit_snr.py            # per-transit SNR (used by filter_toi_catalog)
├── fit_joint_sinusoidal_ttv.py       # joint sinusoidal fit of the transit times
├── qlp_time_fix.py                   # QLP timestamp-error model and correction
├── qlp_time_errors.csv               # affected QLP sectors (read by qlp_time_fix.py)
├── qlp_time_spans.csv                # QLP data spans of the affected stars
│
├── autottv_pipeline_v2/              # the pipeline package (emcee + batman)
│   ├── config.py                     # all configuration constants
│   ├── data_loader.py                # TESS download + cache (lightkurve/MAST)
│   ├── main.py                       # modular entry point, not used for the paper
│   ├── phase_fold_fitter.py          # Step 1 in main.py only (flat b prior)
│   ├── individual_transit_fitter.py  # Step 2
│   ├── ephemeris_analysis.py         # Step 3
│   ├── periodogram.py                # Step 4 in main.py only
│   ├── joint_transit_fitter.py       # joint shape + T_mid fitter
│   ├── convergence.py                # R-hat / ESS / autocorrelation diagnostics
│   ├── limb_darkening.py             # Claret 2017 LD interpolation
│   ├── plotting.py                   # chains, corner, phase-fold, O–C, periodogram
│   └── utils.py                      # batman model, chi², BIC helpers
│
├── tests/                            # pytest suite
├── toi_catalog_240226.csv            # full TOI catalog
├── toi_catalog_240226_for_ttv.csv    # filtered catalog (3,775 TOIs)
├── rejected_TOIs_list.csv            # the 125 TOIs removed before the candidate search
├── c2_loo_survivors.csv              # input list for fit_joint_sinusoidal_ttv.py
├── requirements.txt
└── Dockerfile
```

Heavy outputs (`autottv_results_v2/`, root `*.png`, batch logs) are git-ignored, except
`autottv_results_v2/spoc_cdpp_sampled.npy`, the SPOC CDPP sample that `compute_transit_snr.py` reads.

---

## Citation

If you use this code, please cite the paper that describes it, and the archived version of
the code you used (each release is archived on Zenodo, with its own DOI):

```bibtex
@article{ShporerDrori2026,
  title={AutoTTV: Homogeneous Transit Timing of 3,650 TESS TOIs and the Radius Bias
         of Strict-Period Phase Folding},
  author={Shporer, Avi and Drori, Iddo},
  journal={The Astrophysical Journal Supplement Series},
  year={2026},
  note={submitted}
}
```

The pipeline was first presented at the 247th AAS meeting:

```bibtex
@inproceedings{autottv2026scaling,
  title={Scaling-up autonomous scientific discovery:
         Lessons from AI studying WASP-4b TTV in TESS data},
  author={Shporer, Avi and Drori, Iddo},
  booktitle={247th meeting of the American Astronomical Society (AAS)},
  year={2026},
  month={January}
}
```

Please also cite the underlying mission and tooling:
- Ricker et al. (2015) — TESS mission
- Lightkurve Collaboration (2018) — `lightkurve`
- Foreman-Mackey (2013) — `emcee`
- Kreidberg (2015) — `batman`

---

## Data sources

- **TOI Catalog:** [ExoFOP-TESS](https://exofop.ipac.caltech.edu/tess/)
- **Light curves:** [MAST Archive](https://mast.stsci.edu/) (SPOC PDCSAP 2-min, QLP 200 s / 600 s / 1800 s)
- **TESS mission:** [NASA TESS](https://tess.mit.edu/)

---

## License

MIT — see `LICENSE`.

---

## Acknowledgments

NASA TESS Mission · MAST Archive at STScI · ExoFOP-TESS · Lightkurve · emcee · batman
