# AutoTTV

**Automated MCMC-based transit-timing analysis of TESS planet candidates.**

A pipeline that downloads TESS light curves for each TOI, fits the global transit shape
and every individual transit time with MCMC, then searches for linear/quadratic
ephemerides and Lomb–Scargle TTV signals. A candidate-detection step flags TTV systems,
and a refined fitter re-stacks transits at their individually-fitted times to de-smear
systems that a linear-ephemeris stack would bias.

- **Catalog scope:** 3,776 TOIs (filtered for TTV analysis from the 7,890-line ExoFOP TOI catalog)

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
```

Outputs land in `autottv_results_v2/TOI_<X>/` and include `results.json`, MCMC chains,
and a stack of diagnostic plots.

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
| **1. Phase-fold MCMC** | Fit the global transit shape (P, T₀, Rp/Rs, a/Rs, b, u1, u2, baseline) by stacking all transits on a linear ephemeris. | `phase_fold_fitter.py` |
| **2. Individual-transit MCMC** | Fix the shape from Step 1; fit each transit's T_mid, baseline, slope. | `individual_transit_fitter.py` |
| **3. Ephemeris analysis** | Compare linear vs quadratic ephemerides (ΔBIC ≥ 6 favours quadratic), compute O–C residuals. | `ephemeris_analysis.py` |
| **4. TTV periodogram** | Lomb–Scargle of O–C with bootstrap FAP (requires ≥ 5 transits). | `periodogram.py` |

### Priors (Step 1, free LD)

- Period: σ = 2 × catalog error
- T₀: σ = 2 × catalog error
- Rp/Rs, a/Rs: 50 % Gaussian width around catalog-derived value (sampled uniformly within hard bounds)
- b² ∈ [0, (1 + Rp/Rs)²] — uniform in b² ⇒ geometric prior on b
- u1: σ = 0.15, centered on theoretical Claret value
- u2: σ = 0.10, centered on theoretical Claret value

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
| `find_ttv_candidates.py` | Apply the detection criteria — quadratic ΔBIC (C1), periodogram FAP (C2), O–C-RMS / median-error ratio (C3) — to flag TTV candidates. |
| `refined_transit_params_for_ttv.py` | Re-stack transits at their individually-fitted T_mids and re-fit the shape (de-smears TTV systems that the linear-ephemeris stack biases). |

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
├── refined_transit_params_for_ttv.py # refined shift-and-stack re-fit
├── filter_toi_catalog.py             # catalog filtering / transit counting
├── compute_transit_snr.py            # per-transit SNR (used by filter_toi_catalog)
│
├── autottv_pipeline_v2/              # the pipeline package (emcee + batman)
│   ├── config.py                     # all configuration constants
│   ├── data_loader.py                # TESS download + cache (lightkurve/MAST)
│   ├── phase_fold_fitter.py          # Step 1
│   ├── individual_transit_fitter.py  # Step 2
│   ├── ephemeris_analysis.py         # Step 3
│   ├── periodogram.py                # Step 4
│   ├── joint_transit_fitter.py       # joint shape + T_mid fitter
│   ├── convergence.py                # R-hat / ESS / autocorrelation diagnostics
│   ├── limb_darkening.py             # Claret 2017 LD interpolation
│   ├── plotting.py                   # chains, corner, phase-fold, O–C, periodogram
│   └── utils.py                      # batman model, chi², BIC helpers
│
├── tests/                            # pytest suite
├── toi_catalog_240226.csv            # full TOI catalog
├── toi_catalog_240226_for_ttv.csv    # filtered catalog (3,776 TOIs)
├── requirements.txt
└── Dockerfile
```

Heavy outputs (`autottv_results_v2/`, root `*.png`, batch logs) are git-ignored.

---

## Citation

If you use this code, please cite:

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

A Nature Astronomy manuscript is in preparation.

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
