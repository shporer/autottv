# AutoTTV

**Automated MCMC transit-timing analysis of TESS planet candidates.**

AutoTTV takes a TESS Object of Interest, downloads every light curve MAST holds for it,
fits the transit shape and each individual mid-transit time by MCMC, and searches the
timing residuals for transit timing variations. It is designed to run unattended on a 
group of TOIs, or the entire TOI catalog, producing homogeneous results.

The pipeline was built for [Shporer & Drori (2026)](#citation), which applied it to
**3,650 TOIs** across TESS Sectors 1–96 and produced **111,995 individual transit times**
and **168 TTV candidates**.

**Source code:** [github.com/shporer/autottv](https://github.com/shporer/autottv) — MIT licensed.

---

## What it produces

For each TOI, one directory containing a `results.json`, the MCMC chains, and a set of
diagnostic figures. Across a catalog run, four summary tables: transit times, fitted
transit parameters, and the TTV candidate lists split by detection class.

| Product | Where |
|---|---|
| Per-TOI fit, chains, figures | `autottv_results_v2/TOI_<X>/` |
| Transit-time catalog | `tables/transit_times.csv` |
| Fitted transit parameters | `tables/fit_params_*.csv` |
| TTV candidate lists | `ttv_candidates_canonical_strict_full.csv` |

## The shape of the analysis

```
TOI number
    │
    ├─ Step 0   Retrieve light curves from MAST (SPOC 2-min, else QLP FFI)
    │
    ├─ Step 1   Phase-fold MCMC  →  transit shape (P, T0, Rp/Rs, a/Rs, b, u1, u2)
    │
    ├─ Step 2   Per-transit MCMC →  one mid-transit time per event, shape held fixed
    │
    ├─ Step 3   Ephemeris fits   →  linear vs quadratic, O−C residuals
    │
    └─ Step 4   Periodogram      →  Lomb–Scargle of O−C, bootstrap FAP
                                        │
                                        └─ TTV candidate?  →  iterative TTV-aware refit
```

Steps 1–4 run for every TOI. The iterative refit runs only for TTV candidates, and it is
what removes the radius-ratio bias described in [Physical background](physics.md).

## Where to start

<div class="grid cards" markdown>

- **New here?** → [Installation](installation.md), then [Quick start](quickstart.md) for a
  single TOI end to end.
- **Running a catalog?** → [Running the pipeline](running.md) for batch mode, resource
  planning, and the secondary fitters.
- **Interpreting results?** → [Output files](outputs.md) for every column of every table.
- **Why does it work this way?** → [Physical background](physics.md) and
  [Statistical methods](statistics.md).

</div>

## Requirements at a glance

Python 3.11+, roughly 2 GB of RAM per worker process, and network access to MAST. A
single well-sampled TOI takes minutes to tens of minutes on a laptop; a full catalog run
is a multi-day job on a many-core machine. See [Running the pipeline](running.md#resource-planning).

## Citation

If you use this code or its data products, please cite the AutoTTV paper (Shporer &
Drori 2026) and the underlying tooling: TESS (Ricker et al. 2015), `lightkurve`
(Lightkurve Collaboration 2018), `emcee` (Foreman-Mackey et al. 2013), `batman`
(Kreidberg 2015), and `astropy` (Astropy Collaboration 2013, 2018).

The TESS photometry itself carries MAST dataset DOIs:
[10.17909/t9-nmc8-f686](https://doi.org/10.17909/t9-nmc8-f686) for the SPOC 2-minute light
curves and [10.17909/t9-r086-e880](https://doi.org/10.17909/t9-r086-e880) for the QLP
full-frame-image light curves.

## License

MIT. See `LICENSE` in the repository.
