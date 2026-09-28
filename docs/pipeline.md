# What the code does

Five stages run for every TOI. Steps 1–4 are the published pipeline; the iterative refit
runs only for TTV candidates. Every threshold named here lives in
[`autottv_pipeline_v2/config.py`](configuration.md).

---

## Step 0 — Data retrieval and preparation

**Module:** `data_loader.py`

### Choosing the photometry

All light curves through `MAX_SECTOR` (96) are queried from MAST via `lightkurve`. Where a
sector has SPOC 2-minute PDCSAP photometry, that is used. Otherwise the QLP full-frame-image
light curve is taken, at whatever cadence that sector provides:

| Cadence | Source | Priority |
|---|---|---|
| 120 s | SPOC 2-minute PDCSAP | 1 |
| 200 s | QLP (Sectors 56+) | 2 |
| 600 s | QLP (Sectors 27–55) | 3 |
| 1800 s | QLP (Sectors 1–26) | 4 |

The choice is per sector, so a single target routinely mixes 2-minute and FFI photometry
across its baseline. This matters for the transit model: cadences longer than
`LONG_CADENCE_THRESHOLD` (200 s) are supersampled inside `batman` to account for the
smearing of the transit shape by a long exposure.

### Cleaning

- Non-finite cadences and any point with a non-zero quality flag are dropped.
- Each sector is normalized independently by its out-of-transit median, with both the
  transit and the secondary-eclipse phase windows masked (`TRANSIT_MASK_PHASE_WIDTH` and
  `ECLIPSE_MASK_PHASE_WIDTH`, each 0.15 in phase).
- Transits of **sibling planets** in the same system are masked, so a neighbour's transit
  cannot contaminate the baseline. Disable with `--no-mask`.

### Identifying transits

A transit counts as *full* when data extend to at least `TRANSIT_COVERAGE_FACTOR` (1.5)
transit durations on either side of the predicted mid-time. Partial transits are excluded
from timing: without out-of-transit baseline on both sides, the mid-time is degenerate
with the local slope.

**Output:** a normalized, quality-masked light curve, a per-sector normalization record,
and a list of full transit epochs.

---

## Step 1 — Phase-folded transit fit

**Module:** `FullAnalysisFitter` in `run_full_analysis.py`. The package's `phase_fold_fitter.py`
is a separate implementation that samples $b$ with a flat prior; only
`autottv_pipeline_v2/main.py` uses it.

Every full transit is stacked on a strict linear ephemeris and one transit model is fitted
to the composite by MCMC.

### Free parameters

Eight, or six with `--fix-ld`:

$$P, \quad T_0, \quad k = R_p/R_\star, \quad a/R_\star, \quad b^2, \quad \text{baseline}, \quad u_1, \quad u_2$$

Sampling in $b^2$ rather than $b$ is deliberate: the transit observables depend on $b^2$
near-linearly, which makes the posterior better behaved. The cost is in the prior: uniform
in $b^2$ means $p(b) \propto b$, which gives high impact parameters more weight than
isotropic orbits do (they give a uniform prior on $b$) — see
[Physical background](physics.md#why-b2).

### Priors

| Parameter | Prior | Hard bounds |
|---|---|---|
| $P$ | Gaussian, $\sigma = 2\times$ catalog error | $\pm 2.5\%$ of catalog value |
| $T_0$ | Gaussian, $\sigma = 2\times$ catalog error | $\pm 2.5\%$ of a period |
| $k$ | Gaussian, 50% width around the catalog-derived value | $[0.001,\ 0.5]$ |
| $a/R_\star$ | Gaussian, 50% width | $[1,\ 500]$ |
| $b^2$ | uniform | $[0,\ (1+k)^2]$ |
| baseline | uniform | $[0.9,\ 1.1]$ |
| $u_1$ | Gaussian, $\sigma = 0.15$, centred on Claret | $[0, 1]$ |
| $u_2$ | Gaussian, $\sigma = 0.10$, centred on Claret | $[-0.5, 1]$ |

Limb-darkening coefficients are interpolated from the Claret (2017) TESS-band tables at
the catalog $T_{\rm eff}$ and $\log g$, assuming solar metallicity. Missing stellar
parameters fall back to solar values. The coefficients additionally obey the physical
constraints $u_1 \geq 0$, $u_1 + u_2 \leq 1$, and $u_1 + 2u_2 \geq 0$.

### Sampling

64 walkers, up to 4,000 burn-in steps and 25,000 production steps, with convergence
checked every 500. Two implementation details worth knowing:

- **Bounds are checked before the model is evaluated.** Proposals outside the hard bounds
  get zero prior probability and are rejected without ever calling `batman`, which is where
  nearly all the compute goes.
- **Stuck walkers are rescued.** At the end of burn-in, any walker whose log-probability
  sits more than five robust standard deviations below the ensemble median is
  re-initialized near the median.

Reported values are posterior medians with uncertainties from the central 68% interval.
See [Statistical methods](statistics.md#convergence-diagnostics) for what the convergence
criteria actually test.

**Output:** the transit shape, its posterior, and convergence diagnostics.

---

## Step 2 — Individual transit times

**Module:** `individual_transit_fitter.py`

The shape from Step 1 is now **frozen**. For each full transit, a separate MCMC fits three
parameters over a window of a few transit durations:

$$T_{\rm mid}, \quad \text{local baseline}, \quad \text{local linear slope}$$

The baseline and slope absorb residual trends within the event window, leaving $T_{\rm mid}$
to carry the timing displacement. 16 walkers, up to 10,000 steps, $\hat{R} \leq 1.01$.

Freezing the shape is what makes the timing well posed: fitting depth and duration
per transit would let a shallow, wide solution trade against a shifted mid-time.

### Filtering

Two cuts, both at 10σ, remove transits that would otherwise dominate the ephemeris fit:

- $\sigma_{T_0}$ greater than 10× the median timing error for that target
- an O−C residual more than 10σ from the median

**Output:** one mid-transit time and uncertainty per epoch, plus the O−C series.

---

## Step 3 — Linear versus quadratic ephemeris

**Module:** `ephemeris_analysis.py`

Two models are fitted to the measured mid-times by weighted least squares:

$$T(E) = T_0 + PE \qquad\text{versus}\qquad T(E) = T_0 + PE + QE^2$$

with $dP/dE = 2Q$. Model preference is decided by

$$\Delta{\rm BIC} = {\rm BIC}_{\rm linear} - {\rm BIC}_{\rm quadratic} > 6$$

A curvature term does not by itself mean tidal decay — usually it means the opposite. See
[Physical background](physics.md#what-a-quadratic-ephemeris-does-not-mean).

**Output:** both ephemerides, $\Delta$BIC, $dP/dE$, $dP/dt$, and the O−C residuals against
the linear fit.

---

## Step 4 — Timing-residual periodogram

**Module:** `plot_periodogram` in `run_full_analysis.py`. The package's `periodogram.py`,
used only by `autottv_pipeline_v2/main.py`, differs: its upper frequency comes from the
median spacing of the measured transits.

An error-weighted Lomb–Scargle periodogram of the O−C residuals, on a linear grid of 1,000
frequencies running from $2/T_{\rm baseline}$ to $0.5/P$.

- The **lower limit** guarantees at least two full cycles of any searched period fit inside
  the observing baseline.
- The **upper limit** is the Nyquist frequency of the O−C series: residuals are sampled at
  intervals of the orbital period, so anything faster carries no independent information.

The false-alarm probability is empirical, not analytic: 100,000 label permutations, in each
of which the (O−C, uncertainty) pairs are shuffled among the fixed epochs and the peak
power recorded. The FAP is the fraction of permuted periodograms whose peak meets or
exceeds the observed one, with a reporting floor of $10^{-5}$. Requires at least 5 transits.

**Output:** the periodogram, the peak period, and the bootstrap FAP.

---

## The iterative TTV-aware refit

**Module:** `run_refined_iterations` in `refined_transit_params_for_ttv.py`, run with
`python refined_transit_params_for_ttv.py <TOI>` — runs only for TTV candidates

Step 1 stacked the transits on a strict linear ephemeris. If the system has TTVs, that
stack is *smeared*, and the recovered radius ratio is biased. The refit removes the smear:

1. Shift each transit by its measured O−C, aligning the stack on observed times.
2. Refit the transit shape on the realigned stack.
3. Refit the individual mid-times against the new shape.
4. Repeat on the re-timed transits, up to five more times, until $k$, $a/R_\star$ and $b$
   each move by less than $1\sigma$ from the previous iteration.

The first refit is written to `refined_transit/` and later iterations to
`refined_strict_iter1/`, `refined_strict_iter2/`, and so on. **The last iteration is the
adopted result**; `iter_cascade_summary.json` records it, with the shape change at each
step. `--max-iters=N` changes the cap of five.

The magnitude of the effect is the central finding of the AutoTTV paper: about 20% of TTV
candidates show a radius-ratio change exceeding three times the combined uncertainty, in
both directions. [Physical background](physics.md#timing-smearing) explains the mechanism.

---

## What runs when

| | Every TOI | TTV candidates only |
|---|---|---|
| Step 0 retrieval | ✓ | |
| Step 1 phase fold | ✓ | |
| Step 2 transit times | ✓ | |
| Step 3 ephemerides | ✓ | |
| Step 4 periodogram | ✓ (≥ 5 transits) | |
| Iterative refit | | ✓ |

