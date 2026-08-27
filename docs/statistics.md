# Statistical methods

Every number the pipeline reports comes out of one of the methods below. Thresholds are
in [`config.py`](configuration.md).

---

## MCMC sampling

Posterior sampling uses `emcee`'s affine-invariant ensemble sampler. An ensemble of walkers
proposes moves by stretching along the vector connecting one walker to another, which makes
the sampler insensitive to linear correlations between parameters — useful here, because
the transit posterior is strongly correlated in $(b, k, a/R_\star)$.

| | Phase fold (Step 1) | Individual transits (Step 2) |
|---|---|---|
| Walkers | 64 | 16 |
| Free parameters | 8, or 6 with `--fix-ld` | 3 |
| Burn-in | up to 4,000 | 500 |
| Production | up to 25,000 | up to 10,000 |
| Convergence checked every | 500 steps | 200 steps |

Reported values are posterior medians; uncertainties are the central 68% interval
(15.87th to 84.13th percentiles), so they are asymmetric and the pipeline reports both
sides separately.

### Bad-walker rescue

Ensemble samplers can strand a walker in a low-probability region where the stretch move
rarely rescues it, quietly inflating the autocorrelation time. At the end of burn-in, any
walker whose median log-probability falls more than five robust standard deviations below
the ensemble median is re-initialized near the median. Robust here means
$1.48 \times {\rm MAD}$, not the standard deviation, so a few stranded walkers cannot
widen the criterion that is meant to catch them.

---

## Convergence diagnostics {#convergence-diagnostics}

Three criteria, evaluated cheapest-first so an early failure skips the expensive checks.

### 1. Gelman–Rubin $\hat{R}$

Splits the chain and compares between-chain to within-chain variance:

$$\hat{R} = \sqrt{\frac{\frac{N-1}{N}W + \frac{1}{N}B}{W}}$$

$\hat{R} \to 1$ as the chains forget their initialization. The threshold is
$\hat{R} \leq 1.01$ for every parameter. It is the cheapest diagnostic — means and
variances, no FFT — so it runs first.

### 2. Effective sample size

Correlated draws carry less information than independent ones:

$$N_{\rm eff} = \frac{N}{\tau}, \qquad \tau = 1 + 2\sum_{k=1}^{\infty}\rho_k$$

where $\rho_k$ is the lag-$k$ autocorrelation and $\tau$ the integrated autocorrelation
time. $N_{\rm eff}$ is the number of independent draws giving the same Monte Carlo variance
on the posterior mean.

The sum cannot be evaluated as written — empirical $\hat{\rho}_k$ is noise at large lag and
the sum does not converge. The pipeline truncates using **Geyer's initial positive
sequence**: for a reversible chain the paired sums $\Gamma_m = \rho_{2m} + \rho_{2m+1}$ are
provably positive, so the first non-positive estimated pair marks where signal ends and
noise begins. The estimator is deliberately conservative — biased high in $\tau$, low in
$N_{\rm eff}$. Threshold: $N_{\rm eff} > 1{,}000$.

### 3. Chain length in autocorrelation times

$$N_{\rm steps} > 50\,\tau$$

with $\tau$ from `emcee`'s automated windowing (Sokal), which truncates at the first
$M \geq c\,\tau(M)$ with $c = 5$. This also carries a reliability check on $\tau$ itself: a
chain too short for $\tau$ to be estimable fails outright, where Geyer's rule would still
return a number.

!!! note "Which criterion actually binds"
    Requiring 50 autocorrelation times per walker across 64 walkers implies at least 3,200
    pooled effective samples — more than triple the ESS threshold. In practice the third
    criterion is the binding one and the second acts as an inexpensive pre-filter.

!!! warning "A caveat on absolute-variance floors"
    Both diagnostics carry hardcoded floors (`var < 1e-10` in the ESS, `W + 1e-10` in
    $\hat{R}$) intended to guard against a constant parameter. Because variance is
    dimensional, a tightly constrained period — days, with a precision near $10^{-6}$ d,
    hence a variance near $10^{-12}$ — falls below those floors, and both tests pass
    vacuously for it. The reported summaries use `max_rhat` and `min_ess`, so the degenerate
    values are discarded by the max/min and the gates still operate on the other
    parameters. Worth knowing before you read a per-parameter $\hat{R}$ of 0.18 as meaningful.

### Retention

TOIs that do not converge within 25,000 steps are retained if $\hat{R} < 1.1$. That is
deliberately permissive: a fit can be usable for timing while its shape parameters are
still wandering. Check the fractional uncertainty on $k$ before trusting a non-converged
shape.

---

## Outlier rejection

Photometric outliers are screened with **Chauvenet's criterion**. A point is flagged when
its error-normalized residual from the median-parameter model exceeds

$$\Phi^{-1}\!\left[1 - \frac{1}{4N}\right]$$

where $\Phi^{-1}$ is the inverse normal CDF and $N$ the number of points. Applied two-sided,
this makes the expected number of false flags **0.5 per fit, regardless of sample size** —
the threshold grows with $N$ exactly fast enough to keep it constant.

The pipeline implements an iterative rejection–refit cycle: when points are flagged and the
flagged fraction is below 10%, they are removed and the entire fit — burn-in, production,
convergence checking — is repeated on the cleaned light curve, up to five rounds. A flagged
fraction above 10% is taken to indicate a model or data-quality problem rather than isolated
outliers, and rejection is skipped.

Timing outliers are handled separately in Step 2, with 10σ cuts on both the timing
uncertainty and the O−C residual.

---

## Model selection: BIC

Linear versus quadratic ephemeris is decided by the Bayesian Information Criterion:

$$\text{BIC} = \chi^2 + k \ln n$$

for $k$ parameters and $n$ data points. The pipeline requires

$$\Delta{\rm BIC} = {\rm BIC}_{\rm linear} - {\rm BIC}_{\rm quadratic} > 6$$

conventionally "strong" evidence. BIC rather than AIC because the goal is identifying which
model is true, not best predictive performance, and BIC's $\ln n$ penalty is the harsher of
the two.

!!! caution "The large-sample limit"
    BIC's penalty is derived asymptotically. It is applied uniformly here for consistency,
    but a target with only 6–7 measured transit times is far from that limit and its ΔBIC
    should be read with more scepticism than one with fifty.

---

## Periodogram and bootstrap FAP

An error-weighted Lomb–Scargle periodogram of the O−C residuals, on 1,000 frequencies from
$2/T_{\rm baseline}$ to $0.5/P$. The bounds are physical, not arbitrary: the lower one
guarantees two full cycles inside the baseline, the upper is the Nyquist frequency of a
series sampled at the orbital period.

The false-alarm probability is **empirical**. Analytic FAP formulae assume white Gaussian
noise on a regular grid; O−C series are gappy and often carry red noise, so the analytic
value is optimistic. Instead:

1. Randomly permute the (O−C, uncertainty) pairs among the fixed epochs.
2. Recompute the weighted periodogram and record its peak power.
3. Repeat 100,000 times.
4. FAP is the fraction of permuted periodograms whose peak meets or exceeds the observed one.

Permuting preserves the epoch sampling and the distribution of residual values while
destroying any temporal coherence, which is exactly the null hypothesis of interest. The
reporting floor is $10^{-5}$ — with $10^5$ permutations, zero exceedances cannot be
distinguished from a very small probability.

---

## The three detection criteria {#the-three-detection-criteria}

| | Criterion | Threshold | Config key |
|---|---|---|---|
| **Periodic** | bootstrap FAP | $< 0.01$ | `TTV_FAP_THRESHOLD` |
| **Quadratic** | $\Delta$BIC | $> 6$ | `TTV_DELTA_BIC_THRESHOLD` |
| **Scatter** | O−C rms / median error | $> 2$ | `TTV_OC_RMS_OVER_ERR_THRESHOLD` |

A TOI tripping any criterion becomes a candidate, subject to validation. Classification is
by precedence: Periodic wins where it fires, then Quadratic, then Scatter. Many candidates
trip more than one.

!!! note
    The `C1`/`C2` column labels in `ttv_candidates_canonical_strict_full.csv` are inverted
    relative to the paper's numbering — in the CSV, `C1` tracks ΔBIC and `C2` tracks FAP.
    Correlate against the underlying statistic rather than trusting the column name.

---

## Leave-one-out validation {#leave-one-out-validation}

A criterion firing is not sufficient. Each flagged TOI is re-tested with each transit
dropped in turn: if removing any **single** transit destroys the detection, the signal was
driven by one event and is rejected.

For the Periodic criterion, each drop gets a full bootstrap FAP recomputation, and the
candidate survives only if every drop still yields FAP $< 0.01$.

This works asymmetrically across the three criteria, and it is worth understanding why:

- **Periodic** — well controlled. A noise-driven periodogram peak usually rests on one
  outlying point; drop it and the peak collapses.
- **Quadratic and Scatter** — poorly controlled. A ΔBIC curvature or an inflated O−C
  scatter arising from red noise is a *collective* property of the whole series. No single
  transit drives it, so a drop-one test cannot remove it.

So the Quadratic and Scatter classes carry a higher residual false-positive rate than the
Periodic class, and this is a real limitation of the vetting rather than an implementation
detail.
