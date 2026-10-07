# Physical background

Why the pipeline is built the way it is, and how to read what it produces.

---

## Why transits are not strictly periodic

A single planet orbiting a single star transits on a strict linear ephemeris:

$$T(E) = T_0 + PE$$

Add a second planet and that stops being true. The two exchange angular momentum, the
orbital elements oscillate, and the observed mid-times wander around the linear
prediction. Those deviations are **transit timing variations**, and the residuals

$$(O-C)_E = T_{\rm observed}(E) - T_{\rm linear}(E)$$

encode the perturber's mass and orbit. This is the appeal: TTVs deliver planetary masses
with no radial velocities at all, and they are sensitive to planets that never transit.

### The regimes you will see

**Near-resonance (super-period).** When two planets sit near a mean-motion resonance, the
TTV signal is a sinusoid at a "super-period" much longer than either orbital period. This
is the classic Kepler TTV signal, and it is what the Step 4 periodogram searches for. The
amplitude scales with the perturber's mass and with proximity to exact resonance.

**Chopping.** Successive conjunctions produce a short-period modulation on top of the
super-period. Detecting it requires dense sampling, which TESS's ~27-day sectors rarely give.

**Long-period companions.** A distant companion produces a slow drift or curvature rather
than a resolved cycle. Over a short baseline it looks like a quadratic ephemeris. Some
signal is light-travel time as the star orbits the system barycentre.

**Apsidal precession.** A slow rotation of the orbit's line of apsides shifts transit
times secularly. It looks quadratic over a baseline much shorter than the precession period.

The pipeline does not attempt to distinguish these physically. It classifies by the
*shape* of the timing signal — periodic, quadratic, or excess scatter — and leaves the
dynamical interpretation to follow-up.

---

## Transit shape and its degeneracies

A transit light curve constrains four geometric quantities: depth, duration, ingress
duration, and the shape of the curved bottom. From these the fit recovers

$$k = R_p/R_\star, \qquad a/R_\star, \qquad b = \frac{a\cos i}{R_\star}$$

plus the limb-darkening coefficients.

### The $b$–$k$ degeneracy {#the-b-k-degeneracy}

Depth alone does not give the radius ratio. A transit's depth is roughly $k^2$ only for a
central, uniformly bright star. As the impact parameter rises, the chord crosses the
limb-darkened, fainter part of the stellar disc, so a **larger** planet is needed to
produce the same depth. Duration falls at the same time, since the chord is shorter.

The result is a curved degeneracy: a family of $(b, k)$ pairs fits nearly equally well,
broken only by the ingress duration and the curvature of the transit floor. When the
photometry is noisy or the cadence is long, that information is weak and the degeneracy
stays open.

This degeneracy is why the radius-ratio bias below acts in **both directions** rather than
systematically inflating or deflating radii.

### Why sample in $b^2$? {#why-b2}

The transit observables depend on $b^2$ nearly linearly, so a posterior that is awkward in
$b$ is well behaved in $b^2$. A uniform prior on $b^2$ is not the geometric prior, though.
For randomly oriented orbits $\cos i$ is uniform, so the probability of an impact parameter
below $b$ scales as $b$: the isotropic prior is uniform in $b$. Uniform in $b^2$ instead
gives $p(b) \propto b$, which puts more prior weight on high impact parameters, and so on
the grazing solutions described below.

The upper bound is $b^2 \leq (1+k)^2$, i.e. $b \leq 1+k$ — the geometric limit at which the
planet's disc still grazes the star's.

### Grazing solutions and the prior boundary {#grazing-solutions-and-the-prior-boundary}

When the transit is shallow, V-shaped, or smeared, the sampler can walk up the $b$–$k$
degeneracy until it hits the $b^2$ boundary at $(1+k)^2$. There it returns a large $k$, a
$b$ pressed against the bound, and a suppressed $a/R_\star$.

These are **prior-boundary artefacts, not measurements**. The diagnostic signature:

- $b$ within a percent or two of $1+k$
- a fractional uncertainty on $k$ of tens of percent — the posterior spans the prior
- $a/R_\star$ near its lower clip
- often $\hat{R} > 1.01$, because the chain never localizes

A fit like this can produce a radius ratio nearly an order of magnitude too large. Treat
any result with $b \gtrsim 1$ and a large $\sigma_k/k$ as failed until you have looked at
the corner plot.

---

## Timing smearing {#timing-smearing}

This is the effect AutoTTV was built to measure.

### The mechanism

Standard pipelines measure transit shape from a **phase fold**: stack every transit on a
strict linear ephemeris and fit one model to the composite. That is correct when the
planet is strictly periodic. When it has TTVs, individual transits sit at scattered
offsets from the linear prediction, so the stack blurs them together.

The blurred composite is **wider and shallower** than any individual transit, with softened
ingress and egress — the same signature as a longer, more grazing transit. The fit responds
by moving along the $b$–$k$ degeneracy, and the recovered radius ratio is biased.

The relevant quantity is the timing scatter relative to the transit duration,
$\sigma_{O-C}/T_{\rm dur}$. A few minutes of TTV on a multi-hour transit is negligible; the
same scatter on a short-duration transit is not.

### The fix

Align the stack on the *observed* mid-times instead of the predicted ones, then refit. That
is the [iterative TTV-aware refit](pipeline.md#the-iterative-ttv-aware-refit): shift each
transit by its measured O−C, refit the shape, refit the times, repeat.

Iteration is needed because the two are coupled — the mid-times were measured against the
old shape, so a materially different shape changes them. In practice the shape stops moving
within two or three passes.

### How large is it?

Across the 168 TTV candidates in the published analysis, about **20%** show a radius-ratio
difference between the strict-period and TTV-aware fits exceeding three times the combined
uncertainty. In a matched control sample of non-TTV confirmed planets put through the same
two-step procedure, the rate is far lower — which is what establishes that the effect is
TTV-driven rather than an artefact of refitting.

The bias acts **in both directions**. Where the standard fit walked up the degeneracy to a
grazing solution it returns a radius that is too large; where the smeared composite stayed
box-like at reduced depth it returns one that is too small.

!!! important "Who this affects"
    Any automated catalog that folds on a strict period inherits this bias. It falls
    hardest on compact multi-planet systems near resonance — exactly the systems whose
    masses come from TTVs. Combining a TTV-derived mass with a phase-folded radius means
    the density uses two quantities derived inconsistently from the same photometry.

---

## What a quadratic ephemeris does not mean {#what-a-quadratic-ephemeris-does-not-mean}

A quadratic term $Q$ implies a changing period, $dP/dE = 2Q$. It is tempting to read that
as tidal orbital decay. Usually it is not.

Two checks:

**Sign.** Tidal decay can only shorten a period. A positive $dP/dt$ rules it out.

**Magnitude.** The only compelling measured tidal decay rate is WASP-12 b, at roughly
$-29$ ms yr$^{-1}$. Rates orders of magnitude larger are not decay.

What curvature usually means instead:

- **A segment of a longer TTV cycle.** Any sinusoid looks quadratic over less than a full
  period. With a baseline shorter than the super-period, a resonant TTV is indistinguishable
  from a parabola.
- **A long-period companion**, via dynamical perturbation or light-travel time.
- **Apsidal precession.**
- **An error in the catalog period**, which produces a linear O−C drift that a quadratic
  fit will absorb. Worth ruling out, for example with a high-resolution BLS search,
  before believing a strong ΔBIC.

The published analysis found *no* quadratic candidate with a period derivative consistent
with tidal decay: over half were positive, and every negative value exceeded the WASP-12 b
rate by at least a factor of a few.

---

## Reading the three detection classes

| Class | Timing signature | Most likely cause | Main contaminant |
|---|---|---|---|
| **Periodic** | coherent sinusoid in O−C | near-resonant perturber | a noise-driven periodogram peak |
| **Quadratic** | curvature in O−C | segment of a long cycle, wide companion, precession | an error in the catalog period |
| **Scatter** | excess O−C scatter, no coherent structure | unresolved or aperiodic TTV | stellar activity, spot crossings, systematics |

The Scatter class is the most susceptible to non-dynamical contamination — excess scatter
is what red noise and spot crossings produce too. The Periodic class is best protected,
because a coherent peak is hard to fake and the [leave-one-out
validation](statistics.md#leave-one-out-validation) removes peaks driven by a single
transit.
