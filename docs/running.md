# Running the pipeline

## Command-line flags

`run_full_analysis.py` takes the TOI as its first positional argument and parses the rest
as `--flag` or `--flag=value`. There is no `--help`; this is the complete list.

| Flag | Effect |
|---|---|
| `--cpus=N` | Worker processes for the MCMC pool. Default is `cpu_count() - 1` on machines with ≤ 16 cores, otherwise 16. |
| `--sectors=1,2,28` | Restrict the analysis to these sectors. Default is every sector available up to `MAX_SECTOR`. |
| `--no-cache` | Ignore the cached light curves and re-download from MAST. |
| `--period=X` | Override the period prior centre (days). |
| `--t0=X` | Override the epoch prior centre (BJD_TDB). |
| `--period-err=X` | Override the period prior width. |
| `--t0-err=X` | Override the epoch prior width. |
| `--duration=X` | Override the transit duration (hours) used to size the fitting window. |
| `--dur-prior` | Add a Gaussian prior on $a/R_\star$ derived from the catalog duration. |
| `--fix-ld` | Hold $u_1, u_2$ at the theoretical Claret values instead of fitting them. |
| `--ld-width=X` | Override the limb-darkening prior width. |
| `--no-mask` | Do not mask transits of sibling planets in the same system. |

Unrecognised flags are ignored silently, and malformed values fall back to the default.
Nothing raises an error, so a mistyped flag runs with settings you did not intend:

| You typed | What happens |
|---|---|
| `--cpus=8` | correct — 8 workers |
| `--cpu=8` | no `s`, so the flag is not recognised; **default worker count** |
| `--cpus 8` | space instead of `=`; not recognised; **default worker count** |
| `--cpus=eight` | recognised but unparseable; **default worker count** |

The run prints `CPUs: <n> of <m> available` near the start — check that line against what
you asked for.

### When you need the overrides

`--period` and `--t0` matter when the known ephemeris is wrong or stale. A period error
propagates into a linear drift in the O−C diagram that can masquerade as a quadratic
ephemeris, so if a target shows a strong ΔBIC with a clean parabola, re-deriving the
period with `run_bls_highres.py` and re-running with the override is worth doing before
believing it.

`--dur-prior` helps when $a/R_\star$ is poorly constrained by the photometry alone —
typically a grazing or low-SNR target where the sampler wanders toward the bounds.

`--no-mask` should be rare. Sibling masking exists because an unmasked transit of another
planet in the same system contaminates the out-of-transit baseline.

## Batch processing

For analyzing a group of TOIs the driver runs a range of lines from the catalog 
file included in this repo (toi_catalog_240226_for_ttv.csv; that file can be replaced):

```bash
./run_batch_lines.sh 2 100      # lines 2-100 
./run_batch_lines.sh 5 5        # just line 5
```

Line 1 is the header, so data starts at line 2, and a range is inclusive at both ends.

### The catalog file

Two files are involved, and they have different jobs:

| File | Role |
|---|---|
| `toi_catalog_240226.csv` | the **full** ExoFOP-TESS export. `run_full_analysis.py` looks up every TOI's period, epoch, duration, depth and stellar parameters here, whatever else you do. |
| `toi_catalog_240226_for_ttv.csv` | the same export **filtered** by the cuts in [Configuration](configuration.md#catalog-filtering). Used only as the work list that `run_batch_lines.sh` steps through by line number. |

Both keep ExoFOP's column layout verbatim — **62 columns**, one row per TOI planet entry,
so a multi-planet system appears once per candidate.

The batch driver does not parse the CSV. It reads the line by number and takes column 5
with `cut -d',' -f5`:

```bash
TOI=$(sed -n "${LINE_NUM}p" "$CSV_FILE" | cut -d',' -f5 | tr -d ' "')
```

Everything else the pipeline needs it looks up itself, from the full catalog, by TOI.

The columns that matter:

| # | Column | Used for |
|---|---|---|
| 1 | `TIC ID` | target identifier for the MAST query |
| **5** | **`TOI`** | **what the batch driver reads, and the pipeline's argument** |
| 18 | `TESS Mag` | the CDPP proxy for QLP-only targets |
| 22 | `TFOPWG Disposition` | PC / CP / KP / APC — the disposition filter |
| 31–32 | `Epoch (BJD)` and its error | $T_0$ prior centre and width |
| 33–34 | `Period (days)` and its error | period prior centre and width |
| 35 | `Duration (hours)` | sizes the per-transit fitting window |
| 39 | `Depth (ppm)` | initial $R_p/R_\star$ estimate |
| 48 | `Stellar Eff Temp (K)` | Claret limb-darkening interpolation |
| 50 | `Stellar log(g) (cm/s^2)` | Claret limb-darkening interpolation |
| 52 | `Stellar Radius (R_Sun)` | with the stellar mass, gives the initial $a/R_\star$ |

A representative row:

```
TIC ID                231663901
TOI                   101.01
TFOPWG Disposition    KP
Period (days)         1.43036995
Epoch (BJD)           2458326.009
Duration (hours)      1.616599404
Depth (ppm)           18960.71229
TESS Mag              12.4069
Stellar Radius        0.890774012
Stellar Eff Temp      5600
Stellar log(g)        4.48851
```

Missing stellar parameters fall back to solar values, which is why $T_{\rm eff}$ and
$\log g$ gaps degrade the limb-darkening prior rather than failing the run.

!!! tip "Running your own list of TOIs"
    The simplest way is to skip the batch driver and loop yourself. Put one TOI per line
    in a plain text file — call it whatever you like:

    ```
    105.01
    125.01
    216.01
    ```

    then:

    ```bash
    while read toi; do python run_full_analysis.py "$toi" --cpus=8; done < my_tois.txt
    ```

    This works because `run_full_analysis.py` looks its target up in the **full** catalog
    (`toi_catalog_240226.csv`, set as `CATALOG_FILE` in `config.py`) — it never reads the
    filtered file. The filtered catalog exists only to give `run_batch_lines.sh` something
    to iterate over by line number.

    If you would rather keep using the driver, make a **copy of the filtered catalog** with
    its header row and only the rows you want, point `CSV_FILE` in `run_batch_lines.sh` at
    it, and give a line range. Keep the ExoFOP columns intact — the driver extracts the TOI
    with `cut -f5`, so a reduced or reordered file will feed it the wrong field.

!!! warning "Edit the CPU count first"
    `run_batch_lines.sh` hardcodes a worker count sized for a large machine. Reduce it
    before running on a laptop, or the machine will run out of memory and swap.

Batches are naturally restartable: results are written per TOI, and the light-curve cache
survives, so re-running a range that partly completed skips the download cost.

## Resource planning

| | Typical | Worst case |
|---|---|---|
| Wall time per TOI | minutes | hours for a many-sector, high-cadence target |
| Memory per worker | ~1 GB | ~2 GB |
| Disk per TOI | a few MB | tens of MB, dominated by the chain `.npy` files |
| Light-curve cache | — | grows to tens of GB across a full catalog |

Wall time is dominated by Step 1, and Step 1 scales with the number of cadences, not the
number of transits. A 30-minute-cadence QLP target with 20 transits is fast; a 2-minute
SPOC target observed in 15 sectors is not.

Step 2 is embarrassingly parallel across transits and usually cheaper than Step 1 despite
running one MCMC per event, because each fit has only three free parameters and sees a
window of a few transit durations.

## Secondary fitters

These run **after** the main pipeline and reanalyse an existing
`autottv_results_v2/TOI_<X>/results.json`, writing into named subdirectories. They are
what produce the TTV-aware parameters in the published tables.

| Script | What it does |
|---|---|
| `refined_transit_params_for_ttv.py` | The iterative TTV-aware refit. Re-stacks transits at their individually fitted mid-times, then refits the shape. This is the one that matters — see [Physical background](physics.md#timing-smearing). |
| `run_joint_fixld_full.py` | Joint MCMC of shape and every mid-time simultaneously, limb darkening fixed. |
| `run_joint_freeld_full.py` | The same joint fit with limb darkening free. |
| `fit_with_pdot.py`, `fit_with_pdot_iterative.py` | Phase fold with $dP/dE$ as an additional free parameter. |
| `fit_with_sinusoidal_ttv.py` | Refit the shape with a fixed sinusoidal TTV model whose $A$, $\phi$, $P_{\rm TTV}$ come from a periodogram sine fit. |
| `fit_with_gp.py`, `fit_with_gp_iter.py` | Two-stage Gaussian-process plus transit fit, for hosts with strong stellar variability. |
| `run_bls_highres.py` | High-resolution BLS, for verifying or overriding a catalog period. |

### The iteration, concretely

`refined_transit_params_for_ttv.py` runs a cycle:

1. Start from the Step 1 shape and the Step 2 mid-times.
2. Shift each transit by its measured O−C so the stack aligns on the *observed* times
   rather than a strict linear ephemeris.
3. Refit the shape on the realigned stack.
4. Refit the individual mid-times against the new shape.
5. Repeat until the shape stops moving, up to five iterations.

Output lands in `refined_strict_iter1/`, `refined_strict_iter2/`, and so on. **The highest
available iteration is the adopted fit** — that is the source priority used when building
the published tables, falling back to iteration 0 and then to the standard fit.

## Building the summary tables

```bash
python build_all_transit_times.py     # tables/transit_times.csv, all_transit_times.csv
python build_fit_params_tables.py     # the fit_params_* tables
```

Both ingest every per-TOI `results.json` and can be re-run at any time. For TTV
candidates they pull from the highest available refined iteration; for everything else
they use the standard pipeline output. See [Output files](outputs.md).

## Monitoring long runs

`mem_watchdog.py` samples memory use during a batch and writes it to a log. This is what
lets you identify which target exhausted memory when a long run is killed hours after it
started, since the run itself leaves no record of it.

Batch logs are written per line range, and are the first place to look when a run stops
early. Two failure modes to watch for:

- **A TOI that never converges** and uses all 25,000 production steps. It still finishes,
  just slowly, and its output carries `converged: false`.
- **A MAST download error** partway through. Because light curves are cached, re-running
  that range does not re-download what already succeeded.
