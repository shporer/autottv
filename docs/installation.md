# Installation

## Requirements

- **Python 3.11 or newer.** The Docker image pins 3.11.
- **Network access to MAST.** Light curves are downloaded on first use and cached
  locally, so a run is only network-bound the first time it sees a target.
- **Memory.** Budget roughly 2 GB per worker process. The phase-fold MCMC holds the full
  light curve and the walker ensemble in memory simultaneously; a target with 100,000+
  cadences across many sectors is the demanding case.

## Install from source

```bash
git clone https://github.com/shporer/autottv.git
cd autottv
pip install -r requirements.txt
```

That installs everything the pipeline needs. Nothing else is required to run the analysis
end to end.

### What gets installed

| Package | Used for | Required? |
|---|---|---|
| `lightkurve` | downloading and reading TESS light curves from MAST | yes |
| `batman-package` | the analytic transit model, with cadence-aware supersampling | yes |
| `emcee` | the affine-invariant ensemble sampler | yes |
| `astropy` | FITS I/O and the Lomb–Scargle periodogram | yes |
| `numpy`, `scipy` | numerics; `scipy` also supplies the Claret table interpolation and the ephemeris least-squares fits | yes |
| `pandas` | catalog handling and table construction | yes |
| `matplotlib` | all diagnostic figures | yes |
| `corner` | posterior corner plots | yes |
| `astroquery` | TIC stellar-mass lookup, used to initialise $a/R_\star$ | yes |
| `openpyxl` | the `.xlsx` code paths in a few helper scripts | no |

!!! note "Use a virtual environment"
    `requirements.txt` pins no versions, so an install resolves to whatever is current on
    the day you run it. Dropping that into a shared interpreter will eventually collide
    with another project. And on Python 3.11+ a Homebrew- or distro-managed interpreter
    will refuse the install outright with `externally-managed-environment` (PEP 668), so a
    venv is usually required rather than merely advisable.

    ```bash
    python3 -m venv venv && source venv/bin/activate
    pip install -r requirements.txt
    ```

## Docker

The repository ships a multi-stage `Dockerfile` (builder + slim runtime) if you want a
fixed environment:

```bash
docker build -t autottv .
docker run -it --rm -v "$PWD:/app" autottv
```

The image drops you at a shell in `/app`. Mounting the working tree keeps results and the
light-curve cache on the host, which matters — the cache is expensive to rebuild.

## Verify the install

Run a single well-behaved TOI end to end:

```bash
python run_full_analysis.py 105.01 --cpus=4
```

If it completes, you will find `autottv_results_v2/TOI_105_01/results.json` alongside a
set of PNGs. A first run also populates the light-curve cache, so it is slower than
subsequent ones. See [Quick start](quickstart.md) for what to look at in the output.

## Common installation problems

??? failure "`batman` fails to build"
    `batman-package` compiles C at install time and needs a working compiler toolchain.
    On macOS install the Xcode command-line tools (`xcode-select --install`); on Debian or
    Ubuntu install `build-essential` and the Python headers (`python3-dev`).

??? failure "`pip` refuses with `externally-managed-environment`"
    Python 3.11+ marks system interpreters as externally managed (PEP 668). Create a
    virtual environment as above. `--break-system-packages` will force the install but is
    the wrong fix — it is precisely the collision the marker exists to prevent.

??? failure "`lightkurve` cannot reach MAST"
    Downloads go over HTTPS to `mast.stsci.edu`. Behind a proxy, set `HTTPS_PROXY` before
    running. Transient MAST outages surface as download errors partway through a batch —
    the cache means a re-run resumes rather than starting over.

??? failure "Out of memory during Step 1"
    Lower `--cpus`. Each worker holds its own copy of the light curve, so peak memory
    scales with worker count, not with the size of the machine.
