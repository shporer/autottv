# syntax=docker/dockerfile:1.7
#
# Dockerfile for AutoTTV
# https://github.com/shporer/autottv
#
# Two stages:
#   1) builder  — full dev toolchain, builds a virtualenv at /opt/venv with
#                 all dependencies (compiles the batman C extension)
#   2) runtime  — slim Python image with only the runtime libs, copies the
#                 venv from the builder, runs as a non-root user
#
# Quick usage:
#
#   # build
#   docker build -t autottv .
#
#   # run a single TOI, bind-mounting outputs/cache to your host
#   docker run --rm -it \
#       -v "$PWD/autottv_results_v2:/app/autottv_results_v2" \
#       autottv \
#       python run_full_analysis.py 109.01 --cpus=4
#
#   # interactive shell for batch work
#   docker run --rm -it \
#       -v "$PWD/autottv_results_v2:/app/autottv_results_v2" \
#       autottv

# ----------------------------------------------------------------------
# Stage 1 — builder
# ----------------------------------------------------------------------
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Build-time system deps:
#   build-essential / gcc / g++ / gfortran  — compile batman C extension and any
#                                              source wheels that pip falls back to.
#   libopenblas-dev / liblapack-dev          — numpy/scipy linkage if a wheel is missing.
#   pkg-config                                — used by some scientific deps.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
        gfortran \
        libopenblas-dev \
        liblapack-dev \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Build everything into an isolated venv so we can copy it verbatim
# into the runtime stage.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /tmp/build

# Cache pip layer: requirements.txt is copied alone, then installed.
# Subsequent code changes don't bust the dependency layer.
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel \
 && pip install -r requirements.txt

# ----------------------------------------------------------------------
# Stage 2 — runtime
# ----------------------------------------------------------------------
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg \
    PATH="/opt/venv/bin:$PATH"

# Sensible default thread caps so the BLAS layer doesn't oversubscribe inside
# multiprocessing pools (the pipeline already parallelises across TOIs / walkers).
# Override at runtime with `-e OMP_NUM_THREADS=...` if you want.
ENV OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1

# Runtime libs only — no compilers.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libopenblas0 \
        liblapack3 \
        libstdc++6 \
        libgomp1 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy the prebuilt venv from the builder stage.
COPY --from=builder /opt/venv /opt/venv

# Non-root user.  Override UID/GID at build time to match your host so that
# bind-mounted output files are writeable without chowning:
#   docker build --build-arg UID=$(id -u) --build-arg GID=$(id -g) -t autottv .
ARG UID=1000
ARG GID=1000
RUN groupadd -g ${GID} autottv && useradd -m -u ${UID} -g ${GID} -s /bin/bash autottv

WORKDIR /app

# Copy the project.  .dockerignore prunes results dirs, caches, presentations,
# logs, the venv, and anything else not needed inside the image.
COPY --chown=autottv:autottv . /app/

# Pre-create output dir in case the user runs without bind mounts.
RUN mkdir -p /app/autottv_results_v2 \
 && chown -R autottv:autottv /app/autottv_results_v2

USER autottv

# Bind-mount this for persistent results and to share the MAST data cache
# (autottv_results_v2/data_cache/) across container runs.
VOLUME ["/app/autottv_results_v2"]

# Sanity check: confirm the import chain works at build time.
RUN python -c "import lightkurve, emcee, batman, astropy, numpy, scipy, pandas, matplotlib, corner; \
print('autottv deps OK')"

# Default to an interactive shell.  Override on `docker run` with the script
# you want, e.g.:
#   docker run --rm autottv python run_full_analysis.py 109.01 --cpus=4
CMD ["/bin/bash"]
