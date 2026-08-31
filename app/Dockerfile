# UNTESTED: written, not built — docker is not available on the dev box.
# Build from the REPO ROOT (the COPY paths need both trees):
#
#     docker build -f app/Dockerfile -t deepstaticsim-runner .
#
# This containerizes the RUNNER ONLY — the stateless STL-in/files-out CLI —
# which is the future AWS Batch worker. The web app stays outside; it submits
# jobs that run exactly this entrypoint.
#
# The checkpoint is NOT baked into the image: weights change more often than
# code, and one image should serve every run. Mount it at runtime:
#
#     docker run --rm \
#       -v /path/to/ckpt/best_weights:/ckpt:ro \
#       -v /path/to/work:/data \
#       deepstaticsim-runner /data/bracket.stl --out-dir /data/job --ckpt /ckpt
#
# NOTE: --ckpt must point at a checkpoint directory that still contains its
# run's .hydra/config.yaml two levels up, OR mount the whole run directory and
# pass /run/ckpt/best_weights — the runner reads the architecture from there.

FROM python:3.12-slim

# CPU torch wheel: an order of magnitude smaller than the CUDA one, and Batch
# CPU instances are the target. Everything the runner imports, nothing more
# (no h5py, no zarr, no trame — those belong to training and the web app).
RUN pip install --no-cache-dir \
      --index-url https://download.pytorch.org/whl/cpu torch \
 && pip install --no-cache-dir \
      numpy scipy pyvista einops hydra-core omegaconf safetensors

# Preserve the repo layout: runner.py resolves imports as ../surrogate from
# app/, so both trees are copied side by side and WORKDIR is surrogate/.
WORKDIR /opt/deepstaticsim
COPY surrogate/models/  surrogate/models/
COPY surrogate/engine/  surrogate/engine/
COPY surrogate/utils/   surrogate/utils/
COPY app/runner.py app/export.py app/__init__.py app/

WORKDIR /opt/deepstaticsim/surrogate
ENTRYPOINT ["python", "../app/runner.py"]
CMD ["--help"]
