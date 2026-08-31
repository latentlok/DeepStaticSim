# DeepStaticSim full app image: web UI + surrogate runner + 3D visualization.
#
#   docker build -f deploy/Dockerfile -t deepstaticsim .          # from repo root
#   docker run -d -p 8090:8090 -v $PWD/data:/data deepstaticsim
#
# /data is the ONE mount: put a training run directory's contents there --
#   /data/.hydra/config.yaml      (the model architecture, read by the loader)
#   /data/ckpt/best_weights/      (the weights)
# and jobs land in /data/jobs/. See deploy/DEPLOY.md for the exact copy commands.
#
# CPU-only ON PURPOSE: inference is seconds per bracket on a few cores; a GPU
# instance is only worth paying for when RETRAINING (see surrogate/).
#
# Rendering: VTK draws off-screen under xvfb (a virtual X server) with mesa
# software rasterisation -- no GPU, no display, works on any EC2 instance.

FROM python:3.12-slim

# Headless VTK/pyvista system deps: xvfb + mesa GL + the X client libs the vtk
# wheel links against.
RUN apt-get update && apt-get install -y --no-install-recommends \
        xvfb xauth libgl1 libglx-mesa0 libegl1 \
        libxrender1 libxext6 libsm6 libxt6 libice6 libx11-6 \
    && rm -rf /var/lib/apt/lists/*

# CPU torch first (its own index), then the rest from PyPI. Versions float on
# minor releases deliberately; pin harder if your org requires reproducible pulls.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir \
        numpy einops "hydra-core>=1.3.4" omegaconf safetensors \
        pyvista "trame>=3.13" trame-vtk trame-vuetify aiohttp

# The code layout the app expects: app/ next to surrogate/ (server and runner
# resolve ../surrogate from their own path). dataset/ is deliberately NOT copied:
# it is only imported by the training-side compare server and would drag zarr in.
WORKDIR /opt/dss
COPY surrogate/models/   surrogate/models/
COPY surrogate/engine/   surrogate/engine/
COPY surrogate/utils/__init__.py surrogate/utils/normalize.py surrogate/utils/
COPY app/runner.py app/export.py app/server.py app/__init__.py app/

COPY deploy/start.sh /opt/dss/start.sh

ENV PYTHONUNBUFFERED=1
VOLUME /data
EXPOSE 8090
WORKDIR /opt/dss/surrogate

# start.sh launches Xvfb itself and execs python as PID 1 -- see its comment for
# why xvfb-run must not be the entrypoint.
ENTRYPOINT ["/opt/dss/start.sh"]
