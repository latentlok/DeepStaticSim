# DeepStaticSim release image: web UI + surrogate runner + 3D viz, model BAKED IN.
#
#   docker run -d --gpus all -p 8090:8090 -v dss_jobs:/data/jobs ghcr.io/latentlok/deepstaticsim:latest
#   docker run -d            -p 8090:8090 -v dss_jobs:/data/jobs ghcr.io/latentlok/deepstaticsim:latest   # no GPU
#
# Nothing to mount for the model: the deployed checkpoint lives at /opt/model in
# the loader's layout (<run>/.hydra/config.yaml + <run>/ckpt/best_weights). Jobs
# persist under the /data/jobs volume.
#
# GPU OR CPU, same image: torch is the cu128 wheel (bundles its own CUDA runtime,
# so the base stays plain python:3.12-slim). With `--gpus all` on a host that has
# the NVIDIA driver (>= 570) + nvidia-container-toolkit, start.sh picks cuda;
# otherwise it falls back to cpu automatically (inference is seconds per bracket
# either way). Kernels ship for sm_75 .. sm_120, i.e. T4 (g4dn) through Blackwell
# -- verified in-container with torch.cuda.get_arch_list() at release time.
#
# Rendering: VTK draws off-screen under Xvfb with mesa software rasterisation --
# independent of the GPU, so the viewer works on every instance type.
#
# Maintainers building this themselves: run from the repo root
#   docker build -f deploy/Dockerfile -t ghcr.io/latentlok/deepstaticsim:latest .
# after staging the run directory into deploy/model/ (gitignored):
#   deploy/model/.hydra/config.yaml  deploy/model/ckpt/best_weights/
# A .dockerignore excluding .venv/ outputs/ app_data/ keeps the context small.

FROM python:3.12-slim

# Headless VTK/pyvista system deps: xvfb + mesa GL + the X client libs the vtk
# wheel links against.
RUN apt-get update && apt-get install -y --no-install-recommends \
        xvfb xauth libgl1 libglx-mesa0 libegl1 \
        libxrender1 libxext6 libsm6 libxt6 libice6 libx11-6 \
    && rm -rf /var/lib/apt/lists/*

# torch (cu128 wheel, pinned) from its own index, then the rest from PyPI.
# The checkpoint is plain safetensors weights, loadable by any torch >= 2.4.
RUN pip install --no-cache-dir torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128 \
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

# The deployed model, in the run-directory layout the loader reads.
COPY deploy/model/.hydra/config.yaml        /opt/model/.hydra/config.yaml
COPY deploy/model/ckpt/best_weights/        /opt/model/ckpt/best_weights/

COPY deploy/start.sh /opt/dss/start.sh
RUN chmod +x /opt/dss/start.sh

ENV PYTHONUNBUFFERED=1
VOLUME /data/jobs
EXPOSE 8090
WORKDIR /opt/dss/surrogate

# start.sh picks cuda/cpu, launches Xvfb itself and execs python as PID 1 -- see
# its comment for why xvfb-run must not be the entrypoint.
ENTRYPOINT ["/opt/dss/start.sh"]
