# Deploying DeepStaticSim on AWS EC2 — no build step

Pull one image, run one command, open one port. The image
`ghcr.io/latentlok/deepstaticsim:latest` contains the app, the runner and the trained
checkpoint (at `/opt/model`); jobs persist in a Docker volume mounted at `/data/jobs`.

**GPU is optional.** Inference is seconds per bracket on CPU (~4 s for an 80k-vertex
part on a desktop 12-core; expect ~15–20 s on `c6i.xlarge`). On a GPU the same job is
sub-second and the image uses it automatically when it is exposed with `--gpus all`.
A GPU instance is *required* only for retraining.

## 1. Pick the instance

| want | instance | notes |
|---|---|---|
| cheapest that works | `t3.medium` / `c6i.large` (2 vCPU) | 30–60 s per job, fine for occasional use |
| recommended CPU | **`c6i.xlarge`** (4 vCPU / 8 GB) | ~15–20 s per job, UI stays responsive |
| smallest GPU | **`g4dn.xlarge`** (NVIDIA T4 16 GB, 4 vCPU, 16 GB) | sub-second inference; also the cheapest box that can retrain |
| more GPU | `g5.xlarge` (A10G) / `g6.xlarge` (L4) | faster retraining; overkill for serving |

Avoid `g4ad` (AMD Radeon / ROCm — the image is CUDA). Disk: 30 GB gp3 (the GPU image
is several GB). AMI: **AWS Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04/24.04)**
for GPU instances — NVIDIA driver, Docker and the NVIDIA Container Toolkit come
preinstalled — or plain **Ubuntu 24.04 LTS** for CPU instances.

**Security group:** inbound TCP **8090** from your company CIDR / VPN only. The app has
**no authentication** — anyone who can reach the port can upload and download. Keep it
network-restricted, or front it with an ALB + OIDC or your VPN if it must be shared wider.

## 2. Prepare the host

### Ubuntu 24.04 (CPU instance)

```bash
sudo apt-get update && sudo apt-get install -y docker.io
sudo usermod -aG docker $USER && newgrp docker
```

### GPU instance with a plain Ubuntu 24.04 AMI (skip on the Deep Learning AMI)

```bash
# NVIDIA driver + Docker
sudo apt-get update && sudo apt-get install -y nvidia-driver-570-server docker.io
sudo usermod -aG docker $USER && newgrp docker
sudo reboot   # loads the driver; `nvidia-smi` must work after this

# NVIDIA Container Toolkit (official repo)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker

docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi   # must print the T4
```

(`nvidia-driver-570-server` is the current Ubuntu 24.04 server driver package name;
`ubuntu-drivers list` shows what your AMI offers.)

## 3. Run

```bash
# GPU instance
docker run -d --name deepstaticsim --restart unless-stopped --gpus all \
    -p 8090:8090 -v deepstaticsim-jobs:/data/jobs \
    ghcr.io/latentlok/deepstaticsim:latest

# CPU instance: same line without --gpus all
docker run -d --name deepstaticsim --restart unless-stopped \
    -p 8090:8090 -v deepstaticsim-jobs:/data/jobs \
    ghcr.io/latentlok/deepstaticsim:latest
```

Verify:

```bash
docker logs deepstaticsim | tail -3          # "serving on http://0.0.0.0:8090"
curl -sL -o /dev/null -w '%{http_code}\n' http://localhost:8090/    # 200
```

Then open `http://<EC2_IP>:8090` from your machine.

## 4. Use it

- **Browser:** drawer → pick an `.stl` → **Run** → the job appears in the toolbar
  dropdown. Switch load case / quantity / colormap, drag the **warp** slider for the
  deformed shape, toggle **dark**, download VTP / CSV / summary.
- **Scripted (REST)** — multipart field name is `stl`:

```bash
BASE=http://<EC2_IP>:8090
curl -F "stl=@bracket.stl" $BASE/api/jobs              # 202 {"job": "<id>", "status": "running"}
curl $BASE/api/jobs                                     # every job + status
curl $BASE/api/jobs/<id>                                # status, per-case peaks, download links
curl -O $BASE/download/<id>/result.vtp                  # fields; also result.csv, summary.json, runner.log
```

- **ParaView:** open `result.vtp`, *Color By* `ver_stress` (or any `{case}_stress` /
  `{case}_disp_mag`), *Warp By Vector* → `{case}_disp`. Drag-and-drop works in
  [ParaView Glance](https://kitware.github.io/glance/app/) too.

## 5. Update to a new model version

Releases are tagged; the image tag matches. `docker pull ghcr.io/latentlok/deepstaticsim:<tag>`,
then re-run the `docker run` line with the new tag (jobs in the volume survive). To serve
a checkpoint of your own, mount its run directory and point the app at it:

```bash
docker run -d --gpus all -p 8090:8090 -v deepstaticsim-jobs:/data/jobs \
    -v /path/to/<run>:/opt/custom:ro ghcr.io/latentlok/deepstaticsim:latest \
    --ckpt /opt/custom/ckpt/best_weights
```

(A run directory = `.hydra/config.yaml` + `ckpt/best_weights/`; the loader reads the
config for the architecture. Published runs: GitHub Releases on this repo.)

## 6. docker compose (optional)

`deploy/docker-compose.yml` expresses the same run line (`cd deploy && docker compose up -d`);
see the comments in that file for the GPU/CPU switch.

## 7. Without Docker (bare metal)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
cd DeepStaticSim/surrogate && uv sync
uv run python ../app/server.py --host 0.0.0.0 --port 8090 --ckpt <run>/ckpt/best_weights
```

Same UI/API. Get `<run>` from the GitHub Release assets (see the root README).

## 8. Later: batch workers

The runner (`app/runner.py`) is a stateless file-in/file-out CLI, deliberately separable
from the web tier; `app/Dockerfile` builds a runner-only image. That is the AWS Batch /
SQS-worker seam: the web tier queues `input.stl`, a Batch job runs the same CLI against
S3 paths, the UI polls the same `summary.json`. Nothing in the app changes shape.

## Honest limits

Trained on 27 DeepJEB brackets with the dataset's fixed bolt BCs and four load cases.
On held-out designs: relative L2 0.344, peak-stress error 11–14% for three cases and
23% for torsion (see `surrogate/STATUS.md`). Out-of-family geometry is unvalidated.
Screening tool, not certified analysis.
