# Deploying DeepStaticSim on AWS (EC2, no fuss)

Upload an STL of a DeepJEB-family bracket, get displacement + stress fields for
the four fixed load cases, view them in 3D in the browser, download VTP/CSV.

**No GPU needed.** Inference is CPU-bound and takes seconds per bracket. A GPU
instance (g5/g6) is only worth paying for when *retraining* the surrogate
(see `surrogate/STATUS.md`); the app itself never touches CUDA.

## 1. The instance

- **Type:** `c6i.xlarge` (4 vCPU / 8 GB) is comfortable; `m6i.xlarge` if you
  expect several concurrent jobs. Burstables (t3) work but predictions slow down
  when credits run out.
- **AMI:** Ubuntu 24.04 LTS. **Disk:** 20 GB gp3 is plenty (image ~2.5 GB).
- **Security group:** allow inbound TCP **8090** from your company CIDR / VPN
  range only. The app has **no authentication** — anyone who can reach the port
  can upload and download. Restrict at the network layer (security group), or put
  it behind an ALB with OIDC / your VPN if it must be shared more widely.

## 2. Install and copy

On the instance:

```bash
sudo apt-get update && sudo apt-get install -y docker.io docker-compose-v2
sudo usermod -aG docker $USER && newgrp docker
```

From your machine, ship the repo and the trained run directory (the checkpoint
loader reads the run's `.hydra/config.yaml`, so copy the *run directory*, not
just the weights):

```bash
# the repo
rsync -a --exclude .venv --exclude outputs --exclude app_data \
    DeepStaticSim/  ec2-user@<EC2_IP>:~/DeepStaticSim/

# the trained run -> becomes the app's /data volume. Target layout must be:
#   deploy/data/.hydra/config.yaml        (architecture, read by the loader)
#   deploy/data/ckpt/best_weights/...     (the weights)
RUN=DeepStaticSim/surrogate/outputs/jeb_surface/2026-08-31_21-51-31_750439
ssh ec2-user@<EC2_IP> 'mkdir -p ~/DeepStaticSim/deploy/data/ckpt'
scp -r "$RUN/.hydra"            ec2-user@<EC2_IP>:~/DeepStaticSim/deploy/data/
scp -r "$RUN/ckpt/best_weights" ec2-user@<EC2_IP>:~/DeepStaticSim/deploy/data/ckpt/
```

Then:

```bash
cd ~/DeepStaticSim/deploy && docker compose up -d --build
```

First build downloads ~1.5 GB (CPU torch); subsequent restarts are instant.

## 3. Use it

- **Browser:** `http://<EC2_IP>:8090` → pick an STL in the drawer → **Run** →
  the job appears in the toolbar dropdown; switch load case / quantity, drag the
  *warp* slider for the deformed shape, download VTP/CSV/summary from the drawer.
- **Scripted (REST):**

```bash
# submit                                    (multipart field name: file)
curl -F "file=@bracket.stl" http://<EC2_IP>:8090/api/jobs        # -> {"job": "<id>", ...}
# poll
curl http://<EC2_IP>:8090/api/jobs                               # all jobs + status
curl http://<EC2_IP>:8090/api/jobs/<id>                          # one job (status, summary)
# fetch results
curl -O http://<EC2_IP>:8090/download/<id>/result.vtp
curl -O http://<EC2_IP>:8090/download/<id>/summary.json
```

- **ParaView:** open `result.vtp`, *Color By* `ver_stress` (or any case),
  *Warp By Vector* → `ver_disp` for the deformed shape. Works in
  [ParaView Glance](https://kitware.github.io/glance/app/) too — drag the file in.

## 4. Without Docker (bare metal)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
cd DeepStaticSim/surrogate && uv sync
uv run python ../app/server.py --host 0.0.0.0 --port 8090 \
    --ckpt outputs/jeb_surface/<run>/ckpt/best_weights
```

Same UI/API. The repo's root `Makefile` has `make app` for exactly this.

## 5. Later: batch workers

The runner is a stateless file-in/file-out CLI (`app/runner.py`), deliberately
separable from the web app. `app/Dockerfile` builds a runner-only image — that is
the AWS Batch / SQS-worker seam: the web tier submits `input.stl` to a queue, a
Batch job runs the same CLI against S3 paths, the UI polls the same
`summary.json`. Nothing in the app needs to change shape to get there.

## Honest limits

Trained on 27 DeepJEB brackets with the dataset's fixed bolt BCs and four load
cases. Peak-stress error on held-out designs is 11–16% (see
`surrogate/STATUS.md`), and out-of-family geometry is unvalidated. Screening
tool, not certified analysis.
