# DeepStaticSim

**Structural fields for jet-engine brackets without running FEA.** Upload an STL of a
[DeepJEB](https://arxiv.org/abs/2406.09047)-family bracket; get per-surface-point
displacement (mm) and signed von Mises stress (MPa) for the dataset's four fixed load
cases in seconds, view them in 3D in the browser, download VTP/CSV — or drive it all
through a REST API.

The model is a [Transolver](https://arxiv.org/abs/2402.02366) (slice-attention
transformer for irregular geometry) trained as a surrogate for linear-static FEA on the
DeepJEB bracket family. Geometry is the only input: every DeepJEB design shares the same
bolted boundary conditions and loads (vertical Fz = 35,585.77 N, horizontal
Fx = −37,809.9 N, diagonal 42,258.12 N at 42°, torsion Mz = 564,924.2 N·mm; Ti-6Al-4V).

> **Not a certified analysis.** Screening accuracy on one bracket family — see
> [Results](#results) — use it to rank candidates and locate hot spots, then run real
> FEA on the winners.

## Run it — no build step

The published image has the trained checkpoint baked in.

```bash
# GPU host (NVIDIA driver + container toolkit installed)
docker run -d --restart unless-stopped --gpus all -p 8090:8090 \
    -v deepstaticsim-jobs:/data/jobs ghcr.io/latentlok/deepstaticsim:latest

# CPU-only host -- identical, just drop --gpus all. Inference takes seconds on CPU;
# a GPU is optional for the app (it only matters for retraining).
docker run -d --restart unless-stopped -p 8090:8090 \
    -v deepstaticsim-jobs:/data/jobs ghcr.io/latentlok/deepstaticsim:latest
```

Open `http://<host>:8090`: pick an `.stl`, press **Run**, switch load case / quantity /
colormap, drag the **warp** slider to see the deformed shape, toggle **dark** mode,
download `result.vtp` (ParaView-ready), `result.csv`, `summary.json`.

The same server is a plain HTTP API:

```bash
BASE=http://<host>:8090
curl -F "stl=@bracket.stl" $BASE/api/jobs            # 202 {"job": "<name>", "status": "running"}
curl $BASE/api/jobs/<name>                            # status + per-case peak stress/disp + links
curl -O $BASE/download/<name>/result.vtp              # the fields (Warp By Vector on {case}_disp)
```

EC2 runbook (instance sizing, security group, driver setup): [`deploy/DEPLOY.md`](deploy/DEPLOY.md).
Engineer-facing details of inputs/outputs and limits: [`app/README.md`](app/README.md).

## Results

Held-out test split (4 designs the model never saw), from `surrogate/STATUS.md`:

| metric | baseline (3.86M params) | **deployed** (15.4M params) |
|---|---|---|
| relative L2 on normalized fields | 0.381 | **0.344** |
| MAE (raw units) | 7.08 | **6.37** |
| peak-stress error, vertical | 16.0% | **11.0%** |
| peak-stress error, horizontal | 15.9% | **13.8%** |
| peak-stress error, diagonal | 13.3% | **12.2%** |
| peak-stress error, torsion | **11.4%** | 23.2% |

The deployed model wins on every aggregate and on three of four peak-stress cases;
**torsion peak stress regressed** and is reported as such. Displacement channels are
predicted well (per-channel rel-L2 0.13–0.36); stress fields are harder (0.38–0.51).

**Limits, plainly:** trained on **27 designs** of one family. Geometry outside the
family (different mounting, scale, or part type) gives confidently wrong answers with no
warning. STL tessellation far from the training meshes' ~2 mm element density shifts the
input distribution. Inference is windowed at the training context size because
Transolver pools over its input set. CPU inference: ~1.6 s (baseline) / ~4.4 s (deployed)
for an 80k-vertex bracket on a 12-core desktop CPU.

## Model weights

Published as GitHub Release assets (tag `v0.1.0`). Each tarball is a complete training
run directory — `.hydra/config.yaml` (architecture) + `ckpt/best_weights/` (safetensors,
normalisation buffers included), which is exactly what the loader reads:

```bash
REL=https://github.com/latentlok/DeepStaticSim/releases/download/v0.1.0
curl -LO $REL/deepstaticsim-transolver-big-v0.1.0.tar.gz        # deployed, 15.4M params
curl -LO $REL/deepstaticsim-transolver-baseline-v0.1.0.tar.gz   # baseline, 3.86M params
tar xzf deepstaticsim-transolver-big-v0.1.0.tar.gz              # -> <run>/

cd surrogate && uv run --no-sync python ../app/server.py --host 0.0.0.0 \
    --ckpt ../<run>/ckpt/best_weights
```

## Data

DeepJEB is **not redistributed** here — get it from https://www.narnia.ai/dataset (Open
Data Commons Attribution License). This project used a 50-design subset (`DeepJEB_50`:
every design's `.h5`, `.vtk`, `.stl`, `.step`, `.fem`; 35 with the field `.csv`), split
27 / 4 / 4 over the csv-complete designs.

Three defects of the raw `.h5` files were measured and are repaired in
[`surrogate/utils/fetch_deepjeb.py`](surrogate/utils/fetch_deepjeb.py) — never consume
the h5 nodal fields without it:

1. `nodal_variables` are in OptiStruct node-ID order while `vertices/cells/faces` are not
   — fields are realigned via the `.vtk` points and every design must pass an
   edge-smoothness check.
2. The stored `faces` winding is inconsistent (signed volume 2–60% of the true volume).
3. The stored `faces` index a surface-local numbering, not h5 vertices — the surface is
   rebuilt from the tets, and each design's enclosed volume is checked against
   `bracket_labels.csv` (all 50 within 0.9993–1.0000×).

## Reproduce / retrain

```bash
cd surrogate && uv sync --extra dev               # Python >=3.12,<3.14, uv
export DEEPJEB_RAW=/path/to/DeepJEB_50 DL_DATA=/path/to/processed
uv run python utils/fetch_deepjeb.py --raw $DEEPJEB_RAW --root $DL_DATA
uv run python utils/stats_deepjeb.py --root $DL_DATA
uv run python train.py experiment=jeb_surface                       # baseline config
uv run python train.py experiment=jeb_surface exp_name=jeb_surface_big \
    model.net.n_hidden=512 model.net.slice_num=64 optim.lr=8e-4       # deployed config
uv run python eval.py experiment=jeb_surface ckpt=outputs/<exp>/<run>/ckpt/best_weights data.val_split=test
uv run python utils/compare_server.py --split test --ckpt ...       # truth | prediction | error, in 3D
```

Measured numbers and provenance: [`surrogate/STATUS.md`](surrogate/STATUS.md). Design
spec and implementation plan: [`docs/`](docs/superpowers). Repo layout:

```
app/        web app (trame) + stateless STL->fields runner + REST API + exports
surrogate/  data pipeline, Transolver model, training/eval (fork of a step-first PyTorch template)
deploy/     Docker image, compose file, EC2 runbook
```

Tests: `cd surrogate && uv run pytest tests/ ../app/tests -q`.

## Citations

If you use this, cite the architecture and the data it was trained on:

```bibtex
@inproceedings{wu2024transolver,
  title     = {Transolver: A Fast Transformer Solver for {PDE}s on General Geometries},
  author    = {Wu, Haixu and Luo, Huakun and Wang, Haowen and Wang, Jianmin and Long, Mingsheng},
  booktitle = {Proceedings of the 41st International Conference on Machine Learning},
  series    = {Proceedings of Machine Learning Research},
  volume    = {235},
  pages     = {53681--53705},
  year      = {2024},
  publisher = {PMLR},
  url       = {https://proceedings.mlr.press/v235/wu24r.html}
}

@article{hong2025deepjeb,
  title   = {{DeepJEB}: 3D Deep Learning-Based Synthetic Jet Engine Bracket Dataset},
  author  = {Hong, Seongjun and Kwon, Yongmin and Shin, Dongju and Park, Jangseop and Kang, Namwoo},
  journal = {Journal of Mechanical Design},
  volume  = {147},
  number  = {4},
  pages   = {041703},
  year    = {2025},
  note    = {arXiv:2406.09047}
}

@article{whalen2021simjeb,
  title   = {{SimJEB}: Simulated Jet Engine Bracket Dataset},
  author  = {Whalen, Eamon and Beyene, Azariah and Mueller, Caitlin},
  journal = {Computer Graphics Forum},
  volume  = {40},
  number  = {5},
  pages   = {9--17},
  year    = {2021},
  doi     = {10.1111/cgf.14353}
}
```

`surrogate/models/transolver.py` is a port of the reference implementation in
[thuml/Transolver](https://github.com/thuml/Transolver) (batched, device-agnostic, with a
masked multi-channel loss); the mechanism is theirs. DeepJEB extends the crowdsourced
SimJEB brackets, hence the third entry.

## License

MIT — see [`LICENSE`](LICENSE). The DeepJEB data carries its own license (ODC-By).
