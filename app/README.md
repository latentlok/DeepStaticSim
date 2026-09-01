# DeepStaticSim App — bracket structural results without running FEA

Upload an STL of a jet-engine bracket; get back the displacement and stress
fields a linear-static FEA run would give you, in seconds instead of hours.

## What it predicts

The surrogate is trained on the [DeepJEB](https://arxiv.org/abs/2406.09047)
jet-engine-bracket family. Every bracket in that family shares the **same
mounting interface and the same loads** — that is why the app never asks you
for a load case:

| case | load | direction |
|---|---|---|
| `ver` (vertical)   | Fz = 35,585.77 N  | +z |
| `hor` (horizontal) | Fx = −37,809.9 N  | −x |
| `dia` (diagonal)   | 42,258.12 N       | 42° from vertical |
| `tor` (torsion)    | Mz = 564,924.2 N·mm | −z |

Boundary conditions: the 4 bolt holes fixed, load applied at the interface
point — identical for every design. Material: Ti-6Al-4V (E = 113.8 GPa,
ν = 0.342). Verified by diffing the dataset's OptiStruct decks: only geometry
varies between designs.

**Input:** one STL of a bracket from this family (same bolt pattern and load
interface, roughly the same bounding box, tessellated at about the training
meshes' ~2 mm element density).

**Output:** for every sampled surface point, 16 values — per load case a
displacement vector `{case}_disp` (mm, x/y/z) and a signed von Mises stress
`{case}_stress` (MPa; sign distinguishes tension from compression).

## Honest limits — read before trusting a number

- Trained on **27 designs** of one bracket family. On 4 held-out designs the
  deployed model's mean relative field error (rel-L2) is **0.344**, and the
  **peak-stress error is 11–14%** for the vertical/horizontal/diagonal cases and
  **23% for torsion** (see `surrogate/STATUS.md`). Screening accuracy, not sign-off.
- **Not a certified analysis.** Use it to rank design candidates and spot
  hot-spot locations, then run real FEA on the winners.
- Geometry outside the family — different mounting, different scale, a part
  that is not a DeepJEB-style bracket — is garbage-in, garbage-out. The model
  has no way to warn you; it will happily color a wrong answer.
- STL tessellation far coarser or finer than ~2 mm shifts the input
  distribution away from training and degrades accuracy.

## Quickstart

```bash
cd surrogate && uv sync --extra dev        # once

# Web app: upload STL -> 3D results in the browser
make app                                   # from the repo root; serves on :8090

# One-shot CLI (no browser, results to files)
make predict STL=/path/to/bracket.stl      # writes to jobs/<stl-name>/
make predict STL=/path/to/bracket.stl OUT=/tmp/myjob
```

Both default to the deployed checkpoint's run directory under `surrogate/outputs/`
(pass `--ckpt <run>/ckpt/best_weights` to the underlying scripts to pick another run;
published runs are GitHub Release assets — see the root README). No build, no local
Python at all: `docker run -p 8090:8090 ghcr.io/latentlok/deepstaticsim:latest`
(add `--gpus all` on a GPU host) — details in `deploy/DEPLOY.md`.

## What a job produces

Each run writes a job directory containing:

- **`result.vtp`** — the sampled surface points with all 16 fields attached.
  Open it in [ParaView](https://www.paraview.org/) or drag it into
  [ParaView Glance](https://kitware.github.io/glance/app/) (runs in the
  browser, nothing to install): *Color by* `ver_stress` (or any
  `{case}_stress` / `{case}_disp`), and *Warp By Vector* on `{case}_disp`
  to see the deformed shape.
- **`result.csv`** — the same data as a flat table
  (`x,y,z, ver_disp_x, … tor_stress`), one row per point, for Excel/pandas.
- **`summary.json`** — the numbers an engineer scans first: per case the
  max |stress|, max resultant displacement, and their locations, plus model
  and checkpoint provenance.

## Architecture: the app and the solver are deliberately separable

```
browser ── trame UI (app/server.py) ──▶ runner (app/runner.py) ──▶ result files
                    ▲                        stateless: STL in, files out
                    └────── visualizes ◀─────────────┘
```

The **runner** is a stateless file-in/file-out CLI. The web app just calls it
and renders what it wrote. That contract is the cloud-migration seam: to move
inference to the cloud, containerize the runner (see `Dockerfile` here) and
have the app submit the same CLI as an AWS Batch job with the STL on S3 —
the UI, the file formats, and the visualization do not change.

## Retraining

Accuracy scales with data — see `surrogate/STATUS.md` for current measured
numbers and provenance. With more DeepJEB designs on disk:

```bash
cd surrogate
uv run python utils/fetch_deepjeb.py --raw <DeepJEB dir> --root $DL_DATA
uv run python utils/stats_deepjeb.py --root $DL_DATA
uv run python train.py experiment=jeb_surface
```

## REST API

The web server doubles as a plain HTTP API (same engine, same jobs directory), so
curl, CI, or another service can run analyses without a browser:

```bash
BASE=http://<host>:8090
curl -F "stl=@bracket.stl" $BASE/api/jobs          # -> 202 {"job": "<name>", "status": "running"}
curl $BASE/api/jobs                                 # -> {"jobs": [{"job": ..., "status": ...}]}
curl $BASE/api/jobs/<name>                          # -> status + summary + download links
curl -O $BASE/download/<name>/result.vtp            # the artifacts themselves
```

Statuses: `running` -> `done` (artifacts ready) or `failed` (see
`/download/<name>/runner.log`). Uploads must be multipart/form-data with a file
field named `stl` and an `.stl` filename.

## Dark mode

The toolbar's "dark" switch flips both the UI theme and the 3D viewport
background; the setting is per browser session.
