# DeepStaticSim

Structural surrogate modelling for the [DeepJEB](https://arxiv.org/abs/2406.09047)
jet-engine-bracket dataset: geometry in, per-node displacement + signed von Mises
stress fields out, for the dataset's four fixed static load cases.

```
surrogate/   Transolver surface model + data pipeline (fork of deeplearning-template)
docs/        specs and implementation plans
frontend/    (planned) STL upload + field visualization
```

## Data

Raw DeepJEB subset (50 designs, every format; 35 with the field csv):
`/home/shared/resources/datasets/JEBsim/DeepJEB_50` — read-only, never written.

Processed store (written by `surrogate/utils/fetch_deepjeb.py`):
`$DL_DATA = /home/shared/resources/datasets/JEBsim/processed` →
`deepjeb.zarr` + `splits.json` (27/4/4 over the 35 csv-complete designs) +
`stats_surface.json`.

The raw h5 files carry three defects — fields misaligned with the mesh, garbage
face winding, faces indexed in a surface-local numbering — all measured, all
repaired and guarded in `surrogate/utils/fetch_deepjeb.py` (see its docstring).
Never consume the h5 nodal fields without that realignment.

## Quickstart

```bash
cd surrogate
uv sync --extra dev
export DL_DATA=/home/shared/resources/datasets/JEBsim/processed
uv run python utils/fetch_deepjeb.py --raw /home/shared/resources/datasets/JEBsim/DeepJEB_50 --root $DL_DATA
uv run python utils/stats_deepjeb.py --root $DL_DATA
uv run python train.py experiment=jeb_surface
uv run python eval.py experiment=jeb_surface ckpt=outputs/jeb_surface/<run>/ckpt/best_weights data.val_split=test
```

`surrogate/STATUS.md` records the current measured results.
