# STATUS — DeepJEB surface surrogate

Updated 2026-08-31. Every number below was measured, not estimated.

## Data

- Source: `/home/shared/resources/datasets/JEBsim/DeepJEB_50` (50 designs; the full
  DeepJEB is 2,138 — this is a deliberate small subset).
- Store: `$DL_DATA/deepjeb.zarr`, written by `utils/fetch_deepjeb.py`, which repairs
  three measured defects of the raw h5 (fields misaligned with the mesh, garbage
  face winding, faces in a surface-local numbering — see its docstring). Geometry
  check: every design's enclosed volume within 0.9993–1.0000x of
  `bracket_labels.csv`.
- Split (`splits.json`, seed 0): 27 train / 4 val / 4 test over the **35
  csv-complete designs only** (user decision: no zero-filled ver_x channel in any
  split). Val: 625_434, 630_268, 630_428, 633_321. Test: 625_433, 625_466, 633_55,
  634_291. The 15 csv-less designs sit in the store, unsplit.

## Run

`experiment=jeb_surface`, run `outputs/jeb_surface/2026-08-31_21-51-31_750439`:
Transolver (8 layers, width 256, 8 heads, 32 slices, 4.9M params fun_dim 4 →
out_dim 16), rel-L2 on normalized channels, AdamW 1e-3, cosine to 0 over 20k
steps, clip 1.0, batch 1 × 16,384-point windows, 8 windows/design/epoch.
31 min on the RTX 5090 (~10.7 steps/s). Best val/loss 0.2951 at step 5500
(train kept improving to 0.071 — the val gap is the 27-design data budget
talking, not a bug).

## Results (best checkpoint, step 5500)

| split | masked rel-L2 (norm.) | MAE (raw) | max-stress rel. err per case |
|---|---|---|---|
| val (during training) | 0.295 | 4.80 | dia 4.0%, hor 5.8% (others not logged at best step) |
| **test** (4 designs, `eval.py`) | **0.381** | **7.08** | **ver 16.0%, hor 15.9%, dia 13.3%, tor 11.4%** |

Test per-channel rel-L2: displacements 0.16–0.32; stresses 0.39–0.53
(ch1/ver_y 0.39 … ch7/hor_stress 0.53). Stress fields are the hard part, as
expected — they are rougher than displacements.

Honest read: with 27 training geometries this generalizes but is far from the
paper's 1,900-design regime; peak-stress errors of 11–16% on unseen designs are
a pipeline-proof, not a design tool yet. The pipeline (store, splits, stats,
training, eval) is what this run certifies; accuracy scales with data.

## Reproduce

```bash
export DL_DATA=/home/shared/resources/datasets/JEBsim/processed
uv run python utils/fetch_deepjeb.py --raw /home/shared/resources/datasets/JEBsim/DeepJEB_50 --root $DL_DATA
uv run python utils/stats_deepjeb.py --root $DL_DATA
uv run python train.py experiment=jeb_surface
uv run python eval.py experiment=jeb_surface ckpt=outputs/jeb_surface/<run>/ckpt/best_weights data.val_split=test
```

## Next

- `predict.py`: STL in → sampled surface + checkpoint inference → `.vtp` out
  (view in ParaView Glance, zero install) — agreed direction for the first UI.
- Later: trame+PyVista upload UI in `frontend/`, adapted from
  physics-transolver's `utils/viz_server.py`.
- When the 15 missing csvs arrive: `fetch_deepjeb.py --force --only <ids>`,
  regenerate splits + stats deliberately, retrain.
