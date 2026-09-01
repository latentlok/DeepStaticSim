# STATUS — DeepJEB surface surrogate

Updated 2026-08-31. Every number below was measured, not estimated.

## Data

- Source: `$DEEPJEB_RAW` (50 designs; the full
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

## Runs

| run | params | data | best val | test rel-L2 | test MAE |
|---|---|---|---|---|---|
| jeb_surface `2026-08-31_21-51-31` | 3.86M (8x256, G=32) | fixed 16k windows | 0.295 @5.5k | 0.381 | 7.08 |
| jeb_surface_big (fixed windows, canceled) | 15.4M (8x512, G=64) | fixed 16k windows | 0.313 @8k | -- | -- |
| **jeb_surface_big `2026-08-31_22-55-37`** | 15.4M (8x512, G=64) | **stochastic 32k windows** | **0.303 @12.5k** | **0.344** | **6.37** |

All: rel-L2 loss on normalized channels, AdamW (1e-3 / 8e-4 for big), cosine to 0
over 20k steps, clip 1.0, batch 1, 8 windows/design/epoch. Big run: 86 min on the
RTX 5090. Fresh-windows-per-epoch measurably helped: the same 15.4M model with
fixed windows was canceled at 0.313 best val while this one reached 0.303.

## Results (test split, 4 unseen designs, eval.py)

| metric | baseline 3.86M | big 15.4M (DEPLOYED) |
|---|---|---|
| masked rel-L2 (norm.) | 0.381 | **0.344** |
| MAE (raw) | 7.08 | **6.37** |
| max-stress rel err: ver | 16.0% | **11.0%** |
| max-stress rel err: hor | 15.9% | **13.8%** |
| max-stress rel err: dia | 13.3% | **12.2%** |
| max-stress rel err: tor | **11.4%** | 23.2% |

Per-channel (big): displacements 0.13-0.36, stresses 0.38-0.51. The big model
wins loss, MAE, displacements and 3/4 peak-stress cases; TORSION PEAK REGRESSED
(23.2% vs 11.4%) -- the one number the baseline still does better. Both
checkpoints are kept; app/runner.py and app/server.py default to the big run.

Honest read: 27 training geometries is the binding constraint; these are
screening-grade errors, not sign-off-grade. Accuracy scales with data.

## Reproduce

```bash
export DL_DATA=./data/processed
uv run python utils/fetch_deepjeb.py --raw $DEEPJEB_RAW --root $DL_DATA
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
