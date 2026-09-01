# DeepJEB structural surrogate — design

Date: 2026-08-31. Approved in conversation (Transolver surface model, template fork,
16 target channels, own splits, DeepJEB_50).

## Goal

A surrogate that maps a jet-engine-bracket **surface geometry** to per-node structural
response fields (displacement + signed von Mises stress for 4 fixed load cases), so a
later frontend can accept an STL and visualize predicted fields. Trained on the local
DeepJEB_50 subset (50 designs) at `$DEEPJEB_RAW`.

Loads and BCs are identical for every design (verified by diffing OptiStruct decks:
4 bolted holes RBE2+SPC; Fz 35585.77 N, Fx −37809.9 N, diagonal 42258.12 N,
Mz 564924.2 N·mm; Ti-6Al-4V), so **geometry is the only input** — no load conditioning.

## Repo layout

```
DeepStaticSim/            git root (already initialized, branch main)
  docs/superpowers/specs/ this file, plans
  surrogate/              fork (copy, no history) of ../deeplearning-template
    dataset/deepjeb.py    DataModule: two-level sampling over the zarr store
    models/transolver.py  ported from ../physics-transolver, physics/drag machinery stripped
    utils/fetch_deepjeb.py  raw DeepJEB_50 -> $DL_DATA/deepjeb.zarr + splits.json
    utils/stats_deepjeb.py  train-split per-variable mean/std -> stats_surface.json
    configs/{data,model,experiment}/...
  (frontend/ later — out of scope here)
```

Template conventions all hold: data outside the repo at `$DL_DATA`
(default `./data/processed`), normalisation in model buffers
via `on_data_ready`, stats read from a file never computed in `setup()`, zarr only.

## Preprocessing (`utils/fetch_deepjeb.py`)

Per design (all 50 have h5 + vtk; 35 also have csv):

1. Read h5: `vertices (N,3)` mm, `cells (C,10)` quadratic tets, `faces (F,3)` surface
   tris over the first K vertices, `nodal_variables/*`.
2. **Realign** — the dataset's h5 files are internally inconsistent (verified on both
   downloads): `nodal_variables` are in OptiStruct node-ID order (== vtk point order)
   while `vertices/faces/cells` use a different order. KD-tree match vtk points →
   permutation; permute the **fields** into h5 vertex order so connectivity stays valid.
   Require exact bijection (max NN distance < 1e-2 mm, permutation is a bijection).
3. **Assert alignment**: mean |Δdisp| over tet edges must drop to the smooth level
   (measured ~0.0028 vs ~0.08 misaligned); refuse to write otherwise. No raw h5 field
   can reach training.
4. `ver_x_disp` exists only in the csv. Where the csv is present, read it, align by
   node-ID order (csv rows are already node-ID order — verified), and fill channel;
   where absent, write NaN and record `ver_x_valid=False` as a zarr attr.
5. Surface extraction: surface nodes are `vertices[:K]`, K = `faces.max()+1`
   (contiguous-from-0, verified). Vertex normals = area-weighted mean of incident
   face normals, normalised; vertex area = ⅓ · Σ incident face areas.
6. Write `deepjeb.zarr/<id>/surface/{position(3), normal(3), area(1), ver_disp(3),
   ver_stress(1), hor_disp(3), hor_stress(1), dia_disp(3), dia_stress(1), tor_disp(3),
   tor_stress(1)}` — float32, **rows permuted once** with a per-design RNG so a
   contiguous window is a uniform sample (drivaerml pattern), one permutation per
   design keeping all arrays row-aligned. Attrs: `n_points`, `ver_x_valid`.
7. Write `splits.json` at the store root: our own split (Metadata jsons ignored,
   user's call): seed 0, shuffle 50 ids → 40 train / 5 val / 5 test. Re-running fetch
   must not reshuffle: splits.json is written once and reused if present.
8. Idempotent per design (skip if group exists, `--force` to rewrite), so the 15
   missing csvs can be filled in later with `--force --only <ids…>`.

Mode shapes are **excluded** (eigenvector sign ambiguity makes direct regression
ill-posed). `bracket_labels.csv` scalars are not stored for now (future head).

## Dataset (`dataset/deepjeb.py`)

Two-level sampling, adapted from physics-transolver's `drivaerml.py`: pick a design,
take a contiguous window of `n_points` (default 16384) from its permuted rows;
`samples_per_run` windows per design per epoch. Split by design from `splits.json` —
points from one design never straddle splits. Batch contract:

```
pos (B,N,3) mm   fx (B,N,4) = normal(3)+area(1)   y (B,N,16)   y_mask (16,) bool
```

`y_mask` is False only for the ver_x channel of designs with `ver_x_valid=False`
(NaNs in `y` are zero-filled after masking so they can never propagate).
Stats from `stats_surface.json` via `stats` property → `on_data_ready`.

## Model (`models/transolver.py`)

Port `TransolverNet` + `Transolver` TaskModule from physics-transolver **stripped** of
physics residuals, drag/force, weighting, and freeze_context plumbing (no PDE loss
here — keep `freeze_context` out entirely; it exists only for physics losses). Keep:
batched slice attention, normalizer buffers, rel_l2, per-channel validation metrics.

Changes:
- Masked loss: rel_l2 / mse computed only over channels where `y_mask` is True
  (per-sample mask broadcast; NaN-safe by construction).
- Validation adds, per load case: `max_stress/abs_err` and `max_stress/rel_err` —
  |max|pred stress|| vs |max|true stress|| over the window — the engineering number.
- Config `transolver_surface`: fun_dim 4, out_dim 16, n_hidden 256, n_layers 8,
  n_head 8, slice_num 32, `pos_bounds [[-40,71],[-165,22],[0,66]]` (measured bbox).

Channel order (fixed, documented in config): `[ver_disp xyz, ver_stress, hor_disp xyz,
hor_stress, dia_disp xyz, dia_stress, tor_disp xyz, tor_stress]` — ver_x is index 0.

## Stats (`utils/stats_deepjeb.py`)

Per-variable mean/std over **train-split designs only**, streaming over the zarr
groups; `ver_disp` x-channel statistics computed only over designs with
`ver_x_valid=True`. Written keyed by variable name (template convention).

## Testing

- Synthetic fixture: a tiny fake h5+vtk+csv trio (a few hundred nodes, one tet layer)
  with a *deliberately permuted* h5 — tests that fetch realigns, asserts, extracts
  surface, writes the store; plus a no-csv variant → NaN channel + attr.
- `test_contracts.py`: add `transolver_surface: points_surface` line (synthetic
  point-cloud datamodule config), contract tests come free.
- Masked-loss unit test: NaN in the masked channel must not poison loss/grads; mask
  all-True reduces to plain rel_l2.
- Split test: deterministic, disjoint, written once.
- Template's existing 94 tests must stay green; ruff clean.

## Verification on real data

`fetch` over all 50 → stats → `debug=overfit` on one design (loss must fall hard) →
short smoke train → eval on the 5 test designs with per-channel rel-L2 and max-stress
errors reported honestly (50 designs; expect modest generalisation, pipeline-proof not
paper-beating).

## Out of scope (deliberate)

Frontend/STL inference service, volume model, mode shapes, performance-scalar head,
full-2138 training. The store schema and masked-channel mechanism are the only things
designed ahead for these.
