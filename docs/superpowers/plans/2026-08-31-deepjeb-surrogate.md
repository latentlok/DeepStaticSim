# DeepJEB Surrogate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A Transolver surface surrogate trained on DeepJEB_50: bracket surface point cloud in → 16 structural field channels out (disp + signed von Mises stress for 4 fixed load cases).

**Architecture:** Fork `../deeplearning-template` into `surrogate/`; preprocess the raw dataset (h5+vtk+csv) into a zarr store with a **field↔vertex realignment step** (the h5 files are internally misaligned); a two-level-sampling DataModule serves fixed-size point windows; the model is the physics-transolver Transolver port stripped of physics machinery, with a masked loss for the `ver_x_disp` channel that only 35/50 designs have.

**Tech Stack:** PyTorch 2.13, hydra, zarr 3, h5py, pyvista, scipy, einops, uv. Python 3.12/3.13.

**Spec:** `docs/superpowers/specs/2026-08-31-deepjeb-surrogate-design.md`

## Global Constraints

- All commands run from `surrogate/` with `uv run …`; repo git root is `DeepStaticSim/`.
- Raw data (read-only): `/home/shared/resources/datasets/JEBsim/DeepJEB_50` — 50 designs; every design has `FieldMesh/<id>.h5`, `VolumeMesh/<id>.vtk`; 35 have `Field/<id>.csv`.
- Processed data root: `DL_DATA=/home/shared/resources/datasets/JEBsim/processed` (never inside the repo).
- Template conventions (surrogate/CLAUDE.md) are binding: stats from a file, normalisation in model buffers via `on_data_ready`, `@hydra.main` stays in root `train.py`/`eval.py`, zarr only.
- Channel order (fixed everywhere): `y = [ver_disp x,y,z, ver_stress, hor_disp x,y,z, hor_stress, dia_disp x,y,z, dia_stress, tor_disp x,y,z, tor_stress]` → 16 channels; stress indices 3, 7, 11, 15; `ver_x` is index 0 and is the only maskable channel.
- Units stay raw: mm, MPa. `pos_bounds` for the model: `[[-40, 71], [-165, 22], [0, 66]]`.
- Splits: our own `splits.json` (seed 0, 40/5/5) — never `Metadata/*.json`.
- Reference code (read, then adapt): `../physics-transolver/models/transolver.py`, `dataset/drivaerml.py`, `dataset/examples.py::PointCloudData`, `utils/fetch_drivaerml.py`, `utils/stats_drivaerml.py`. Do not import across repos — copy into `surrogate/`.
- After every task: `uv run pytest tests/ -q` green and `uv run ruff check engine models dataset utils tests train.py eval.py` clean, then commit (from repo root, paths prefixed `surrogate/`).

---

### Task 1: Fork the template

**Files:**
- Create: `surrogate/` (copy of `../deeplearning-template`, no git history)
- Modify: `surrogate/pyproject.toml` (project name + new deps)

**Interfaces:**
- Produces: a working `surrogate/` tree where `uv run pytest tests/ -q` passes (94 tests) with `h5py`, `pyvista`, `scipy` importable.

- [ ] **Step 1: Copy the template**

```bash
cd /home/vishal/Documents/projects/DeepStaticSim
rsync -a --exclude .git --exclude .venv --exclude outputs --exclude .pytest_cache \
  --exclude .ruff_cache --exclude __pycache__ --exclude .graphify --exclude graphify-out \
  ../deeplearning-template/ surrogate/
```

- [ ] **Step 2: Edit `surrogate/pyproject.toml`**: change `name = "dlt"` → `name = "surrogate"`, and add to `dependencies`: `"h5py>=3.12"`, `"pyvista>=0.44"`, `"scipy>=1.14"`.

- [ ] **Step 3: Install and verify**

```bash
cd surrogate && uv sync --extra dev
uv run python -c "import h5py, pyvista, scipy, torch; print('ok')"
uv run pytest tests/ -q          # expect: 94 passed
uv run ruff check engine models dataset utils tests train.py eval.py
```

- [ ] **Step 4: Commit** — `git add surrogate && git commit -m "chore: fork deeplearning-template into surrogate/ with h5py/pyvista/scipy"`

---

### Task 2: Transolver model with masked loss

**Files:**
- Create: `surrogate/models/transolver.py`, `surrogate/dataset/points.py`, `surrogate/configs/model/transolver_surface.yaml`, `surrogate/configs/data/points_surface.yaml`, `surrogate/tests/test_transolver.py`
- Modify: `surrogate/tests/test_contracts.py` (one `MODEL_DATA` line)

**Interfaces:**
- Consumes: template `TaskModule`/`DataModule` ABCs, `utils/normalize.py::Normalizer`.
- Produces:
  - `models.transolver.TransolverNet(space_dim, fun_dim, out_dim, n_layers, n_hidden, n_head, slice_num, mlp_ratio, dropout, act, unified_pos, ref, pos_bounds)` with `forward(pos: (B,N,3), fx: (B,N,fun_dim)|None) -> (B,N,out_dim)`
  - `models.transolver.Transolver(TaskModule)` — batch contract `{"pos": (B,N,3), "fx": (B,N,4), "y": (B,N,16), "y_mask": (B,16) bool}` (`y_mask` optional; missing ⇒ all-true)
  - `models.transolver.masked_rel_l2(pred, target, mask) -> Tensor` and `CASES = ("ver","hor","dia","tor")`, `STRESS_IDX = {"ver":3,"hor":7,"dia":11,"tor":15}`
  - `dataset.points.PointCloudData(DataModule)` with `mask_channel: int|None` — synthetic clouds for contract tests.

- [ ] **Step 1: Write failing tests** in `surrogate/tests/test_transolver.py`:

```python
import torch
from models.transolver import Transolver, masked_rel_l2, relative_l2

def _batch(b=2, n=64, mask_ok=True):
    g = torch.Generator().manual_seed(0)
    y = torch.randn(b, n, 16, generator=g)
    mask = torch.ones(b, 16, dtype=torch.bool)
    if not mask_ok:
        mask[:, 0] = False
        y[..., 0] = 0.0          # dataset zero-fills masked channels
    return {"pos": torch.randn(b, n, 3, generator=g),
            "fx": torch.randn(b, n, 4, generator=g), "y": y, "y_mask": mask}

def test_masked_rel_l2_all_true_equals_plain():
    b = _batch(); pred = torch.randn_like(b["y"])
    assert torch.allclose(masked_rel_l2(pred, b["y"], b["y_mask"]),
                          relative_l2(pred, b["y"]))

def test_masked_channel_does_not_affect_loss_or_grad():
    b = _batch(mask_ok=False)
    pred = torch.randn_like(b["y"]).requires_grad_()
    loss = masked_rel_l2(pred, b["y"], b["y_mask"]); loss.backward()
    assert pred.grad[..., 0].abs().max() == 0          # no gradient into masked channel
    pred2 = pred.detach().clone(); pred2[..., 0] += 100.0
    assert torch.allclose(loss, masked_rel_l2(pred2, b["y"], b["y_mask"]))

def test_training_and_validation_step_run():
    m = Transolver(net=dict(fun_dim=4, out_dim=16, n_hidden=32, n_head=4,
                            n_layers=2, slice_num=8))
    out = m.training_step(_batch(mask_ok=False), state=None)
    assert out["loss"].isfinite()
    val = m.validation_step(_batch(mask_ok=False), state=None)
    for case in ("ver", "hor", "dia", "tor"):
        assert f"max_stress/{case}_abs_err" in val
    assert "rel_l2/ch0" not in val or val["rel_l2/ch0"].isfinite()
```

- [ ] **Step 2: Run** `uv run pytest tests/test_transolver.py -q` — expect ImportError.

- [ ] **Step 3: Implement `surrogate/models/transolver.py`.** Start from `../physics-transolver/models/transolver.py` and apply:
  1. **Delete**: `freeze_context` (everywhere — plain forwards only), physics config/`_physics`/`_check_tau`, `force`/`_drag`, `Weighting`/`StaticWeights`, imports from `models.pinn`, `utils.force`, `utils.physics_loss`; delete `eval_requires_grad` logic.
  2. **Keep verbatim**: `MLP`, `PhysicsAttention` (minus the freeze branch), `Block`, `TransolverNet` (minus freeze), `relative_l2`, `LOSSES`, normalizer plumbing (`pos_norm`/`fx_norm`/`y_norm`, `on_data_ready`), `configure_optimizers`.
  3. **Add**:

```python
CASES = ("ver", "hor", "dia", "tor")
STRESS_IDX = {"ver": 3, "hor": 7, "dia": 11, "tor": 15}

def masked_rel_l2(pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    """rel_l2 over the channels mask marks valid. mask (B,C) broadcasts over N.
    Masked channels are zeroed in BOTH tensors, so they add 0 to both norms and
    receive no gradient. The dataset zero-fills masked target channels already;
    the multiply here is what guarantees it for pred."""
    m = mask.to(pred.dtype).unsqueeze(1)                     # B,1,C
    num = torch.linalg.vector_norm((pred - target) * m, dim=(1, 2))
    den = torch.linalg.vector_norm(target * m, dim=(1, 2)).clamp_min(1e-8)
    return (num / den).mean()
```

  4. `training_step`: `pred = self(batch["pos"], batch["fx"])`; `mask = batch.get("y_mask", torch.ones_like-shaped all-True)`; `loss = masked_rel_l2(self.y_norm.norm(pred), self.y_norm.norm(batch["y"]) * mask.unsqueeze(1), mask)` — note: normalising a zero-filled channel gives `(0-mean)/std ≠ 0`, hence the extra `* mask.unsqueeze(1)` on the target before the loss (the loss masks again; belt and braces). Log `rel_l2` (raw units, masked) alongside.
  5. `validation_step`: masked loss + `mae` over valid channels + per-channel `rel_l2/ch{c}` **only for channels valid across the whole batch** + per case `max_stress/{case}_abs_err = ||pred_s|.max() − |y_s|.max()|` and `max_stress/{case}_rel_err = abs_err / |y_s|.max().clamp_min(1e-8)` where `s = STRESS_IDX[case]`, maxima over the window dim.

- [ ] **Step 4: Implement `surrogate/dataset/points.py`.** Copy `PointCloudData` from `../physics-transolver/dataset/examples.py`, add `out_dim: int = 16` and `mask_channel: int | None = 0`: each item gains `"y_mask": torch.ones(out_dim, dtype=bool)` with `y_mask[mask_channel] = False` for odd item indices (and the corresponding `y[..., mask_channel] = 0`), so contract tests exercise the mask path.

- [ ] **Step 5: Configs.** `configs/model/transolver_surface.yaml` — copy the physics-transolver one, then: `fun_dim: 4`, `out_dim: 16`, `pos_bounds: [[-40, 71], [-165, 22], [0, 66]]`, delete the whole `force:` and any `physics:` block, add a comment naming the 16-channel order. `configs/data/points_surface.yaml`:

```yaml
_target_: dataset.points.PointCloudData
n_points: 256
fun_dim: 4
out_dim: 16
mask_channel: 0
n_train: 8
n_val: 2
batch_size: 2
num_workers: 0
seed: 0
```

- [ ] **Step 6:** add `"transolver_surface": "points_surface"` to `MODEL_DATA` in `surrogate/tests/test_contracts.py`.

- [ ] **Step 7:** `uv run pytest tests/ -q` (all green, incl. new contract params) and ruff. Commit: `feat: Transolver surface model with masked 16-channel loss`.

---

### Task 3: Preprocessing — realign, extract surface, write zarr

**Files:**
- Create: `surrogate/utils/fetch_deepjeb.py`, `surrogate/tests/test_fetch_deepjeb.py`

**Interfaces:**
- Produces (importable functions, all pure except `write_design`/`main`):
  - `nodeid_to_h5(vtk_points: (N,3) f64, h5_vertices: (N,3) f64) -> np.ndarray` — `inv` with `field_h5order = field_nodeid[inv]`; raises `ValueError` unless bijective with max NN dist < 1e-2.
  - `edge_roughness(field: (N,), cells: (C,10)) -> float`
  - `surface_features(vertices: (N,3), faces: (F,3)) -> (normal (K,3), area (K,1))`, K = `faces.max()+1`; outward orientation via signed volume.
  - `load_design(raw: Path, id: str) -> dict` with keys `position (K,3), normal (K,3), area (K,1), {case}_disp (K,3), {case}_stress (K,1), ver_x_valid: bool` — realigned, surface-only, row-permuted.
  - `make_splits(ids: list[str], seed=0, n_val=5, n_test=5) -> dict[str, list[str]]`
  - CLI: `uv run python utils/fetch_deepjeb.py --raw <dir> --root $DL_DATA [--only id ...] [--force]` → `$DL_DATA/deepjeb.zarr/<id>/surface/*` + `$DL_DATA/splits.json`.

- [ ] **Step 1: Fixture + failing tests** in `surrogate/tests/test_fetch_deepjeb.py`. Fixture builds a fake raw design **with the real dataset's bug baked in** (h5 vertices reordered, fields left in node-ID order):

```python
import numpy as np, h5py, pyvista as pv, pytest

def make_fake_raw(root, id="1_2", with_csv=True, seed=0):
    """Delaunay tet mesh of ~60 pts -> quadratic tets (midside nodes on unique
    edges) -> subdivided boundary tris (each 6-node face as 4 linear tris, so
    surface nodes include midside nodes, like DeepJEB). Fields = smooth fn of
    coords, in NODE-ID order. h5 gets vertices in a DIFFERENT order (surface
    nodes first, then shuffled) with cells/faces remapped -- fields NOT remapped."""
    rng = np.random.default_rng(seed)
    from scipy.spatial import Delaunay
    pts = rng.uniform(0, 10, (60, 3)); tet = Delaunay(pts).simplices        # (T,4)
    # midside nodes
    edges = {tuple(sorted((a, b))) for t in tet for a, b in
             [(t[0],t[1]),(t[1],t[2]),(t[2],t[0]),(t[0],t[3]),(t[1],t[3]),(t[2],t[3])]}
    edges = sorted(edges); mid_of = {e: len(pts)+i for i, e in enumerate(edges)}
    verts = np.vstack([pts, [(pts[a]+pts[b])/2 for a, b in edges]])          # node-ID order
    def m(a, b): return mid_of[tuple(sorted((a, b)))]
    cells = np.array([[a,b,c,d, m(a,b), m(b,c), m(c,a), m(a,d), m(b,d), m(c,d)]
                      for a,b,c,d in tet])                                   # CTETRA10 order
    # boundary faces of the linear tets -> subdivide each into 4
    from collections import Counter
    fc = Counter(tuple(sorted(f)) for t in tet for f in
                 [(t[0],t[1],t[2]),(t[0],t[1],t[3]),(t[0],t[2],t[3]),(t[1],t[2],t[3])])
    faces = []
    for (a,b,c), n in fc.items():
        if n == 1:
            ab, bc, ca = m(a,b), m(b,c), m(c,a)
            faces += [[a,ab,ca],[ab,b,bc],[ca,bc,c],[ab,bc,ca]]
    faces = np.array(faces)
    fields = {}          # node-ID order, smooth
    for case in ("ver","hor","dia","tor"):
        for ax, name in zip(range(3), "xyz"):
            if case == "ver" and name == "x": continue     # absent in h5, like DeepJEB
            fields[f"{case}_{name}_disp(mm)"] = np.sin(verts[:, ax] / 3) * 0.1
        fields[f"{case}_stress(MPa)"] = np.cos(verts[:, 0] / 3) * 100
        fields[f"{case}_resultant_disp(mm)"] = np.abs(np.sin(verts[:, 0] / 3)) * 0.1
    # the bug: reorder vertices surface-first + shuffled, remap cells/faces only
    surf = np.unique(faces); interior = np.setdiff1d(np.arange(len(verts)), surf)
    new2old = np.concatenate([rng.permutation(surf), rng.permutation(interior)])
    old2new = np.empty(len(verts), int); old2new[new2old] = np.arange(len(verts))
    (root / "FieldMesh").mkdir(parents=True, exist_ok=True)
    (root / "VolumeMesh").mkdir(exist_ok=True); (root / "Field").mkdir(exist_ok=True)
    with h5py.File(root / "FieldMesh" / f"{id}.h5", "w") as f:
        f["vertices"] = verts[new2old].astype(np.float32)
        f["cells"] = old2new[cells]; f["faces"] = old2new[faces]
        for k, v in fields.items(): f[f"nodal_variables/{k}"] = v.astype(np.float32)
    grid = pv.UnstructuredGrid({24: cells}, verts)        # node-ID order, like DeepJEB
    grid.save(root / "VolumeMesh" / f"{id}.vtk", binary=True)
    if with_csv:
        hdr = "nodeID,coord_x(mm),coord_y(mm),coord_z(mm),ver_x_disp(mm)"
        rows = np.column_stack([np.arange(1, len(verts)+1), verts,
                                np.sin(verts[:, 0] / 3) * 0.1])
        np.savetxt(root / "Field" / f"{id}.csv", rows, delimiter=",",
                   header=hdr, comments="")
    return verts, cells, faces
```

Tests (each a `def test_*` using a `tmp_path` fixture):
- `test_nodeid_to_h5_recovers_permutation`: build fixture, load h5+vtk, `inv = nodeid_to_h5(...)`; assert `np.allclose(h5_vertices[..], vtk_points[inv-inverse...])` — concretely: `fields_aligned = f["nodal_variables/ver_stress(MPa)"][:][inv]` has `edge_roughness` < 0.3 × the misaligned value.
- `test_nodeid_to_h5_rejects_non_bijection`: pass `vtk_points` with one row perturbed by 1.0 → `ValueError`.
- `test_surface_features`: on a unit-cube-ish fixture, normals are unit norm, `area > 0`, and total area ≈ sum of face areas (1/3 rule ⇒ equality); orientation: `((position - centroid) * normal).sum(1).mean() > 0` (outward on a convex-ish cloud).
- `test_load_design_shapes_and_mask`: `d = load_design(tmp, "1_2")` → all arrays share K rows, `ver_x_valid is True`, `d["ver_disp"][:, 0]` finite; `with_csv=False` variant → `ver_x_valid is False` and `np.isnan(d["ver_disp"][:, 0]).all()`.
- `test_write_and_reread`: run `main`-level `write_store(raw, store_root, ids=["1_2"])`; open with zarr, check arrays + attrs `ver_x_valid`, `n_points`; run again without `--force` → skipped (mtimes unchanged); `make_splits(list of 50 fake ids)` → disjoint, sizes 40/5/5, deterministic across calls.

- [ ] **Step 2:** run — expect ImportError.

- [ ] **Step 3: Implement `surrogate/utils/fetch_deepjeb.py`.** Core pieces:

```python
CASES = ("ver", "hor", "dia", "tor")

def nodeid_to_h5(vtk_points, h5_vertices):
    from scipy.spatial import cKDTree
    d, idx = cKDTree(h5_vertices).query(vtk_points)      # idx[i] = h5 index of node i
    if d.max() > 1e-2 or len(np.unique(idx)) != len(h5_vertices):
        raise ValueError(f"vtk/h5 vertex sets do not match bijectively "
                         f"(max NN dist {d.max():.3g} mm)")
    inv = np.empty(len(idx), np.int64); inv[idx] = np.arange(len(idx))
    return inv                                            # field_h5 = field_nodeid[inv]

def edge_roughness(field, cells):
    e = np.vstack([cells[:, [0, 4]], cells[:, [4, 1]], cells[:, [1, 5]]])
    return float(np.abs(field[e[:, 0]] - field[e[:, 1]]).mean())

def surface_features(vertices, faces):
    K = int(faces.max()) + 1
    tri = vertices[faces]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])   # 2*area*normal
    # outward orientation: signed volume of the closed surface (divergence thm)
    vol6 = np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum()
    if vol6 < 0: cross = -cross
    fa = 0.5 * np.linalg.norm(cross, axis=1)
    vn = np.zeros((K, 3)); va = np.zeros(K)
    for k in range(3):
        np.add.at(vn, faces[:, k], cross)
        np.add.at(va, faces[:, k], fa / 3.0)
    vn /= np.clip(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12, None)
    return vn.astype(np.float32), va[:, None].astype(np.float32)
```

`load_design(raw, id)`: read h5 (`vertices, cells, faces, nodal_variables`), read vtk points via `pyvista.read` (`[:len(vertices)]` — the vtk may carry RBE extra points at the END; verify by the bijection check), compute `inv`, align every field, **assert** for each case `edge_roughness(aligned_resultant) < 0.5 * edge_roughness(stored_resultant)` (measured real gap is ~30×; 0.5 is the refusal line — raise `RuntimeError` naming the id otherwise). `ver_x` from csv column `ver_x_disp(mm)` (locate by header name, not position) if the csv exists, else NaN column + `ver_x_valid=False`. Slice all arrays to `[:K]`, stack disp as `(K,3)` in x,y,z order, apply one `rng = np.random.default_rng(hash(id) % 2**32)` row permutation to every array, return dict.

`write_store(raw, root, ids, force=False)`: `zarr.open_group(root / "deepjeb.zarr", mode="a")`; skip existing groups unless `force`; `create_array(name, shape, dtype="float32", chunks=(65536, width))` then assign (see `../physics-transolver/utils/fetch_drivaerml.py:153,209` for the zarr-3 idioms); set `attrs["ver_x_valid"]`, `attrs["n_points"]`. `make_splits`: `rng = np.random.default_rng(seed)`, shuffle sorted ids, slice test/val/train; `main` writes `splits.json` only if absent.

- [ ] **Step 4:** `uv run pytest tests/test_fetch_deepjeb.py -q` → green; full suite + ruff. Commit: `feat: DeepJEB preprocessing with field-vertex realignment`.

---

### Task 4: Stats script

**Files:**
- Create: `surrogate/utils/stats_deepjeb.py`, `surrogate/tests/test_stats_deepjeb.py`

**Interfaces:**
- Consumes: the zarr store + `splits.json` from Task 3.
- Produces: `$DL_DATA/stats_surface.json` keyed `{var}_mean` / `{var}_std` for `position, normal, area, ver_disp, ver_stress, hor_disp, hor_stress, dia_disp, dia_stress, tor_disp, tor_stress` (lists, one value per column). `ver_disp` stats use only `ver_x_valid` designs. CLI: `uv run python utils/stats_deepjeb.py --root $DL_DATA`.

- [ ] **Step 1: Failing test**: build two fake designs with the Task-3 fixture (`with_csv=True` and `False`), `write_store`, hand-write a `splits.json` putting both in train; run `compute_stats(root)`; assert keys exist, `len(stats["position_mean"]) == 3`, `ver_disp` stats are finite (NaN design excluded), and `area_std > 0`.
- [ ] **Step 2:** run, expect failure. **Step 3:** implement — stream Welford or sum/sumsq per variable over train ids only (`ver_disp` loop skips `ver_x_valid=False` groups; every other variable uses all train designs). **Step 4:** tests + ruff green. **Step 5:** Commit `feat: per-variable train-split stats for the DeepJEB store`.

---

### Task 5: DataModule

**Files:**
- Create: `surrogate/dataset/deepjeb.py`, `surrogate/configs/data/deepjeb_surface.yaml`, `surrogate/tests/test_deepjeb_data.py`

**Interfaces:**
- Consumes: store + `splits.json` + `stats_surface.json`.
- Produces: `dataset.deepjeb.DeepJEBData(DataModule)` with constructor `(root, store="deepjeb.zarr", n_points=16384, samples_per_run=8, batch_size=1, num_workers=0, pin_memory=False, val_split="val", stats=None, seed=0)`; `.stats` dict with `pos_mean/pos_std (3), fx_mean/fx_std (4), y_mean/y_std (16)` assembled in the Global-Constraints channel order (adapt `_load_stats`/`_assemble` from `../physics-transolver/dataset/drivaerml.py:253-277`); items `{"pos": (n,3), "fx": (n,4), "y": (n,16), "y_mask": (16,)}` float32/bool tensors, `y` zero-filled where masked; windows are contiguous slices `start = rng.integers(0, K - n_points + 1)` (clamp `n_points` to min K with a logged warning); split by design id from `splits.json`; `val_split` chooses which split the val loader serves (`"val"` or `"test"`).

- [ ] **Step 1: Failing tests** (fixture: 3 fake designs → `write_store` → hand `splits.json` 1/1/1 → `compute_stats`):
  - shapes/dtypes/keys as above; `n_points` larger than K clamps rather than crashes;
  - no NaN anywhere in `y`; the no-csv design's `y_mask[0] == False` and `y[..., 0] == 0`;
  - determinism: same seed ⇒ same first item; train/val designs disjoint;
  - `stats` property returns 3/4/16-wide vectors matching hand-computed means for one variable;
  - `val_split="test"` serves the test design.
- [ ] **Step 2:** run, expect failure. **Step 3:** implement (model on `drivaerml.py`: per-item `(design_idx, window)` sampling, `samples_per_run` windows per design per epoch; open the zarr group once per worker in `__getitem__` via a cached handle). **Step 4:** config `configs/data/deepjeb_surface.yaml`:

```yaml
_target_: dataset.deepjeb.DeepJEBData
root: ${paths.data_root}
store: deepjeb.zarr
n_points: 16384
samples_per_run: 8
val_split: val
batch_size: 1
num_workers: 0
pin_memory: false
stats: null        # null -> <root>/stats_surface.json
seed: 0
```

- [ ] **Step 5:** full suite + ruff. Commit `feat: DeepJEB two-level-sampling DataModule`.

---

### Task 6: Experiment config + real-data verification

**Files:**
- Create: `surrogate/configs/experiment/jeb_surface.yaml`, `surrogate/STATUS.md`
- Modify: none

**Interfaces:**
- Consumes: everything above; raw DeepJEB_50.
- Produces: populated `$DL_DATA` (store, splits, stats), a trained checkpoint under `surrogate/outputs/jeb_surface/`, measured numbers in `surrogate/STATUS.md`.

- [ ] **Step 1: Experiment config** `configs/experiment/jeb_surface.yaml`:

```yaml
# @package _global_
defaults:
  - override /model: transolver_surface
  - override /data: deepjeb_surface
  - override /sched: cosine

exp_name: jeb_surface
seed: 0

trainer:
  max_steps: 20000
  log_every: 50
  val_every: 500

optim:
  lr: 1.0e-3
```

(Check `configs/sched/cosine.yaml` interpolates `trainer.max_steps`; if it needs `T_max`, set it to 20000 explicitly.)

- [ ] **Step 2: Run the pipeline on real data** (each command must be run and its output read, not assumed):

```bash
export DL_DATA=/home/shared/resources/datasets/JEBsim/processed
uv run python utils/fetch_deepjeb.py --raw /home/shared/resources/datasets/JEBsim/DeepJEB_50 --root $DL_DATA
# expect: 50 designs written, 35 with ver_x, 0 alignment refusals; splits.json 40/5/5
uv run python utils/stats_deepjeb.py --root $DL_DATA
```

- [ ] **Step 3: Overfit gate** — `uv run python train.py experiment=jeb_surface debug=overfit` (single repeated batch); masked rel_l2 must fall well below 0.1 within its budget, else stop and debug (systematic-debugging skill) before any long run.
- [ ] **Step 4: Smoke** — `uv run python train.py experiment=jeb_surface trainer.max_steps=200 trainer.val_every=100` and read the val metrics once.
- [ ] **Step 5: Train** — `uv run python train.py experiment=jeb_surface` (run in background; watch `metrics.jsonl`). On a 32 GB RTX 5090 with N=16384, C=256, L=8, B=1 this fits comfortably; if a step OOMs halve `data.n_points`.
- [ ] **Step 6: Eval on the test split** — `uv run python eval.py ckpt=outputs/jeb_surface/<run>/ckpt/best_weights data.val_split=test`.
- [ ] **Step 7: `surrogate/STATUS.md`** — record: dataset provenance (DeepJEB_50, 50 designs, realignment applied + measured roughness gap), split ids, train/val/test masked rel-L2, per-channel rel-L2, per-case max-stress errors, honest caveat that 50 designs bounds generalisation. **Step 8:** Commit `feat: jeb_surface experiment + first training run results`.

---

## Self-review notes

- Spec coverage: preprocessing (T3), stats (T4), datamodule (T5), model+masked loss (T2), splits (T3), experiment/verification (T6), fork (T1). Mode shapes / scalars / frontend: out of scope per spec.
- Type consistency: `y_mask` is `(16,)` per item, `(B,16)` in batch everywhere; `inv` convention stated identically in T3 interface and code; channel order pinned in Global Constraints and referenced in T2/T4/T5.
- The `ver_x` channel: NaN in store (T3) → zero-filled + mask in dataset (T5) → excluded from stats (T4) → masked in loss/metrics (T2). One mechanism, four touchpoints, all named.
