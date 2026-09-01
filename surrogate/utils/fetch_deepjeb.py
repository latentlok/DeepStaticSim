"""DeepJEB_50 raw files -> $DL_DATA/deepjeb.zarr + splits.json.

    uv run python utils/fetch_deepjeb.py \
        --raw $DEEPJEB_RAW --root $DL_DATA

Per design this reads FieldMesh/<id>.h5 (mesh + fields), VolumeMesh/<id>.vtk
(coordinates in node-ID order) and, when present, Field/<id>.csv (the only source
of ver_x_disp), and writes one zarr group of row-aligned surface arrays:

    deepjeb.zarr/<id>/surface/{position, normal, area,
                               ver_disp, ver_stress, hor_disp, hor_stress,
                               dia_disp, dia_stress, tor_disp, tor_stress}

THE REALIGNMENT IS THE POINT OF THIS FILE. DeepJEB's h5 is internally
inconsistent (verified on two independent downloads, two designs each):
`nodal_variables` are stored in OptiStruct node-ID order -- identical to the csv
rows, the .fem GRID order and the .vtk points -- while `vertices`/`cells`/`faces`
use a different vertex order. Used raw, every field is effectively scrambled
against the mesh: mean |disp difference| across tet edges measures ~0.08 as
stored vs ~0.0028 realigned vs ~0.13 under a random permutation. The fix is to
KD-tree-match the vtk points (node-ID order) onto the h5 vertices and permute the
FIELDS into h5 vertex order, so `faces`/`cells` stay valid. `load_design` then
refuses any design whose realigned field is not dramatically smoother than the
stored one, so a silently wrong alignment can never reach training -- a scrambled
field still trains, which is exactly why the check must live here and be loud.

Rows are permuted once at write time (per-design seed), so a contiguous window of
the store is a uniform random sample of the surface AND costs one chunk read --
the same trade as the DrivAerML store this repo's conventions come from.

ver_x_disp: absent from the h5, present only in the csv. Designs without a csv
get NaN in that column and attrs["ver_x_valid"] = False; the dataset masks the
channel. Re-run with `--only <ids> --force` once the missing csvs arrive.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import zlib
from pathlib import Path

import h5py
import numpy as np

log = logging.getLogger(__name__)

CASES = ("ver", "hor", "dia", "tor")
PERM_SEED = 1234
CHUNK = 65536
# The refusal line for the smoothness check. Measured on real data the realigned /
# stored roughness ratio is ~0.03; anything above 0.5 means the permutation did not
# actually fix the field (e.g. the two files describe different meshes).
ROUGHNESS_RATIO = 0.5


def nodeid_to_h5(nodeid_points: np.ndarray, h5_vertices: np.ndarray) -> np.ndarray:
    """Match node-ID-order coordinates onto h5 vertex order.

    Returns `inv` such that `field_h5order = field_nodeidorder[inv]`. Raises unless
    the two point sets match bijectively to within 1e-2 mm -- a partial or sloppy
    match would silently scramble fields, which is the disease being cured.
    """
    from scipy.spatial import cKDTree

    if nodeid_points.shape != h5_vertices.shape:
        raise ValueError(
            f"point sets differ in shape: {nodeid_points.shape} vs {h5_vertices.shape} "
            f"-- cannot be bijective"
        )
    d, idx = cKDTree(h5_vertices).query(nodeid_points)
    if d.max() > 1e-2 or len(np.unique(idx)) != len(h5_vertices):
        raise ValueError(
            f"vtk/h5 vertex sets do not match bijectively "
            f"(max NN dist {d.max():.3g} mm, {len(np.unique(idx))}/{len(h5_vertices)} "
            f"unique matches)"
        )
    inv = np.empty(len(idx), np.int64)
    inv[idx] = np.arange(len(idx))
    return inv


def edge_roughness(field: np.ndarray, cells: np.ndarray) -> float:
    """Mean |field difference| across corner-midside tet edges.

    On a correctly ordered nodal field this is small (neighbours agree); on a
    scrambled one it approaches the random-pair level. The three edge families
    sampled here are plenty -- the statistic only has to separate ~0.003 from ~0.08.
    """
    e = np.vstack([cells[:, [0, 4]], cells[:, [4, 1]], cells[:, [1, 5]]])
    return float(np.abs(field[e[:, 0]] - field[e[:, 1]]).mean())


_FACE_LOCAL = (
    # (corner local idxs, opposite local idx, midside cols for edges (i,j),(j,k),(k,i))
    ((0, 1, 2), 3, (4, 5, 6)),
    ((0, 1, 3), 2, (4, 8, 7)),
    ((1, 2, 3), 0, (5, 9, 8)),
    ((0, 2, 3), 1, (6, 9, 7)),
)


def boundary_faces(vertices: np.ndarray, cells: np.ndarray) -> np.ndarray:
    """Outward-oriented triangulation of the tet-mesh boundary, built FROM THE CELLS.

    The stored `faces` cannot be used for normals: measured on DeepJEB_50 their
    winding is per-face garbage (signed enclosed volume 2-60% of the labelled
    volume, random sign), and they are a free re-triangulation of the surface
    nodes that does not follow tet faces (triangles mix midside nodes of 2-6
    distinct parent corners), so no local repair against an owning tet exists.
    The tets themselves are unambiguous: a corner triple appearing in exactly one
    tet is a boundary face; its outward direction is away from that tet's fourth
    corner. Each quadratic boundary face is emitted as 4 sub-triangles through its
    midside nodes, so the triangulation covers exactly the corner+midside surface
    node set the h5 uses.
    """
    tris = np.concatenate([cells[:, list(f)] for f, _, _ in _FACE_LOCAL])
    opps = np.concatenate([cells[:, o] for _, o, _ in _FACE_LOCAL])
    mids = np.concatenate([cells[:, list(m)] for _, _, m in _FACE_LOCAL])
    key = np.sort(tris, axis=1)
    _, inverse, counts = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    on_boundary = counts[inverse] == 1
    tris, opps, mids = tris[on_boundary].copy(), opps[on_boundary], mids[on_boundary].copy()

    p = vertices[tris]
    n_vec = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    inward = np.einsum("ij,ij->i", n_vec, p.mean(1) - vertices[opps]) < 0
    tris[inward] = tris[inward][:, [0, 2, 1]]
    # swapping corners b<->c maps edges (ab, bc, ca) -> (ac, cb, ba) = (ca, bc, ab)
    mids[inward] = mids[inward][:, [2, 1, 0]]

    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    ab, bc, ca = mids[:, 0], mids[:, 1], mids[:, 2]
    return np.concatenate(
        [
            np.stack([a, ab, ca], 1),
            np.stack([ab, b, bc], 1),
            np.stack([ca, bc, c], 1),
            np.stack([ab, bc, ca], 1),
        ]
    )


def surface_features(
    vertices: np.ndarray, faces: np.ndarray, cells: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """Surface node indices, per-node outward unit normal, lumped area, volume.

    Returns (normal (K,3), area (K,1), volume mm^3, surf_idx (K,)) where
    `surf_idx` are h5-vertex indices of the surface nodes, sorted ascending, and
    normal/area rows follow that order.

    EVERYTHING comes from the tets. The stored `faces` are unusable for geometry:
    their winding is per-face garbage AND their indices live in a compact
    surface-local numbering, not h5 vertex indices (measured: the boundary node
    set rebuilt from the cells has the same size as the stored faces' node set
    but different indices, and taking the stored faces against h5 vertices gives
    a signed volume 2-60% of the labelled volume with random sign). They serve
    only as a cross-check that both descriptions agree on the surface node COUNT.
    Normals are area-weighted means of incident rebuilt-face normals; the vertex
    area is 1/3 of each incident face's area, so vertex areas sum exactly to the
    surface area. The returned volume (divergence theorem over the closed
    oriented surface) is the caller's cross-check against bracket_labels.csv --
    a wrong orientation or a wrong node set collapses it, loudly.
    """
    sub = boundary_faces(vertices, cells)
    surf_idx = np.unique(sub)
    n_stored = len(np.unique(faces))
    if len(surf_idx) != n_stored:
        raise ValueError(
            f"boundary rebuilt from cells has {len(surf_idx)} nodes but the stored "
            f"faces reference {n_stored} -- the two surface descriptions disagree"
        )
    compact = np.full(int(surf_idx[-1]) + 1, -1, np.int64)
    compact[surf_idx] = np.arange(len(surf_idx))
    sub_c = compact[sub]

    tri = vertices[sub].astype(np.float64)
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])  # 2*area*normal
    volume = np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() / 6.0
    if volume <= 0:
        raise ValueError(f"non-positive enclosed volume {volume:.3g} -- orientation is wrong")
    fa = 0.5 * np.linalg.norm(cross, axis=1)
    K = len(surf_idx)
    vn = np.zeros((K, 3))
    va = np.zeros(K)
    for k in range(3):
        np.add.at(vn, sub_c[:, k], cross)
        np.add.at(va, sub_c[:, k], fa / 3.0)
    vn /= np.clip(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12, None)
    return vn.astype(np.float32), va[:, None].astype(np.float32), float(volume), surf_idx


def _csv_ver_x(csv_path: Path) -> np.ndarray:
    """The ver_x_disp column, located BY HEADER NAME -- a positional index would
    silently read a different column the day the csv layout changes."""
    with open(csv_path) as fh:
        header = fh.readline().strip().split(",")
    col = header.index("ver_x_disp(mm)")
    return np.loadtxt(csv_path, delimiter=",", skiprows=1, usecols=col)


def load_design(raw: Path, design: str) -> dict:
    """One design: realigned, surface-only, row-permuted arrays in raw units."""
    raw = Path(raw)
    with h5py.File(raw / "FieldMesh" / f"{design}.h5", "r") as f:
        vertices = f["vertices"][...].astype(np.float64)
        cells = f["cells"][...]
        faces = f["faces"][...]
        nodal = {k: v[...] for k, v in f["nodal_variables"].items()}

    import pyvista as pv

    # The vtk may carry a handful of extra points (RBE reference nodes) at the END;
    # the first len(vertices) rows are the mesh nodes in node-ID order (verified).
    points = np.asarray(pv.read(raw / "VolumeMesh" / f"{design}.vtk").points)[: len(vertices)]
    inv = nodeid_to_h5(points, vertices)

    # Realign every field into h5 vertex order, then PROVE it worked: the realigned
    # resultant displacement must be far smoother across mesh edges than the stored
    # one. Refuse the design otherwise -- a scrambled field trains without error.
    aligned = {k: v[inv] for k, v in nodal.items()}
    for case in CASES:
        key = f"{case}_resultant_disp(mm)"
        rough_stored = edge_roughness(nodal[key], cells)
        rough_aligned = edge_roughness(aligned[key], cells)
        if rough_aligned >= ROUGHNESS_RATIO * rough_stored:
            raise RuntimeError(
                f"design {design}: realignment did not smooth {key} "
                f"(stored {rough_stored:.4g}, realigned {rough_aligned:.4g}). The vtk "
                f"and h5 disagree about this design -- refusing to write it."
            )

    normal, area, volume, surf_idx = surface_features(vertices, faces, cells)

    csv_path = raw / "Field" / f"{design}.csv"
    ver_x_valid = csv_path.exists()
    if ver_x_valid:
        ver_x = _csv_ver_x(csv_path)[inv]
    else:
        ver_x = np.full(len(vertices), np.nan)

    out: dict = {
        "position": vertices[surf_idx].astype(np.float32),
        "normal": normal,
        "area": area,
        "ver_x_valid": bool(ver_x_valid),
        "volume_mm3": volume,
    }
    for case in CASES:
        cols = [
            ver_x if (case == "ver" and ax == "x") else aligned[f"{case}_{ax}_disp(mm)"]
            for ax in "xyz"
        ]
        out[f"{case}_disp"] = np.stack(cols, axis=1)[surf_idx].astype(np.float32)
        out[f"{case}_stress"] = aligned[f"{case}_stress(MPa)"][surf_idx, None].astype(np.float32)

    # One permutation for every array, so rows stay aligned and a contiguous window
    # of the store is a uniform sample of the surface.
    perm = np.random.default_rng(PERM_SEED + zlib.crc32(design.encode())).permutation(len(surf_idx))
    for k, v in out.items():
        if isinstance(v, np.ndarray):
            out[k] = np.ascontiguousarray(v[perm])
    return out


def _read_labels(raw: Path) -> dict[str, float]:
    """item_name -> volume(mm3) from bracket_labels.csv, {} if the file is absent."""
    import csv as csvmod

    path = Path(raw) / "bracket_labels.csv"
    if not path.exists():
        return {}
    with open(path) as fh:
        return {r["item_name"]: float(r["volume(mm3)"]) for r in csvmod.DictReader(fh)}


def write_store(raw: Path, root: Path, ids: list[str], force: bool = False) -> list[str]:
    """Write each design's surface group; returns the ids actually written."""
    import zarr

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    store = zarr.open_group(str(root / "deepjeb.zarr"), mode="a")
    labels = _read_labels(raw)
    written = []
    for i, design in enumerate(ids, 1):
        if not force and design in store:
            log.info("[%d/%d] %s already in store, skipping", i, len(ids), design)
            continue
        t0 = time.time()
        d = load_design(Path(raw), design)
        grp = store.require_group(design).require_group("surface")
        n = d["position"].shape[0]
        for name, arr in d.items():
            if not isinstance(arr, np.ndarray):
                continue
            z = grp.create_array(
                name,
                shape=arr.shape,
                chunks=(min(CHUNK, arr.shape[0]),) + arr.shape[1:],
                dtype="float32",
                overwrite=True,
            )
            z[:] = arr
        grp.attrs.update(
            n_points=n,
            ver_x_valid=d["ver_x_valid"],
            permuted=True,
            volume_mm3=d["volume_mm3"],
        )
        # Independent geometry check: the enclosed volume of the repaired surface
        # against the dataset's own label. A wrong orientation repair or a wrong
        # vertex matching collapses this ratio -- it cannot drift a little.
        if design in labels:
            ratio = d["volume_mm3"] / labels[design]
            if abs(ratio - 1.0) > 0.05:
                raise RuntimeError(
                    f"design {design}: enclosed volume {d['volume_mm3']:.0f} mm3 is "
                    f"{ratio:.3f}x the labelled {labels[design]:.0f} mm3 -- geometry "
                    f"processing is wrong, refusing to continue."
                )
        written.append(design)
        log.info(
            "[%d/%d] %s: %d surface nodes, ver_x=%s, vol %.0f mm3%s, %.1fs",
            i,
            len(ids),
            design,
            n,
            d["ver_x_valid"],
            d["volume_mm3"],
            f" ({d['volume_mm3'] / labels[design]:.4f}x label)" if design in labels else "",
            time.time() - t0,
        )
    return written


def make_splits(ids: list[str], seed: int = 0, n_val: int = 5, n_test: int = 5) -> dict:
    """Deterministic design-level split. Sorted first, so the caller's ordering is
    irrelevant; points from one design never straddle splits by construction."""
    ids = sorted(ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    return {
        "test": sorted(ids[:n_test]),
        "val": sorted(ids[n_test : n_test + n_val]),
        "train": sorted(ids[n_test + n_val :]),
        "seed": seed,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--raw", type=Path, required=True, help="DeepJEB_50 directory")
    p.add_argument("--root", type=Path, required=True, help="processed data root ($DL_DATA)")
    p.add_argument("--only", nargs="*", default=None, help="subset of design ids")
    p.add_argument("--force", action="store_true", help="rewrite designs already in the store")
    p.add_argument("--n-val", type=int, default=4, help="designs held out for validation")
    p.add_argument("--n-test", type=int, default=4, help="designs held out for test")
    a = p.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    all_ids = sorted(f.stem for f in (a.raw / "FieldMesh").glob("*.h5"))
    ids = a.only if a.only else all_ids
    missing = [i for i in ids if i not in all_ids]
    if missing:
        raise SystemExit(f"no FieldMesh h5 for: {missing}")

    written = write_store(a.raw, a.root, ids, force=a.force)
    log.info("%d design(s) written, %d skipped", len(written), len(ids) - len(written))

    splits_path = a.root / "splits.json"
    if splits_path.exists():
        log.info("splits.json already exists, leaving it alone")
    else:
        # Split over COMPLETE designs only (csv present, so all 16 channels are
        # real): the user's call -- no zero-filled ver_x in any split. Designs
        # without a csv are still written to the store but stay unsplit until
        # their csvs arrive and the split is deliberately regenerated.
        complete = [i for i in all_ids if (a.raw / "Field" / f"{i}.csv").exists()]
        log.info("%d/%d designs are csv-complete; splitting those", len(complete), len(all_ids))
        splits = make_splits(complete, n_val=a.n_val, n_test=a.n_test)
        splits_path.write_text(json.dumps(splits, indent=2))
        log.info(
            "wrote %s: %d train / %d val / %d test",
            splits_path,
            len(splits["train"]),
            len(splits["val"]),
            len(splits["test"]),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
