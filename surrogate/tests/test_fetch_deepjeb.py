"""fetch_deepjeb against a fake raw dataset WITH THE REAL BUG BAKED IN.

DeepJEB's h5 files are internally inconsistent: `nodal_variables` are in OptiStruct
node-ID order (== vtk point order) while `vertices/cells/faces` use a different
vertex order (verified on both downloads; disp edge-roughness ~0.08 stored vs
~0.0028 realigned vs ~0.13 random). The fixture reproduces exactly that, so these
tests fail against a fetch that forgets the realignment -- which is the only way a
loader bug here would ever be caught, because a scrambled field still trains.
"""

from __future__ import annotations

from collections import Counter

import h5py
import numpy as np
import pytest
import zarr

from utils.fetch_deepjeb import (
    CASES,
    edge_roughness,
    load_design,
    make_splits,
    nodeid_to_h5,
    surface_features,
    write_store,
)


def make_fake_raw(root, id="1_2", with_csv=True, seed=0, break_alignment=False):
    """Delaunay tets -> quadratic tets (midside nodes on unique edges) -> boundary
    tris subdivided 4-way (surface nodes include midside nodes, like DeepJEB).
    Fields are smooth functions of position in NODE-ID order; the h5 stores
    vertices in a DIFFERENT order (surface-first, shuffled) with cells/faces
    remapped -- and the fields deliberately NOT remapped. The vtk keeps node-ID
    order. `break_alignment=True` additionally shuffles the fields so no
    permutation can smooth them -- the fetch must refuse such a design."""
    from scipy.spatial import Delaunay

    rng = np.random.default_rng(seed)
    pts = rng.uniform(0, 10, (300, 3))
    tet = Delaunay(pts).simplices
    edges = sorted(
        {
            tuple(sorted((int(t[a]), int(t[b]))))
            for t in tet
            for a, b in [(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)]
        }
    )
    mid_of = {e: len(pts) + i for i, e in enumerate(edges)}
    verts = np.vstack([pts, [(pts[a] + pts[b]) / 2 for a, b in edges]])  # node-ID order

    def m(a, b):
        return mid_of[tuple(sorted((int(a), int(b))))]

    cells = np.array(
        [[a, b, c, d, m(a, b), m(b, c), m(c, a), m(a, d), m(b, d), m(c, d)] for a, b, c, d in tet]
    )
    # Boundary faces with INCONSISTENT winding, like the real data: the sorted()
    # here scrambles orientation deliberately (measured on DeepJEB_50: the raw
    # faces' signed enclosed volume is 2-60% of the true volume, either sign).
    # surface_features must repair orientation from the tets, not trust it.
    fc = Counter(
        tuple(sorted(f))
        for t in tet
        for f in [
            (t[0], t[1], t[2]),
            (t[0], t[1], t[3]),
            (t[0], t[2], t[3]),
            (t[1], t[2], t[3]),
        ]
    )
    faces = []
    for (a, b, c), n in fc.items():
        if n == 1:
            ab, bc, ca = m(a, b), m(b, c), m(c, a)
            faces += [[a, ab, ca], [ab, b, bc], [ca, bc, c], [ab, bc, ca]]
    faces = np.array(faces)

    fields = {}  # node-ID order, smooth in space
    for case in CASES:
        for ax, name in zip(range(3), "xyz", strict=True):
            if case == "ver" and name == "x":
                continue  # absent in the h5, like DeepJEB
            fields[f"{case}_{name}_disp(mm)"] = verts[:, ax] * 0.01
        fields[f"{case}_stress(MPa)"] = (verts[:, 0] - 2 * verts[:, 1] + verts[:, 2]) * 5
        fields[f"{case}_resultant_disp(mm)"] = verts @ np.array([0.01, 0.02, 0.03])
    if break_alignment:
        shuf = rng.permutation(len(verts))
        fields = {k: v[shuf] for k, v in fields.items()}

    # THE BUG: reorder vertices surface-first + shuffled, remap cells/faces ONLY.
    surf = np.unique(faces)
    interior = np.setdiff1d(np.arange(len(verts)), surf)
    new2old = np.concatenate([rng.permutation(surf), rng.permutation(interior)])
    old2new = np.empty(len(verts), int)
    old2new[new2old] = np.arange(len(verts))

    (root / "FieldMesh").mkdir(parents=True, exist_ok=True)
    (root / "VolumeMesh").mkdir(exist_ok=True)
    (root / "Field").mkdir(exist_ok=True)
    with h5py.File(root / "FieldMesh" / f"{id}.h5", "w") as f:
        f["vertices"] = verts[new2old].astype(np.float32)
        f["cells"] = old2new[cells]
        f["faces"] = old2new[faces]
        for k, v in fields.items():
            f[f"nodal_variables/{k}"] = v.astype(np.float32)

    import pyvista as pv

    grid = pv.UnstructuredGrid({24: cells}, verts)  # node-ID order, like DeepJEB
    grid.save(root / "VolumeMesh" / f"{id}.vtk", binary=True)

    if with_csv:
        hdr = "nodeID,coord_x(mm),coord_y(mm),coord_z(mm),junk,ver_x_disp(mm)"
        rows = np.column_stack(
            [
                np.arange(1, len(verts) + 1),
                verts,
                np.full(len(verts), 9.9),  # a column the loader must NOT pick up
                verts[:, 0] * 0.01,
            ]
        )
        np.savetxt(root / "Field" / f"{id}.csv", rows, delimiter=",", header=hdr, comments="")
    return verts, cells, faces, old2new


def test_nodeid_to_h5_recovers_permutation(tmp_path):
    make_fake_raw(tmp_path)
    with h5py.File(tmp_path / "FieldMesh" / "1_2.h5", "r") as f:
        v = f["vertices"][...].astype(np.float64)
        cells = f["cells"][...]
        stored = f["nodal_variables"]["ver_resultant_disp(mm)"][...]
    import pyvista as pv

    cc = np.asarray(pv.read(tmp_path / "VolumeMesh" / "1_2.vtk").points)[: len(v)]
    inv = nodeid_to_h5(cc, v)
    assert edge_roughness(stored[inv], cells) < 0.3 * edge_roughness(stored, cells)


def test_nodeid_to_h5_rejects_non_bijection(tmp_path):
    make_fake_raw(tmp_path)
    with h5py.File(tmp_path / "FieldMesh" / "1_2.h5", "r") as f:
        v = f["vertices"][...].astype(np.float64)
    cc = v.copy()
    cc[0] += 1.0
    with pytest.raises(ValueError, match="bijectiv"):
        nodeid_to_h5(cc, v)


def test_surface_features(tmp_path):
    make_fake_raw(tmp_path)
    with h5py.File(tmp_path / "FieldMesh" / "1_2.h5", "r") as f:
        v = f["vertices"][...].astype(np.float64)
        faces = f["faces"][...]
        cells = f["cells"][...]
    normal, area, vol, surf_idx = surface_features(v, faces, cells)
    K = int(faces.max()) + 1
    assert len(surf_idx) == K  # fixture: same surface node count both ways
    assert normal.shape == (K, 3) and area.shape == (K, 1)
    assert np.allclose(np.linalg.norm(normal, axis=1), 1.0, atol=1e-6)
    assert (area > 0).all()
    tri = v[faces]
    face_area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    assert np.isclose(area.sum(), face_area.sum(), rtol=1e-6)  # the 1/3 rule
    # The fixture's stored winding is scrambled; after repair the enclosed volume
    # must be positive and the normals outward. On this convex-ish blob "outward"
    # means pointing away from the centroid for the overwhelming majority.
    assert vol > 0
    sv = v[surf_idx]
    out = ((sv - sv.mean(0)) * normal).sum(1)
    assert (out > 0).mean() > 0.9


def test_load_design_shapes_and_mask(tmp_path):
    make_fake_raw(tmp_path, with_csv=True)
    d = load_design(tmp_path, "1_2")
    K = d["position"].shape[0]
    widths = {"position": 3, "normal": 3, "area": 1}
    widths |= {f"{c}_disp": 3 for c in CASES} | {f"{c}_stress": 1 for c in CASES}
    for name, w in widths.items():
        assert d[name].shape == (K, w), name
        assert d[name].dtype == np.float32, name
    assert d["ver_x_valid"] is True
    assert np.isfinite(d["ver_disp"]).all()
    assert not np.allclose(d["ver_disp"][:, 0], 9.9)  # header-located, not positional

    make_fake_raw(tmp_path / "b", with_csv=False)
    d2 = load_design(tmp_path / "b", "1_2")
    assert d2["ver_x_valid"] is False
    assert np.isnan(d2["ver_disp"][:, 0]).all()
    assert np.isfinite(d2["ver_disp"][:, 1:]).all()


def test_load_design_realigns_fields(tmp_path):
    """The point of the whole module: the surface stress must be smooth over the
    surface after loading, which it is NOT in the raw h5 ordering."""
    verts, cells, faces, old2new = make_fake_raw(tmp_path)
    d = load_design(tmp_path, "1_2")
    # position rows are a permutation of the surface vertices; match and check the
    # field value at each position equals the node-ID-order truth at that vertex.
    from scipy.spatial import cKDTree

    truth = (verts[:, 0] - 2 * verts[:, 1] + verts[:, 2]) * 5  # ver_stress, node-ID order
    _, idx = cKDTree(verts).query(d["position"].astype(np.float64))
    assert np.allclose(d["ver_stress"][:, 0], truth[idx], atol=1e-3)


def test_load_design_refuses_unfixable_fields(tmp_path):
    make_fake_raw(tmp_path, break_alignment=True)
    with pytest.raises(RuntimeError, match="1_2"):
        load_design(tmp_path, "1_2")


def test_write_store_and_splits(tmp_path):
    raw = tmp_path / "raw"
    make_fake_raw(raw, id="1_2", with_csv=True)
    make_fake_raw(raw / ".unused", id="9_9")  # not passed in ids -> not written
    store_root = tmp_path / "processed"
    write_store(raw, store_root, ids=["1_2"])
    g = zarr.open_group(str(store_root / "deepjeb.zarr"), mode="r")["1_2/surface"]
    assert g.attrs["ver_x_valid"] is True
    assert g.attrs["n_points"] == g["position"].shape[0]
    assert g["ver_disp"].shape == (g.attrs["n_points"], 3)
    # idempotent: second call skips, mtime unchanged
    p = store_root / "deepjeb.zarr" / "1_2" / "surface" / "position"
    files = sorted(p.rglob("*"))
    before = [f.stat().st_mtime_ns for f in files]
    write_store(raw, store_root, ids=["1_2"])
    assert [f.stat().st_mtime_ns for f in sorted(p.rglob("*"))] == before

    ids = [f"{i}_{i}" for i in range(50)]
    s = make_splits(ids, seed=0, n_val=5, n_test=5)
    assert sorted(s["train"] + s["val"] + s["test"]) == sorted(ids)
    assert len(s["val"]) == 5 and len(s["test"]) == 5 and len(s["train"]) == 40
    assert s == make_splits(list(reversed(ids)), seed=0, n_val=5, n_test=5)  # deterministic
