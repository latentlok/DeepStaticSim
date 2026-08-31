"""app/runner + app/export against synthetic STLs and an injected tiny model.

No real checkpoint is needed: run_job accepts a model instance, so these tests
exercise the whole STL -> features -> windowed prediction -> exports path with a
32-wide Transolver. CPU only -- the GPU belongs to training runs.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # DeepStaticSim root

from app import export, runner  # noqa: E402


@pytest.fixture(scope="module")
def tiny_model():
    from models.transolver import Transolver

    return Transolver(
        net=dict(fun_dim=4, out_dim=16, n_hidden=32, n_head=4, n_layers=2, slice_num=8)
    ).eval()


@pytest.fixture()
def sphere_stl(tmp_path):
    import pyvista as pv

    path = tmp_path / "sphere.stl"
    pv.Sphere(radius=10.0, theta_resolution=48, phi_resolution=48).save(str(path))
    return path


def test_stl_features_sphere(sphere_stl):
    f = runner.stl_features(sphere_stl)
    n = len(f["position"])
    assert f["position"].shape == (n, 3) and f["position"].dtype == np.float32
    assert f["normal"].shape == (n, 3) and f["area"].shape == (n, 1)
    assert np.allclose(np.linalg.norm(f["normal"], axis=1), 1.0, atol=1e-5)
    assert (f["area"] > 0).all()
    true_vol = 4.0 / 3.0 * np.pi * 10.0**3
    assert abs(f["volume_mm3"] - true_vol) / true_vol < 0.05  # tessellation slack
    center = f["position"].mean(0)
    outward = ((f["position"] - center) * f["normal"]).sum(1)
    assert (outward > 0).mean() > 0.9
    # vertex areas sum exactly to the surface area (the 1/3 rule)
    assert np.isclose(f["area"].sum(), f["area_mm2"], rtol=1e-5)


def test_stl_features_merges_duplicate_vertices(tmp_path):
    import pyvista as pv

    path = tmp_path / "cube.stl"
    pv.Cube().triangulate().save(str(path))
    f = runner.stl_features(path)
    # STL stores 3 vertices per facet (12 tris = 36); a merged cube has 8 corners.
    assert len(f["position"]) == 8
    assert abs(f["volume_mm3"] - 1.0) < 1e-6


def test_predict_shape_and_determinism(tiny_model, sphere_stl):
    f = runner.stl_features(sphere_stl)
    a = runner.predict(tiny_model, f, window=512)
    b = runner.predict(tiny_model, f, window=512)
    assert a.shape == (len(f["position"]), 16)
    assert np.isfinite(a).all()
    assert np.array_equal(a, b)  # fixed seed -> bitwise deterministic


def test_run_job_writes_all_artifacts(tiny_model, sphere_stl, tmp_path):
    import pyvista as pv

    out = tmp_path / "job"
    summary = runner.run_job(sphere_stl, out, tiny_model, window=512)

    mesh = pv.read(out / "result.vtp")
    n = mesh.n_points
    for case in export.CASES:
        assert mesh[f"{case}_disp"].shape == (n, 3)
        assert mesh[f"{case}_disp_mag"].shape == (n,)
        assert mesh[f"{case}_stress"].shape == (n,)
    assert mesh["normal"].shape == (n, 3)

    header = (out / "result.csv").read_text().splitlines()[0]
    assert header == ",".join(export.CSV_FIELDS)
    assert len(export.CSV_FIELDS) == 19  # xyz + 16 channels

    on_disk = json.loads((out / "summary.json").read_text())
    assert on_disk == summary
    assert set(summary["cases"]) == {"ver", "hor", "dia", "tor"}
    for case in summary["cases"].values():
        assert case["max_abs_stress_MPa"] >= 0
        assert case["max_resultant_disp_mm"] >= 0
    assert summary["n_points"] == n
    assert (
        "not a certified FEA replacement" in summary["disclaimer"].lower()
        or "Not a certified" in summary["disclaimer"]
    )


def test_cli_help_and_missing_file(tmp_path):
    root = Path(__file__).resolve().parents[2]
    ok = subprocess.run(
        [sys.executable, str(root / "app" / "runner.py"), "--help"], capture_output=True, text=True
    )
    assert ok.returncode == 0 and "--out-dir" in ok.stdout

    bad = subprocess.run(
        [
            sys.executable,
            str(root / "app" / "runner.py"),
            str(tmp_path / "missing.stl"),
            "--out-dir",
            str(tmp_path / "j"),
        ],
        capture_output=True,
        text=True,
    )
    assert bad.returncode == 1
    assert "error:" in bad.stderr
