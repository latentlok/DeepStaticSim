"""stats_deepjeb over a store built by the real write_store on the fake raw data.

The properties pinned here are the ones a wrong stats script would get silently
wrong: train-split-only accumulation, the ver_disp channel drawing ONLY from
designs whose csv (the ver_x source) exists -- a NaN from an invalid design would
poison the mean without raising -- and a loud refusal when no train design can
define the ver_x statistics at all.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import zarr

from tests.test_fetch_deepjeb import make_fake_raw
from utils.fetch_deepjeb import write_store
from utils.stats_deepjeb import VARS, compute_stats, main


@pytest.fixture()
def store_root(tmp_path):
    raw = tmp_path / "raw"
    make_fake_raw(raw, id="1_2", with_csv=True, seed=0)
    make_fake_raw(raw, id="3_4", with_csv=False, seed=1)
    root = tmp_path / "processed"
    write_store(raw, root, ids=["1_2", "3_4"])
    (root / "splits.json").write_text(
        json.dumps({"train": ["1_2", "3_4"], "val": [], "test": [], "seed": 0})
    )
    return root


def test_compute_stats_keys_and_values(store_root):
    stats = compute_stats(store_root)
    for var in VARS:
        assert f"{var}_mean" in stats and f"{var}_std" in stats, var
        assert np.isfinite(stats[f"{var}_mean"]).all(), var
        assert np.isfinite(stats[f"{var}_std"]).all(), var
    assert len(stats["position_mean"]) == 3
    assert len(stats["area_mean"]) == 1
    assert len(stats["ver_disp_std"]) == 3
    assert all(s > 0 for s in stats["area_std"])

    z = zarr.open_group(str(store_root / "deepjeb.zarr"), mode="r")
    # every variable but ver_disp pools ALL train designs...
    both = np.concatenate(
        [np.asarray(z[d]["surface"]["area"][:], dtype=np.float64) for d in ("1_2", "3_4")]
    )
    assert np.allclose(stats["area_mean"], both.mean(0), rtol=1e-6)
    assert np.allclose(stats["area_std"], both.std(0), rtol=1e-6)
    # ...while ver_disp uses only the design whose csv exists (3_4's column 0 is NaN)
    valid_only = np.asarray(z["1_2"]["surface"]["ver_disp"][:], dtype=np.float64)
    assert np.allclose(stats["ver_disp_mean"], valid_only.mean(0), rtol=1e-6)
    assert np.allclose(stats["ver_disp_std"], valid_only.std(0), rtol=1e-6)


def test_refuses_when_no_train_design_has_ver_x(store_root):
    (store_root / "splits.json").write_text(
        json.dumps({"train": ["3_4"], "val": [], "test": [], "seed": 0})
    )
    with pytest.raises(SystemExit, match="ver_x"):
        compute_stats(store_root)


def test_refuses_missing_design(store_root):
    (store_root / "splits.json").write_text(
        json.dumps({"train": ["1_2", "9_9"], "val": [], "test": [], "seed": 0})
    )
    with pytest.raises(SystemExit, match="9_9"):
        compute_stats(store_root)


def test_cli_writes_json(store_root):
    assert main(["--root", str(store_root)]) == 0
    out = store_root / "stats_surface.json"
    assert out.exists()
    stats = json.loads(out.read_text())
    assert set(stats) == {f"{v}_{k}" for v in VARS for k in ("mean", "std")}
    assert all(isinstance(x, float) for v in stats.values() for x in v)
