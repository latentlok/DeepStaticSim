"""DeepJEBData against a miniature store built by the real fetch pipeline.

The fixture reuses tests/test_fetch_deepjeb.make_fake_raw -- the fake raw dataset
with the real ordering bug baked in -- and runs utils.fetch_deepjeb.write_store, so
these tests exercise the exact store layout training will read, not a hand-rolled
imitation of it.
"""

from __future__ import annotations

import json

import torch

from dataset.deepjeb import DeepJEBData
from tests.test_fetch_deepjeb import make_fake_raw
from utils.fetch_deepjeb import write_store

IDS = ["1_2", "3_4", "5_6", "7_8"]  # 5_6 has no csv -> ver_x masked


def _build(tmp_path):
    """raw -> store -> splits.json -> stats_surface.json; returns the data root."""
    raw = tmp_path / "raw"
    make_fake_raw(raw, id="1_2", with_csv=True, seed=0)
    make_fake_raw(raw, id="3_4", with_csv=True, seed=1)
    make_fake_raw(raw, id="5_6", with_csv=False, seed=2)
    make_fake_raw(raw, id="7_8", with_csv=True, seed=3)
    root = tmp_path / "processed"
    write_store(raw, root, ids=IDS)
    (root / "splits.json").write_text(
        json.dumps({"train": ["1_2", "5_6"], "val": ["3_4"], "test": ["7_8"], "seed": 0})
    )
    stats = {}
    for var, w in [("position", 3), ("normal", 3), ("area", 1)]:
        stats[f"{var}_mean"] = [0.0] * w
        stats[f"{var}_std"] = [1.0] * w
    for case in ("ver", "hor", "dia", "tor"):
        stats[f"{case}_disp_mean"] = [0.0] * 3
        stats[f"{case}_disp_std"] = [1.0] * 3
        stats[f"{case}_stress_mean"] = [0.0]
        stats[f"{case}_stress_std"] = [1.0]
    # Sentinels that only correct BY-NAME assembly can place correctly:
    stats["area_mean"] = [42.0]  # -> fx_mean[3]
    stats["tor_stress_mean"] = [7.0]  # -> y_mean[15]
    (root / "stats_surface.json").write_text(json.dumps(stats))
    return root


def _module(root, **kw):
    dm = DeepJEBData(root=root, n_points=kw.pop("n_points", 128), **kw)
    dm.setup("fit")
    return dm


def test_item_shapes_dtypes_keys(tmp_path):
    dm = _module(_build(tmp_path))
    item = dm.train_ds[0]
    assert set(item) == {"pos", "fx", "y", "y_mask"}
    n = item["pos"].shape[0]
    assert item["pos"].shape == (n, 3) and item["pos"].dtype == torch.float32
    assert item["fx"].shape == (n, 4) and item["fx"].dtype == torch.float32
    assert item["y"].shape == (n, 16) and item["y"].dtype == torch.float32
    assert item["y_mask"].shape == (16,) and item["y_mask"].dtype == torch.bool
    batch = next(iter(dm.train_dataloader()))
    assert batch["pos"].ndim == 3 and batch["y_mask"].ndim == 2


def test_no_nan_and_mask_semantics(tmp_path):
    dm = _module(_build(tmp_path))
    seen = {}
    for i in range(len(dm.train_ds)):
        item = dm.train_ds[i]
        assert not torch.isnan(item["y"]).any()
        design = dm.train_ds.designs[i % len(dm.train_ds.designs)]
        seen[design] = item
    assert seen["5_6"]["y_mask"][0].item() is False or not seen["5_6"]["y_mask"][0]
    assert bool(seen["1_2"]["y_mask"].all())
    assert (seen["5_6"]["y"][:, 0] == 0).all()  # masked channel zero-filled
    assert not (seen["1_2"]["y"][:, 0] == 0).all()  # valid ver_x carries data


def test_clamps_oversized_n_points(tmp_path):
    dm = _module(_build(tmp_path), n_points=10**6)
    item = dm.train_ds[0]
    assert item["pos"].shape[0] > 0
    # every item has the SAME length (min over the split), so batches stack
    lengths = {dm.train_ds[i]["pos"].shape[0] for i in range(len(dm.train_ds))}
    assert len(lengths) == 1


def test_determinism_and_split_disjointness(tmp_path):
    root = _build(tmp_path)
    a = _module(root).train_ds[0]
    b = _module(root).train_ds[0]
    assert torch.equal(a["pos"], b["pos"]) and torch.equal(a["y"], b["y"])
    dm = _module(root)
    assert set(dm.train_ds.designs).isdisjoint(dm.val_ds.designs)
    assert dm.val_ds.designs == ["3_4"]


def test_stats_assembled_by_name(tmp_path):
    dm = _module(_build(tmp_path))
    assert len(dm.stats["pos_mean"]) == 3 and len(dm.stats["pos_std"]) == 3
    assert len(dm.stats["fx_mean"]) == 4 and len(dm.stats["y_mean"]) == 16
    assert dm.stats["fx_mean"][3] == 42.0  # area, by name -- not by position
    assert dm.stats["y_mean"][15] == 7.0  # tor_stress lands on channel 15


def test_val_split_test_serves_test_designs(tmp_path):
    dm = _module(_build(tmp_path), val_split="test")
    assert dm.val_ds.designs == ["7_8"]
    batch = next(iter(dm.val_dataloader()))
    assert batch["y"].shape[-1] == 16


def test_windows_vary_within_epoch(tmp_path):
    dm = _module(_build(tmp_path))
    # same design, different item index -> different window (samples_per_run > 1)
    n_designs = len(dm.train_ds.designs)
    a, b = dm.train_ds[0], dm.train_ds[n_designs]
    assert not torch.equal(a["pos"], b["pos"])


def test_train_windows_differ_across_epochs_val_fixed(tmp_path):
    dm = _module(_build(tmp_path))
    # TRAIN: revisiting the same item on a second pass draws a different window.
    first = [dm.train_ds[i]["pos"] for i in range(len(dm.train_ds))]
    second = [dm.train_ds[i]["pos"] for i in range(len(dm.train_ds))]
    assert any(not torch.equal(a, b) for a, b in zip(first, second, strict=True))
    # VAL: the same item is the same window every time, forever.
    v1, v2 = dm.val_ds[0], dm.val_ds[0]
    assert torch.equal(v1["pos"], v2["pos"])
