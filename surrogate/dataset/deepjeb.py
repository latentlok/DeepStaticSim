"""Two-level sampling over the DeepJEB zarr store: pick a design, then pick points.

The store is written by `utils/fetch_deepjeb.py`; what matters here is one property
of it: **rows were permuted once at write time**, one permutation per design, so a
contiguous slice

    arr[start : start + n_points]

is a uniform random sample of that design's surface AND costs one zarr chunk read.

Deliberate choices, mirrored from the DrivAerML pipeline this repo's conventions
come from:

  1. Designs have DIFFERENT surface node counts. `n_points` is clamped to the
     smallest design in the split (with a logged warning), so every item has the
     same length and batches stack without ragged shapes.
  2. setup() NEVER scans data for statistics; it reads stats_surface.json, written
     once by `utils/stats_deepjeb.py`, keyed BY VARIABLE NAME so reordering the
     channel list in code can never silently pair a variable with another
     variable's mean.
  3. The split is BY DESIGN, read from splits.json (written once by
     fetch_deepjeb.py) -- points from one bracket never straddle train/val/test.
  4. Windows are a pure function of (seed, item index). Validation must be
     comparable across evaluations; for training the price is that epochs repeat
     the same `samples_per_run` windows per design, which at 8 windows x 16k
     points against ~65k surface nodes still covers each design about twice over.

The ver_x channel: designs whose csv was missing at fetch time store NaN in
`ver_disp[:, 0]` and `attrs["ver_x_valid"] = False`. Here that becomes
`y_mask[0] = False` and the NaNs are ZERO-FILLED, so no NaN can ever reach the
model -- the masked loss (models/transolver.py) guarantees the zeros train nothing.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from engine.base import DataModule

log = logging.getLogger(__name__)

# (variable, width) in the FIXED channel order shared with models/transolver.py
# (CASES / STRESS_IDX) and utils/stats_deepjeb.py. y_mask index 0 is ver_disp x.
INPUT_VARS = (("normal", 3), ("area", 1))
TARGET_VARS = (
    ("ver_disp", 3),
    ("ver_stress", 1),
    ("hor_disp", 3),
    ("hor_stress", 1),
    ("dia_disp", 3),
    ("dia_stress", 1),
    ("tor_disp", 3),
    ("tor_stress", 1),
)
N_CHANNELS = sum(w for _, w in TARGET_VARS)  # 16


def open_store(path: Path) -> Any:
    """Open the zarr store read-only, with an error that says what to fix."""
    import zarr

    try:
        return zarr.open_group(str(path), mode="r")
    except (FileNotFoundError, KeyError) as e:
        raise FileNotFoundError(
            f"no zarr store at {path}. The data lives outside the repo -- point DL_DATA "
            f"(or paths.data_root) at the directory holding deepjeb.zarr, and run "
            f"`python utils/fetch_deepjeb.py` if it is not there yet."
        ) from e


class DesignPointSamples(Dataset):
    """One item = one contiguous window of surface points from one design."""

    def __init__(
        self,
        path: Path,
        designs: list[str],
        n_points: int,
        samples_per_run: int,
        seed: int,
    ) -> None:
        self.path, self.designs = path, list(designs)
        self.n_points, self.samples_per_run, self.seed = n_points, samples_per_run, seed
        self._store: Any = None  # opened lazily: zarr handles do not survive forking

    def __len__(self) -> int:
        return len(self.designs) * self.samples_per_run

    def _group(self, design: str) -> Any:
        if self._store is None:
            self._store = open_store(self.path)
        return self._store[design]["surface"]

    def _read(self, g: Any, name: str, start: int) -> np.ndarray:
        block = np.atleast_1d(g[name][start : start + self.n_points])
        return block if block.ndim > 1 else block[:, None]

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        design = self.designs[item % len(self.designs)]
        g = self._group(design)
        n_rows = g["position"].shape[0]  # zarr 3 arrays have no __len__
        # A pure function of (seed, item): validation metrics stay comparable across
        # evaluations, and a run is reproducible bit-for-bit from its config.
        rng = np.random.default_rng(self.seed * 1_000_003 + item)
        start = int(rng.integers(0, n_rows - self.n_points + 1))

        pos = self._read(g, "position", start)
        fx = np.concatenate([self._read(g, n, start) for n, _ in INPUT_VARS], axis=-1)
        y = np.concatenate([self._read(g, n, start) for n, _ in TARGET_VARS], axis=-1)

        mask = np.ones(N_CHANNELS, dtype=bool)
        if not g.attrs.get("ver_x_valid", True):
            mask[0] = False
        # Zero-fill AFTER masking is decided: the only NaNs in the store are the
        # masked ver_x column, and nothing NaN may ever reach the model.
        y = np.nan_to_num(y, nan=0.0)

        return {
            "pos": torch.from_numpy(np.ascontiguousarray(pos)),
            "fx": torch.from_numpy(np.ascontiguousarray(fx)),
            "y": torch.from_numpy(np.ascontiguousarray(y)),
            "y_mask": torch.from_numpy(mask),
        }


class DeepJEBData(DataModule):
    """<data_root>/deepjeb.zarr, split by design (splits.json), sampled by point."""

    def __init__(
        self,
        root: str | Path = "data",
        store: str = "deepjeb.zarr",
        n_points: int = 16384,
        samples_per_run: int = 8,
        batch_size: int = 1,
        num_workers: int = 0,
        pin_memory: bool = False,
        val_split: str = "val",
        stats: str | Path | None = None,
        seed: int = 0,
    ) -> None:
        # __init__ touches NO files -- tests/test_configs.py instantiates every config
        # in configs/data/ on machines where the store is not mounted.
        if val_split not in ("val", "test"):
            raise ValueError(f"val_split must be 'val' or 'test', got {val_split!r}")
        self.root = Path(root).expanduser()
        self.store_name = store
        self.n_points, self.samples_per_run = n_points, samples_per_run
        self.batch_size, self.num_workers, self.pin_memory = batch_size, num_workers, pin_memory
        self.val_split, self.seed = val_split, seed
        self.stats_path = Path(stats).expanduser() if stats else None
        self.stats: dict[str, list[float]] = {}
        self.train_ds: DesignPointSamples | None = None
        self.val_ds: DesignPointSamples | None = None

    # -- statistics ------------------------------------------------------------------

    def _load_stats(self) -> None:
        """stats_surface.json -> the pos_/fx_/y_ vectors on_data_ready copies out.

        On disk it is keyed `{variable}_{mean,std}`, so reordering or subsetting the
        channel lists in code cannot silently pair a variable with another
        variable's statistic. Assembly concatenates in INPUT_VARS/TARGET_VARS order
        -- the same order __getitem__ concatenates the arrays.
        """
        path = self.stats_path or self.root / "stats_surface.json"
        if not path.exists():
            log.warning(
                "no %s -- the model gets no normalisation and trains on raw units "
                "(stress spans hundreds of MPa, displacement fractions of a mm). "
                "Run `python utils/stats_deepjeb.py --root %s` once.",
                path,
                self.root,
            )
            return
        per_var = json.loads(path.read_text())
        groups = (("pos", (("position", 3),)), ("fx", INPUT_VARS), ("y", TARGET_VARS))
        for prefix, variables in groups:
            for key in ("mean", "std"):
                vec: list[float] = []
                for name, width in variables:
                    if f"{name}_{key}" not in per_var:
                        raise KeyError(
                            f"{path} has no '{name}_{key}'. It is keyed by variable "
                            f"name; rerun `python utils/stats_deepjeb.py`."
                        )
                    vals = [float(v) for v in np.atleast_1d(per_var[f"{name}_{key}"])]
                    if len(vals) != width:
                        raise ValueError(
                            f"{path}: '{name}_{key}' has {len(vals)} values, expected {width}"
                        )
                    vec.extend(vals)
                self.stats[f"{prefix}_{key}"] = vec
        log.info("loaded stats from %s", path)

    # -- contract --------------------------------------------------------------------

    def setup(self, stage: str) -> None:
        self._load_stats()
        path = self.root / self.store_name
        store = open_store(path)

        splits_path = self.root / "splits.json"
        if not splits_path.exists():
            raise FileNotFoundError(
                f"no {splits_path} -- utils/fetch_deepjeb.py writes it; the split is "
                f"never invented at train time, or two runs could disagree about it."
            )
        splits = json.loads(splits_path.read_text())
        train, held = list(splits["train"]), list(splits[self.val_split])
        if overlap := set(train) & set(held):
            raise ValueError(f"designs in both splits: {sorted(overlap)} -- that leaks")
        for design in train + held:
            if design not in store:
                raise KeyError(f"splits.json names {design!r} but the store has no such group")

        # Clamp the window to the smallest design in play so every item has the same
        # length and batches stack. Shape metadata only -- no chunk is read.
        n_min = min(store[d]["surface"]["position"].shape[0] for d in train + held)
        n_points = self.n_points
        if n_points > n_min:
            log.warning(
                "n_points=%d exceeds the smallest design's %d surface nodes; clamping",
                n_points,
                n_min,
            )
            n_points = n_min

        log.info(
            "%d train / %d %s designs (%d points x %d samples each)",
            len(train),
            len(held),
            self.val_split,
            n_points,
            self.samples_per_run,
        )
        common = dict(path=path, n_points=n_points, seed=self.seed)
        if stage == "fit":
            self.train_ds = DesignPointSamples(
                designs=train, samples_per_run=self.samples_per_run, **common
            )
        # ONE fixed window per held-out design, so the metric is comparable design
        # to design and evaluation to evaluation.
        self.val_ds = DesignPointSamples(designs=held, samples_per_run=1, **common)

    def _loader(self, ds: Dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=_collate,
            persistent_workers=self.num_workers > 0,
        )

    def train_dataloader(self) -> DataLoader:
        assert self.train_ds is not None, "call setup('fit') first"
        return self._loader(self.train_ds, shuffle=True)

    def val_dataloader(self) -> DataLoader | None:
        # Never shuffled: a val metric must be comparable across evaluations.
        return self._loader(self.val_ds, shuffle=False) if self.val_ds is not None else None


def _collate(samples: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """List of items -> one batch. Every item is already n_points long, so this stacks."""
    return {k: torch.stack([s[k] for s in samples]) for k in samples[0]}
