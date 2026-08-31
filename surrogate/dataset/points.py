"""Synthetic point clouds shaped like the DeepJEB surface task.

Exists so the transolver_surface contract tests run without the DeepJEB store
mounted. The target is a deterministic smooth function of position, so a model that
trains sees the loss fall and one that is wired up wrong does not -- a check, not a
benchmark. Replace with dataset/deepjeb.py for real runs.

The one DeepJEB-specific twist is `mask_channel`: odd-indexed items mark that target
channel invalid (`y_mask` False, values zero-filled), mirroring the 15/50 designs
whose csv -- the only source of ver_x_disp -- is missing. Contract tests therefore
exercise the masked-loss path for free.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from engine.base import DataModule


class _PointClouds(Dataset):
    """One item per "geometry": n_points sampled points and their field values."""

    def __init__(
        self,
        n_items: int,
        n_points: int,
        fun_dim: int,
        out_dim: int,
        mask_channel: int | None,
        seed: int,
    ) -> None:
        self.n_items, self.n_points = n_items, n_points
        self.fun_dim, self.out_dim = fun_dim, out_dim
        self.mask_channel, self.seed = mask_channel, seed

    def __len__(self) -> int:
        return self.n_items

    def __getitem__(self, i: int) -> dict[str, Tensor]:
        g = torch.Generator().manual_seed(self.seed * 10_000 + i)
        pos = torch.rand(self.n_points, 3, generator=g) * 200 - 100  # bracket-ish mm
        phase = torch.arange(self.out_dim, dtype=torch.float32)[None, :]
        y = torch.sin(pos.sum(-1, keepdim=True) / 50 + phase) * 100.0
        mask = torch.ones(self.out_dim, dtype=torch.bool)
        if self.mask_channel is not None and i % 2 == 1:
            mask[self.mask_channel] = False
            y[..., self.mask_channel] = 0.0  # zero-filled, exactly like the real store
        item = {"pos": pos, "y": y, "y_mask": mask}
        if self.fun_dim:
            item["fx"] = torch.rand(self.n_points, self.fun_dim, generator=g)
        return item


class PointCloudData(DataModule):
    """Two-level sampling in miniature: an item is a geometry, not a row."""

    def __init__(
        self,
        n_points: int = 256,
        fun_dim: int = 4,
        out_dim: int = 16,
        mask_channel: int | None = 0,
        n_train: int = 8,
        n_val: int = 2,
        batch_size: int = 2,
        num_workers: int = 0,
        seed: int = 0,
    ) -> None:
        self.n_points, self.fun_dim, self.out_dim = n_points, fun_dim, out_dim
        self.mask_channel = mask_channel
        self.n_train, self.n_val = n_train, n_val
        self.batch_size, self.num_workers, self.seed = batch_size, num_workers, seed
        self.stats: dict[str, list[float]] = {}

    def setup(self, stage: str) -> None:
        # Statistics the model copies into buffers via on_data_ready. Real runs read
        # these from stats_surface.json; here they are the known generating constants.
        self.stats = {
            "pos_mean": [0.0, 0.0, 0.0],
            "pos_std": [57.7, 57.7, 57.7],
            "y_mean": [0.0] * self.out_dim,
            "y_std": [70.7] * self.out_dim,
            "fx_mean": [0.5] * max(self.fun_dim, 1),
            "fx_std": [0.289] * max(self.fun_dim, 1),
        }
        self.train_ds = _PointClouds(
            self.n_train, self.n_points, self.fun_dim, self.out_dim, self.mask_channel, self.seed
        )
        self.val_ds = _PointClouds(
            self.n_val, self.n_points, self.fun_dim, self.out_dim, self.mask_channel, self.seed + 1
        )

    def _loader(self, ds: Dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=lambda s: {k: torch.stack([x[k] for x in s]) for k in s[0]},
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_ds, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_ds, shuffle=False)
