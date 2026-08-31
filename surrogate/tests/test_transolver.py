"""Masked-loss behaviour and the metric surface of the Transolver TaskModule.

The contract tests in test_contracts.py already cover "it instantiates, steps and
backprops" for the transolver_surface config. What is tested here is the one piece
of logic that is new to this fork: the ver_x channel mask. A wrong mask fails
SILENTLY -- the loss stays finite and falls -- so each property is pinned:
masked == plain when everything is valid, and a masked channel contributes neither
loss nor gradient.
"""

from __future__ import annotations

import torch

from engine.base import TrainState
from models.transolver import CASES, STRESS_IDX, Transolver, masked_rel_l2, relative_l2


def _batch(b: int = 2, n: int = 64, mask_ok: bool = True) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(0)
    y = torch.randn(b, n, 16, generator=g)
    mask = torch.ones(b, 16, dtype=torch.bool)
    if not mask_ok:
        mask[:, 0] = False
        y[..., 0] = 0.0  # the dataset zero-fills masked channels
    return {
        "pos": torch.randn(b, n, 3, generator=g),
        "fx": torch.randn(b, n, 4, generator=g),
        "y": y,
        "y_mask": mask,
    }


def _tiny() -> Transolver:
    return Transolver(
        net=dict(fun_dim=4, out_dim=16, n_hidden=32, n_head=4, n_layers=2, slice_num=8)
    )


def test_masked_rel_l2_all_true_equals_plain() -> None:
    b = _batch()
    pred = torch.randn_like(b["y"])
    assert torch.allclose(
        masked_rel_l2(pred, b["y"], b["y_mask"]), relative_l2(pred, b["y"])
    )


def test_masked_channel_does_not_affect_loss_or_grad() -> None:
    b = _batch(mask_ok=False)
    pred = torch.randn_like(b["y"]).requires_grad_()
    loss = masked_rel_l2(pred, b["y"], b["y_mask"])
    loss.backward()
    assert pred.grad is not None
    assert pred.grad[..., 0].abs().max() == 0  # no gradient into the masked channel
    assert pred.grad[..., 1:].abs().max() > 0  # ...while the rest trains
    pred2 = pred.detach().clone()
    pred2[..., 0] += 100.0  # garbage in the masked channel changes nothing
    assert torch.allclose(loss, masked_rel_l2(pred2, b["y"], b["y_mask"]))


def test_missing_mask_means_all_valid() -> None:
    b = _batch()
    del b["y_mask"]
    out = _tiny().training_step(b, TrainState())
    assert out["loss"].isfinite()


def test_training_step_masked_batch() -> None:
    out = _tiny().training_step(_batch(mask_ok=False), TrainState())
    assert out["loss"].ndim == 0 and out["loss"].isfinite()
    assert out["rel_l2"].isfinite()


def test_validation_metrics_surface() -> None:
    val = _tiny().validation_step(_batch(mask_ok=False), TrainState())
    for case in CASES:
        assert f"max_stress/{case}_abs_err" in val
        assert f"max_stress/{case}_rel_err" in val
        assert val[f"max_stress/{case}_abs_err"].isfinite()
    # channel 0 is masked for the whole batch -> no per-channel metric for it
    assert "rel_l2/ch0" not in val
    assert val["rel_l2/ch1"].isfinite()
    assert set(STRESS_IDX.values()) == {3, 7, 11, 15}
