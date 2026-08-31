"""Transolver: a transformer whose attention runs over learned slices, not points.

Ported from ../physics-transolver/models/transolver.py, itself a port of
thuml/Transolver (Wu et al., ICML 2024). The mechanism is the reference one --
slice, attend, deslice, learned temperature, orthogonal init on the slice
projection -- and the reasons it is written the way it is live in that file's
docstrings. What this fork DELETES from the physics version:

  * physics residuals, `freeze_context`, loss weighting -- structural statics here
    has no PDE residual term; the data loss is the loss.
  * drag/force integration -- a bracket has no Cd.

What it ADDS: a 16-channel target with ONE maskable channel. `ver_x_disp` exists
only in the csv files and 15/50 designs lack the csv, so the batch carries
`y_mask (B,16)` and `masked_rel_l2` zeroes that channel out of both loss and
gradient. Everything else -- batched slice attention, raw-units-in/raw-units-out
normalisation living in model buffers -- is kept verbatim.

WHY IT IS NOT QUADRATIC: every point is softly assigned to one of `slice_num`
slices, attention runs among the G slices (G x G, not N x N), and the result is
scattered back with the same weights. Cost O(N*G + G^2), linear in points -- which
is what makes 16-92k-node bracket surfaces trainable.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from einops import rearrange
from torch import Tensor, nn

from engine.base import DataModule, OptimSpec, TaskModule, TrainState
from utils.normalize import Normalizer

ACTIVATION = {
    "gelu": nn.GELU,
    "relu": nn.ReLU,
    "silu": nn.SiLU,
    "tanh": nn.Tanh,
}

# The fixed channel layout of y. Order is load-bearing: fetch_deepjeb.py writes it,
# stats_deepjeb.py and dataset/deepjeb.py assemble it, and the metrics below index it.
CASES = ("ver", "hor", "dia", "tor")
STRESS_IDX = {"ver": 3, "hor": 7, "dia": 11, "tor": 15}


class MLP(nn.Module):
    """Reference's MLP: widen, activate, narrow. n_layers=0 (used everywhere here)
    makes this exactly Linear -> act -> Linear."""

    def __init__(
        self,
        n_input: int,
        n_hidden: int,
        n_output: int,
        n_layers: int = 0,
        act: str = "gelu",
        res: bool = True,
    ) -> None:
        super().__init__()
        if act not in ACTIVATION:
            raise ValueError(f"unknown activation {act!r}, expected one of {list(ACTIVATION)}")
        a = ACTIVATION[act]
        self.res = res
        self.linear_pre = nn.Sequential(nn.Linear(n_input, n_hidden), a())
        self.linear_post = nn.Linear(n_hidden, n_output)
        self.linears = nn.ModuleList(
            [nn.Sequential(nn.Linear(n_hidden, n_hidden), a()) for _ in range(n_layers)]
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.linear_pre(x)
        for layer in self.linears:
            x = layer(x) + x if self.res else layer(x)
        return self.linear_post(x)


class PhysicsAttention(nn.Module):
    """Slice -> attend -> deslice, for points on an irregular mesh.

    `temperature` is learned, initialised at 0.5: low sharpens the assignment toward
    hard clustering, high keeps it diffuse. `in_project_slice` is orthogonally
    initialised so the slices start decorrelated -- that init is load-bearing.
    """

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        slice_num: int = 32,
    ) -> None:
        super().__init__()
        inner_dim = dim_head * heads
        self.heads, self.dim_head = heads, dim_head
        self.scale = dim_head**-0.5
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.temperature = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)

        self.in_project_x = nn.Linear(dim, inner_dim)
        self.in_project_fx = nn.Linear(dim, inner_dim)
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        nn.init.orthogonal_(self.in_project_slice.weight)

        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def _heads(self, x: Tensor) -> Tensor:
        """(B,N,H*d) -> (B,H,N,d)."""
        b, n, _ = x.shape
        return x.reshape(b, n, self.heads, self.dim_head).permute(0, 2, 1, 3).contiguous()

    def forward(self, x: Tensor) -> Tensor:
        # (1) Slice: soft-assign every point to a slice, average into slice tokens.
        fx_mid = self._heads(self.in_project_fx(x))  # B H N d -- the content
        x_mid = self._heads(self.in_project_x(x))  # B H N d -- decides the assignment
        weights = self.softmax(self.in_project_slice(x_mid) / self.temperature)  # B H N G
        norm = weights.sum(dim=2)  # B H G -- points per slice, softly counted

        # Contract over n: a batched matmul, (B,H,G,N) @ (B,H,N,d) -> (B,H,G,d).
        # Dividing by `norm` turns the sum into a MEAN, so a slice holding 40k points
        # and one holding 12 come out on the same scale.
        token = weights.transpose(-2, -1) @ fx_mid  # B H G d
        token = token / (norm + 1e-5).unsqueeze(-1)

        # (2) Attend among slices only. This matrix is G x G.
        q, k, v = self.to_q(token), self.to_k(token), self.to_v(token)
        attn = self.dropout(self.softmax(torch.matmul(q, k.transpose(-1, -2)) * self.scale))
        out_token = torch.matmul(attn, v)  # B H G d

        # (3) Deslice with the SAME weights, so a point 70% in slice 3 gets 70% of it.
        # (B,H,N,G) @ (B,H,G,d) -> (B,H,N,d); no normalisation -- the weights already
        # sum to 1 across G for every point.
        out = weights @ out_token  # B H N d
        return self.to_out(rearrange(out, "b h n d -> b n (h d)"))


class Block(nn.Module):
    """Pre-norm transformer block. The last one in the stack also carries the head."""

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        dropout: float,
        act: str = "gelu",
        mlp_ratio: int = 2,
        last_layer: bool = False,
        out_dim: int = 1,
        slice_num: int = 32,
    ) -> None:
        super().__init__()
        self.last_layer = last_layer
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.attn = PhysicsAttention(
            hidden_dim,
            heads=num_heads,
            dim_head=hidden_dim // num_heads,
            dropout=dropout,
            slice_num=slice_num,
        )
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(
            hidden_dim, hidden_dim * mlp_ratio, hidden_dim, n_layers=0, res=False, act=act
        )
        if last_layer:
            self.ln_3 = nn.LayerNorm(hidden_dim)
            self.head = nn.Linear(hidden_dim, out_dim)

    def forward(self, fx: Tensor) -> Tensor:
        fx = self.attn(self.ln_1(fx)) + fx
        fx = self.mlp(self.ln_2(fx)) + fx
        return self.head(self.ln_3(fx)) if self.last_layer else fx


class TransolverNet(nn.Module):
    """The architecture on its own -- no loss, no optimizer, no normalisation."""

    def __init__(
        self,
        space_dim: int = 3,
        fun_dim: int = 0,
        out_dim: int = 16,
        n_layers: int = 8,
        n_hidden: int = 256,
        n_head: int = 8,
        slice_num: int = 32,
        mlp_ratio: int = 2,
        dropout: float = 0.0,
        act: str = "gelu",
        unified_pos: bool = False,
        ref: int = 8,
        pos_bounds: Sequence[Sequence[float]] = ((-40.0, 71.0), (-165.0, 22.0), (0.0, 66.0)),
    ) -> None:
        super().__init__()
        if n_hidden % n_head:
            raise ValueError(f"n_hidden ({n_hidden}) must be divisible by n_head ({n_head})")
        self.space_dim, self.fun_dim = space_dim, fun_dim
        self.unified_pos, self.ref = unified_pos, ref
        self.register_buffer("pos_bounds", torch.tensor(pos_bounds, dtype=torch.float32))

        in_dim = fun_dim + (ref**3 if unified_pos else space_dim)
        self.preprocess = MLP(in_dim, n_hidden * 2, n_hidden, n_layers=0, res=False, act=act)
        self.placeholder = nn.Parameter((1 / n_hidden) * torch.rand(n_hidden, dtype=torch.float32))

        self.blocks = nn.ModuleList(
            [
                Block(
                    num_heads=n_head,
                    hidden_dim=n_hidden,
                    dropout=dropout,
                    act=act,
                    mlp_ratio=mlp_ratio,
                    out_dim=out_dim,
                    slice_num=slice_num,
                    last_layer=(i == n_layers - 1),
                )
                for i in range(n_layers)
            ]
        )
        self.apply(self._init)

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def grid_distances(self, pos: Tensor) -> Tensor:
        """Distance from every point to a ref^3 lattice spanning `pos_bounds`.

        A positional encoding that is meaningful on an irregular mesh. Costs ref^3
        input channels, so ref=8 means 512 -- the reason it defaults off.
        """
        b, device = pos.shape[0], pos.device
        axes = [
            torch.linspace(lo, hi, self.ref, device=device, dtype=pos.dtype)
            for lo, hi in self.pos_bounds
        ]
        grid = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1)  # ref,ref,ref,3
        grid = grid.reshape(1, -1, 3).expand(b, -1, -1)  # B, ref^3, 3
        return torch.cdist(pos, grid)  # B, N, ref^3

    def forward(self, pos: Tensor, fx: Tensor | None = None) -> Tensor:
        """pos (B,N,space_dim), optional features fx (B,N,fun_dim) -> (B,N,out_dim)."""
        x = self.grid_distances(pos) if self.unified_pos else pos
        if fx is not None:
            x = torch.cat((x, fx), dim=-1)
        h = self.preprocess(x)
        if fx is None:
            # No function input: the reference adds a learned constant so the first
            # block sees something other than a pure function of position.
            h = h + self.placeholder[None, None, :]
        for block in self.blocks:
            h = block(h)
        return h


def relative_l2(pred: Tensor, target: Tensor) -> Tensor:
    """Per-sample ||pred - target|| / ||target||, averaged. The PDE-surrogate standard:
    scale-free, so a large-magnitude channel cannot drown a small one."""
    dims = tuple(range(1, pred.ndim))
    num = torch.linalg.vector_norm(pred - target, dim=dims)
    den = torch.linalg.vector_norm(target, dim=dims).clamp_min(1e-8)
    return (num / den).mean()


def masked_rel_l2(pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    """relative_l2 over the channels `mask` marks valid.

    pred/target (B,N,C), mask (B,C) broadcasting over N. Masked channels are zeroed
    in BOTH tensors, so they add zero to both norms and receive no gradient. The
    dataset already zero-fills masked target channels; the multiply here is what
    guarantees the same for pred (and for a target that arrives normalised, where a
    zero-filled raw channel is no longer zero).
    """
    m = mask.to(pred.dtype).unsqueeze(1)  # B,1,C
    num = torch.linalg.vector_norm((pred - target) * m, dim=(1, 2))
    den = torch.linalg.vector_norm(target * m, dim=(1, 2)).clamp_min(1e-8)
    return (num / den).mean()


def masked_mse(pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    """MSE over valid channels only, averaged over the valid entries (not over C)."""
    m = mask.to(pred.dtype).unsqueeze(1).expand_as(pred)
    return (((pred - target) * m) ** 2).sum() / m.sum().clamp_min(1.0)


def masked_l1(pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    """L1 over valid channels only, averaged over the valid entries."""
    m = mask.to(pred.dtype).unsqueeze(1).expand_as(pred)
    return (((pred - target) * m).abs()).sum() / m.sum().clamp_min(1.0)


LOSSES = {
    "rel_l2": masked_rel_l2,
    "mse": masked_mse,
    "l1": masked_l1,
}


class Transolver(TaskModule):
    """The trainable unit: architecture + normalisation + masked loss + optimizer.

    Batch contract (dataset/deepjeb.py and dataset/points.py both honour it):

        pos     (B, N, 3)    point coordinates, raw mm
        fx      (B, N, 4)    normal(3) + area(1), raw units
        y       (B, N, 16)   targets, raw mm / MPa, masked channels zero-filled
        y_mask  (B, 16)      bool; optional -- absent means every channel is valid

    Normalisation is applied inside, so `forward` is raw-in/raw-out and a checkpoint
    can denormalise its own predictions with no dataset mounted.
    """

    def __init__(
        self,
        net: dict | None = None,
        optim=None,
        sched=None,
        loss: str = "rel_l2",
        lr: float = 1e-3,
        **net_kwargs,
    ) -> None:
        super().__init__()
        if loss not in LOSSES:
            raise ValueError(f"unknown loss {loss!r}, expected one of {list(LOSSES)}")
        self.net = TransolverNet(**{**(net or {}), **net_kwargs})
        self.loss_name, self.loss_fn = loss, LOSSES[loss]
        self.optim_factory, self.sched_factory, self.lr = optim, sched, lr

        # Identity until on_data_ready fills them; they ride in the checkpoint.
        self.pos_norm = Normalizer(self.net.space_dim)
        self.fx_norm = Normalizer(max(self.net.fun_dim, 1))
        self.y_norm = Normalizer(self.net.blocks[-1].head.out_features)

    # -- statistics ------------------------------------------------------------------

    def on_data_ready(self, datamodule: DataModule) -> None:
        """Copy offline statistics into buffers. Never computes anything itself."""
        stats = getattr(datamodule, "stats", {}) or {}
        for key, norm in (("pos", self.pos_norm), ("fx", self.fx_norm), ("y", self.y_norm)):
            mean, std = stats.get(f"{key}_mean"), stats.get(f"{key}_std")
            if mean is not None and std is not None:
                norm.fit(mean, std)

    # -- forward ---------------------------------------------------------------------

    def forward(self, pos: Tensor, fx: Tensor | None = None) -> Tensor:
        """Raw units in, raw units out."""
        fx_n = self.fx_norm.norm(fx) if (fx is not None and self.net.fun_dim) else None
        return self.y_norm.denorm(self.net(self.pos_norm.norm(pos), fx_n))

    def _mask(self, batch: dict, pred: Tensor) -> Tensor:
        mask = batch.get("y_mask")
        if mask is None:
            mask = torch.ones(pred.shape[0], pred.shape[-1], dtype=torch.bool, device=pred.device)
        return mask

    def _step(self, batch: dict) -> tuple[Tensor, Tensor, Tensor]:
        pred = self(batch["pos"], batch.get("fx"))
        return pred, batch["y"], self._mask(batch, pred)

    # -- the contract methods ----------------------------------------------------------

    def _loss(self, pred: Tensor, y: Tensor, mask: Tensor) -> Tensor:
        """Loss on NORMALISED values so every channel contributes comparably --
        stress in MPa would otherwise swamp displacement in mm. The normalised
        target is re-masked because norm() maps a zero-filled channel to
        (0 - mean)/std != 0; masked_* would zero it again, but only symmetric
        zeroing keeps `den` honest for rel_l2."""
        m = mask.to(pred.dtype).unsqueeze(1)
        return self.loss_fn(self.y_norm.norm(pred), self.y_norm.norm(y) * m, mask)

    def training_step(self, batch: dict, state: TrainState) -> dict[str, Tensor]:
        pred, y, mask = self._step(batch)
        return {
            "loss": self._loss(pred, y, mask),
            "rel_l2": masked_rel_l2(pred, y, mask).detach(),
        }

    def validation_step(self, batch: dict, state: TrainState) -> dict[str, Tensor]:
        pred, y, mask = self._step(batch)
        out = {
            "loss": self._loss(pred, y, mask),
            "rel_l2": masked_rel_l2(pred, y, mask),
            "mae": masked_l1(pred, y, mask),
        }
        # Per-channel relative error in raw units -- what says the model got stress
        # right but displacement wrong, which the mean hides. A channel invalid
        # anywhere in the batch is skipped rather than reported half-true.
        for c in range(y.shape[-1]):
            if bool(mask[:, c].all()):
                out[f"rel_l2/ch{c}"] = relative_l2(pred[..., c : c + 1], y[..., c : c + 1])
        # Peak |stress| per load case: THE number a bracket is judged by. Maxima are
        # over the sampled window, so this is an estimate whose sampling noise is
        # common-mode between pred and true.
        for case in CASES:
            s = STRESS_IDX[case]
            pk_pred = pred[..., s].abs().amax(dim=1)
            pk_true = y[..., s].abs().amax(dim=1)
            out[f"max_stress/{case}_abs_err"] = (pk_pred - pk_true).abs().mean()
            out[f"max_stress/{case}_rel_err"] = (
                ((pk_pred - pk_true).abs() / pk_true.clamp_min(1e-8)).mean()
            )
        return out

    def configure_optimizers(self) -> OptimSpec:
        opt = (
            self.optim_factory(self.parameters())
            if self.optim_factory is not None
            else torch.optim.AdamW(self.parameters(), lr=self.lr)
        )
        if self.sched_factory is None:
            return OptimSpec.of(opt)
        return OptimSpec.of(opt, self.sched_factory(opt), interval="step")
