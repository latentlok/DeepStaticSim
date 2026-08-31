"""One surrogate job: STL in -> predicted fields out (.vtp + .csv + summary.json).

    uv run python ../app/runner.py bracket.stl --out-dir jobs/0001
    uv run python ../app/runner.py bracket.stl --out-dir jobs/0001 --ckpt <run>/ckpt/best_weights

This is the piece a future service (local web app, Docker worker, AWS Batch job)
wraps: no server assumptions, plain files in and out, exit code says success.

What the model expects and what this file must therefore produce from a bare STL:
per-point position (mm), outward unit normal, and lumped vertex area (mm^2) --
the same features `utils/fetch_deepjeb.py` derives from the FE mesh at training
time. The load cases are NOT inputs: every DeepJEB design is solved under the
same four bolted-bracket load cases, so geometry alone determines all 16 output
channels [per case: disp x,y,z (mm) + signed von Mises stress (MPa)].

Two things are easy to get silently wrong:
  * STL files repeat vertices per facet -- `clean()` merges them first, or the
    "surface" would be triangle soup with triple-counted areas.
  * Transolver POOLS over the point set it is given, so predictions depend on
    context size. Inference therefore runs on random windows of the same size
    the model saw in training (default 16384), permuted then unpermuted --
    feeding all N points in one forward is a distribution shift, not a speedup.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# Imports resolve the same whether this runs as `python app/runner.py` or is
# imported as `app.runner`: the repo root (for `app.*`) and the surrogate fork
# (for `models`/`engine`) both go on the path.
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "surrogate"))
sys.path.insert(0, str(_ROOT))

from app.export import write_csv, write_summary, write_vtp  # noqa: E402

DEFAULT_CKPT = (
    Path(__file__).resolve().parent.parent
    / "surrogate/outputs/jeb_surface/2026-08-31_21-51-31_750439/ckpt/best_weights"
)


def load_model(ckpt: Path, device: str = "cpu"):
    """Rebuild the model from its run's composed config, load weights, eval().

    The checkpoint is self-contained: normalizers are buffers, so forward is
    raw-mm-in / raw-units-out with no stats file or datamodule mounted.
    """
    import hydra
    from omegaconf import OmegaConf

    ckpt = Path(ckpt)
    run_dir = ckpt.parent.parent
    cfg_path = run_dir / ".hydra" / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"{cfg_path} not found -- pass a checkpoint directory inside a training "
            f"run, e.g. .../ckpt/best_weights"
        )
    model = hydra.utils.instantiate(OmegaConf.load(cfg_path).model)

    from engine.checkpoint import load_checkpoint

    load_checkpoint(ckpt, model, weights_only=True)
    return model.to(device).eval()


def stl_features(stl_path: Path) -> dict:
    """Position / outward unit normal / lumped area per merged vertex, from an STL.

    Same math as utils/fetch_deepjeb.surface_features, with one difference: STL
    facets are consistently wound (they come from one CAD export), so a single
    global flip -- decided by the sign of the enclosed volume -- fixes outwardness.
    The DeepJEB h5 needed per-face repair; a watertight STL does not.
    """
    import pyvista as pv

    mesh = pv.read(str(stl_path))
    if mesh.n_points == 0 or mesh.n_cells == 0:
        raise ValueError(f"{stl_path}: empty mesh")
    mesh = mesh.triangulate().clean()  # merge per-facet duplicate vertices
    faces = mesh.faces.reshape(-1, 4)[:, 1:]
    v = np.asarray(mesh.points, dtype=np.float64)

    tri = v[faces]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])  # 2*area*normal
    vol6 = np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum()
    if vol6 < 0:
        cross = -cross
        vol6 = -vol6
    fa = 0.5 * np.linalg.norm(cross, axis=1)
    if not fa.sum() > 0:
        raise ValueError(f"{stl_path}: degenerate surface (zero total area)")

    n_pts = len(v)
    vn = np.zeros((n_pts, 3))
    va = np.zeros(n_pts)
    for k in range(3):
        np.add.at(vn, faces[:, k], cross)
        np.add.at(va, faces[:, k], fa / 3.0)
    vn /= np.clip(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12, None)

    return {
        "position": v.astype(np.float32),
        "normal": vn.astype(np.float32),
        "area": va[:, None].astype(np.float32),
        "volume_mm3": float(vol6 / 6.0),
        "area_mm2": float(fa.sum()),
    }


def predict(model, feats: dict, device: str = "cpu", window: int = 16384, seed: int = 0):
    """(N,16) prediction, computed on training-sized random windows.

    Transolver pools over its input set, so context size is part of the input
    distribution. Points are permuted once, run in `window`-sized chunks (the
    training window), and unpermuted -- deterministic for a fixed seed.
    """
    import torch

    pos = np.asarray(feats["position"], dtype=np.float32)
    fx = np.concatenate([feats["normal"], feats["area"]], axis=-1).astype(np.float32)
    n = len(pos)
    perm = np.random.default_rng(seed).permutation(n)
    out = np.empty((n, 16), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, n, window):
            idx = perm[s : s + window]
            p = torch.from_numpy(pos[idx])[None].to(device)
            f = torch.from_numpy(fx[idx])[None].to(device)
            out[idx] = model(p, f)[0].cpu().numpy()
    return out


def run_job(
    stl_path: Path,
    out_dir: Path,
    model_or_ckpt=DEFAULT_CKPT,
    device: str = "cpu",
    window: int = 16384,
) -> dict:
    """STL -> out_dir/{result.vtp, result.csv, summary.json}; returns the summary."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}

    t0 = time.time()
    if isinstance(model_or_ckpt, (str, Path)):
        print(f"loading checkpoint {model_or_ckpt}", flush=True)
        model = load_model(Path(model_or_ckpt), device)
        ckpt_label = str(model_or_ckpt)
    else:
        model, ckpt_label = model_or_ckpt, "<injected model>"
    timings["load_model"] = time.time() - t0

    t0 = time.time()
    feats = stl_features(Path(stl_path))
    timings["stl_features"] = time.time() - t0
    print(
        f"{stl_path}: {len(feats['position']):,} merged vertices, "
        f"volume {feats['volume_mm3']:,.0f} mm3",
        flush=True,
    )

    t0 = time.time()
    pred = predict(model, feats, device=device, window=window)
    timings["predict"] = time.time() - t0
    print(f"predicted 16 channels in {timings['predict']:.1f}s on {device}", flush=True)

    write_vtp(out_dir / "result.vtp", feats, pred)
    write_csv(out_dir / "result.csv", feats, pred)
    summary = write_summary(out_dir / "summary.json", feats, pred, ckpt_label, timings)
    print(f"wrote {out_dir}/result.vtp, result.csv, summary.json", flush=True)
    return summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("stl", type=Path, help="input STL (mm units)")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    p.add_argument("--device", default="cpu")
    p.add_argument("--window", type=int, default=16384, help="inference window (training size)")
    a = p.parse_args(argv)
    try:
        summary = run_job(a.stl, a.out_dir, a.ckpt, device=a.device, window=a.window)
    except Exception as e:  # noqa: BLE001 - CLI boundary: report, don't traceback-spam
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(json.dumps(summary["cases"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
