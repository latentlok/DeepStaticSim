"""Ground truth vs prediction, side by side, served to a browser.

    uv run python utils/compare_server.py                      # latest best_weights
    uv run python utils/compare_server.py --ckpt outputs/jeb_surface/<run>/ckpt/best_weights
    uv run python utils/compare_server.py --split val --port 8081

Three linked 3D views of one held-out bracket -- truth | prediction | |error| --
with dropdowns for the design and for any of the 16 target channels. Adapted from
physics-transolver/utils/viz_server.py, which established the serving pattern for
this box: it is headless, so PyVista renders off-screen through VTK's EGL path and
trame streams images to the browser; with `--renderer mesa` (default) rendering is
a software rasteriser and does not care that training holds the GPU.

The checkpoint is self-contained: normalisation lives in model buffers, so
`forward` is raw-mm-in / raw-units-out with no stats file mounted. The model
config is read from the run's own .hydra/config.yaml, so this never has to guess
architecture hyperparameters. Predictions for every design in the split are
computed once at startup (CPU by default; a 4M-param model over ~60k points takes
seconds) and cached.

Truth and prediction share one colour scale (2-98 percentile of the truth), so a
prediction that looks identical IS identical to that tolerance; the error view has
its own 0..p98 scale. Captions carry the per-channel relative L2 for the design on
screen.
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

log = logging.getLogger("compare")

MESA_VENDOR = "/usr/share/glvnd/egl_vendor.d/50_mesa.json"

# (channel name, y index, units) -- the fixed order of models/transolver.py.
CHANNELS: list[tuple[str, int, str]] = []
_i = 0
for _case in ("ver", "hor", "dia", "tor"):
    for _ax in "xyz":
        CHANNELS.append((f"{_case}_disp_{_ax}", _i, "mm"))
        _i += 1
    CHANNELS.append((f"{_case}_stress", _i, "MPa"))
    _i += 1


def setup_renderer(mode: str) -> None:
    """Point VTK at a headless GL backend. MUST run before pyvista is imported.

    Measured on this box (see physics-transolver/utils/viz_server.py): EGL with the
    default glvnd vendor picks the NVIDIA driver and dies when the card is busy;
    the mesa vendor renders in software and always works. xvfb is the escape hatch.
    """
    os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
    if mode == "xvfb":
        return  # caller runs the process under `xvfb-run -a`
    os.environ["VTK_DEFAULT_OPENGL_WINDOW"] = "vtkEGLRenderWindow"
    if mode == "mesa":
        if not Path(MESA_VENDOR).exists():
            raise SystemExit(f"{MESA_VENDOR} not found; try --renderer nvidia or xvfb")
        os.environ["__EGL_VENDOR_LIBRARY_FILENAMES"] = MESA_VENDOR


def tailscale_ip() -> str | None:
    try:
        out = subprocess.run(["tailscale", "ip", "-4"], capture_output=True, text=True, timeout=10)
        lines = out.stdout.strip().splitlines()
        return lines[0].strip() if lines and lines[0].strip() else None
    except (FileNotFoundError, subprocess.SubprocessError):
        return None


def latest_best_weights(outputs: Path) -> Path:
    """Newest run directory under outputs/ that holds a ckpt/best_weights."""
    candidates = sorted(
        outputs.glob("*/*/ckpt/best_weights"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    if not candidates:
        raise SystemExit(f"no */*/ckpt/best_weights under {outputs} -- train first, or pass --ckpt")
    return candidates[0]


def load_model(ckpt: Path, device: str):
    """Rebuild the model from the run's own composed config, then load weights.

    The normalizers are buffers inside the checkpoint, so nothing else is needed:
    no stats file, no datamodule, no on_data_ready.
    """
    import hydra
    import torch
    from omegaconf import OmegaConf

    run_dir = ckpt.parent.parent  # .../<run>/ckpt/best_weights -> <run>
    cfg_path = run_dir / ".hydra" / "config.yaml"
    if not cfg_path.exists():
        raise SystemExit(f"{cfg_path} not found -- pass a checkpoint inside a run directory")
    cfg = OmegaConf.load(cfg_path)
    model = hydra.utils.instantiate(cfg.model)

    from engine.checkpoint import load_checkpoint

    load_checkpoint(ckpt, model, weights_only=True)
    model.to(device).eval()
    n_par = sum(p.numel() for p in model.parameters())
    log.info("model from %s | %.2fM params | device %s", run_dir.name, n_par / 1e6, device)
    return model, torch.no_grad


def predict_design(model, group, device: str, chunk: int = 65536) -> dict[str, np.ndarray]:
    """Full-surface truth and prediction for one store group, raw units."""
    import torch

    pos = np.asarray(group["position"][:], dtype=np.float32)
    fx = np.concatenate(
        [np.asarray(group["normal"][:]), np.asarray(group["area"][:])], axis=-1
    ).astype(np.float32)
    y = np.concatenate(
        [
            np.asarray(group[f"{c}_{k}"][:])
            for c in ("ver", "hor", "dia", "tor")
            for k in ("disp", "stress")
        ],
        axis=-1,
    ).astype(np.float32)

    preds = []
    with torch.no_grad():
        for s in range(0, len(pos), chunk):
            p = torch.from_numpy(pos[s : s + chunk])[None].to(device)
            f = torch.from_numpy(fx[s : s + chunk])[None].to(device)
            preds.append(model(p, f)[0].cpu().numpy())
    pred = np.concatenate(preds, axis=0)
    return {"pos": pos, "y": y, "pred": pred}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--ckpt", type=Path, default=None, help="default: newest */*/ckpt/best_weights")
    p.add_argument("--root", type=Path, default=None, help="default: $DL_DATA")
    p.add_argument("--store", default="deepjeb.zarr")
    p.add_argument("--split", default="test", choices=("test", "val"))
    p.add_argument("--device", default="cpu", help="cpu keeps the GPU free for training")
    p.add_argument("--host", default=None, help="default: this host's tailscale IP")
    p.add_argument("--port", type=int, default=8081)
    p.add_argument(
        "--renderer",
        default="mesa",
        choices=("mesa", "nvidia", "xvfb"),
        help="mesa: EGL software, immune to GPU contention (default)",
    )
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    host = a.host or tailscale_ip() or "127.0.0.1"

    setup_renderer(a.renderer)
    import pyvista as pv

    pv.OFF_SCREEN = True
    pv.set_plot_theme("document")

    import json

    root = a.root or Path(os.environ.get("DL_DATA", "data"))
    from dataset.deepjeb import open_store

    store = open_store(root / a.store)
    designs = json.loads((root / "splits.json").read_text())[a.split]

    ckpt = a.ckpt or latest_best_weights(Path(__file__).resolve().parent.parent / "outputs")
    model, _ = load_model(Path(ckpt), a.device)

    log.info("predicting %d %s design(s) on %s ...", len(designs), a.split, a.device)
    data: dict[str, dict[str, np.ndarray]] = {}
    for d in designs:
        data[d] = predict_design(model, store[d]["surface"], a.device)
        err = data[d]["pred"] - data[d]["y"]
        log.info(
            "  %s: %s points | mean rel_l2 %.3f",
            d,
            f"{len(data[d]['pos']):,}",
            float(
                np.mean(
                    np.linalg.norm(err, axis=0)
                    / np.clip(np.linalg.norm(data[d]["y"], axis=0), 1e-8, None)
                )
            ),
        )

    # ---- rendering ------------------------------------------------------------
    pl = pv.Plotter(shape=(1, 3), border=False, window_size=(1800, 700))
    ui = {"design": designs[0], "channel": CHANNELS[15][0]}  # tor_stress: stress is the story
    by_name = {name: (idx, units) for name, idx, units in CHANNELS}

    def draw() -> None:
        d = data[ui["design"]]
        idx, units = by_name[ui["channel"]]
        truth, pred = d["y"][:, idx], d["pred"][:, idx]
        err = np.abs(pred - truth)
        rel = float(np.linalg.norm(pred - truth) / max(np.linalg.norm(truth), 1e-8))

        lo, hi = (float(np.percentile(truth, 2)), float(np.percentile(truth, 98)))
        if lo == hi:
            lo, hi = lo - 1e-6, hi + 1e-6
        e_hi = float(np.percentile(err, 98)) or 1e-6
        cloud = pv.PolyData(d["pos"])
        style = dict(point_size=3, render_points_as_spheres=True, cmap="coolwarm")

        panes = (
            ("truth", truth, (lo, hi), f"truth [{units}]", "coolwarm"),
            ("prediction", pred, (lo, hi), f"prediction [{units}]", "coolwarm"),
            ("|error|", err, (0.0, e_hi), f"|error| [{units}]", "inferno"),
        )
        for col, (pane, vals, clim, bar, cmap) in enumerate(panes):
            pl.subplot(0, col)
            mesh = cloud.copy(deep=False)
            mesh[pane] = vals
            style["cmap"] = cmap
            pl.add_mesh(
                mesh,
                scalars=pane,
                clim=clim,
                scalar_bar_args={"title": bar},
                name=f"mesh_{col}",
                **style,
            )
            title = {
                "truth": f"{ui['design']}  ground truth",
                "prediction": f"prediction   rel L2 {rel:.3f}",
                "|error|": f"|error|   p98 {e_hi:.3g} {units}",
            }[pane]
            pl.add_text(title, position="upper_left", font_size=10, name=f"cap_{col}", color="#333333")

    draw()
    pl.link_views()
    pl.camera_position = "iso"

    from trame.app import get_server
    from trame.ui.vuetify3 import SinglePageLayout
    from trame.widgets import vtk as vtk_widgets
    from trame.widgets import vuetify3 as v3

    server = get_server(client_type="vue3")
    state, ctrl = server.state, server.controller

    def refresh(**_):
        ui.update(design=state.design, channel=state.channel)
        draw()
        ctrl.view_update()

    state.change("design")(refresh)
    state.change("channel")(refresh)

    with SinglePageLayout(server) as layout:
        layout.title.set_text(f"DeepJEB {a.split}: truth vs prediction")
        with layout.toolbar as tb:
            tb.density = "compact"
            v3.VSpacer()
            v3.VSelect(
                v_model=("design", ui["design"]),
                items=("designs", [{"title": d, "value": d} for d in designs]),
                density="compact",
                hide_details=True,
                variant="outlined",
                style="max-width:180px",
                classes="mx-1",
            )
            v3.VSelect(
                v_model=("channel", ui["channel"]),
                items=("channels", [{"title": n, "value": n} for n, _, _ in CHANNELS]),
                density="compact",
                hide_details=True,
                variant="outlined",
                style="max-width:220px",
                classes="mx-1",
            )
        with layout.content:
            with v3.VContainer(fluid=True, classes="pa-0 fill-height"):
                view = vtk_widgets.VtkRemoteView(pl.ren_win, interactive_ratio=1)
                ctrl.view_update = view.update
                ctrl.view_reset_camera = view.reset_camera

    log.info("serving on http://%s:%d", host, a.port)
    server.start(host=host, port=a.port, open_browser=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
