"""DeepStaticSim web app: upload an STL, run the surrogate, inspect the fields.

    cd surrogate && uv run --no-sync python ../app/server.py            # tailscale
    cd surrogate && uv run --no-sync python ../app/server.py --host 127.0.0.1

Browser flow: pick an .stl -> Run -> the runner CLI (app/runner.py) executes in a
subprocess (CPU by default, so GPU training is untouched) -> the finished job
appears in the job list and renders in an interactive 3D view with load-case /
quantity / colormap controls and a deformation-warp slider. result.vtp,
result.csv and summary.json are served as real HTTP downloads.

Same engine, two doors: the browser UI and a REST API --

    POST /api/jobs                multipart/form-data, field "stl"  -> 202 {job, status}
    GET  /api/jobs                every job with its status
    GET  /api/jobs/<job>          status + summary + download links

so curl / CI / another service can drive analyses without a browser. A toolbar
switch flips the UI and the 3D viewport to dark mode.

Serving pattern is the one measured to work on this headless box (see
surrogate/utils/compare_server.py): VTK renders off-screen through EGL with the
mesa software vendor -- immune to whatever holds the GPU -- and trame streams
images; `timeout=0` because wslink otherwise exits 300 s after the last client.

The server never loads the model itself; only the runner does. A dead or corrupt
job directory is skipped with a log line, never a crash.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import json
import logging
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

APP_DIR = Path(__file__).resolve().parent
SURROGATE = APP_DIR.parent / "surrogate"
sys.path.insert(0, str(SURROGATE))

log = logging.getLogger("app")

MESA_VENDOR = "/usr/share/glvnd/egl_vendor.d/50_mesa.json"
CASES = ("ver", "hor", "dia", "tor")
QUANTITIES = {
    "stress": ("signed von Mises", "MPa"),
    "disp_mag": ("|displacement|", "mm"),
    "disp_x": ("displacement x", "mm"),
    "disp_y": ("displacement y", "mm"),
    "disp_z": ("displacement z", "mm"),
}
CMAPS = ("coolwarm", "viridis", "inferno", "turbo")
DOWNLOADABLE = ("result.vtp", "result.csv", "summary.json", "runner.log")

# ---- job bookkeeping: module-level and pure, so the API surface is unit-testable
# without a model, a checkpoint, or a running server. --------------------------------


def sanitize_stem(filename: str) -> str:
    """Filesystem-safe job stem from a user-supplied filename."""
    return re.sub(r"[^A-Za-z0-9_.-]", "_", Path(filename).stem)[:40] or "job"


def new_job_dir(jobs_dir: Path, filename: str, content: bytes) -> Path:
    """Create <stamp>_<stem>/input.stl. Raises ValueError on invalid input."""
    if not filename.lower().endswith(".stl"):
        raise ValueError(f"'{filename}' is not an .stl file")
    if not content:
        raise ValueError("empty upload")
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    jobdir = jobs_dir / f"{stamp}_{sanitize_stem(filename)}"
    n = 0
    while jobdir.exists():  # same-second uploads must not collide
        n += 1
        jobdir = jobs_dir / f"{stamp}_{sanitize_stem(filename)}_{n}"
    jobdir.mkdir(parents=True)
    (jobdir / "input.stl").write_bytes(content)
    return jobdir


def job_status(jobs_dir: Path, name: str, live: set[str]) -> str | None:
    """'running' | 'done' | 'failed', or None for a job that does not exist."""
    if name in live:
        return "running"
    d = jobs_dir / name
    if not d.is_dir():
        return None
    return "done" if (d / "result.vtp").exists() else "failed"


def completed_jobs(jobs_dir: Path) -> list[str]:
    """Jobs with a result.vtp, newest first."""
    dirs = [d for d in jobs_dir.iterdir() if d.is_dir() and (d / "result.vtp").exists()]
    return [d.name for d in sorted(dirs, key=lambda d: d.stat().st_mtime, reverse=True)]


def list_jobs(jobs_dir: Path, live: set[str]) -> list[dict]:
    """Running jobs first, then completed, for GET /api/jobs."""
    out = [{"job": n, "status": "running"} for n in sorted(live)]
    out += [{"job": n, "status": "done"} for n in completed_jobs(jobs_dir)]
    return out


def job_payload(jobs_dir: Path, name: str, live: set[str]) -> dict | None:
    """The GET /api/jobs/<job> body, or None for 404."""
    status = job_status(jobs_dir, name, live)
    if status is None:
        return None
    payload: dict = {"job": name, "status": status}
    if status == "done":
        sj = jobs_dir / name / "summary.json"
        if sj.exists():
            try:
                payload["summary"] = json.loads(sj.read_text())
            except json.JSONDecodeError:
                payload["summary"] = {}
        payload["downloads"] = [
            f"/download/{name}/{f}" for f in DOWNLOADABLE if (jobs_dir / name / f).exists()
        ]
    return payload


def setup_renderer(mode: str) -> None:
    """Headless GL backend. MUST run before pyvista is imported (see compare_server)."""
    os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
    if mode == "xvfb":
        return
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


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--host", default=None, help="default: this host's tailscale IP")
    p.add_argument("--port", type=int, default=8090)
    p.add_argument("--renderer", default="mesa", choices=("mesa", "nvidia", "xvfb"))
    p.add_argument(
        "--ckpt",
        type=Path,
        default=SURROGATE / "outputs/jeb_surface/2026-08-31_21-51-31_750439/ckpt/best_weights",
    )
    p.add_argument("--jobs-dir", type=Path, default=APP_DIR.parent / "app_data" / "jobs")
    p.add_argument("--device", default="cpu", help="cpu keeps the GPU free for training")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    host = a.host or tailscale_ip() or "127.0.0.1"
    jobs_dir: Path = a.jobs_dir
    jobs_dir.mkdir(parents=True, exist_ok=True)
    runner = APP_DIR / "runner.py"

    setup_renderer(a.renderer)
    import pyvista as pv

    pv.OFF_SCREEN = True
    pv.set_plot_theme("document")

    # ---- jobs ----------------------------------------------------------------
    live: set[str] = set()  # job names with a subprocess in flight
    cache: dict[str, dict] = {}  # job name -> {mesh, summary}

    def load_job(name: str) -> dict | None:
        if name in cache:
            return cache[name]
        d = jobs_dir / name
        try:
            mesh = pv.read(d / "result.vtp")
            needed = [f"{c}_{q}" for c in CASES for q in ("stress", "disp_mag")]
            missing = [k for k in needed if k not in mesh.point_data]
            if missing:
                raise ValueError(f"result.vtp missing arrays {missing}")
            summary = {}
            sj = d / "summary.json"
            if sj.exists():
                summary = json.loads(sj.read_text())
            cache[name] = {
                "mesh": mesh,
                "points": np.asarray(mesh.points).copy(),
                "summary": summary,
            }
            return cache[name]
        except Exception as e:  # a corrupt job must never take the server down
            log.warning("skipping job %s: %s", name, e)
            return None

    # ---- rendering -----------------------------------------------------------
    pl = pv.Plotter(window_size=(1500, 850))
    pl.add_axes()

    ui = {
        "job": "",
        "case": "ver",
        "quantity": "stress",
        "cmap": "coolwarm",
        "full_range": False,
        "psize": 3,
        "warp": 0.0,
        "dark": False,
    }

    def draw() -> None:
        cap = "#dddddd" if ui["dark"] else "#333333"
        pl.set_background("#1a1a1a" if ui["dark"] else "white")
        name = ui["job"]
        data = load_job(name) if name else None
        if data is None:
            pl.remove_actor("mesh", render=False)
            pl.add_text(
                "No completed job selected.\nUpload an STL and press RUN.",
                position="upper_left",
                font_size=12,
                name="caption",
                color=cap,
            )
            return
        mesh, base, summary = data["mesh"], data["points"], data["summary"]
        case, q = ui["case"], ui["quantity"]
        label, units = QUANTITIES[q]
        if q == "stress":
            vals = np.asarray(mesh.point_data[f"{case}_stress"]).ravel()
        elif q == "disp_mag":
            vals = np.asarray(mesh.point_data[f"{case}_disp_mag"]).ravel()
        else:
            vals = np.asarray(mesh.point_data[f"{case}_disp"])[:, "xyz".index(q[-1])]

        shown = mesh.copy(deep=False)
        w = float(ui["warp"])
        if w > 0 and f"{case}_disp" in mesh.point_data:
            shown = mesh.copy()
            shown.points = base + w * np.asarray(mesh.point_data[f"{case}_disp"])
        shown["_active"] = vals

        if ui["full_range"]:
            clim = (float(vals.min()), float(vals.max()))
        else:
            clim = (float(np.percentile(vals, 2)), float(np.percentile(vals, 98)))
        if clim[0] == clim[1]:
            clim = (clim[0] - 1e-6, clim[1] + 1e-6)

        style = {}
        if shown.n_cells == 0 or shown.n_cells == shown.n_points:  # bare point cloud
            style = {"point_size": int(ui["psize"]), "render_points_as_spheres": True}
        pl.add_mesh(
            shown,
            scalars="_active",
            cmap=ui["cmap"],
            clim=clim,
            scalar_bar_args={"title": f"{case} {label} [{units}]"},
            name="mesh",
            **style,
        )
        per_case = summary.get("cases", {}).get(case, {})
        pl.add_text(
            f"{name}   {shown.n_points:,} points\n"
            f"{case}: max |stress| {per_case.get('max_abs_stress_MPa', float('nan')):.1f} MPa, "
            f"max |disp| {per_case.get('max_resultant_disp_mm', float('nan')):.3f} mm"
            + (f"\nwarp x{w:.0f}" if w > 0 else ""),
            position="upper_left",
            font_size=10,
            name="caption",
            color=cap,
        )

    draw()
    pl.camera_position = "iso"

    # ---- trame ---------------------------------------------------------------
    from trame.app import get_server
    from trame.app.file_upload import ClientFile
    from trame.ui.vuetify3 import SinglePageWithDrawerLayout
    from trame.widgets import vtk as vtk_widgets
    from trame.widgets import vuetify3 as v3

    server = get_server(client_type="vue3")
    state, ctrl = server.state, server.controller

    state.jobs = completed_jobs(jobs_dir)
    state.job = state.jobs[0] if state.jobs else ""
    state.status_text = f"{len(state.jobs)} completed job(s)"
    state.stl_file = None
    ui["job"] = state.job

    def refresh(**_):
        ui.update(
            job=state.job or "",
            case=state.load_case,
            quantity=state.quantity,
            cmap=state.cmap,
            full_range=state.full_range,
            psize=state.psize,
            warp=state.warp,
            dark=state.dark_mode,
        )
        draw()
        ctrl.view_update()

    for key in ("job", "load_case", "quantity", "cmap", "full_range", "psize", "warp", "dark_mode"):
        state.change(key)(refresh)

    async def _watch(proc, jobdir: Path) -> None:
        rc = await proc.wait()
        live.discard(jobdir.name)
        with state:
            state.jobs = completed_jobs(jobs_dir)
            if rc == 0 and (jobdir / "result.vtp").exists():
                state.status_text = f"{jobdir.name}: done"
                state.job = jobdir.name  # triggers refresh via state.change
            else:
                state.status_text = f"{jobdir.name}: FAILED -- see runner.log in the job dir"

    def launch_job(filename: str, content: bytes) -> str:
        """One door for both the UI button and POST /api/jobs. Raises ValueError."""
        jobdir = new_job_dir(jobs_dir, filename, content)
        live.add(jobdir.name)

        async def _launch():
            log_fh = open(jobdir / "runner.log", "wb")  # noqa: SIM115 - lives with the proc
            proc = await asyncio.create_subprocess_exec(
                sys.executable,
                str(runner),
                str(jobdir / "input.stl"),
                "--out-dir",
                str(jobdir),
                "--ckpt",
                str(a.ckpt),
                "--device",
                a.device,
                stdout=log_fh,
                stderr=asyncio.subprocess.STDOUT,
                cwd=str(SURROGATE),
            )
            await _watch(proc, jobdir)
            log_fh.close()

        asyncio.get_event_loop().create_task(_launch())
        return jobdir.name

    def start_job() -> None:
        f = ClientFile(state.stl_file)
        if not f or not f.name:
            state.status_text = "pick an .stl file first"
            return
        try:
            name = launch_job(f.name, f.content)
        except ValueError as e:
            state.status_text = str(e)
            return
        state.status_text = f"{name}: running ..."

    ctrl.start_job = start_job

    # Real HTTP downloads: a plain aiohttp route on the wslink server.
    from aiohttp import web

    async def handle_download(request):
        job, fname = request.match_info["job"], request.match_info["fname"]
        if fname not in DOWNLOADABLE or not re.fullmatch(r"[A-Za-z0-9_.-]+", job):
            return web.Response(status=400, text="bad request")
        path = jobs_dir / job / fname
        if not path.exists():
            return web.Response(status=404, text="not found")
        return web.FileResponse(
            path, headers={"Content-Disposition": f'attachment; filename="{job}_{fname}"'}
        )

    # ---- REST API: the same launch_job the UI uses, over plain HTTP ------------
    async def api_post_job(request):
        if not (request.content_type or "").startswith("multipart/"):
            return web.json_response(
                {"error": "multipart/form-data with a file field 'stl' is required"}, status=400
            )
        # Streamed multipart, deliberately: request.post() enforces aiohttp's 1MB
        # client_max_size, and a bracket STL is ~30MB.
        reader = await request.multipart()
        filename, content = None, b""
        async for part in reader:
            if part.name == "stl":
                filename = part.filename or ""
                chunks = []
                while True:
                    c = await part.read_chunk(1 << 20)
                    if not c:
                        break
                    chunks.append(c)
                content = b"".join(chunks)
                break
        if filename is None:
            return web.json_response(
                {"error": "multipart field 'stl' with a file is required"}, status=400
            )
        try:
            name = launch_job(filename, content)
        except ValueError as e:
            return web.json_response({"error": str(e)}, status=400)
        return web.json_response({"job": name, "status": "running"}, status=202)

    async def api_list_jobs(request):
        return web.json_response({"jobs": list_jobs(jobs_dir, live)})

    async def api_get_job(request):
        name = request.match_info["job"]
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
            return web.json_response({"error": "bad job name"}, status=400)
        payload = job_payload(jobs_dir, name, live)
        if payload is None:
            return web.json_response({"error": "unknown job"}, status=404)
        return web.json_response(payload)

    def on_server_bind(wslink_server):
        wslink_server.app.add_routes(
            [
                web.get("/download/{job}/{fname}", handle_download),
                web.post("/api/jobs", api_post_job),
                web.get("/api/jobs", api_list_jobs),
                web.get("/api/jobs/{job}", api_get_job),
            ]
        )

    ctrl.on_server_bind.add(on_server_bind)

    with SinglePageWithDrawerLayout(server) as layout:
        layout.title.set_text("DeepStaticSim")
        # Vuetify 3 theme switch; the 3D viewport follows in draw().
        layout.root.theme = ("dark_mode ? 'dark' : 'light'",)
        with layout.toolbar as tb:
            tb.density = "compact"
            v3.VSpacer()
            v3.VSwitch(
                v_model=("dark_mode", False),
                label="dark",
                density="compact",
                hide_details=True,
                classes="mx-2",
            )
            v3.VSelect(
                v_model=("job", state.job),
                items=("jobs", state.jobs),
                label="job",
                density="compact",
                hide_details=True,
                variant="outlined",
                style="max-width:260px",
                classes="mx-1",
            )
            v3.VSelect(
                v_model=("load_case", "ver"),
                items=("cases", [{"title": c, "value": c} for c in CASES]),
                label="load case",
                density="compact",
                hide_details=True,
                variant="outlined",
                style="max-width:130px",
                classes="mx-1",
            )
            v3.VSelect(
                v_model=("quantity", "stress"),
                items=("quantities", [{"title": v[0], "value": k} for k, v in QUANTITIES.items()]),
                label="quantity",
                density="compact",
                hide_details=True,
                variant="outlined",
                style="max-width:190px",
                classes="mx-1",
            )
        with layout.drawer as drawer:
            drawer.width = 330
            with v3.VCard(flat=True, classes="pa-3"):
                v3.VCardTitle("New analysis", classes="pa-0 pb-2")
                v3.VFileInput(
                    v_model=("stl_file", None),
                    accept=".stl",
                    label="STL file",
                    density="compact",
                    variant="outlined",
                    show_size=True,
                    hide_details=True,
                )
                v3.VBtn(
                    "Run",
                    click=ctrl.start_job,
                    color="primary",
                    block=True,
                    classes="mt-2",
                )
                v3.VCardText("{{ status_text }}", classes="px-0 py-2 text-caption")
                v3.VDivider()
                v3.VCardTitle("Display", classes="pa-0 py-2")
                v3.VSelect(
                    v_model=("cmap", "coolwarm"),
                    items=("cmaps", list(CMAPS)),
                    label="colormap",
                    density="compact",
                    variant="outlined",
                    hide_details=True,
                    classes="mb-2",
                )
                v3.VSlider(
                    v_model=("warp", 0.0),
                    min=0,
                    max=500,
                    step=10,
                    label="warp",
                    density="compact",
                    hide_details=True,
                    thumb_label=True,
                )
                v3.VSlider(
                    v_model=("psize", 3),
                    min=1,
                    max=8,
                    step=1,
                    label="pt size",
                    density="compact",
                    hide_details=True,
                )
                v3.VSwitch(
                    v_model=("full_range", False),
                    label="full color range",
                    density="compact",
                    hide_details=True,
                )
                v3.VDivider(classes="my-2")
                v3.VCardTitle("Export", classes="pa-0 py-2")
                for fname, label in (
                    ("result.vtp", "VTP (ParaView)"),
                    ("result.csv", "CSV"),
                    ("summary.json", "summary"),
                ):
                    v3.VBtn(
                        label,
                        href=(f"'/download/' + job + '/{fname}'",),
                        disabled=("!job",),
                        variant="tonal",
                        block=True,
                        classes="mb-1",
                    )
                v3.VCardText(
                    "Surrogate prediction, not FEA. Trained on 27 DeepJEB brackets with "
                    "fixed bolt BCs and the dataset's four load cases; accuracy on "
                    "out-of-family shapes is unverified.",
                    classes="px-0 pt-2 text-caption text-medium-emphasis",
                )
        with layout.content:
            with v3.VContainer(fluid=True, classes="pa-0 fill-height"):
                view = vtk_widgets.VtkRemoteView(pl.ren_win, interactive_ratio=1)
                ctrl.view_update = view.update
                ctrl.view_reset_camera = view.reset_camera

    log.info("jobs dir %s | runner %s | ckpt %s", jobs_dir, runner, a.ckpt)
    log.info("serving on http://%s:%d", host, a.port)
    server.start(host=host, port=a.port, open_browser=False, timeout=0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
