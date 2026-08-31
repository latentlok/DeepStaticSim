"""Exports for one surrogate job: .vtp (3D), .csv (tabular), summary.json.

The .vtp carries each load case's displacement as a VECTOR array so ParaView's
Warp By Vector works out of the box, alongside its magnitude and the signed von
Mises stress as scalars. The csv is the same 16 channels flat, one row per
surface point, for spreadsheets and downstream scripts. summary.json is the
engineer-facing digest: peak stress / peak displacement per load case.

Channel order everywhere (fixed by models/transolver.py):
[ver_disp x,y,z, ver_stress, hor_disp x,y,z, hor_stress,
 dia_disp x,y,z, dia_stress, tor_disp x,y,z, tor_stress].
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

CASES = ("ver", "hor", "dia", "tor")
DISP_SLICE = {"ver": slice(0, 3), "hor": slice(4, 7), "dia": slice(8, 11), "tor": slice(12, 15)}
STRESS_IDX = {"ver": 3, "hor": 7, "dia": 11, "tor": 15}

CSV_FIELDS = ["x_mm", "y_mm", "z_mm"]
for _case in CASES:
    CSV_FIELDS += [f"{_case}_disp_{_ax}_mm" for _ax in "xyz"]
    CSV_FIELDS += [f"{_case}_stress_MPa"]

DISCLAIMER = (
    "Surrogate prediction from a Transolver model trained on 27 DeepJEB bracket "
    "designs under the dataset's four fixed load cases (identical bolted boundary "
    "conditions for every design). Not a certified FEA replacement: verify "
    "critical designs with a full solver."
)


def write_vtp(path: Path, feats: dict, pred: np.ndarray) -> None:
    import pyvista as pv

    points = np.asarray(feats["position"], dtype=np.float32)
    if "faces" in feats:
        # A real surface: the STL's own triangulation, so viewers shade an
        # interpolated surface instead of drawing points.
        f = np.asarray(feats["faces"], dtype=np.int64)
        cloud = pv.PolyData(points, faces=np.hstack([np.full((len(f), 1), 3, np.int64), f]))
    else:
        cloud = pv.PolyData(points)
    for case in CASES:
        disp = pred[:, DISP_SLICE[case]]
        cloud[f"{case}_disp"] = disp.astype(np.float32)  # vector -> Warp By Vector
        cloud[f"{case}_disp_mag"] = np.linalg.norm(disp, axis=1).astype(np.float32)
        cloud[f"{case}_stress"] = pred[:, STRESS_IDX[case]].astype(np.float32)
    cloud["normal"] = np.asarray(feats["normal"], dtype=np.float32)
    cloud["area"] = np.asarray(feats["area"], dtype=np.float32).ravel()
    cloud.save(str(path))


def write_csv(path: Path, feats: dict, pred: np.ndarray) -> None:
    rows = np.concatenate([np.asarray(feats["position"], dtype=np.float32), pred], axis=1)
    np.savetxt(path, rows, delimiter=",", header=",".join(CSV_FIELDS), comments="", fmt="%.6g")


def write_summary(path: Path, feats: dict, pred: np.ndarray, ckpt, timings: dict) -> dict:
    per_case = {}
    for case in CASES:
        disp = pred[:, DISP_SLICE[case]]
        per_case[case] = {
            "max_abs_stress_MPa": float(np.abs(pred[:, STRESS_IDX[case]]).max()),
            "max_resultant_disp_mm": float(np.linalg.norm(disp, axis=1).max()),
        }
    summary = {
        "n_points": int(len(pred)),
        "volume_mm3": float(feats["volume_mm3"]),
        "area_mm2": float(feats["area_mm2"]),
        "cases": per_case,
        "checkpoint": str(ckpt),
        "timings_s": {k: round(float(v), 3) for k, v in timings.items()},
        "disclaimer": DISCLAIMER,
    }
    Path(path).write_text(json.dumps(summary, indent=2))
    return summary
