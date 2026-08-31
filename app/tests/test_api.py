"""The REST API's pure helpers, against fabricated job directories.

The aiohttp handlers in server.py are thin wrappers over these functions plus
`launch_job` (which needs a model and a subprocess); everything decidable --
sanitization, status derivation, listing, payload shape -- is decided here, so
these tests need no checkpoint, no server, no network.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.server import (  # noqa: E402
    completed_jobs,
    job_payload,
    job_status,
    list_jobs,
    new_job_dir,
    sanitize_stem,
)


def _done_job(jobs_dir: Path, name: str, summary: dict | None = None) -> Path:
    d = jobs_dir / name
    d.mkdir(parents=True)
    (d / "result.vtp").write_bytes(b"x" * 16)
    (d / "result.csv").write_text("x_mm\n")
    if summary is not None:
        (d / "summary.json").write_text(json.dumps(summary))
    return d


def test_sanitize_stem():
    assert sanitize_stem("bracket v2 (final).stl") == "bracket_v2__final_"
    assert sanitize_stem("../../etc/passwd.stl") == "passwd"  # Path().stem first
    assert sanitize_stem("") == "job"  # never empty
    assert len(sanitize_stem("x" * 200 + ".stl")) <= 40


def test_new_job_dir_validation(tmp_path):
    with pytest.raises(ValueError, match="not an .stl"):
        new_job_dir(tmp_path, "part.step", b"data")
    with pytest.raises(ValueError, match="empty"):
        new_job_dir(tmp_path, "part.stl", b"")
    d = new_job_dir(tmp_path, "part.stl", b"solid x")
    assert (d / "input.stl").read_bytes() == b"solid x"
    d2 = new_job_dir(tmp_path, "part.stl", b"solid y")  # same second must not collide
    assert d2 != d and (d2 / "input.stl").exists()


def test_job_status(tmp_path):
    _done_job(tmp_path, "a_done")
    (tmp_path / "b_failed").mkdir()
    live = {"c_running"}
    assert job_status(tmp_path, "a_done", live) == "done"
    assert job_status(tmp_path, "b_failed", live) == "failed"
    assert job_status(tmp_path, "c_running", live) == "running"
    assert job_status(tmp_path, "nope", live) is None


def test_list_and_completed(tmp_path):
    import os
    import time

    _done_job(tmp_path, "old")
    time.sleep(0.02)
    _done_job(tmp_path, "new")
    os.utime(tmp_path / "old" / "result.vtp")  # mtime of the DIR decides, not the file
    (tmp_path / "half").mkdir()  # no result.vtp -> not listed as done
    assert completed_jobs(tmp_path) == ["new", "old"]
    jobs = list_jobs(tmp_path, {"run1"})
    assert jobs[0] == {"job": "run1", "status": "running"}
    assert {j["job"] for j in jobs} == {"run1", "new", "old"}


def test_job_payload(tmp_path):
    summary = {"n_points": 7, "cases": {"ver": {"max_abs_stress_MPa": 1.0}}}
    _done_job(tmp_path, "a", summary)
    p = job_payload(tmp_path, "a", set())
    assert p["status"] == "done"
    assert p["summary"] == summary
    assert "/download/a/result.vtp" in p["downloads"]
    assert "/download/a/result.csv" in p["downloads"]
    assert all(d.startswith("/download/a/") for d in p["downloads"])

    (tmp_path / "bad").mkdir()
    (tmp_path / "bad" / "result.vtp").write_bytes(b"x")
    (tmp_path / "bad" / "summary.json").write_text("{not json")
    assert job_payload(tmp_path, "bad", set())["summary"] == {}  # corrupt -> {}, not crash

    assert job_payload(tmp_path, "missing", set()) is None
    assert job_payload(tmp_path, "r", {"r"})["status"] == "running"
