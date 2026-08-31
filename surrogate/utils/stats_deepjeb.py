"""Train-split per-variable mean/std for the DeepJEB store -> stats_surface.json.

    uv run python utils/stats_deepjeb.py --root $DL_DATA

Statistics are computed OFFLINE and read from a file at train time -- never in
DataModule.setup() -- for the template's standing reasons: scanning the store per
run costs time, and worse, the val/test splits must normalise against the TRAIN
numbers, not against whatever split happens to be mounted. The json is keyed by
variable name ("position_mean", ...), never by column position, so reordering a
config cannot silently pair a variable with another variable's statistics.

One wrinkle is ver_disp: its x column exists only for designs whose csv was
present at fetch time (attrs["ver_x_valid"]) and is NaN elsewhere. All three
ver_disp columns are therefore accumulated ONLY over ver_x_valid train designs --
a NaN would poison the mean without raising, and normalising x against
statistics from a different design subset than y/z would be quietly wrong. If no
train design can define the channel, this refuses loudly rather than writing a
stats file that undefines channel 0.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

# Store variables, in the y-channel order documented in configs/model/transolver_surface.yaml.
VARS = (
    "position",
    "normal",
    "area",
    "ver_disp",
    "ver_stress",
    "hor_disp",
    "hor_stress",
    "dia_disp",
    "dia_stress",
    "tor_disp",
    "tor_stress",
)


def compute_stats(root: Path, store: str = "deepjeb.zarr") -> dict[str, list[float]]:
    """Stream sum/sumsq over the train split; return {var}_mean / {var}_std lists."""
    import zarr

    root = Path(root)
    splits_path = root / "splits.json"
    if not splits_path.exists():
        raise SystemExit(f"no {splits_path} -- run utils/fetch_deepjeb.py first")
    train = json.loads(splits_path.read_text())["train"]
    if not train:
        raise SystemExit(f"{splits_path} has an empty train split")

    z = zarr.open_group(str(root / store), mode="r")
    acc: dict[str, list] = {v: [0, None, None] for v in VARS}  # n, sum, sumsq per column
    ver_x_designs = 0
    for design in train:
        if design not in z:
            raise SystemExit(
                f"train design {design} is in splits.json but not in {root / store} -- "
                f"fetch it before computing statistics"
            )
        g = z[design]["surface"]
        valid = bool(g.attrs["ver_x_valid"])
        ver_x_designs += valid
        for var in VARS:
            if var == "ver_disp" and not valid:
                continue  # its x column is NaN here; see module docstring
            arr = np.asarray(g[var][:], dtype=np.float64)
            n, s, sq = acc[var]
            acc[var] = [
                n + arr.shape[0],
                arr.sum(0) if s is None else s + arr.sum(0),
                (arr**2).sum(0) if sq is None else sq + (arr**2).sum(0),
            ]
    if ver_x_designs == 0:
        raise SystemExit(
            "no train design has ver_x_valid=True, so ver_disp statistics are "
            "undefined. Fetch the missing csvs (utils/fetch_deepjeb.py --force "
            "--only <ids>) or change the split."
        )

    stats: dict[str, list[float]] = {}
    for var, (n, s, sq) in acc.items():
        mean = s / n
        var_ = np.maximum(sq / n - mean**2, 0.0)
        stats[f"{var}_mean"] = mean.tolist()
        stats[f"{var}_std"] = np.sqrt(var_).tolist()
    log.info(
        "stats over %d train design(s); ver_disp from the %d with ver_x",
        len(train),
        ver_x_designs,
    )
    return stats


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--root", type=Path, required=True, help="processed data root ($DL_DATA)")
    p.add_argument("--store", default="deepjeb.zarr")
    p.add_argument("--out", default="stats_surface.json")
    a = p.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    stats = compute_stats(a.root, a.store)
    out = a.root / a.out
    out.write_text(json.dumps(stats, indent=2))
    log.info("wrote %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
