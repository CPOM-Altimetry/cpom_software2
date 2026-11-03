#!/usr/bin/env python
"""cpom.altimetry.projects.clev2er_liiw.uncertainty.aggregate_p2p_diffs

Aggregate the monthly point-to-point difference files written by
``validate_l2_altimetry_elevations.py`` into one yearly ``.npz`` per (area, band),
which is the form consumed by the CLEV2ER POCA uncertainty-LUT builders
(``clev2er.utils.uncertainty.create_luts_from_is2_diffs`` and ``create_4d_luts``).

``validate_l2_altimetry_elevations`` writes one
``<outdir>/<year>/<month>/p2p_diffs_<area>.npz`` per run, each holding equal-length
1-D arrays (``dh``, ``lats``, ``lons``, ``sig0``, ``coherence`` ...). Different months
can hold a different set of keys - e.g. a month with only LR-Ku data has no
``coherence`` key - so the files are concatenated on the **union** of keys, filling a
file's missing keys with NaN (so the downstream HR/LR split by coherence still works).

Usage
-----
::

    python -m cpom.altimetry.projects.clev2er_liiw.uncertainty.aggregate_p2p_diffs \\
        --in-dir   /.../is2_diffs/antarctica/ku \\
        --out-file /.../is2_diffs/antarctica/ku/cs2_minus_is2_2020_antarctica_ku.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def find_monthly_files(in_dir: Path, pattern: str = "**/p2p_diffs_*.npz") -> list[Path]:
    """Return the sorted list of monthly p2p-diff npz files under ``in_dir``.

    Args:
        in_dir (Path): directory searched recursively.
        pattern (str): glob pattern for the monthly files.

    Returns:
        list[Path]: matching files, sorted by path.
    """
    return sorted(Path(in_dir).glob(pattern))


def _row_count(data: dict[str, np.ndarray]) -> int:
    """Number of records in one file's arrays (uses ``dh`` if present)."""
    if "dh" in data:
        return int(len(data["dh"]))
    return int(max((len(value) for value in data.values()), default=0))


def aggregate_npz_files(
    files: list[Path],
) -> tuple[dict[str, np.ndarray], list[tuple[Path, int]]]:
    """Concatenate npz files on the union of keys, NaN-filling missing keys.

    Args:
        files: list of ``.npz`` paths, each holding equal-length 1-D arrays.

    Returns:
        (aggregated, counts) where ``aggregated`` maps each key (the union over all
        files) to the concatenated array (values missing in a file are NaN), and
        ``counts`` is the per-file ``(path, n_records)`` list.
    """
    loaded: list[tuple[dict[str, np.ndarray], int]] = []
    counts: list[tuple[Path, int]] = []
    keys: list[str] = []  # union of keys, in first-seen order
    for file in files:
        with np.load(file) as npz:
            data = {key: npz[key] for key in npz.files}
        n_records = _row_count(data)
        loaded.append((data, n_records))
        counts.append((file, n_records))
        for key in data:
            if key not in keys:
                keys.append(key)

    columns: dict[str, list[np.ndarray]] = {key: [] for key in keys}
    for data, n_records in loaded:
        for key in keys:
            columns[key].append(
                np.asarray(data[key]) if key in data else np.full(n_records, np.nan)
            )
    aggregated = {
        key: (np.concatenate(parts) if parts else np.array([])) for key, parts in columns.items()
    }
    return aggregated, counts


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point. Returns a process exit code."""
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate monthly p2p_diffs_<area>.npz files (from "
            "validate_l2_altimetry_elevations) into one yearly npz per (area, band)."
        )
    )
    parser.add_argument(
        "--in-dir",
        type=Path,
        required=True,
        help="directory searched recursively for the monthly p2p_diffs_*.npz files",
    )
    parser.add_argument("--out-file", type=Path, required=True, help="output aggregated .npz path")
    parser.add_argument(
        "--pattern",
        default="**/p2p_diffs_*.npz",
        help="[optional] glob pattern for the monthly files (default %(default)s)",
    )
    args = parser.parse_args(argv)

    files = find_monthly_files(args.in_dir, args.pattern)
    if not files:
        print(f"No files matching '{args.pattern}' under {args.in_dir}", file=sys.stderr)
        return 1

    print(f"Aggregating {len(files)} file(s) from {args.in_dir}")
    aggregated, counts = aggregate_npz_files(files)
    for file, n_records in counts:
        print(f"  {n_records:8d}  {file}")

    n_total = len(aggregated.get("dh", []))
    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    # numpy's savez stub mistypes **kwds (expects bool), so ignore arg-type below
    np.savez(args.out_file, **aggregated)  # type: ignore[arg-type]
    print(f"Wrote {args.out_file}  ({n_total} points, keys: {', '.join(aggregated)})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
