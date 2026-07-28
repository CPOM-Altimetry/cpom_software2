# pylint: disable=C0302
"""
cpom.altimetry.tools.sec_tools.grid_and_surface_fit

Apply the surface fit pipeline to a parquet altimetry archive,
gridding the data. This script is intended to be used at lancaster and
replaces the grid_for_elev_change.py -> surface_fit.py workflow.

See surface_fit.py for the surface fit algorithm details.
"""

import argparse
import json
import logging
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict

import numpy as np
import polars as pl
import pyarrow.parquet as pq

from cpom.altimetry.datasets.parquet_tools.spatio_temporal import ParquetFilter
from cpom.altimetry.tools.sec_tools.metadata_helper import (
    elapsed,
    get_algo_name,
    get_metadata_params,
    write_metadata,
)
from cpom.altimetry.tools.sec_tools.surface_fit import (
    clean_directory,
    filter_to_mode,
    fit_linear_fit_per_group,
    fit_power_correction_per_group,
    fit_surface_model_per_group,
)
from cpom.altimetry.tools.sec_tools.surface_fit import (
    parse_arguments as surface_fit_parse_arguments,
)
from cpom.gridding.gridareas import GridArea
from cpom.masks.masks import Mask

SECONDS_PER_YEAR = 31557600


# ----------------------------------------------------------------------------
# SETUP FUNCTIONS - Initialize config, logging, time bounds, chunks
# ---------------------------------------------------------------------------


def parse_arguments(args):
    """Parse command line arguments for grid + surface fitting.

    Reuses the surface fit argument parser and adds gridding arguments.
    """
    grid_parser = argparse.ArgumentParser(
        description=("Compute fitted elevations from parquet altimetry store.")
    )
    grid_parser.add_argument(
        "--data_crs",
        type=int,
        default=3413,
        help="EPSG code of the archive's x/y columns.",
    )
    grid_parser.add_argument(
        "--mask_name",
        type=str,
        required=True,
        help="CPOM grid mask name to filter the archive to.",
    )
    grid_parser.add_argument(
        "--gridarea",
        type=str,
        required=True,
        help="CPOM GridArea name to bin the archive onto, e.g. 'greenland'.",
    )
    grid_parser.add_argument(
        "--binsize",
        type=int,
        required=True,
        help="Grid binsize in metres to bin the archive onto, e.g. 5000.",
    )
    grid_parser.add_argument(
        "--partition_xy_chunking",
        type=int,
        default=200,
        help=(
            "Number of grid cells per processing chunk side (x_bin/y_bin block size). "
            "Purely a memory-chunking knob - has no effect on results."
        ),
    )
    grid_parser.add_argument(
        "--force_regrid",
        action="store_true",
        help="Rebuild the gridded cache even if a matching one already exists.",
    )
    grid_parser.add_argument(
        "--memory_limit",
        type=str,
        default="100GB",
        help="Memory limit for the DuckDB gridding query (e.g. '100GB', '4GB').",
    )
    grid_parser.add_argument(
        "--max_workers",
        type=int,
        default=None,
        help="Number of worker processes for per-cell fitting; defaults to os.cpu_count().",
    )

    grid_params, remaining_args = grid_parser.parse_known_args(args)
    params = surface_fit_parse_arguments(remaining_args)
    for key, value in vars(grid_params).items():
        setattr(params, key, value)
    return params


def get_surface_fit_objects(
    params: argparse.Namespace,
    standard_epoch: str,
    logger: logging.Logger,
    cache_dir: str,
    min_secs: float,
    max_secs: float,
    loaded_stats: dict[str, int],
) -> dict[str, Any]:
    """Initialise the objects required for the surface fit pipeline.

    Computes time boundaries, minimum cell time span, unique spatial chunks,
    and status counters seeded with the gridding stage's loaded stats.

    Args:
        params (argparse.Namespace): Command line parameters.
        standard_epoch (str): Reference epoch for time calculations (ISO format string)
        logger (logging.Logger): Logger object
        cache_dir (str): Directory of the cached, finalized gridded archive to read chunks from.
        min_secs (float): Surface fit window start in seconds since epoch.
        max_secs (float): Surface fit window end in seconds since epoch.
        loaded_stats (dict[str, int]): "loaded" status counts from `build_gridded_archive`.
    Returns:
        Dict of mintime, maxtime, min_secs, max_secs, pc_min_secs, pc_max_secs,
        status, part_df, cache_dir - the full surface fit initialization state.
    """

    def _init_status() -> dict[str, int]:
        status = {
            k: 0
            for k in [
                "n_cells_with_data_loaded",
                "n_measurements_loaded",
                "n_cells_time_identical",
                "n_cells_with_timespan_too_short",
                "n_cells_with_too_few_measurements",
                "n_cells_with_too_few_measurements_after_fit_sigma_filter",
                "n_cells_fit_failed",
                "n_cells_didnot_converge",
                "n_cells_power_corrected",
                "n_cells_too_few_values_in_linear_fit",
                "n_cells_too_few_vals_after_pctime_filter",
                "n_cells_fitted",
            ]
        }
        status.update(loaded_stats)
        return status

    epoch_dt = datetime.fromisoformat(standard_epoch)
    mintime = (epoch_dt + timedelta(seconds=min_secs)).timestamp()
    maxtime = (epoch_dt + timedelta(seconds=max_secs)).timestamp()
    logger.info("Surface fit time range %s: %s : %s", standard_epoch, mintime, maxtime)

    if params.powercorrect:
        if params.pcmintime is not None and params.pcmaxtime is not None:
            _, pc_min_secs = _parse_date_to_secs(epoch_dt, params.pcmintime)
            _, pc_max_secs = _parse_date_to_secs(epoch_dt, params.pcmaxtime)
        else:
            pc_min_secs, pc_max_secs = min_secs, max_secs
    else:
        pc_min_secs, pc_max_secs = None, None

    params.min_timespan_in_cell_in_secs = (maxtime - mintime) * (
        params.min_percent_timespan_in_cell / 100.0
    )

    part_df = (
        pl.scan_parquet(
            os.path.join(cache_dir, "x_part=*", "y_part=*", "*.parquet"), hive_partitioning=True
        )
        .select("x_part", "y_part")
        .unique()
        .sort(["x_part", "y_part"])
        .collect()
    )
    logger.info("Found %d chunks to process", len(part_df))

    return {
        "standard_epoch": standard_epoch,
        "mintime": mintime,
        "maxtime": maxtime,
        "min_secs": min_secs,
        "max_secs": max_secs,
        "pc_min_secs": pc_min_secs,
        "pc_max_secs": pc_max_secs,
        "status": _init_status(),
        "part_df": part_df,
        "cache_dir": cache_dir,
    }


# ---------------------------------------------------------------------------
# CREATE GRIDDED DATASET
# ---------------------------------------------------------------------------


def _parse_date_to_secs(epoch_time: datetime, timedt: str) -> tuple[datetime, float]:
    """Parse a DD/MM/YYYY or DD.MM.YYYY date string, return (datetime, seconds_since_epoch)."""
    if "/" in timedt:
        time_dt = datetime.strptime(timedt, "%d/%m/%Y")
    elif "." in timedt:
        time_dt = datetime.strptime(timedt, "%d.%m.%Y")
    else:
        raise ValueError(f"Unrecognized date format: {timedt}, pass as DD/MM/YYYY or DD.MM.YYYY ")
    return time_dt, (time_dt - epoch_time).total_seconds()


def _gridded_cache_fingerprint(params: argparse.Namespace, standard_epoch: str) -> dict[str, Any]:
    """Parameters that determine whether an existing gridded cache can be reused."""
    return {
        "in_dir": str(Path(params.in_dir).resolve()),
        "parquet_glob": params.parquet_glob,
        "data_crs": params.data_crs,
        "mask_name": params.mask_name,
        "gridarea": params.gridarea,
        "binsize": params.binsize,
        "partition_xy_chunking": params.partition_xy_chunking,
        "x_column": params.x_column,
        "y_column": params.y_column,
        "elevation_column": params.elevation_column,
        "heading_column": params.heading_column,
        "standard_epoch": standard_epoch,
        "mintime": params.mintime,
        "maxtime": params.maxtime,
        "mode_values": params.mode_values,
        "mode_column": params.mode_column,
    }


def _compute_time_bounds(cache_dir: str, time_column: str = "time") -> tuple[float, float]:
    """Return (min, max) of the cached archive's time column via a DuckDB aggregate scan.

    Args:
        cache_dir: Directory of the cached gridded archive.
        time_column: Name of the time column.
    Returns:
        (min_secs, max_secs)
    """
    glob = os.path.join(cache_dir, "x_part=*", "y_part=*", "*.parquet")
    pfilter = ParquetFilter(glob, engine="duckdb")
    try:
        bounds = pfilter.select(
            [f"MIN({time_column}) AS min_secs", f"MAX({time_column}) AS max_secs"]
        ).run()
    finally:
        pfilter.close()
    assert isinstance(bounds, pl.DataFrame)
    return float(bounds["min_secs"][0]), float(bounds["max_secs"][0])


def _finalise_gridded_cache(
    raw_dir: Path,
    out_dir: str,
    params: argparse.Namespace,
    min_secs: float,
    max_secs: float,
) -> dict[str, int]:
    """Filter one raw gridded chunk down to surface-fit-ready rows, in place.

    Applies the fit time window, --mode_values, each cell's dominant mode, and
    keeps only cells with a valid elevation. Overwrites the chunk's parquet
    file(s) with the finalized result.

    Args:
        raw_dir: Directory of this chunk's raw partition file(s).
        params: Command line arguments.
        min_secs: Surface-fit window start in seconds since epoch.
        max_secs: Surface-fit window end in seconds since epoch.
    Returns:
        stats (n_measurements_loaded, n_cells_with_data_loaded)
    """
    raw_glob = str(raw_dir / "x_part=*" / "y_part=*" / "*.parquet")
    lf = pl.scan_parquet(raw_glob, hive_partitioning=True)
    schema_cols = lf.collect_schema().names()
    extra_cols = [
        c
        for c in [params.power_column, params.mode_column, params.weight_column]
        if c and c in schema_cols
    ]

    stats = lf.select(
        pl.len().alias("n_measurements_loaded"),
        pl.struct(["x_bin", "y_bin"])
        .filter(pl.col("height").is_not_null())
        .n_unique()
        .alias("n_cells_with_data_loaded"),
    )

    if bool(params.mode_column) and params.mode_column in schema_cols:
        if params.mode_values:
            lf = lf.filter(pl.col(params.mode_column).is_in(params.mode_values))
        lf = filter_to_mode(lf)

    lf = lf.join(
        lf.filter(
            (pl.col("time") >= min_secs)
            & (pl.col("time") <= max_secs)
            & (pl.col("height").is_not_null())
        )
        .select(["x_bin", "y_bin"])
        .unique(),
        on=["x_bin", "y_bin"],
        how="inner",
    ).select(
        [
            "x_part",
            "y_part",
            "x_bin",
            "y_bin",
            "x",
            "y",
            "x2",
            "y2",
            "xy",
            "height",
            "heading",
            "time",
            "time_years",
        ]
        + extra_cols
    )

    ParquetFilter(lf, engine="polars").run(output_path=out_dir, partition_by=["x_part", "y_part"])

    stats = stats.collect()

    return {
        "n_measurements_loaded": int(stats["n_measurements_loaded"][0]),
        "n_cells_with_data_loaded": int(stats["n_cells_with_data_loaded"][0]),
    }


def build_gridded_archive(
    params: argparse.Namespace, standard_epoch: str, logger: logging.Logger
) -> tuple[str, float, float, dict[str, int]]:
    """Grid the raw parquet archive into a cached, disk-partitioned, surface-fit-ready dataset.

    Reuses an existing cache if its fingerprint (`_gridded_cache_fingerprint`) matches;
    pass --force_regrid to always rebuild.

    Args:
        params: Command line parameters.
        standard_epoch: Reference epoch for time calculations (ISO format string).
        logger: Logger object.
    Returns:
        (cache_dir, min_secs, max_secs, loaded_stats)
    """
    cache_dir = str(Path(params.out_dir) / "grid_cache")
    fingerprint = _gridded_cache_fingerprint(params, standard_epoch)
    meta_path = Path(cache_dir) / "cache_meta.json"

    if not params.force_regrid and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta["fingerprint"] == fingerprint:
            logger.info("Reusing existing gridded cache at %s", cache_dir)
            return cache_dir, meta["min_secs"], meta["max_secs"], meta["loaded_stats"]

    if Path(cache_dir).exists():
        shutil.rmtree(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    raw_dir = Path(cache_dir) / "_raw"
    os.makedirs(raw_dir, exist_ok=True)

    parquet_glob = os.path.join(params.in_dir, params.parquet_glob)
    logger.info(
        "Building gridded cache at %s from %s (mask=%s, grid_area=%s@%sm)",
        cache_dir,
        parquet_glob,
        params.mask_name,
        params.gridarea,
        params.binsize,
    )

    temp_directory = os.path.join(cache_dir, "duckdb_temp")
    pfilter = (
        ParquetFilter(
            parquet_glob,
            data_crs=params.data_crs,
            engine="duckdb",
            memory_limit=params.memory_limit,
            temp_directory=temp_directory,
        )
        .get_cpom_grid_mask(mask=Mask(params.mask_name))
        .get_grid_cells_from_grid_area(GridArea(params.gridarea, params.binsize))
        .custom_clause(
            [
                f"CAST(date_diff('second', TIMESTAMP '{standard_epoch}', datetime) AS DOUBLE) "
                "AS time",
                f"time / {SECONDS_PER_YEAR}.0 AS time_years",
                f"CAST(x_bin / {params.partition_xy_chunking} AS BIGINT) AS x_part",
                f"CAST(y_bin / {params.partition_xy_chunking} AS BIGINT) AS y_part",
                f"{params.x_column} AS x",
                f"{params.y_column} AS y",
                f"({params.x_column}) * ({params.x_column}) AS x2",
                f"({params.y_column}) * ({params.y_column}) AS y2",
                f"({params.x_column}) * ({params.y_column}) AS xy",
                f"{params.elevation_column} AS height",
                f"CAST({params.heading_column} AS INTEGER) AS heading",
            ]
        )
    )
    try:
        pfilter.run(output_path=str(raw_dir), partition_by=["x_part", "y_part"])
    finally:
        pfilter.close()

    # Resolve the fit time window now, from the raw (unfiltered) cache just built.
    epoch_dt = datetime.fromisoformat(standard_epoch)
    if params.mintime is not None and params.maxtime is not None:
        _, min_secs = _parse_date_to_secs(epoch_dt, params.mintime)
        _, max_secs = _parse_date_to_secs(epoch_dt, params.maxtime)
    else:
        min_secs, max_secs = _compute_time_bounds(str(raw_dir))

    loaded_stats = _finalise_gridded_cache(raw_dir, cache_dir, params, min_secs, max_secs)

    shutil.rmtree(raw_dir)
    if os.path.isdir(temp_directory):
        shutil.rmtree(temp_directory, ignore_errors=True)

    meta_path.write_text(
        json.dumps(
            {
                "fingerprint": fingerprint,
                "min_secs": min_secs,
                "max_secs": max_secs,
                "loaded_stats": loaded_stats,
            },
            indent=2,
        )
    )
    logger.info("Gridded cache build complete")

    return cache_dir, min_secs, max_secs, loaded_stats


# ------------------------
# Output Functions
# ------------------------
def write_chunk_output(
    params: argparse.Namespace,
    timeseries_frames: list[pl.DataFrame] | None = None,
    row: dict | None = None,
) -> None:
    """Write this chunk's timeseries output to <out_dir>/x_part=X/y_part=Y/dh_time_grid.parquet.

    Args:
        params: Command line arguments.
        timeseries_frames: Per-cell timeseries frames to write.
        row: x_part/y_part for directory partitioning; if None, writes to the root output dir.
    """
    if not timeseries_frames:
        return

    chunk_outdir = (
        Path(params.out_dir) / f"x_part={row['x_part']}" / f"y_part={row['y_part']}"
        if row is not None
        else Path(params.out_dir)
    )
    os.makedirs(chunk_outdir, exist_ok=True)
    pl.concat(timeseries_frames).lazy().sink_parquet(
        chunk_outdir / "dh_time_grid.parquet", compression="zstd"
    )


class GridWriter:
    """Appends each chunk's grid-cell records straight to a single grid_data.parquet."""

    def __init__(self, out_dir: str):
        self._path = Path(out_dir) / "grid_data.parquet"
        self._writer: pq.ParquetWriter | None = None

    def write(self, grid_records: list[dict]) -> None:
        """Append this chunk's grid-cell records, opening the file on first use."""
        if not grid_records:
            return
        table = pl.DataFrame(grid_records).to_arrow()
        if self._writer is None:
            self._writer = pq.ParquetWriter(self._path, table.schema, compression="zstd")
        self._writer.write_table(table)

    def close(self) -> None:
        """Close the underlying parquet file, if anything was written."""
        if self._writer is not None:
            self._writer.close()


def get_metadata_json(
    params: argparse.Namespace,
    status: Dict[str, int],
    start_time: float,
    logger: logging.Logger,
) -> None:
    """Write surface fit run metadata to <out_dir>/surface_fit_meta.json."""
    meta_json_path = Path(params.out_dir)
    try:
        write_metadata(
            params,
            get_algo_name(__file__),
            meta_json_path,
            {
                **vars(params),
                **status,
                "execution_time": elapsed(start_time),
            },
        )
        logger.info("Wrote data_set metadata to folder %s", meta_json_path)

    except OSError as e:
        logger.error("Failed to write surface_fit_meta.json with %s", e)


# --------------------------------------------
# Main Processing Workflow
# --------------------------------------------

# Per-worker-process context, set once via ProcessPoolExecutor's initializer
# instead of being pickled and sent through the task queue for every cell.
_worker_ctx: dict[str, Any] = {}


def _init_worker(
    params: argparse.Namespace,
    min_secs: float,
    max_secs: float,
    pc_min_secs: float | None,
    pc_max_secs: float | None,
) -> None:
    """Pool initializer: stash the run-constant context once per worker process."""
    global _worker_ctx  # pylint: disable=global-statement
    _worker_ctx = {
        "params": params,
        "min_secs": min_secs,
        "max_secs": max_secs,
        "pc_min_secs": pc_min_secs,
        "pc_max_secs": pc_max_secs,
    }


def _process_one_cell(task: dict) -> dict:
    """Worker entry point: fit one grid cell end-to-end (surface fit -> power
    correction -> linear fit). Run-constant state (params, time window) comes
    from `_worker_ctx`, set once per worker by `_init_worker`.

    Args:
        task: gridcell_np (this cell's raw arrays) and cell_key (x_bin, y_bin,
            x_part, y_part).
    Returns:
        {"cell_key", "status"} on failure (status is a STATUS_KEYS error string), or
        {"cell_key", "status": None, "status_extra", "grid_record", "timeseries"} on success.
    """
    logger = logging.getLogger("surface_fit_worker")
    ctx = _worker_ctx
    params = ctx["params"]
    cell_key = task["cell_key"]
    x_bin, y_bin, x_part, y_part = cell_key

    surface_result = fit_surface_model_per_group(
        params=params,
        group_np=task["gridcell_np"],
        min_seconds=ctx["min_secs"],
        max_seconds=ctx["max_secs"],
        logger=logger,
    )
    if isinstance(surface_result, str):
        return {"cell_key": cell_key, "status": surface_result}
    res, group_np = surface_result

    status_extra = None
    if params.powercorrect:
        power_result = fit_power_correction_per_group(
            params,
            group_np,
            time_params={
                "pc_min_secs": ctx["pc_min_secs"],
                "pc_max_secs": ctx["pc_max_secs"],
                "mintime": ctx["min_secs"],
                "maxtime": ctx["max_secs"],
            },
            logger=logger,
        )
        if isinstance(power_result, str):
            return {"cell_key": cell_key, "status": power_result}
        group_np = power_result
        status_extra = "n_cells_power_corrected"

    linear_result = fit_linear_fit_per_group(params, group_np)
    if isinstance(linear_result, str):
        return {"cell_key": cell_key, "status": linear_result}

    grid_record = {
        "x_part": x_part,
        "y_part": y_part,
        "x_bin": x_bin,
        "y_bin": y_bin,
        "dhdt": linear_result["m"],
        "slope": (180.0 / np.pi) * np.sqrt((res[1] ** 2) + (res[2] ** 2)),
        "sigma": linear_result["sigma"],
        "rms": linear_result["rms"],
    }

    fitted = linear_result["group_np"]
    timeseries = {
        "x_part": x_part,
        "y_part": y_part,
        "x_bin": x_bin,
        "y_bin": y_bin,
        "time": fitted["time"],
        "time_years": fitted["time_years"],
        "dh": fitted["dH"],
    }
    if params.weight_column and params.weight_column in fitted:
        timeseries[params.weight_column] = fitted[params.weight_column]

    return {
        "cell_key": cell_key,
        "status": None,
        "status_extra": status_extra,
        "grid_record": grid_record,
        "timeseries": timeseries,
    }


_GRIDCELL_NP_COLUMNS = [
    "x",
    "y",
    "x2",
    "y2",
    "xy",
    "height",
    "power",
    "heading",
    "time",
    "time_years",
]


def _build_chunk_tasks(
    gridcell_lazy: pl.LazyFrame,
    params: argparse.Namespace,
) -> list[dict]:
    """Build one picklable `_process_one_cell` task per grid cell in this chunk."""
    columns = _GRIDCELL_NP_COLUMNS + ([params.weight_column] if params.weight_column else [])
    grouped = gridcell_lazy.collect().group_by(["x_bin", "y_bin", "x_part", "y_part"])
    return [
        {
            "gridcell_np": {
                col: gridcell[col].to_numpy() for col in columns if col in gridcell.columns
            },
            "cell_key": (x_bin, y_bin, x_part, y_part),
        }
        for (x_bin, y_bin, x_part, y_part), gridcell in grouped
    ]


def fit_surface_fit_models_per_group(
    params: argparse.Namespace, sf_objects: dict, logger: logging.Logger
) -> dict[str, int]:
    """Run the surface fit pipeline over all chunks and grid cells.

    Chunks are processed one at a time; within a chunk, grid cells are fit in
    parallel by `_process_one_cell` (surface fit -> power correction -> linear
    fit) across a worker pool created once for the whole run. Each chunk's
    outputs are written before moving to the next chunk.

    Args:
        params: Command line arguments.
        sf_objects: Surface fit objects from `get_surface_fit_objects`.
        logger: Logger object.
    Returns:
        Status dictionary containing result counts.
    """
    status = sf_objects["status"]
    min_secs = sf_objects["min_secs"]
    max_secs = sf_objects["max_secs"]
    pc_min_secs = sf_objects["pc_min_secs"]
    pc_max_secs = sf_objects["pc_max_secs"]
    cache_dir: str = sf_objects["cache_dir"]
    grid_writer = GridWriter(params.out_dir)

    n_workers = params.max_workers or os.cpu_count() or 1
    logger.info(
        "Fitting grid cells with a pool of %d worker process(es); "
        "chunks are still processed one at a time",
        n_workers,
    )

    try:
        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_init_worker,
            initargs=(params, min_secs, max_secs, pc_min_secs, pc_max_secs),
        ) as executor:
            for chunk_id, row in enumerate(sf_objects["part_df"].iter_rows(named=True), start=1):
                logger.info("Processing chunk : %s / %s", chunk_id, len(sf_objects["part_df"]))

                # This chunk's already-finalized partition file(s) - time window, mode
                # selection, and valid-elevation cells were applied at cache-build time.
                chunk_glob = os.path.join(
                    cache_dir, f"x_part={row['x_part']}", f"y_part={row['y_part']}", "*.parquet"
                )
                gridcell_lazy = pl.scan_parquet(chunk_glob, hive_partitioning=True)
                tasks = _build_chunk_tasks(gridcell_lazy, params)

                if not tasks:
                    write_chunk_output(params, [], row)
                    grid_writer.write([])
                    continue

                # Fit every cell in this chunk in parallel, waiting for all of them
                # before moving on, so at most one chunk is ever in flight.
                chunksize = max(1, len(tasks) // (n_workers * 4))
                results = list(executor.map(_process_one_cell, tasks, chunksize=chunksize))

                grid_records = []
                timeseries_frames = []
                for result in results:
                    if result["status"] is not None:
                        status[result["status"]] += 1
                        continue
                    if result.get("status_extra"):
                        status[result["status_extra"]] += 1
                    grid_records.append(result["grid_record"])
                    timeseries_frames.append(pl.DataFrame(result["timeseries"]))
                    status["n_cells_fitted"] += 1

                write_chunk_output(params, timeseries_frames, row)
                grid_writer.write(grid_records)
    finally:
        grid_writer.close()

    return status


def grid_and_surface_fit(args: list[str] | None = None) -> None:
    """
    Main entry point for the surface fit pipeline.
    Args:
        args: Command line arguments. If None, uses sys.argv[1:]
    """
    params = parse_arguments(args)
    try:
        standard_epoch = get_metadata_params(
            params, ["standard_epoch"], algo_name="grid_for_elev_change"
        )["standard_epoch"]
    except ValueError:
        sys.exit(
            "Standard epoch must be provided either in grid metadata or as a command line argument"
        )

    start_time = time.time()

    # Validate weighted fit config
    if params.weighted_surface_fit and params.weight_column is None:
        sys.exit("Weight name must be provided for weighted surface fit")

    logger = clean_directory(params, confirm_regrid=False)

    cache_dir, min_secs, max_secs, loaded_stats = build_gridded_archive(
        params, standard_epoch, logger
    )

    sf_objects = get_surface_fit_objects(
        params, standard_epoch, logger, cache_dir, min_secs, max_secs, loaded_stats
    )

    status = fit_surface_fit_models_per_group(params, sf_objects=sf_objects, logger=logger)

    # Write final metadata
    get_metadata_json(params, status, start_time, logger)


if __name__ == "__main__":
    grid_and_surface_fit(sys.argv[1:])
