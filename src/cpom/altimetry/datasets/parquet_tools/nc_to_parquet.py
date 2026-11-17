"""Convert altimetry data from NetCDF to partitioned Parquet format."""

import argparse
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import lru_cache, partial
from pathlib import Path
from typing import Any, Generator

import duckdb
import numpy as np
import polars as pl
import polars.selectors as cs
import pyarrow as pa
import pyarrow.parquet as pq
from pyproj import Transformer

from cpom.altimetry.datasets.dataset_helper import DatasetHelper
from cpom.altimetry.datasets.parquet_tools.correction_pipeline import CorrectionPipeline
from cpom.altimetry.tools.sec_tools.metadata_helper import (
    elapsed,
    get_algo_name,
    write_metadata,
)
from cpom.logging_funcs.logging import set_loggers

# pylint: disable=too-many-lines,too-many-locals,too-many-statements, too-many-branches
PATH_PREFIX_COLUMNS = {"hemisphere", "satellite", "dataset"}
HEMISPHERE_BBOX = {
    "north": (-180.0, 60.0, 180.0, 90.0),
    "south": (-180.0, -90.0, 180.0, -60.0),
}


# -------------------------------------#
# Set up Functions                     #
# -------------------------------------#
def parse_arguments(args):
    """
    Args:
        args (list[str]): Raw CLI arguments, e.g. sys.argv[1:].
    Returns:
        argparse.Namespace: Parsed parameters.
    """
    parser = argparse.ArgumentParser(
        description="""Convert altimetry data from NetCDF to Parquet format."""
    )
    # I/O parameters
    parser.add_argument(
        "--data_input_dir",
        type=str,
        required=True,
        help="Root directory containing L2 input files to grid.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to a dataset YAML file with dataset config.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        required=True,
        help="Root output directory. Without 'root/hemisphere/satellite/dataset'.",
    )
    parser.add_argument("--hemisphere", type=str, default="north")
    # Time filtering parameters
    parser.add_argument(
        "--standard_epoch",
        type=str,
        default="1970-01-01T00:00:00",
        help="Reference epoch for time values. Default: '1970-01-01T00:00:00'",
    )
    parser.add_argument("--min_date", type=str, help="Minimum date to include (e.g., '2000.01.01')")
    parser.add_argument("--max_date", type=str, help="Maximum date to include (e.g., '2020.12.31')")
    parser.add_argument(
        "--threaded_search",
        action="store_true",
        help=(
            "Search immediate subdirectories of data_input_dir concurrently when "
            "finding input files. Speeds up file discovery on network-mounted "
            "archives with many subdirectories, where the walk is latency-bound."
        ),
    )
    parser.add_argument(
        "--search_workers",
        type=int,
        default=8,
        help="Thread pool size used when --threaded_search is set. Default: 8.",
    )

    # Multiprocessing parameters
    parser.add_argument(
        "--max_workers",
        type=int,
        default=max(1, os.cpu_count() or 1),
        help="Number of parallel worker processes. Default: number of CPU cores.",
    )
    parser.add_argument(
        "--flush_every",
        type=int,
        default=60,
        help="Number of files to buffer before flushing to disk. Default: 60.",
    )

    # Output partitioning parameters
    parser.add_argument(
        "--partition_columns",
        type=str,
        nargs="+",
        default=["year"],
        help="Parquet partition columns. "
        "By default Hemisphere/satellite/dataset are added as a path prefix.",
    )
    # Spatial index / sort
    parser.add_argument(
        "--h3_res",
        type=int,
        default=7,
        help="H3 resolution for the cell index column.",
    )
    parser.add_argument(
        "--sort_method",
        default="datetime",
        choices=["hilbert", "h3", "datetime"],
        help=("Row sort order written into each partition file. "),
    )
    parser.add_argument(
        "--memory_limit",
        type=str,
        default="100GB",
        help="Memory limit for DuckDB operations (e.g., '100GB', '4GB').",
    )
    parser.add_argument(
        "--compression",
        default="zstd",
        choices=["zstd", "snappy", "lz4", "uncompressed"],
    )
    parser.add_argument("--compression_level", type=int, default=3)
    parser.add_argument("--debug", action="store_true", help="Enable DEBUG level logging")

    args = parser.parse_args(args)
    dataset = DatasetHelper(data_dir=args.data_input_dir, dataset_yaml=args.dataset)
    args.outdir = _get_prefixed_output_dir(args, dataset)
    os.makedirs(args.outdir, exist_ok=True)

    return args, dataset


def _get_partition_context(params: argparse.Namespace, dataset: DatasetHelper) -> dict[str, str]:
    """Build the static partition fields added to every output row."""
    return {
        "hemisphere": params.hemisphere,
        "satellite": dataset.mission,
        "dataset": dataset.long_name.replace(" ", "_"),
    }


def _get_prefixed_output_dir(params: argparse.Namespace, dataset: DatasetHelper) -> str:
    """Build out_dir with hemisphere/satellite/dataset as plain path segments."""
    partition_context = _get_partition_context(params, dataset)
    base = Path(getattr(params, "outdir", "") or "").expanduser()
    suffix = (
        partition_context["hemisphere"],
        partition_context["satellite"],
        partition_context["dataset"],
    )

    if len(base.parts) >= 3 and tuple(base.parts[-3:]) == suffix:
        return str(base)

    return str(base / suffix[0] / suffix[1] / suffix[2])


@lru_cache(maxsize=None)
def _get_transformer(src_crs: str, dst_crs: str) -> Transformer:
    """Build (and cache per worker process) a pyproj Transformer for src_crs -> dst_crs."""
    return Transformer.from_crs(src_crs, dst_crs, always_xy=True)


def _get_projected_coordinates(
    lats: np.ndarray,
    lons: np.ndarray,
    projected_crs: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Project lat/lon to x/y using the specified projected CRS."""
    transformer = _get_transformer("EPSG:4326", projected_crs)
    xs, ys = transformer.transform(lons, lats)
    return np.array(xs), np.array(ys)


def _get_geographic_coordinates(
    xs: np.ndarray,
    ys: np.ndarray,
    projected_crs: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Inverse-project x/y to lat/lon using the specified projected CRS."""
    transformer = _get_transformer(projected_crs, "EPSG:4326")
    lons, lats = transformer.transform(xs, ys)
    return np.array(lons), np.array(lats)


# -----------------------#
# DuckDB connection     #
# -----------------------#


def get_db_conn(params, logger) -> "duckdb.DuckDBPyConnection":
    """
    Create an in-memory DuckDB connection with the spatial and H3 extensions loaded.

    Args:
        params (argparse.Namespace): Command line parameters.
        logger (logging.Logger): Logger object.
    Returns:
        duckdb.DuckDBPyConnection: DuckDB connection object.
    """
    con = duckdb.connect()
    logger.info("DuckDB connection established")
    for ext, source in (("spatial", None), ("h3", "community")):
        try:
            con.execute(f"LOAD {ext};")
        except duckdb.IOException:
            install_cmd = f"INSTALL {ext} FROM {source};" if source else f"INSTALL {ext};"
            con.execute(install_cmd)
            con.execute(f"LOAD {ext};")
    logger.info("DuckDB extensions loaded: spatial, h3")

    con.execute(f"SET memory_limit = '{params.memory_limit}';")
    con.execute(f"SET temp_directory = '{Path(params.outdir) / 'duckdb_temp'}';")

    logger.info("DuckDB connection established with memory_limit=%s", params.memory_limit)
    logger.info("DuckDB temp_directory set to: %s", Path(params.outdir) / "duckdb_temp")
    return con


def close_db_conn(
    params: argparse.Namespace, con: "duckdb.DuckDBPyConnection", logger: logging.Logger
) -> None:
    """Close the DuckDB connection and delete the temporary directory.
    Args:
        params (argparse.Namespace): Command line parameters.
        con (duckdb.DuckDBPyConnection): DuckDB connection object.
        logger (logging.Logger): Logger object.
    """
    # delete temporary directory if it exists
    temp_dir = Path(params.outdir) / "duckdb_temp"
    if temp_dir and os.path.exists(temp_dir):
        # delete the directory if it exists
        os.rmdir(temp_dir)
    con.close()
    logger.info("DuckDB connection closed and temporary directory deleted: %s", temp_dir)


def _log_file_stats(logger, file_stats):
    """
    Log per-file processing statistics at DEBUG level.

    Args:
        logger (logging.Logger): Logger object.
        file_stats (dict): Per-file counts and rejection reason from process_file.
    """
    logger.debug(
        "file=%s | accepted=%d/%d | rejected(area/mask=%d, quality/time/nan=%d, elev_range=%d)%s%s",
        file_stats.get("path"),
        int(file_stats.get("accepted", 0)),
        int(file_stats.get("total", 0)),
        int(file_stats.get("outside_area_or_mask", 0)),
        int(file_stats.get("outside_quality_time_elev_or_nan", 0)),
        int(file_stats.get("outside_elevation_range", 0)),
        (
            f" | reason={file_stats.get('file_reject_reason')}"
            if file_stats.get("file_reject_reason")
            else ""
        ),
        (
            f" | detail={file_stats.get('file_reject_detail')}"
            if file_stats.get("file_reject_detail")
            else ""
        ),
    )


# ---------------------------------------#
# Data Loading and Processing Functions #
# ---------------------------------------#
def get_processing_objects(
    params: argparse.Namespace, dataset: DatasetHelper, correction_pipeline: CorrectionPipeline
) -> tuple[np.ndarray, Any, logging.Logger]:
    """Set up objects needed to process altimetry files.

    Args:
        params (argparse.Namespace): Command line parameters
        dataset (DatasetHelper): CPOM DatasetHelper object.
        correction_pipeline (CorrectionPipeline): Correction pipeline object.

    Returns:
        tuple:
            - file_and_dates (np.ndarray): Structured array of file paths and date information.
                        (one row per file, or one row per file/beam for multi-beam datasets).
            - worker (partial): Partial function to be used by process_file.
            - logger (logging.Logger): Logger object.
    """

    run_logger = set_loggers(
        log_dir=params.outdir,
        default_log_level=logging.DEBUG if params.debug else logging.INFO,
    )
    run_logger.info("Output_dir=%s", params.outdir)
    run_logger.info("Finding files and dates")

    file_and_dates = dataset.get_files_and_dates(
        hemisphere=params.hemisphere,
        min_dt_time=params.min_date,
        max_dt_time=params.max_date,
        threaded=params.threaded_search,
        max_workers=params.search_workers,
    )
    run_logger.info("Found %s files", len(file_and_dates))
    if len(file_and_dates) == 0:
        run_logger.warning("No files found matching criteria. Exiting.")
        sys.exit(0)

    # For multi-beam missions (IS2), expand to one task per (file, beam)
    if dataset.beams:
        n, nb = len(file_and_dates), len(dataset.beams)
        new_dtype = np.dtype(file_and_dates.dtype.descr + [("beam", "U10")])
        expanded = np.empty(n * nb, dtype=new_dtype)

        field_names = file_and_dates.dtype.names or ()
        for name in field_names:
            expanded[name] = np.repeat(file_and_dates[name], nb)
        expanded["beam"] = np.tile(dataset.beams, n)

        file_and_dates = expanded
        run_logger.info("Expanded to %s per-beam tasks (%s beams)", len(file_and_dates), nb)

    worker = partial(
        process_file,
        dataset=dataset,
        correction_pipeline=correction_pipeline,
        offset=dataset.get_unified_time_epoch_offset(params.standard_epoch, dataset.dataset_epoch),
        params=params,
        projected_crs="EPSG:3413" if params.hemisphere == "north" else "EPSG:3031",
    )

    return file_and_dates, worker, run_logger


# pylint: disable= R0911, R0912, R0913, R0914, R0915,R0917
def process_file(
    file_and_date: np.void,
    dataset: DatasetHelper,
    correction_pipeline: CorrectionPipeline,
    params: argparse.Namespace,
    offset: float,
    projected_crs: str,
) -> pl.LazyFrame | tuple[pl.LazyFrame | None, dict | None] | None:
    """
    Extract and grid data from a single altimetry file.

    Runs a full per file pipeline:
        1. Load coordinates
        2. Spatial filter to area/mask and convert to x/y
        3. Load variables, apply corrections and mask.
        4. Compute grid cell positions.
        5. Cast dtypes
        6. Create Polars LazyFrame and apply elevation range filter.

    Args:
        file_and_date (np.void): File metadata (must include: 'path', 'year', 'month' keys)
        dataset (DatasetHelper): CPOM DatasetHelper object (dataset configuration).
        offset (float): Time offset in seconds to convert to standard epoch.
        projected_crs (str): Target CRS EPSG string.
        params (argparse.Namespace): Command line arguments

    Returns:
        pl.LazyFrame | [pl.LazyFrame | None, dict | None] | None: Processed data frame,
        or None if the file is rejected.
        In debug mode returns a tuple of (frame or None, stats dict).
    """

    field_names = file_and_date.dtype.names or ()
    beam = file_and_date["beam"] if "beam" in field_names else None
    if beam:
        dataset.beams = [str(beam)]
    elif dataset.beams:
        dataset.beams = []

    stats: dict | None = None
    if params.debug:
        stats = {
            "path": file_and_date["path"],
            "accepted": 0,
            "outside_area_or_mask": 0,
            "outside_quality_time_elev_or_nan": 0,
            "file_reject_reason": None,
            "file_reject_detail": None,
            "total": 0,
        }

    variable_dict = {}
    try:
        with dataset.get_file_handle(file_and_date["path"]) as nc:
            for param in dataset.params:
                variable_path = dataset.get_param_path(param)
                if variable_path is None:
                    raise ValueError(f"No path configured for dataset parameter {param!r}")
                variable_dict[param] = dataset.get_variable(
                    nc,
                    variable_path,
                    raise_if_missing=True,
                )
                if param in ["longitude", "longitude_nadir"]:
                    variable_dict[param] = variable_dict[param] % 360
                if dataset.beams != [] and param == "latitude":
                    variable_dict["beam"] = dataset.get_variable(
                        nc,
                        variable_path,
                        return_beams=True,
                        raise_if_missing=True,
                    )
            if "x" in variable_dict and "y" in variable_dict:
                if "latitude" not in variable_dict or "longitude" not in variable_dict:
                    variable_dict["longitude"], variable_dict["latitude"] = (
                        _get_geographic_coordinates(
                            variable_dict["x"], variable_dict["y"], projected_crs
                        )
                    )

            latitude_mask = (
                variable_dict["latitude"] >= 0
                if params.hemisphere == "north"
                else variable_dict["latitude"] <= 0
            )
            variable_dict = correction_pipeline.run(
                nc, latitude_mask, variable_dict, not params.debug
            )
            if "x" not in variable_dict or "y" not in variable_dict:
                variable_dict["x"], variable_dict["y"] = _get_projected_coordinates(
                    variable_dict["latitude"], variable_dict["longitude"], projected_crs
                )

            variable_dict["idx"] = np.arange(len(variable_dict["latitude"]))
            if "latitude_nadir" in variable_dict or "latitude" in variable_dict:
                variable_dict["ascending"] = dataset.get_file_orbital_direction(
                    latitude=(
                        variable_dict["latitude_nadir"]
                        if "latitude_nadir" in variable_dict
                        else variable_dict["latitude"]
                    ),
                    nc=nc,
                )
    except KeyError as e:
        if params.debug:
            assert stats is not None
            stats["file_reject_reason"] = "missing_beam_or_variable"
            stats["file_reject_detail"] = str(e)
            return None, stats
        return None
    except OSError as e:
        if params.debug:
            assert stats is not None
            stats["file_reject_reason"] = "unreadable_file"
            stats["file_reject_detail"] = str(e)
            return None, stats
        return None

    frame = (
        pl.LazyFrame(variable_dict)
        .with_columns(
            pl.from_epoch(pl.col("time") + offset, time_unit="s").alias("datetime"),
            pl.lit(Path(file_and_date["path"]).stem).alias("source"),
        )
        .drop("time")
    )
    frame = frame.with_columns(
        pl.col("datetime").dt.year().cast(pl.Int16).alias("year"),
        pl.col("datetime").dt.month().cast(pl.Int8).alias("month"),
        pl.col("datetime").dt.day().cast(pl.Int8).alias("day"),
    )

    frame = (
        frame.filter(pl.col("elevation").is_finite())
        .with_columns(
            (cs.float() - cs.by_name("x", "y", "latitude", "longitude", require_all=False)).cast(
                pl.Float32
            )
        )
        .with_columns(pl.col("idx").cast(pl.Int32))
        .with_columns(pl.col("source").cast(pl.Categorical))
    )
    if params.debug:
        return frame, stats
    return frame


# --------------------------------------#
# File Writing Helpers                  #
# --------------------------------------#


def _group_periods(
    file_and_dates: np.ndarray, use_month: bool
) -> Generator[tuple[str, np.ndarray], None, None]:
    """
    Group files by processing period and yield each period's label and file subset.

    Args:
        file_and_dates (np.ndarray): Structured array of file paths and date information.
        use_month (bool): If True, group by year/month. If False, group by year only.

    Yields:
        tuple[str, np.ndarray]: Period label (e.g. "2020" or "2020-03") and
            the subset of file_and_dates for that period.
    """
    if use_month:
        keys = np.unique(
            np.stack([file_and_dates["year"], file_and_dates["month"]], axis=1),
            axis=0,
        )
        for year, month in keys:
            mask = (file_and_dates["year"] == year) & (file_and_dates["month"] == month)
            period_label = f"{int(year)}-{int(month):02d}"
            yield period_label, file_and_dates[mask]
    else:
        years = sorted(set(file_and_dates["year"]))
        for year in years:
            mask = file_and_dates["year"] == year
            yield str(int(year)), file_and_dates[mask]


def _flush_buffer(
    buffer: list[pl.LazyFrame],
    out_dir: str,
    partition_columns: list[str],
    writer_cache: dict[str, pq.ParquetWriter],
) -> int:
    """Concatenate buffered LazyFrames, write them to disk, and clear the buffer.

    Extracted so the flush sequence exists in exactly one place (it's needed
    both for periodic flushes and the final trailing flush).

    Args:
        buffer (list[pl.LazyFrame]): Per-file frames accumulated so far; cleared in place.
        out_dir (str): Output directory path.
        partition_columns (list[str]): Ordered partition columns.
        writer_cache (dict): Open ParquetWriters keyed by output file path. Must be
            shared across the whole run (not recreated per period) so that
            periods contributing rows to the same partition (e.g. a file dated
            31 Dec with a few records timestamped past midnight, landing in
            the next year's partition) append to the same writer instead of
            each opening their own "data.parquet".

    Returns:
        int: Number of rows written by this flush.
    """
    rows_written = write_partitions_to_disk(
        pl.concat(buffer, rechunk=False).collect(),
        out_dir,
        partition_columns,
        writer_cache,
    )
    buffer.clear()
    return rows_written


def write_partitions_to_disk(
    final_df: pl.DataFrame,
    out_dir: str,
    partition_columns: list[str],
    writer_cache: dict[str, pq.ParquetWriter] | None = None,
) -> int:
    """
    Write a DataFrame to disk as hive-partitioned Parquet files.

    Args:
        final_df (pl.DataFrame): The data to write.
        out_dir (str): Output directory path.
        partition_columns (list[str]): Ordered partition columns.
        writer_cache (dict | None): Open ParquetWriters keyed by output file path.
            Must be shared across all callers writing into the same output
            directory (e.g. across processing periods) so that every partition
            ends up with a single "data.parquet" rather than one file per caller.

    Returns:
        int: Total number of rows written to disk.
    """

    missing = [col for col in partition_columns if col not in final_df.columns]
    if missing:
        raise ValueError(f"Missing partition columns in frame: {missing}")

    if writer_cache is None:
        writer_cache = {}

    partition_columns = list(partition_columns)
    total_rows = 0
    for group in final_df.partition_by(*partition_columns, maintain_order=False):
        subdir = os.path.join(
            out_dir,
            *[
                f"{col}={val}"
                for col, val in zip(
                    partition_columns, tuple(group[col][0] for col in partition_columns)
                )
            ],
        )

        outfile = os.path.join(subdir, "data.parquet")
        os.makedirs(subdir, exist_ok=True)

        table = group.to_arrow()
        if outfile not in writer_cache:
            writer_cache[outfile] = pq.ParquetWriter(
                outfile,
                table.schema,
                compression="zstd",
            )
        writer_cache[outfile].write_table(table)
        total_rows += len(group)

    return total_rows


# -----------------------------#
# Multi-Processing Function    #
# ------------------------------#


def get_data_and_status_multiprocessed(
    params: argparse.Namespace,
    file_and_dates: np.ndarray,
    worker: partial,
    run_logger: logging.Logger,
) -> dict[str, Any]:  # pylint: disable=too-many-arguments
    """
    Process altimetry files in parallel and write results to partitioned Parquet files.

    Iterates over periods (years or year/months), submits files to a process pool,
    collects LazyFrames into a buffer, and flushes to disk every flush_every files.
    Returns a status dictionary with per-period and total ingestion counts.

    Args:
        params (argparse.Namespace): Command line parameters.
        file_and_dates (np.ndarray): Structured array of file paths and date information.
        worker (partial): Partial function for processing files.
        logger (logging.Logger): Logger object.

    Returns:
        dict: Processing status dictionary, containing information about the files processed.
    """

    status: dict = {
        "empty_periods": [],
        "total_l2_files_rejected": 0,
        "total_l2_files_ingested": 0,
    }
    use_month = "month" in params.partition_columns
    writer_cache: dict[str, pq.ParquetWriter] = {}

    try:
        with ProcessPoolExecutor(max_workers=params.max_workers) as executor:
            try:
                for period_label, files_in_period in _group_periods(file_and_dates, use_month):

                    n_valid = n_rejected = total_rows = 0
                    buffer: list[pl.LazyFrame] = []
                    chunksize = max(1, len(files_in_period) // max(1, params.max_workers * 4))

                    for result in executor.map(worker, files_in_period, chunksize=chunksize):

                        frame, file_stats = (
                            result
                            if isinstance(result, tuple) and len(result) == 2
                            else (result, None)
                        )

                        if params.debug and file_stats and isinstance(file_stats, dict):
                            _log_file_stats(run_logger, file_stats)

                        if frame is not None:
                            buffer.append(frame)
                            n_valid += 1

                            if len(buffer) >= params.flush_every:
                                total_rows += _flush_buffer(
                                    buffer,
                                    params.outdir,
                                    params.partition_columns,
                                    writer_cache,
                                )
                        else:
                            n_rejected += 1

                    if buffer:
                        total_rows += _flush_buffer(
                            buffer,
                            params.outdir,
                            params.partition_columns,
                            writer_cache,
                        )

                    run_logger.info(
                        "Period %s: %d valid, %d rejected, %d rows",
                        period_label,
                        n_valid,
                        n_rejected,
                        total_rows,
                    )

                    if n_valid == 0:
                        status["empty_periods"].append(period_label)
                        continue

                    status["total_l2_files_ingested"] += n_valid
                    status["total_l2_files_rejected"] += n_rejected

            except (ValueError, OSError, RuntimeError) as e:
                run_logger.error("Error collecting results: %s", e, exc_info=True)
                if buffer:
                    total_rows += _flush_buffer(
                        buffer,
                        params.outdir,
                        params.partition_columns,
                        writer_cache,
                    )
            finally:
                for w in writer_cache.values():
                    w.close()

    except (OSError, ValueError) as e:
        run_logger.error("Multiprocessing failed: %s", e)
        sys.exit(1)

    return status


# -------------
# Sorting
# -------------
def _iter_partition_files(out_dir: str) -> Generator[str, None, None]:
    """Walk *out_dir* and yield every partition file path (data.parquet)."""
    for root, _, files in os.walk(out_dir):
        for fname in files:
            if fname.startswith("data") and fname.endswith(".parquet"):
                yield os.path.join(root, fname)


def sort_partitions_pass(
    params: argparse.Namespace,
    out_dir: str,
    con: "duckdb.DuckDBPyConnection",
    logger: logging.Logger,
) -> None:
    """Add an H3 index and sort every partition file written in the ingest pass.

    Each partition file is rewritten in place, adding an h3_cell_id column
    and ordering rows by Hilbert curve / H3 cell / datetime (per
    --sort_method).

    Args:
        params (argparse.Namespace): Command line parameters.
        out_dir (str): Output directory path.
        con (duckdb.DuckDBPyConnection): DuckDB connection object.
        logger (logging.Logger): Logger object.
    """
    files = list(_iter_partition_files(out_dir))
    if not files:
        logger.warning("sort_partitions_pass: no data.parquet files found under %s", out_dir)
        return

    logger.info("Sort pass: %d partition file(s) to process", len(files))
    if params.sort_method == "h3":
        order_clause = "h3_cell_id"
    elif params.sort_method == "datetime":
        order_clause = "datetime"
    else:
        min_lon, min_lat, max_lon, max_lat = HEMISPHERE_BBOX[params.hemisphere]
        order_clause = (
            f"ST_Hilbert(longitude, latitude, "
            f"ST_MakeBox2D(ST_Point({min_lon}, {min_lat}), ST_Point({max_lon}, {max_lat})))"
        )

    def _sort_single_partition(fpath: str) -> None:
        logger.debug("Sort pass: %s", fpath)
        cursor = con.cursor()
        try:
            cursor.execute(f"""
                COPY (
                    SELECT *,
                        CAST(
                            h3_latlng_to_cell(latitude, longitude, {params.h3_res}) AS UBIGINT
                        ) AS h3_cell_id
                    FROM read_parquet('{fpath}')
                    ORDER BY {order_clause}
                )
                TO '{fpath}' (
                    FORMAT parquet,
                    COMPRESSION {params.compression},
                    COMPRESSION_LEVEL {params.compression_level},
                    ROW_GROUP_SIZE 100000
                )
            """)
        finally:
            cursor.close()

    max_workers = max(1, min(params.max_workers, len(files)))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(executor.map(_sort_single_partition, files))

    logger.info("Sort pass complete")


# --------------------------------------#
# Metadata Output Function              #
# --------------------------------------#
def _get_parquet_output_metadata(out_dir: str, partition_columns: list[str]) -> dict[str, Any]:
    """
    Inspect the Parquet files actually written under out_dir and summarize them.

    Args:
        out_dir (str): Output directory.
        partition_columns (list[str]): Partition columns used in the output.

    Returns:
        dict: Parquet output summary.
    """
    files = list(_iter_partition_files(out_dir))
    if not files:
        return {"files_found": 0}

    total_rows, total_bytes = 0, 0
    row_group_sizes: list[int] = []
    partition_values: dict[str, set[str]] = {col: set() for col in partition_columns}
    schema: pa.Schema | None = None

    for fpath in files:
        meta = pq.ParquetFile(fpath).metadata  # footer-only read, no row data
        total_rows += meta.num_rows
        total_bytes += os.path.getsize(fpath)
        for rg in range(meta.num_row_groups):
            row_group_sizes.append(meta.row_group(rg).num_rows)
        if schema is None:
            schema = meta.schema.to_arrow_schema()

        # Recover partition values from the hive-style path, e.g. .../year=2020/data.parquet
        rel_parts = Path(fpath).relative_to(out_dir).parts
        for part in rel_parts:
            if "=" in part:
                key, _, val = part.partition("=")
                if key in partition_values:
                    partition_values[key].add(val)

    return {
        "file_partitions": len(files),
        "total_rows": total_rows,
        "total_bytes": total_bytes,
        "total_size_human": _human_size(total_bytes),
        "avg_rows_per_file": total_rows // len(files),
        "avg_row_group_size": (
            sum(row_group_sizes) // len(row_group_sizes) if row_group_sizes else 0
        ),
        "num_row_groups": len(row_group_sizes),
        "columns": (
            {name: str(schema.field(name).type) for name in schema.names}
            if schema is not None
            else {}
        ),
    }


def _human_size(num_bytes: int) -> str:
    """Format a byte count as a human-readable string (e.g. '1.3 GB')."""
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} PB"


def _alias_roots(obj: Any) -> Any:
    """Recursively replace configured absolute roots with env-variable tokens."""
    # Longest root first, so a nested root (e.g. a subdir of another aliased root)
    # is matched before its shorter parent.
    replacements = sorted(
        (
            (os.environ[var].rstrip("/"), f"${var}")
            for var in ("CPDATA_DIR", "CPOM_SOFTWARE_DIR")
            if os.environ.get(var)
        ),
        key=lambda pair: len(pair[0]),
        reverse=True,
    )

    def _replace_str(value: str) -> str:
        for root, token in replacements:
            if value == root:
                return token
            if value.startswith(root + "/"):
                return token + value[len(root) :]
        return value

    def _walk(o: Any) -> Any:
        if isinstance(o, str):
            return _replace_str(o)
        if isinstance(o, dict):
            return {key: _walk(v) for key, v in o.items()}
        if isinstance(o, (list, tuple)):
            return type(o)(_walk(v) for v in o)
        return o

    return _walk(obj)


def get_metadata_json(
    params: argparse.Namespace,
    dataset: DatasetHelper,
    correction_pipeline: CorrectionPipeline,
    status: dict,
    start_time: float,
) -> None:
    """
    Write processing metadata to a JSON file.

    Output is grouped into four tagged sections:
        "script_arguments"    — CLI parameters as passed to nc_to_parquet.
        "dataset_arguments"   — DatasetHelper config (mission, params, configured
                                correction defaults, etc.) as declared in the YAML.
        "corrections_applied" — What CorrectionPipeline actually resolved and ran
                                (module.qualname + docstring summary per stage),
                                which reflects any CLI overrides of the YAML
                                defaults. Columns added via a correction's
                                extra_params already appear in parquet_output.columns.
        "parquet_output"      — Summary of the actual Parquet output (schema, row/file
                                counts, size, partition values, sort/compression settings)
                                plus the ingestion status produced during processing.

    Args:
        params (argparse.Namespace): Command line parameters.
        dataset (DatasetHelper): CPOM DatasetHelper object.
        correction_pipeline (CorrectionPipeline): Correction pipeline actually run.
        status (dict): Processing status dictionary.
        start_time (float): Processing start time.
    """
    script_arguments = dict(vars(params))
    dataset_arguments = dict(vars(dataset))

    parquet_output: dict[str, Any] = {
        **_get_parquet_output_metadata(params.outdir, params.partition_columns),
        **status,
        "gridding_time": elapsed(start_time),
    }

    output: dict[str, Any] = {
        "script_arguments": script_arguments,
        "dataset_arguments": dataset_arguments,
        "corrections_applied": correction_pipeline.describe(),
        "parquet_output": parquet_output,
    }

    output = _alias_roots(output)

    write_metadata(
        params,
        get_algo_name(__file__),
        Path(params.outdir),
        output,
    )


# --------------------------------------#
# Main Function #
# --------------------------------------#


def nc_to_parquet(args: list[str]) -> None:
    """Main entry point to grid altimetry data for elevation change processing.

    Loads the dataset from a YAML file, resolves the output directory,
    processes all files in parallel (pass 1), sorts each partition file
    (pass 2), and writes a metadata JSON summarizing the run (pass 3). See
    the module docstring for the full pipeline description.

    Args:
        args (list[str]): Command line arguments, e.g. sys.argv[1:].
    """
    parsed_args, this_dataset = parse_arguments(args)

    start_time = time.time()

    correction_pipeline = CorrectionPipeline.from_args(this_dataset)

    file_and_dates, worker, run_logger = get_processing_objects(
        parsed_args, this_dataset, correction_pipeline
    )

    status = get_data_and_status_multiprocessed(
        params=parsed_args,
        file_and_dates=file_and_dates,
        worker=worker,
        run_logger=run_logger,
    )

    run_logger.info("Starting sort pass (sort_method=%s)", parsed_args.sort_method)
    con = get_db_conn(parsed_args, run_logger)
    sort_partitions_pass(parsed_args, parsed_args.outdir, con, run_logger)
    close_db_conn(parsed_args, con, run_logger)

    get_metadata_json(parsed_args, this_dataset, correction_pipeline, status, start_time)


if __name__ == "__main__":
    nc_to_parquet(sys.argv[1:])
