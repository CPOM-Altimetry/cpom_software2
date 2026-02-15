# pylint: disable=C0302
"""
cpom.altimetry.tools.sec_tools.surface_fit

Purpose:
    Fit surface models to elevation time series for each grid cell.
    Step 1 - Setup: Initialize configuration, logging, time bounds, chunks
    Step 2 - Loading: Load, preprocess, and compute features per chunk
    Step 3 - Fitting: Fit surface, apply power correction, estimate dH/dt per cell

Output:
    - Surface fit results: <out_dir>/grid_data.parquet
    - Metadata: <out_dir>/metadata.json

Processing Workflow:

    Gridded Altimetry Data
            ↓
    clean_directory() - Initialize output directory and logging
    get_surface_fit_objects() - Setup time bounds, chunks, status dictionary
            ↓
    surface_fit() - Main surface fitting loop
        - For each chunk (x_part, y_part):
            - Load data with get_grid_data()
            - For each grid cell:
                - fit_surface_model_per_group() - Fit surface model to elevation data
                - fit_power_correction_per_group() - Apply power correction (optional)
                - fit_linear_fit_per_group() - Estimate dH/dt from residuals

            Per chunk output:
                x_part=X/y_part=Y/dh_time_grid.parquet (time, dh, [weights])
            Per data output:
                grid_data.parquet (x_bin, y_bin, dhdt, slope, sigma, rms)
            ↓
    get_metadata_json() - Write metadata JSON with execution details
"""

import argparse
import json
import logging
import os
import shutil
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Tuple

import duckdb
import numpy as np
import polars as pl
import statsmodels.api as sm
from scipy import stats

from cpom.altimetry.tools.sec_tools.basin_selection_helper import (
    add_basin_selection_arguments,
    get_basins_to_process,
)
from cpom.logging_funcs.logging import set_loggers

SECONDS_PER_YEAR = 31557600


# ----------------------------------------------------------------------------
# SETUP FUNCTIONS - Initialize config, logging, time bounds, chunks
# ---------------------------------------------------------------------------
def parse_arguments(args):
    """
    Parse command line arguments for surface fitting.
    Args:
        args (list): List of command line arguments
    Returns:
        argparse.Namespace: Parsed command line arguments
    """

    def auto_type(value: str) -> int | float | str:
        """Try to convert to int or float, else keep as string."""
        try:
            return int(value)
        except ValueError:
            try:
                return float(value)
            except ValueError:
                return value  # leave as string

    parser = argparse.ArgumentParser(
        description=(
            "Compute fitted elevation per grid cell from"
            " gridded altimetry data stored in parquet files."
        )
    )
    # I/O arguments
    parser.add_argument(
        "--in_dir",
        help=("Path of the grid dir containing parquet files"),
        type=str,
        required=True,
    )
    parser.add_argument(
        "--out_dir",
        help="Path of output directory for surface fit results",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--grid_info_json",
        help="Path to the grid metadata JSON file.",
        required=True,
    )
    parser.add_argument(
        "--parquet_glob",
        help="(optional) Glob pattern to match parquet files",
        type=str,
        default="**/*.parquet",
    )
    # Surface fit parameters
    parser.add_argument(
        "--min_vals_in_cell",
        default=20,
        type=int,
        help="Minimum measurements in cell to perform plane fit",
    )
    parser.add_argument(
        "--min_percent_timespan_in_cell",
        default=70.0,
        type=float,
        help="Minimum time-span in cell to perform plane fit as percent of (maxtime-mintime),"
        " default is 70%%",
    )
    parser.add_argument(
        "--n_sigma",
        default=2.0,
        type=float,
        help="Sigma to filter out measurements from plane fit",
    )
    parser.add_argument(
        "--max_surfacefit_iterations",
        default=30,
        type=int,
        help="Maximum number of iterations of surface fit",
    )
    parser.add_argument(
        "--max_linearfit_iterations",
        default=3,
        type=int,
        help="Maximum number of iterations of linear fit",
    )
    parser.add_argument(
        "--mintime",
        type=str,
        help="(optional) start time for plane fit (DD/MM/YYYY or DD.MM.YYYY) \
            | min time in the data if not provided.",
    )
    parser.add_argument(
        "--maxtime",
        type=str,
        help="(optional) End time for plane fit (DD/MM/YYYY or DD.MM.YYYY) \
            | max time in the grid if not provided.",
    )
    parser.add_argument(
        "--powercorrect",
        help="Enable power (backscatter) correction",
        action="store_true",
    )
    parser.add_argument(
        "--mode_values",
        nargs="+",
        default=None,
        type=auto_type,
        help="List of mode values to include in surface fit",
    )
    parser.add_argument(
        "--pcmintime",
        type=str,
        help="(optional) Start date of power correction (DD/MM/YYYY or DD.MM.YYYY) \
        | mintime if not provided.",
    )
    parser.add_argument(
        "--pcmaxtime",
        type=str,
        help="(optional) End date of power correction (DD/MM/YYYY or DD.MM.YYYY) \
            | maxtime if not provided.",
    )
    # Optional column name arguments
    parser.add_argument(
        "--time_column",
        type=str,
        default="time",
        help="Name of the time column in the parquet files",
    )
    parser.add_argument(
        "--x_column",
        type=str,
        default="x_cell_offset",
        help="Name of the x coordinate column in the parquet files",
    )
    parser.add_argument(
        "--y_column",
        type=str,
        default="y_cell_offset",
        help="Name of the y coordinate column in the parquet files",
    )
    parser.add_argument(
        "--elevation_column",
        type=str,
        default="elevation",
        help="Name of the elevation column in the parquet files",
    )
    parser.add_argument(
        "--heading_column",
        type=str,
        default="ascending",
        help="Name of the heading column in the parquet files",
    )
    parser.add_argument(
        "--power_column",
        type=str,
        default="power",
        help="Name of the power column in the parquet files",
    )
    parser.add_argument(
        "--mode_column",
        type=str,
        default="mode",
        help="Name of the mode column in the parquet files",
    )
    # Is2 specific options
    parser.add_argument(
        "--weighted_surface_fit",
        action="store_true",
        help="Use weighted surface fit for is2 data",
    )
    parser.add_argument(
        "--weighted_power_fit",
        action="store_true",
        help="Use weighted power fit for power correction",
    )
    parser.add_argument(
        "--weight_column",
        type=str,
        help="(optional) Name of the weight column to use for weighted surface fit",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable DEBUG level logging",
    )

    # Add shared basin/region selection arguments for consistency across tools
    add_basin_selection_arguments(parser)
    return parser.parse_args(args)


def clean_directory(params: argparse.Namespace, confirm_regrid: bool = False):
    """
    Create output directory.

    Load grid metadata from the specified JSON file, and initialize logging.

    Args:
        params (argparse.Namespace): Command line parameters
        confirm_regrid (bool): If True,
            prompts user before removing existing output directory.

    Returns [logging.Logger, dict]:
        - Logger object
        - grid metadata from - grid_for_elev_change.
    """

    with open(Path(params.grid_info_json), "r", encoding="utf-8") as f:
        grid_meta = json.load(f)

    # Full regrid => remove entire directory, then create fresh
    if confirm_regrid:
        if os.path.exists(params.out_dir):
            if params.out_dir != "/" and grid_meta["mission"] in params.out_dir:  # safety check
                response = (
                    input("Confirm removal of previous surface fit archive? (y/n): ")
                    .strip()
                    .lower()
                )
                if response == "y":
                    shutil.rmtree(params.out_dir)
                else:
                    print("Exiting as user requested not to overwrite surface fit archive")
                    sys.exit(0)
            else:
                sys.exit(1)

    # Create output directory before setting up file logging
    os.makedirs(params.out_dir, exist_ok=True)

    logger = set_loggers(
        log_dir=params.out_dir,
        default_log_level=logging.DEBUG if params.debug else logging.INFO,
    )
    logger.info("output_dir=%s", params.out_dir)

    return logger, grid_meta


def get_min_max_time(
    epoch_time: datetime,
    parquet_glob: str,
    mintime: str | None = None,
    maxtime: str | None = None,
    time_column: str = "time",
) -> tuple[float, float, float, float]:
    """
    Get the min/max time as a timestamp and in seconds.
    If mintime and maxtime are not provided, data daterange is used.
    Args:
        epoch_time (datetime): Standard epoch time to calculate seconds from.
        parquet_glob (str): Path to the parquet files
        mintime (str: Optional): Start time (DD/MM/YYYY or DD.MM.YYYY).
        maxtime (str: Optional): End time (DD/MM/YYYY or DD.MM.YYYY).
        time_column (str): Name of the time column in the parquet files.
    Returns:
        tuple[float, float, float, float]:
          (mintime_timestamp, maxtime_timestamp, min_secs, max_secs)
    """

    def get_date(epoch_time: datetime, timedt: str) -> Tuple[datetime, float]:
        """
        Convert a date string to a datetime object
        and calculate the seconds from the epoch.
        """
        if "/" in timedt:
            time_dt = datetime.strptime(timedt, "%d/%m/%Y")
        elif "." in timedt:
            time_dt = datetime.strptime(timedt, "%d.%m.%Y")
        else:
            raise ValueError(
                f"Unrecognized date format: {timedt}, pass as DD/MM/YYYY or DD.MM.YYYY "
            )
        seconds = (time_dt - epoch_time).total_seconds()

        return time_dt, seconds

    if mintime is not None and maxtime is not None:
        min_dt, min_secs = get_date(epoch_time, mintime)
        max_dt, max_secs = get_date(epoch_time, maxtime)
    else:
        df = pl.scan_parquet(parquet_glob)
        min_max = df.select(
            [
                pl.col(time_column).min().alias("min_time"),
                pl.col(time_column).max().alias("max_time"),
            ]
        ).collect()

        min_secs = min_max["min_time"][0]
        max_secs = min_max["max_time"][0]
        min_dt = epoch_time + timedelta(seconds=min_secs)
        max_dt = epoch_time + timedelta(seconds=max_secs)
    return min_dt.timestamp(), max_dt.timestamp(), min_secs, max_secs


def get_min_timespan(
    params: argparse.Namespace, mintime: float, maxtime: float
) -> argparse.Namespace:
    """
    Calculate minimum required time span in seconds for surface fit per grid cell.
    Args:
        params (argparse.Namespace): Command line parameters
        mintime (float): Minimum time for surface fit (timestamp)
        maxtime (float): Maximum time for surface fit (timestamp)
    Returns:
        argparse.Namespace: Updated command line parameters with
            min_timespan_in_cell_in_secs attribute set.
    """
    params.min_timespan_in_cell_in_secs = (maxtime - mintime) * (
        params.min_percent_timespan_in_cell / 100.0
    )
    return params


def get_unique_chunks(params: argparse.Namespace) -> pl.DataFrame:
    """
    Get all unique (x_part, y_part) chunks to process from input parquet files.
    Args:
        params (argparse.Namespace): Command line parameters including in_dir
    Returns:
        pl.DataFrame: DataFrame of unique (x_part, y_part) chunks to process
    """
    conn = duckdb.connect()

    # Try to read with hive partitioning (directory structure)
    part_df = conn.execute(
        f"""
        SELECT DISTINCT x_part, y_part
        FROM read_parquet('{params.in_dir}/**/x_part=*/y_part=*/*.parquet',
        hive_partitioning=1);
        """
    ).pl()
    conn.close()

    # Sort
    part_df = part_df.sort(["x_part", "y_part"])
    # Check if we got any partitions from directory structure
    if part_df.height > 0:
        return part_df
    # No directory partitioning found - data has x_part/y_part columns
    return pl.DataFrame([{"x_part": None, "y_part": None}])


def get_surface_fit_objects(
    params: argparse.Namespace, parquet_glob: str, grid_meta: dict, logger: logging.Logger
) -> dict[str, Any]:
    """
    Get objects required for surface fit :

        1. Get time boundaries for surface fit and optional power correction. (get_min_max_time)
        2. Calculates minimum required time span per grid cell based on configured percentage
            (get_min_timespan)
        3. Get all unique spatial chunks (x_part, y_part) that need processing
            (get_unique_chunks)
        4. Initialize status counters for surface fit processing.

    Args:
        params (argparse.Namespace): Command line parameters including:
            - mintime/maxtime: Optional time range for surface fitting
            - pcmintime/pcmaxtime: Optional time range for power correction
            - min_percent_timespan_in_cell: Minimum required time coverage per cell (%)
            - powercorrect: Whether power correction is enabled
            - in_dir: Input directory containing partitioned parquet files
        parquet_glob (str): Glob pattern to match input parquet files
        grid_meta (dict): Grid metadata dictionary containing:
            - standard_epoch: Reference epoch for time calculations (ISO format string)
        logger (logging.Logger): Logger object

    Returns:
        dict[str, Any]: Dictionary containing all surface fit initialization data:
            - mintime (float): Surface fit start time as timestamp
            - maxtime (float): Surface fit end time as timestamp
            - min_secs (float): Surface fit start time in seconds since standard_epoch
            - max_secs (float): Surface fit end time in seconds since standard_epoch
            - pc_min_secs (float | None): Power correction start time in seconds
            - pc_max_secs (float | None): Power correction end time in seconds
            - status (dict): Initialized status counters (all set to 0)
            - part_df (pl.DataFrame): DataFrame of unique (x_part, y_part) chunks to process
    """

    def _init_status() -> dict[str, int]:
        return {
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

    # 1. Get the min/max time period for surface fit and power correction
    mintime, maxtime, min_secs, max_secs = get_min_max_time(
        datetime.fromisoformat(grid_meta["standard_epoch"]),
        parquet_glob,
        params.mintime,
        params.maxtime,
        params.time_column,
    )
    logger.info(f"Surface fit time range : {mintime} : {maxtime}")

    if params.powercorrect:
        _, _, pc_min_secs, pc_max_secs = get_min_max_time(
            datetime.fromisoformat(grid_meta["standard_epoch"]),
            parquet_glob,
            params.pcmintime,
            params.pcmaxtime,
            params.time_column,
        )
    else:
        pc_min_secs, pc_max_secs = None, None

    # 2. Get the minimum time span in a cell in seconds
    params = get_min_timespan(params, mintime, maxtime)
    status = _init_status()

    # 3. Get unique chunks in the dataset
    part_df = get_unique_chunks(params)
    logger.info(f"Found {len(part_df)} chunks to process")

    return {
        "mintime": mintime,
        "maxtime": maxtime,
        "min_secs": min_secs,
        "max_secs": max_secs,
        "pc_min_secs": pc_min_secs,
        "pc_max_secs": pc_max_secs,
        "status": status,
        "part_df": part_df,
    }


# ----------------------------------------------------------------------------
# DATA LOADING FUNCTIONS - Load, preprocess, compute features
# ----------------------------------------------------------------------------


def filter_to_mode(lf: pl.LazyFrame) -> pl.LazyFrame:
    """
    Filter to the most common mode in each grid cell.
    If a cell has multiple modes, keep only the mode with the highest count of observations.

    Args:
        lf (pl.LazyFrame): Input Polars LazyFrame with :
            - Columns: x_bin, y_bin, mode

    Returns:
        pl.LazyFrame: Filtered LazyFrame with only most common mode per cell.
    """

    # Count observations per mode per cell (excluding null modes)
    mode_counts = lf.filter(pl.col("mode").is_not_null()).group_by(["x_bin", "y_bin", "mode"]).len()

    # Get cells with multiple modes
    multimode_cells = (
        mode_counts.group_by(["x_bin", "y_bin"])
        .len()
        .filter(pl.col("len") > 1)
        .select(["x_bin", "y_bin"])
    )

    mode_to_keep = (
        mode_counts.join(multimode_cells, on=["x_bin", "y_bin"])
        .with_columns(
            pl.col("len").rank("dense", descending=True).over(["x_bin", "y_bin"]).alias("rank")
        )
        .filter(pl.col("rank") == 1)
        .select(["x_bin", "y_bin", "mode"])
    )

    # For multi-mode cells: keep most common valid mode + all null modes
    # For single-mode cells: keep everything
    return pl.concat(
        [
            # Cells without multiple valid modes (keep all data)
            lf.join(multimode_cells, on=["x_bin", "y_bin"], how="anti"),
            # Multi-mode cells: keep only the most common non-null mode
            lf.join(multimode_cells, on=["x_bin", "y_bin"]).join(
                mode_to_keep, on=["x_bin", "y_bin", "mode"], how="semi"
            ),
            # lf.join(multimode_cells, on=["x_bin", "y_bin"]).filter(pl.col("mode").is_null()),
        ]
    )


def get_grid_data(
    parquet_glob: str, params: argparse.Namespace, min_secs: int, max_secs: int
) -> tuple[pl.LazyFrame, dict[str, int]]:
    """
    Load gridded altimetry data into a polars LazyFrame.
        - Load data
        - Filter to rows within specified time range
        - Filter to the mode with the most measurements in each grid cell
        - Filter to valid elevations
        - Add computed columns required for surface fitting

    Args:
        parquet_glob (str): Path to the parquet files
        params (argparse.Namespace): Configuration parameters
        min_secs (int): Minimum time in seconds
        max_secs (int): Maximum time in seconds
        mission (str): Mission identifier (e.g., 'cs2', 'ev', 'e1', 'e2')
    Returns:
        tuple[pl.LazyFrame, dict[str, int]]:
            - Filtered Polars DataFrame :
                - Columns: x_part, y_part, x_bin,y_bin, x,y,x2,y2,xy,height,heading,time,time_years
                - Optionally power, mode, weight_column
            - Status dictionary with counts of loaded and filtered measurements
    """
    # Load gridding data
    lf = pl.scan_parquet(parquet_glob)

    cells_before = (
        lf.filter(pl.col(params.elevation_column).is_not_null())
        .select(["x_bin", "y_bin"])
        .unique()
        .collect()
        .height
    )
    total_measurements = lf.select(pl.len()).collect().item()

    status = {
        "n_cells_with_data_loaded": cells_before,
        "n_measurements_loaded": total_measurements,
    }
    schema = lf.collect_schema()

    # Filter to mode values if specified
    if params.mode_values is not None:
        lf = lf.filter(pl.col(params.mode_column).is_in(params.mode_values))

    # Filter to most common mode per cell (using ALL time data)
    if params.mode_column and params.mode_column in schema:
        lf = filter_to_mode(lf)

    # Filter out data outside time range
    lf = lf.filter(
        (pl.col(params.time_column) >= min_secs) & (pl.col(params.time_column) <= max_secs)
    )

    # Filter to cells with valid elevations in each grid cell
    # Join back to only keep cells with valid elevations
    # Add computed columns required for surface fitting
    lf = lf.join(
        lf.filter(pl.col(params.elevation_column).is_not_null())
        .select(["x_bin", "y_bin"])
        .unique(),
        on=["x_bin", "y_bin"],
        how="inner",
    ).with_columns(
        [
            (pl.col(params.x_column).alias("x")),
            (pl.col(params.y_column).alias("y")),
            (pl.col(params.x_column) ** 2).alias("x2"),
            (pl.col(params.y_column) ** 2).alias("y2"),
            (pl.col(params.x_column) * pl.col(params.y_column)).alias("xy"),
            (pl.col(params.elevation_column).alias("height")),
            (pl.col(params.heading_column).cast(pl.Int32).alias("heading")),
            (pl.col(params.time_column) / SECONDS_PER_YEAR).alias("time_years"),
        ]
    )
    schema = lf.collect_schema()
    columns = [
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
    if params.power_column and params.power_column in schema:
        columns.append(params.power_column)
    if params.mode_column and params.mode_column in schema:
        columns.append(params.mode_column)
    if params.weight_column and params.weight_column in schema:
        columns.append(params.weight_column)

    return lf.select([pl.col(col) for col in columns]), status


# ----------------------------------------------------------------------------
# SURFACE FITTING FUNCTIONS - Per-cell surface fit, power correction
# ----------------------------------------------------------------------------
# ---------------------
# Surface Model Fit
# ---------------------
def _apply_dh(group_np: dict[str, np.ndarray], fit_params: np.ndarray) -> dict[str, np.ndarray]:
    """
    Update the dH column by removing surface components from elevation.
    Remove modelled heights from measured elevation
    This leave the temporal change + residuals

    Args:
        group_np (dict[str, np.ndarray]): Cell data arrays.
        fit_params (np.ndarray): Surface fit parameters
    Returns:
        dict[str, np.ndarray]: Corrected dH
    """
    modelled_heights_surface_component = (
        fit_params[0]
        + fit_params[1] * group_np["x"]
        + fit_params[2] * group_np["y"]
        + fit_params[3] * group_np["x2"]
        + fit_params[4] * group_np["y2"]
        + fit_params[5] * group_np["xy"]
        + fit_params[6] * group_np["heading"]
    )
    group_np["dH"] = group_np["height"] - modelled_heights_surface_component
    return group_np


def _get_modelled_heights_and_sigma_filter(
    group_np: dict[str, np.ndarray],
    fit_params: np.ndarray,
    ref_time: np.ndarray,
    n_sigma: float,
) -> np.ndarray:
    """
    Get modelled heights and filter outliers based on n_sigma threshold.
    Get the standard deviation (sigma) of the absolute differences
    between heights and the modelled heights.

    Filter to where the differences > n_sigma * sigma

    Args:
        group_np (dict[str, np.ndarray]): Cell data arrays.
        fit_params (np.ndarray): Surface fit parameters
        ref_time (np.ndarray): Reference time array for surface fit
        n_sigma (float): Sigma threshold for filtering

    Returns:
        Boolean mask: True where residuals are within n_sigma * sigma.
    """
    modelled_heights = (
        fit_params[0]
        + fit_params[1] * group_np["x"]
        + fit_params[2] * group_np["y"]
        + fit_params[3] * group_np["x2"]
        + fit_params[4] * group_np["y2"]
        + fit_params[5] * group_np["xy"]
        + fit_params[6] * group_np["heading"]
        + fit_params[7] * ref_time
    )
    differences = modelled_heights - group_np["height"]
    sigma = np.std(differences, ddof=1)
    return np.abs(differences) <= n_sigma * sigma


def _get_fit_params(
    params: argparse.Namespace,
    group_np: Dict[str, np.ndarray],
    ref_time: np.ndarray,
    logger: logging.Logger,
) -> tuple[np.ndarray | None, str | None]:
    """
    Solve the surface model for a single cell.

    Returns:
        tuple[np.ndarray | None, str | None]: (fit_params, error_status)
            - fit_params: ndarray of params if successful, else None
            - error_status: None on success, or "n_cells_fit_failed" on failure
    """
    iv = np.column_stack(
        (
            np.ones(ref_time.size),  # constant column
            group_np["x"],
            group_np["y"],
            group_np["x2"],
            group_np["y2"],
            group_np["xy"],
            group_np["heading"],
            ref_time,
        )
    )

    # --------------------------
    # Weighted surface fit
    # ---------------------------
    if params.weighted_surface_fit:
        weights = group_np[params.weight_column]
        w = np.where(weights > 0, 1.0 / (weights**2), 0.0)
        wt = w / np.nansum(w)
        try:
            res = sm.WLS(group_np["height"], iv, weights=wt, missing="drop").fit()
            if res.model.data.param_names != [
                "const",
                "x1",
                "x2",
                "x3",
                "x4",
                "x5",
                "x6",
                "x7",
            ]:
                return None, "n_cells_fit_failed"

        # pylint: disable=W0718
        except Exception as e:
            logger.warning(f"WLS failed with: {e}")
            return None, "n_cells_fit_failed"
        return res.params, None
    # ----------------------------------------------------
    # Least Squares fit to a surface function
    # -----------------------------------------------------
    try:
        fit_params, *_ = np.linalg.lstsq(iv, group_np["height"], rcond=None)
        return fit_params, None
    except np.linalg.LinAlgError as e:
        logger.warning(f"lstsq failed with: {e}")
        return None, "n_cells_fit_failed"


# pylint: disable=R0911
def fit_surface_model_per_group(
    params: argparse.Namespace,
    group_np: dict[str, np.ndarray],
    min_seconds: float,
    max_seconds: float,
    logger: logging.Logger,
) -> tuple[np.ndarray, dict[str, np.ndarray]] | str:
    """
    Fit a surface model per grid cell.

    Model form (per point):
        z(x, y, t, heading) = a0 + a1*x + a2*y + a3*x^2 + a4*y^2 + a5*xy + a6*heading + a7*t

    Workflow (per cell):
    1) Build design matrix from spatial terms, heading, and centered time.
    2) Solve either weighted least squares (when weighted_surface_fit) or ordinary least squares.
        (_get_fit_params)
    3) Compute residuals and drop points outside n_sigma until convergence or iteration cap.
        (_get_modelled_heights_and_sigma_filter)
    4) On convergence, compute dH (elevation minus spatial surface component) and
    return params + filtered data (_apply_dh)
    Args:
        params (argparse.Namespace): Command Line Parameters.
                Includes : n_sigma, max_surfacefit_iterations, weighted_surface_fit,weight_column,
                min_vals_in_cell, min_timespan_in_cell_in_secs
        group_np (dict[str,np.ndarray]): Cell data arrays.
        Required keys: "height", "time", "time_years", "x", "y", "x2", "y2", "xy", "heading".
                If weighted, also weight_column key.
        min_seconds (float): Surface-fit window start. (Seconds since epoch)
        max_seconds (float): Surface-fit window end. (Seconds since epoch)a
        logger (logging.Logger): Logger Object.

    Returns:
        tuple[np.ndarray, dict[str, np.ndarray]] | str:
            On success, returns (fit_params, filtered_data_dict) where:
                - fit_params: numpy array with [const, x, y, x2, y2, xy, heading, time]
                - filtered_data_dict: dictionary with filtered arrays
            On failure, returns error status string.
    """
    # --------------------------------------------------------------------------
    # Iterate a surface model fit to the cell
    # --------------------------------------------------------------------------

    if np.ptp(group_np["time"]) < params.min_timespan_in_cell_in_secs:
        return "n_cells_with_timespan_too_short"

    for _ in range(params.max_surfacefit_iterations):
        if group_np["height"].size < params.min_vals_in_cell:
            return "n_cells_with_too_few_measurements"

        # Reference time for surface fit is centred on the mid-point of the time window
        ref_time = group_np["time_years"] - ((min_seconds + max_seconds) / (2.0 * SECONDS_PER_YEAR))
        # Perform surface fit
        fit_params, error_status = _get_fit_params(params, group_np, ref_time, logger)
        if error_status is not None:
            return error_status
        assert fit_params is not None

        mask = _get_modelled_heights_and_sigma_filter(
            group_np, fit_params, ref_time, params.n_sigma
        )
        if np.count_nonzero(mask) < params.min_vals_in_cell:
            return "n_cells_with_too_few_measurements_after_fit_sigma_filter"

        # ----------------------------
        # If the surface fit has converged / All points within n_sigma
        # ----------------------------
        if np.count_nonzero(~mask) == 0:
            group_np = _apply_dh(group_np, fit_params)
            return (
                fit_params,
                {k: v for k, v in group_np.items() if k not in ["x", "y", "x2", "y2", "xy"]},
            )

        # ---------------------------------------------
        # Apply mask to all arrays in group dictionary
        # Remove outliers
        # ----------------------------------------------
        for key in group_np:
            group_np[key] = group_np[key][mask]

    if params.weighted_surface_fit:
        if fit_params is not None:
            group_np = _apply_dh(group_np, fit_params)
            return (
                fit_params,
                {k: v for k, v in group_np.items() if k not in ["x", "y", "x2", "y2", "xy"]},
            )
        return "n_cells_fit_failed"

    return "n_cells_didnot_converge"


# ------------------------
# Power Correction
# ------------------------


def fit_power_correction_per_group(
    params: argparse.Namespace,
    group_np: dict[str, np.ndarray],
    time_params: dict[str, float],
    logger: logging.Logger,
) -> dict[str, np.ndarray] | str:
    """
    Apply power (backscatter) correction to elevation residuals.
    Fits an OLS model to power as a function of time and heading (p = a + bt + ch),
    then removes height components correlated with time-dependent power changes.
    The correction is applied as: dH = dH - (dH/dP) * dPif params.powercorrect:, where dH/dP is the
    gradient calculated over the power correction time window.

    Args:
        params (argparse.Namespace): Configuration parameters
        group_np (dict[str, np.ndarray]): Grid cell containing:
        "heading", "time", "time_years", "power", "dH"
        time_params (dict[str, float]): Dictionary containing time parameters:
            - pc_min_secs (float): Minimum time for power correction period (seconds)
            - pc_max_secs (float): Maximum time for power correction period (seconds)
            - mintime (float): Minimum time for surface fit reference (seconds)
            - maxtime (float): Maximum time for surface fit reference (seconds)
        logger (logging.Logger): Logger Object
    Returns:
        dict[str, np.ndarray] | str: Dictionary with corrected data ("time", "time_years", "dH")
                                    or error status string.
    """

    pc_timeok_indices = np.where(
        (group_np["time"] >= time_params["pc_min_secs"])
        & (group_np["time"] <= time_params["pc_max_secs"])
    )[0]

    if pc_timeok_indices.size >= params.min_vals_in_cell:
        ref_time = group_np["time_years"] - (
            (time_params["mintime"] + time_params["maxtime"]) / (2.0 * SECONDS_PER_YEAR)
        )

        # ----------------------------------------------------------------------------------------
        # Fit a model to power as a function of time and heading, p = a + bt + ch
        # ----------------------------------------------------------------------------------------
        iv = np.column_stack((np.ones(ref_time.size), ref_time, group_np["heading"]))
        # Check if weighted power fit is enabled and weight_column is available
        if (
            hasattr(params, "weighted_power_fit")
            and params.weighted_power_fit
            and params.weight_column
            and params.weight_column in group_np
        ):
            weights = group_np[params.weight_column]
            w = np.where(weights > 0, 1.0 / (weights**2), 0.0)
            wt = w / np.nansum(w)
            power_res = sm.WLS(group_np["power"], iv, weights=wt, missing="drop").fit()
            if power_res.model.data.param_names != ["const", "x1", "x2"]:
                return "n_cells_fit_failed"
        else:
            try:
                power_res, *_ = np.linalg.lstsq(iv, group_np["power"], rcond=None)
            except np.linalg.LinAlgError as e:
                logger.warning(f"lstsq failed with: {e}")
                return "n_cells_fit_failed"
        # -----------------------------------------------------------------
        # Get the time dependent component of power is: bt = p -(a  + ch)
        # ------------------------------------------------------------------
        dp = group_np["power"] - (power_res[0] + power_res[2] * group_np["heading"])
        # --------------------------------------------------------------
        # Remove component of dH due to correlated power change
        # Find the gradient of dH/dP over time interval (pcmintime,pcmaxtime)
        # --------------------------------------------------------------
        m, _ = np.polyfit(dp[pc_timeok_indices], group_np["dH"][pc_timeok_indices], 1)  # m=dH/dP
        group_np["dH"] -= m * dp

        # Return appropriate keys based on whether weights are available
        return_keys = ["time", "time_years", "dH", "heading"]
        if params.weight_column and params.weight_column in group_np:
            return_keys.append(params.weight_column)

        return {key: value for key, value in group_np.items() if key in return_keys}

    # Return uncorrected data if insufficient time points (matches old_sf.py behavior)
    return_keys = ["time", "time_years", "dH", "heading"]
    if params.weight_column and params.weight_column in group_np:
        return_keys.append(params.weight_column)
    return {key: value for key, value in group_np.items() if key in return_keys}


# ------------------------
# Linear Fit
# ------------------------


def fit_linear_fit_per_group(
    params: argparse.Namespace, group_np: dict[str, np.ndarray]
) -> dict[str, Any] | str:
    """
    Calculate (dH/dt), from heights and time :  h = a + bt
    Iterate until the difference between the dH and modelled dH are
    are within +/-2 * rms or the max_linearfit_iterations limit is reached.

    Args:
        params (argparse.Namespace): Configuration parameters
        group_np (dict[str, np.ndarray]): Grid cell data dictionary with keys:
            "time_years", "dH" (elevation residuals after surface/power correction)

    Returns:
        dict[str, Any] | str:
            On success, returns dictionary with keys:
                - "m": slope (float)
                - "rms": root mean square error (float)
                - "sigma": standard deviation (float)
                - "mask": boolean array (np.ndarray)
                - "group_np": filtered data dictionary (dict[str, np.ndarray])
                - "std_err": standard error (float or None)
            On failure, returns error status string.
    """
    for _ in range(params.max_linearfit_iterations):
        if (
            hasattr(params, "weighted_power_fit")
            and params.weighted_power_fit
            and params.weight_column
            and params.weight_column in group_np
        ):
            if np.nanmax(group_np["time"]) == np.nanmin(group_np["time"]) or np.nanmax(
                group_np["time_years"]
            ) == np.nanmin(group_np["time_years"]):
                return "n_cells_time_identical"

            m, c, _, _, std_err = stats.linregress(group_np["time_years"], group_np["dH"])
            modelled_dh = m * group_np["time_years"] + c
        else:
            m, c = np.polyfit(group_np["time_years"], group_np["dH"], 1)
            poly1d_fn = np.poly1d((m, c))
            modelled_dh = poly1d_fn(group_np["time_years"])
            std_err = None

        differences = group_np["dH"] - modelled_dh
        rms = np.sqrt(np.mean(differences**2))
        sigma = np.std(differences, ddof=1)
        mask = np.absolute(differences) <= rms * 2.0

        if np.count_nonzero(mask) == 0:
            break
        if np.count_nonzero(mask) <= 3:
            break

        for key in group_np:  # Apply mask to all arrays in group dictionary
            group_np[key] = group_np[key][mask]

    if group_np["time"].size <= 3:
        return "n_too_few_values_in_linear_fit"

    return {
        "m": m,
        "rms": rms,
        "sigma": sigma,
        "mask": mask,
        "group_np": group_np,
        "std_err": std_err,
    }


# ------------------------
# Output Functions
# ------------------------


def write_chunk_output(
    params: argparse.Namespace,
    timeseries_records: list[dict] | None = None,
    row: dict | None = None,
    grid_records: list[dict] | None = None,
    chunk_id: int = 0,
) -> None:
    """
    Write per-chunk surface fit outputs to parquet files.

    1. Timeseries data (time, dh, weights):
        <out_dir>/x_part=X/y_part=Y/dh_time_grid.parquet
    2. Grid cell metadata (x_bin, y_bin, dhdt, slope, sigma, rms):
            Temporary chunk file - consolidated with 'consolidate_grid_chunks'

    Args:
        params (argparse.Namespace): Command line parameters with out_dir.
        timeseries_records (list[dict] | None): List of timeseries records to write.
        row (dict | None): Dictionary with x_part and y_part keys for directory partitioning.
                          If None, uses root output directory.
        grid_records (list[dict] | None): List of grid cell records for temporary chunk file.
        chunk_id (int): Unique chunk identifier for temporary file naming.
    """
    if timeseries_records is None:
        timeseries_records = []
    if grid_records is None:
        grid_records = []

    if row is not None:
        # Directory-partitioned case
        chunk_outdir = Path(params.out_dir) / f"x_part={row['x_part']}" / f"y_part={row['y_part']}"
    else:
        # Column-partitioned case
        chunk_outdir = Path(params.out_dir)

    os.makedirs(chunk_outdir, exist_ok=True)
    if timeseries_records:
        pl.DataFrame(timeseries_records).lazy().sink_parquet(
            chunk_outdir / "dh_time_grid.parquet", compression="zstd"
        )

    # Write to temporary chunk file
    if grid_records:
        chunk_path = Path(params.out_dir) / f".grid_chunk_{chunk_id:06d}.parquet"
        pl.DataFrame(grid_records).write_parquet(chunk_path, compression="zstd")


def consolidate_grid_chunks(params: argparse.Namespace, logger: logging.Logger) -> None:
    """
    Consolidate temporary chunk parquet files into a single grid_data.parquet output.

    During per-chunk surface fitting, grid results are written to temporary files
    (.grid_chunk_XXXXXX.parquet). This function streams all chunks together into a
    single consolidated output file, then removes the temporary chunk files.

    Args:
        params (argparse.Namespace): Command line parameters
        logger (logging.Logger): Logger object
    """
    output_path = Path(params.out_dir) / "grid_data.parquet"
    chunk_pattern = str(Path(params.out_dir) / ".grid_chunk_*.parquet")

    # Check if any chunk files exist
    chunk_files = list(Path(params.out_dir).glob(".grid_chunk_*.parquet"))
    if not chunk_files:
        logger.warning("No grid chunk files found to consolidate")
        return

    logger.info(f"Consolidating {len(chunk_files)} grid chunk files into {output_path}")

    # Stream-concatenate all chunks without loading into memory
    pl.scan_parquet(chunk_pattern).sink_parquet(output_path, compression="zstd")

    # Clean up temporary chunk files
    for chunk_file in chunk_files:
        chunk_file.unlink()

    logger.info(f"Consolidation complete, removed {len(chunk_files)} temporary files")


# ------------------------
# Metadata JSON
# ------------------------
def get_metadata_json(
    params: argparse.Namespace, status: Dict[str, int], start_time: float, logger: logging.Logger
) -> None:
    """
    Create and save surface fit metadata.json in the output directory.
    Metadata includes:
        - Command line parameters
        - Processing status
        - Execution time
    Args:
        params (argparse.Namespace): Command line parameters
        status (dict): Processing status information
        start_time (float): Start time of the processing
        logger (logging.Logger): Logger object
    """
    meta_json_path = Path(params.out_dir) / "metadata.json"
    hours, remainder = divmod(int(time.time() - start_time), 3600)
    minutes, seconds = divmod(remainder, 60)

    try:
        with open(meta_json_path, "w", encoding="utf-8") as f_meta:
            json.dump(
                {
                    **vars(params),
                    **status,
                    "execution_time": f"{hours:02}:{minutes:02}:{seconds:02}",
                },
                f_meta,
                indent=2,
            )
        logger.info("Wrote data_set metadata to %s", meta_json_path)

    except OSError as e:
        logger.error("Failed to write surface_fit_meta.json with %s", e)


# --------------------------------------------
# Main Processing Workflow
# --------------------------------------------


# pylint: disable=R0914
def surface_fit(
    params: argparse.Namespace, sf_objects: dict, logger: logging.Logger
) -> dict[str, int]:
    """
    Perform surface fit on grid data per grid cell.

    Steps:
        1. Loop though chunks
        2. Load grid data for chunk using 'get_grid_data'
        3. Loop through grid cells :
            a. Convert gridcell data to a numpy arrays
            b. Perform surface / plane fit 'fit_surface_model_per_group'
            c. Perform power correction 'fit_power_correction_per_group'
            d. Perform linear fit 'fit_linear_fit_per_group'
            e. Construct output dictionaries
        4. Write parquet files
            - A parquet of grid cells metadata
            - A timeseries grid

    Args:
        params (argparse.Namespace): Configuration parameters
        sf_objects (dict): Surface fit objects from 'get_surface_fit_objects'
        logger (logging.Logger): Logger Object
    Returns:
        dict[str, int]: Status dictionary containing result counts
    """

    status = sf_objects["status"]
    min_secs = sf_objects["min_secs"]
    max_secs = sf_objects["max_secs"]
    pc_min_secs = sf_objects["pc_min_secs"]
    pc_max_secs = sf_objects["pc_max_secs"]
    chunk_id = 0

    # 1. Loop through grid chunks
    for row in sf_objects["part_df"].iter_rows(named=True):
        grid_records = []
        timeseries_records = []
        chunk_id = chunk_id + 1
        logger.info(f"Processing chunk : {chunk_id} / {len(sf_objects['part_df'])}")

        # 2. Load gridded data for this chunk
        gridcell_lazy, chunk_status = get_grid_data(
            f"{params.in_dir}/**/x_part={row['x_part']}/y_part={row['y_part']}/*.parquet",
            params,
            min_secs,
            max_secs,
        )

        # Merge chunk status into overall status
        for key, value in chunk_status.items():
            status[key] += value

        grouped = gridcell_lazy.collect().group_by(["x_bin", "y_bin", "x_part", "y_part"])

        # 3. Loop through grid cells in this chunk
        for (x_bin, y_bin, x_part, y_part), gridcell in grouped:
            # 3a. Construct numpy arrays of grid cell data used in fitting
            gridcell_np = {
                col: gridcell[col].to_numpy()
                for col in [
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
                    params.weight_column,  # Only loaded for is2
                ]
                if col in gridcell.columns
            }

            # 3b. Plane fit to surface
            surface_result = fit_surface_model_per_group(
                params=params,
                group_np=gridcell_np,
                min_seconds=min_secs,
                max_seconds=max_secs,
                logger=logger,
            )

            if isinstance(surface_result, str):
                status[surface_result] += 1
                continue
            res, group_np = surface_result

            # 3c. Power Correction (if enabled)
            if params.powercorrect:
                power_result = fit_power_correction_per_group(
                    params,
                    group_np,
                    time_params={
                        "pc_min_secs": pc_min_secs,
                        "pc_max_secs": pc_max_secs,
                        "mintime": min_secs,
                        "maxtime": max_secs,
                    },
                    logger=logger,
                )

                if isinstance(power_result, str):
                    status[power_result] += 1
                    continue
                status["n_cells_power_corrected"] += 1
                group_np = power_result

            # 3d. Linear fit to get dhdt
            linear_result = fit_linear_fit_per_group(params, group_np)
            if isinstance(linear_result, str):
                status[linear_result] += 1
                continue

            # 3e. Construct output record for grid cell
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
            if linear_result.get("std_err") is not None:
                grid_record["std_err"] = linear_result.get("std_err")
            grid_records.append(grid_record)

            for idx, (ts_val, ty_val, dh_val) in enumerate(
                zip(
                    linear_result["group_np"]["time"],
                    linear_result["group_np"]["time_years"],
                    linear_result["group_np"]["dH"],
                )
            ):
                ts_record = {
                    "x_part": x_part,
                    "y_part": y_part,
                    "x_bin": x_bin,
                    "y_bin": y_bin,
                    "time": ts_val,
                    "time_years": ty_val,
                    "dh": dh_val,
                }

                if params.weight_column in linear_result["group_np"]:
                    ts_record[params.weight_column] = linear_result["group_np"][
                        params.weight_column
                    ][idx]
                timeseries_records.append(ts_record)

            status["n_cells_fitted"] += 1

        # 3. Write Parquet Files per chunk
        write_chunk_output(params, timeseries_records, row, grid_records, chunk_id)

    # 4. Consolidate all chunk grid files into single output file
    consolidate_grid_chunks(params, logger=logger)

    return status


def main(args: list[str] | None = None) -> None:
    """
    Main entry point orchestrating the three-phase surface fit pipeline.

    Supports:
        - root processing (default or when structure='root')
        - basin-wise processing for non-root structure

    Setup: Parse args, load grid metadata, initialize logging and status counters
    Initialize: Compute time bounds, identify spatial chunks to process
    Process: For each chunk, load data, then fit surface per grid cell

    Args:
        args: Command line arguments. If None, uses sys.argv[1:]
    """
    params = parse_arguments(args)

    def run_once(run_params: argparse.Namespace) -> None:
        """Run the surface fit pipeline for a single root or basin directory."""
        start_time = time.time()

        # Validate weighted fit config
        if run_params.weighted_surface_fit and run_params.weight_column is None:
            sys.exit("Weight name must be provided for weighted surface fit")

        # Initialize logging and load grid metadata
        parquet_glob = os.path.join(run_params.in_dir, run_params.parquet_glob)
        logger, grid_meta = clean_directory(run_params, confirm_regrid=False)
        sf_objects = get_surface_fit_objects(run_params, parquet_glob, grid_meta, logger)

        status = surface_fit(run_params, sf_objects=sf_objects, logger=logger)

        # Write final metadata
        get_metadata_json(run_params, status, start_time, logger)

    # Basin-wise processing for single-tier or two-tier structures
    if params.basin_structure is True:
        # Create a logger for basin selection
        selector_logger = logging.getLogger("surface_fit.basin_selection")
        selector_logger.setLevel(logging.INFO)
        basins = get_basins_to_process(params, Path(params.in_dir), selector_logger)

        for basin in basins:
            basin_params = argparse.Namespace(**vars(params))
            basin_params.in_dir = (
                str(Path(params.in_dir) / basin) if basin != "None" else params.in_dir
            )
            basin_params.out_dir = (
                str(Path(params.out_dir) / basin) if basin != "None" else params.out_dir
            )
            run_once(basin_params)
        return

    run_once(params)


if __name__ == "__main__":
    main(sys.argv[1:])
