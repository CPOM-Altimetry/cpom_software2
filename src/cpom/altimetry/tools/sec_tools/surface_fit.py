# pylint: disable=C0302
"""
cpom.altimetry.tools.sec_tools.surface_fit.py

Purpose:
  Perform planefit to get dh/dt time series from
  gridded elevation and time data.
  Output dh time series as parquet files
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

from cpom.logging_funcs.logging import set_loggers


def get_set_up_objects(params: argparse.Namespace, confirm_regrid: bool = False):
    """
    Creates the output directory and logger for surface fit.
    Reads grid metadata, verifies and recreates the output directory,
    initializes file-based logging.
    Args:
        params (argparse.Namespace): Command line parameters
        confirm_regrid (bool): If True,
            prompts user before removing existing output directory.

    Returns:
        Tuple: Logger object, grid metadata.
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
        log_file_info=Path(params.out_dir) / "info.log",
        log_file_error=Path(params.out_dir) / "errors.log",
    )
    logger.info("output_dir=%s", params.out_dir)

    return logger, grid_meta


def get_metadata_json(
    params: argparse.Namespace, status: Dict[str, int], start_time: float, logger: logging.Logger
) -> None:
    """
    Create and save metadata.json in the output directory.
    Metadata includes:
        - Command line parameters
        - Processing Status
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


def get_min_max_time(
    epoch_time: datetime, parquet_glob: str, mintime: str | None = None, maxtime: str | None = None
) -> tuple[float, float, float, float]:
    """
    Get the min/max time as a timestamp and in seconds.
    Calculated from :
        mintime and maxtime date strings
        or
        The date range in the data if not provided.
    Args:
        mintime (str: Optional): Start time in (DD/MM/YYYY or DD.MM.YYYY).
        maxtime (str: Optional): End time in (DD/MM/YYYY or DD.MM.YYYY).
        epoch_time (datetime): Epoch time is referenced to.
        parquet_glob (str): Directory glob for parquet files
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
            [pl.col("time").min().alias("min_time"), pl.col("time").max().alias("max_time")]
        ).collect()

        min_secs = min_max["min_time"][0]
        max_secs = min_max["max_time"][0]
        min_dt = epoch_time + timedelta(seconds=min_secs)
        max_dt = epoch_time + timedelta(seconds=max_secs)
    return min_dt.timestamp(), max_dt.timestamp(), min_secs, max_secs


def get_grid_data(
    parquet_glob: str, params: argparse.Namespace, min_secs: int, max_secs: int
) -> tuple[pl.LazyFrame, dict[str, int]]:
    """
    Load data from parquet files into a Polars DataFrame.
    Filters to :
        - Specified time range
        - The mode with the most measurements in each grid cell
        - Valid elevations
    Args:
        parquet_glob (str): Path to the parquet files
        params (argparse.Namespace): Configuration parameters
        min_secs (int): Minimum time in seconds
        max_secs (int): Maximum time in seconds
        mission (str): Mission identifier (e.g., 'cs2', 'ev', 'e1', 'e2')
    Returns:
        tuple[pl.LazyFrame, dict[str, int]]: Filtered Polars DataFrame and status counts
    """
    # Get all data first to count total cells
    lf = pl.scan_parquet(parquet_glob)
    total_cells_with_data = (
        lf.filter(pl.col("elevation").is_not_null())
        .select(["x_bin", "y_bin"])
        .unique()
        .collect()
        .height
    )
    n_measurements_loaded = lf.select(pl.len()).collect().item()

    # Get cells with valid elevations after time filter
    valid_cells = lf.filter(pl.col("elevation").is_not_null()).select(["x_bin", "y_bin"]).unique()

    # Update status with loaded counts
    # Calculate status
    status = {
        "n_cells_with_data_loaded": total_cells_with_data,
        "n_measurements_loaded": n_measurements_loaded,
    }

    # Filter out data outside time range
    lf = lf.filter((pl.col("time") >= min_secs) & (pl.col("time") <= max_secs))
    # get counts after time filter
    total_cells_with_data_after_time_filter = (
        lf.filter(pl.col("elevation").is_not_null())
        .select(["x_bin", "y_bin"])
        .unique()
        .collect()
        .height
    )
    status["n_measurements_with_time_outside_minmax"] = (
        total_cells_with_data - total_cells_with_data_after_time_filter
    )

    # Join with valid cells and add computed columns
    lf = lf.join(valid_cells, on=["x_bin", "y_bin"], how="inner").with_columns(
        [
            (pl.col("x_cell_offset").alias("x")),
            (pl.col("y_cell_offset").alias("y")),
            (pl.col("x_cell_offset") ** 2).alias("x2"),
            (pl.col("y_cell_offset") ** 2).alias("y2"),
            (pl.col("x_cell_offset") * pl.col("y_cell_offset")).alias("xy"),
            (pl.col("elevation").alias("height")),
            (pl.col("ascending").cast(pl.Int32).alias("heading")),
            (pl.col("time") / (31557600)).alias("time_years"),
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
    if "power" in schema:
        columns.append("power")
    if "mode" in schema:
        columns.append("mode")
    if params.weight_name and params.weight_name in schema:
        columns.append(params.weight_name)

    # Filter to mode
    if params.mode_values is not None:
        lf = lf.filter(pl.col("mode").is_in(params.mode_values))

    if "mode" in schema:
        lf = filter_to_mode(lf)

    return lf.select([pl.col(col) for col in columns]), status


def filter_to_mode(lf: pl.LazyFrame) -> pl.LazyFrame:
    """
    Filter the DataFrame to keep only the most common mode in cells with multiple modes.

    Mission Mode Definitions:
        - cs2: 1=LRM, 2=SAR, 3=SIN. Filter if both LRM and SIN present.
        - ev: 0=320MHz, 1=80MHz, 2=20MHz. Filter if multiple bandwidth modes present.
        - e1/e2: 2=ocean mode, 3=ice mode. Filter if both present.

    Args:
        lf (pl.LazyFrame): Input Polars LazyFrame with mode information.
        mission (str): Mission identifier ('cs2', 'ev', 'e1', 'e2')

    Returns:
        pl.LazyFrame: Filtered LazyFrame with only most common mode per cell where multiple existed.
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

    if multimode_cells.collect().height == 0:
        return lf

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
            # Multi-mode cells: filter to keep only most common mode OR null modes
            lf.join(multimode_cells, on=["x_bin", "y_bin"]).join(
                mode_to_keep, on=["x_bin", "y_bin", "mode"], how="semi"
            ),
            lf.join(multimode_cells, on=["x_bin", "y_bin"]).filter(pl.col("mode").is_null()),
        ]
    )


def get_surface_fit_objects(
    params: argparse.Namespace, parquet_glob: str, grid_meta: dict, logger: logging.Logger
) -> dict[str, Any]:
    """
    Prepare objects required for surface fitting.
    1. Get the min/max time period for surface fit and power correction
    2. Get the minimum time span in a cell in seconds
    3. Get unique chunks (x_part, y_part) in the dataset
    Args:
        params (argparse.Namespace): Configuration parameters
        parquet_glob (str): Path to the input parquet files
        grid_meta (dict): Metadata for the grid
        logger (logging.Logger): Logger object
    Returns:
        dict[str, Any]: Surface fit objects containing time parameters, status, and unique chunks
        keys :
            - mintime (datetime)
            - maxtime (datetime)
            - min_secs (float)
            - max_secs (float)
            - pc_min_secs (float)
            - pc_max_secs (float)
            - status (dict)
            - part_df (pl.DataFrame)
    """

    def _get_min_timespan(
        params: argparse.Namespace, mintime: float, maxtime: float
    ) -> argparse.Namespace:
        params.min_timespan_in_cell_in_secs = (maxtime - mintime) * (
            params.min_percent_timespan_in_cell / 100.0
            if params.min_percent_timespan_in_cell
            else (maxtime - mintime) * 0.7
        )  # 70% of required time span
        return params

    def _init_status() -> dict[str, int]:
        return {
            k: 0
            for k in [
                "n_cells_with_data_loaded",
                "n_measurements_loaded",
                "n_measurements_with_time_outside_minmax",
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

    def _get_unique_chunks(params: argparse.Namespace) -> pl.DataFrame:
        conn = duckdb.connect()
        part_df = conn.execute(
            f"""
            SELECT DISTINCT x_part, y_part
            FROM read_parquet('{params.in_dir}/**/x_part=*/y_part=*/*.parquet',
            hive_partitioning=1);
            """
        ).pl()
        conn.close()
        return part_df

    # 1. Get the min/max time period for surface fit and power correction
    mintime, maxtime, min_secs, max_secs = get_min_max_time(
        datetime.fromisoformat(grid_meta["standard_epoch"]),
        parquet_glob,
        params.mintime,
        params.maxtime,
    )
    logger.info(f"Surface fit time range : {mintime} : {maxtime}")

    if params.powercorrect:
        _, _, pc_min_secs, pc_max_secs = get_min_max_time(
            datetime.fromisoformat(grid_meta["standard_epoch"]),
            parquet_glob,
            params.pcmintime,
            params.pcmaxtime,
        )
    else:
        pc_min_secs, pc_max_secs = None, None

    # 2. Get the minimum time span in a cell in seconds
    params = _get_min_timespan(params, mintime, maxtime)
    status = _init_status()

    # 3. Get unique chunks in the dataset
    part_df = _get_unique_chunks(params)
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


def _fit_surface_model_per_group(
    params: argparse.Namespace,
    group_np: dict[str, np.ndarray],
    mintime: float,
    maxtime: float,
    logger: logging.Logger,
) -> tuple[np.ndarray, dict[str, np.ndarray]] | str:
    """
    Fits a weighted linear regression model to a
    surface fit function for each grid cell.

    Function :
    z(x,y,t,heading) = a0 + a1x + a2y + a3x^2 + a4y^2 + a5xy + a6heading+ a7t
    Iteratively filters out outliers beyond n_sigma standard deviations until convergence
    or maximum iterations reached.
    Compute dH removing fitted surface components from elevations.

    Args:
        params (argparse.Namespace): Configuration parameters.
        group_np (dict[str, np.ndarray]): Grid cell data containing:
            "height", "time", "time_years", "x", "y", "x2", "y2", "xy", "heading"
        mintime (float): Minimum time for surface fit (seconds since 1991-01-01)
        maxtime (float): Maximum time for surface fit (seconds since 1991-01-01)
        logger (logging.Logger): Logger Object
    Returns:
        tuple[np.ndarray, dict[str, np.ndarray]] | str:
            On success, returns (fit_params, filtered_data_dict) where:
                - fit_params: numpy array with [const, x, y, x2, y2, xy, heading, time]
                - filtered_data_dict: dictionary with filtered arrays
            On failure, returns error status string.
    """

    def _get_surface_fit(
        params: argparse.Namespace, group_np: Dict[str, np.ndarray], ref_time: np.ndarray
    ) -> tuple[np.ndarray | None, str | None]:
        """
        Get a design matrix for the surface fit.
        Fit the surface model to the data.

        Returns:
            tuple[np.ndarray | None, str | None]: (fit_params, error_status)
                - fit_params: numpy array if successful, None if failed
                - error_status: None if successful, error string if failed
        """
        iv = np.zeros((ref_time.size, 7))
        iv[:, 0] = group_np["x"]
        iv[:, 1] = group_np["y"]
        iv[:, 2] = group_np["x2"]
        iv[:, 3] = group_np["y2"]
        iv[:, 4] = group_np["xy"]
        iv[:, 5] = group_np["heading"]
        iv[:, 6] = ref_time
        iv = sm.add_constant(iv, has_constant="add")

        # --------------------------
        # Weighted surface fit
        #   ---------------------------
        if params.weighted_surface_fit:

            weights = group_np[params.weight_name]
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

    def _get_modelled_heights_and_sigma_filter(
        group_np: dict[str, np.ndarray],
        fit_params: np.ndarray,
        ref_time: np.ndarray,
        n_sigma: float,
    ) -> np.ndarray:
        """
        Get modelled heights and filter outliers based on n_sigma threshold.
        Get the standard deviation (sigma) of the absolute differences between heights and
        the modelled heights. Filter to where the differences > n_sigma * sigma

        Returns:
            np.ndarray: Boolean mask where True indicates points within n_sigma threshold
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

    def _apply_dh(group_np: dict[str, np.ndarray], fit_params: np.ndarray) -> dict[str, np.ndarray]:
        """
        Update the dH column by removing surface components from elevation.
        Remove modelled heights from measured elevation
        This leave the temporal change + residuals

        Returns:
            dict[str, np.ndarray]: Updated group_np dictionary with dH column modified
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

    # --------------------------------------------------------------------------
    # Iterate a surface model fit to the cell
    # --------------------------------------------------------------------------
    for _ in range(params.max_surfacefit_iterations):

        if np.ptp(group_np["time"]) < params.min_timespan_in_cell_in_secs:
            return "n_cells_with_timespan_too_short"

        if group_np["height"].size < params.min_vals_in_cell:
            return "n_cells_with_too_few_measurements"

        # Reference time for surface fit is centred on the mid-point of the time window
        ref_time = group_np["time_years"] - ((mintime + maxtime) / (2.0 * 31557600))
        # Perform surface fit
        fit_params, error_status = _get_surface_fit(params, group_np, ref_time)
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
        if not params.weighted_surface_fit and np.count_nonzero(~mask) == 0:
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

    return "n_cells_didnot_converge"


def _fit_power_correction_per_group(
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
    ref_time = group_np["time_years"] - (
        (time_params["mintime"] + time_params["maxtime"]) / (2.0 * 31557600)
    )

    if pc_timeok_indices.size >= params.min_vals_in_cell:
        # ----------------------------------------------------------------------------------------
        # Fit a model to power as a function of time and heading, p = a + bt + ch
        # ----------------------------------------------------------------------------------------
        iv = np.zeros((ref_time.size, 2))
        iv[:, 0] = ref_time
        iv[:, 1] = group_np["heading"]
        iv = sm.add_constant(iv, has_constant="add")

        # Check if weighted power fit is enabled and weight_name is available
        if (
            hasattr(params, "weighted_power_fit")
            and params.weighted_power_fit
            and params.weight_name
            and params.weight_name in group_np
        ):
            weights = group_np[params.weight_name]
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
        return_keys = ["time", "time_years", "dH"]
        if params.weight_name and params.weight_name in group_np:
            return_keys.append(params.weight_name)

        return {key: value for key, value in group_np.items() if key in return_keys}
    return "n_cells_too_few_vals_after_pctime_filter"


def _fit_linear_fit_per_group(
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
            and params.weight_name
            and params.weight_name in group_np
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

        if np.count_nonzero(mask) <= 3:
            return "n_too_few_values_in_linear_fit"

        for key in group_np:  # Apply mask to all arrays in group dictionary
            group_np[key] = group_np[key][mask]

    return {
        "m": m,
        "rms": rms,
        "sigma": sigma,
        "mask": mask,
        "group_np": group_np,
        "std_err": std_err,
    }


# pylint: disable=R0914
def surface_fit(
    params: argparse.Namespace, sf_objects: dict, logger: logging.Logger
) -> dict[str, int]:
    """
    Perform surface fit on grid data per grid cell. Write to parquet files
    in the output directory.
    Steps:
    1. Loop though chunks and load data using 'get_grid_data'
    2. Loop through grid cells :
        - Convert gridcell data to a numpy array
        - Perform surface fit 'fit_surface_model_per_group'
        - Perform power correction 'fit_power_correction_per_group'
        - Perform linear fit 'fit_linear_fit_per_group'
        - Construct output dictionaries
    3. Write parquet files
        - A parquet of grid cells metadata
        - A timeseries grid

    Args:
        params (argparse.Namespace): Configuration parameters
        sf_objects (dict): Surface fit objects from 'get_surface_fit_objects'
        logger (logging.Logger): Logger Object
    Returns:
        dict[str, int]: Status dictionary containing result counts
    """

    def _write_chunk_output(
        params: argparse.Namespace, row: dict, timeseries_records: list[dict]
    ) -> None:
        chunk_outdir = Path(params.out_dir) / f"x_part={row['x_part']}" / f"y_part={row['y_part']}"
        os.makedirs(chunk_outdir, exist_ok=True)
        pl.DataFrame(timeseries_records).lazy().sink_parquet(
            chunk_outdir / "dh_time_grid.parquet", compression="zstd"
        )

    def _write_grid_output(params: argparse.Namespace, grid_records: list[dict]) -> None:
        pl.DataFrame(grid_records).lazy().sink_parquet(
            Path(params.out_dir) / "grid_data.parquet", compression="zstd"
        )

    # -------------------#
    # Start Processing  #
    # -------------------#

    status = sf_objects["status"]
    min_secs = sf_objects["min_secs"]
    max_secs = sf_objects["max_secs"]
    pc_min_secs = sf_objects["pc_min_secs"]
    pc_max_secs = sf_objects["pc_max_secs"]
    grid_records = []
    i = 0
    for row in sf_objects["part_df"].iter_rows(named=True):
        timeseries_records = []
        i = i + 1
        logger.info(f"Processing chunk : {i} / {len(sf_objects['part_df'])}")

        # 1. Get grid data for chunk
        gridcell_lazy, chunk_status = get_grid_data(
            f"{params.in_dir}/*/x_part={row['x_part']}/y_part={row['y_part']}/*.parquet",
            params,
            min_secs,
            max_secs,
        )

        # Merge chunk status into overall status
        for key, value in chunk_status.items():
            status[key] += value

        gridcell_data = gridcell_lazy.collect()
        grouped = gridcell_data.group_by(["x_bin", "y_bin", "x_part", "y_part"])

        # 2. Execute surface fit process
        for (x_bin, y_bin, x_part, y_part), gridcell in grouped:
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
                    params.weight_name,  # Only loaded for is2
                ]
                if col in gridcell.columns
            }
            # --------------------------------------------------------------------------
            # Perform a Least Squares fit to a surface function
            # --------------------------------------------------------------------------
            surface_result = _fit_surface_model_per_group(
                params=params,
                group_np=gridcell_np,
                mintime=min_secs,
                maxtime=max_secs,
                logger=logger,
            )

            if isinstance(surface_result, str):
                status[surface_result] += 1
                continue
            res, group_np = surface_result
            # --------------------------------------------------------------------
            #  Apply a Power Correction if --apply_power_correction is set
            #  power correction can be performed using the whole time period, or
            #  a short period (ie 60 months) set by pcmintime,pcmaxtime
            # --------------------------------------------------------------------
            if params.powercorrect:

                power_result = _fit_power_correction_per_group(
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

            # ------------------------------------------------
            # Get modelled dh/dt
            # ------------------------------------------------
            linear_result = _fit_linear_fit_per_group(params, group_np)
            if isinstance(linear_result, str):
                status[linear_result] += 1
                continue

            if len(linear_result["mask"]) > 3:
                grid_record = {
                    "x_part": x_part,
                    "y_part": y_part,
                    "x_bin": x_bin,
                    "y_bin": y_bin,
                    "dhdt": linear_result["m"],
                    "slope": (180.0 / np.pi) * np.sqrt(res[1] ** 2 + res[2] ** 2),
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
                    if params.weight_name in linear_result["group_np"]:
                        ts_record[params.weight_name] = linear_result["group_np"][
                            params.weight_name
                        ][idx]
                    timeseries_records.append(ts_record)

                status["n_cells_fitted"] += 1
            else:
                status["n_too_few_values_in_linear_fit"] += 1

        # 3. Write Parquet Files per chunk
        _write_chunk_output(params, row, timeseries_records)

    _write_grid_output(params, grid_records)
    return status


def main(args: list[str] | None = None) -> None:
    """
    Main function to parse arguments and execute surface fit.
    1. Load command line arguments
    2. Load grid metadata from the grid_dir
    3. Set output directory for surface fit results
    4. Get the min/max time period for surface fit and power correction
    5. Get the minimum time span in a cell in seconds
    6. Get unique chunks (x_part, y_part) in the dataset
    7. Perform surface fit on each grid cell
    8. Write metadata json file

    Args:
        args (list[str] | None): Command line arguments. If None, uses sys.argv
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

    # ------------------------------#
    # 1.Load command line arguments#
    # ------------------------------#
    parser = argparse.ArgumentParser(
        description=(
            "Compute fitted elevation per grid cell from"
            " gridded altimetry data stored in parquet files."
        )
    )
    parser.add_argument(
        "--in_dir",
        help=("Path of the grid dir containing parquet files"),
        type=str,
        required=True,
    )
    parser.add_argument(
        "--parquet_glob",
        help="(optional) Glob pattern to match parquet files",
        type=str,
        default="**/*.parquet",
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
        "--min_vals_in_cell",
        default=20,
        type=int,
        help="(integer, default=20), minimum measurements in cell to perform plane fit",
    )
    parser.add_argument(
        "--min_percent_timespan_in_cell",
        default=70.0,
        type=float,
        help="(optional), minimum time-span in cell to perform plane fit \
            as percent of (maxtime-mintime), default is 70%",
    )
    parser.add_argument(
        "--n_sigma",
        default=2.0,
        type=float,
        help="(float, default=2.0). Sigma to filter out measurements from plane fit",
    )
    parser.add_argument(
        "--max_surfacefit_iterations",
        default=50,
        type=int,
        help="(integer, default=50) maximum number of iterations of plane modelfit",
    )
    parser.add_argument(
        "--max_linearfit_iterations",
        default=3,
        type=int,
        help="(integer, default=3) maximum number of iterations of linear fit",
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
        help="(optional) if set apply a power (backscatter) correction",
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
        "--weight_name",
        type=str,
        help="(optional) Name of the weight column to use for weighted surface fit",
    )
    start_time = time.time()

    # -----------------------------------------#
    # 2. Load grid metadata from the grid_dir
    # 3. Set output directory for surface fit results
    # -----------------------------------------#
    params = parser.parse_args(args)
    if params.weighted_surface_fit and params.weight_name is None:
        sys.exit("Weight name must be provided for weighted surface fit")
    parquet_glob = os.path.join(params.in_dir, params.parquet_glob)
    logger, grid_meta = get_set_up_objects(params, confirm_regrid=False)

    # -----------------------------------------#
    # 4. Perform surface fit
    # -----------------------------------------#
    sf_objects = get_surface_fit_objects(params, parquet_glob, grid_meta, logger)
    status = surface_fit(params, sf_objects=sf_objects, logger=logger)

    # --------------------#
    # 5. Write metadata json file #
    # --------------------#
    get_metadata_json(params, status, start_time, logger)


if __name__ == "__main__":
    main(sys.argv[1:])
