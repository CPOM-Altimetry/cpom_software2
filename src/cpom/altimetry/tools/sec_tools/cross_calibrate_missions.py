"""
cpom.altimetry.tools.sec_tools.surface_fit.py

Purpose:
  Perform planefit to get dh/dt time series from
  gridded elevation and time data.
  Output dh time series as parquet files.
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.spatial import KDTree
from sklearn.linear_model import LinearRegression

from cpom.logging_funcs.logging import set_loggers

log = logging.getLogger(__name__)


def get_sub_basins(params, logger) -> list[str]:
    """
    Get available sub-basins by scanning the directory structure.

        - If params.sub_basins is None or ["None"], returns ["None"] and logs
          that root-level data will be processed.
        - If params.sub_basins is "all" or ["all"], returns a sorted list of
          sub-basins that exist across all missions and contain Parquet files.
                input_directory/
                    sub_basin_1/
                        *.parquet
                    sub_basin_2/
                        *.parquet
        - Otherwise, returns the list provided in params.sub_basins as-is.

    Args:
        params (argparse.Namespace): Command line arguments.
            Includes : mission_mapper (dict), sub_basins (list or str)
        logger (logging.Logger): Logger

    Returns:
        list: Sorted list of sub-basin directory names to process.
    """
    # Determine which sub-basins to proces
    if params.sub_basins is None or params.sub_basins == ["None"]:
        sub_basins_to_process = ["None"]
        logger.info("Processing root level data")
    elif params.sub_basins in ("all", ["all"]):

        all_sub_basins = set()

        # Get subdirectories from the first mission directory
        first_mission = list(params.mission_mapper.keys())[0]
        first_dir = Path(params.mission_mapper[first_mission])

        if first_dir.exists():
            # Look for subdirectories that contain parquet files
            for subdir in first_dir.iterdir():
                if subdir.is_dir():
                    parquet_files = list(subdir.glob("*.parquet"))
                    if parquet_files:
                        all_sub_basins.add(subdir.name)
        # Verify sub-basins exist across all missions
        valid_sub_basins = []
        for sub_basin in all_sub_basins:
            exists_in_all = True
            for _, mission_dir in params.mission_mapper.items():
                sub_basin_path = Path(mission_dir) / sub_basin
                if not sub_basin_path.exists() or not list(sub_basin_path.glob("*.parquet")):
                    exists_in_all = False
                    break

            if exists_in_all:
                valid_sub_basins.append(sub_basin)
        sub_basins_to_process = sorted(valid_sub_basins)

        logger.info("Processing sub-basins: %s", sub_basins_to_process)
    else:
        sub_basins_to_process = params.sub_basins
        logger.info("Processing sub-basins: %s", sub_basins_to_process)

    return sub_basins_to_process


def get_path(params, mission: str, sub_basin: str = "None") -> Path:
    """
    Get file path pointing to Parquet files for a specific mission
    and (optionally) a specific sub-basin.

    Uses params.mission_mapper, mapping mission identifiers
    (e.g. "e1", "env", "cs2") to their corresponding base directory paths.

    If a sub-basin : <mission_root>/<sub_basin>/*.parquet
    If sub_basin is "None" or ["None"]:  <mission_root>/*.parquet

    Args:
        params (argparse.Namespace): Command line arguments.
                                    Includes : mission_mapper (dict)
        mission (str): Mission identifier.
        sub_basin (str): Name of a sub-basin directory. If "None" or ["None"], no sub-basin
            directory is appended. Defaults to "None".
    Returns:
        Path: Wildcard pattern to match all Parquet files for the selected mission
        (and optionally sub-basin).
    """
    if sub_basin in ["None", ["None"], None]:
        return Path(params.mission_mapper[mission]) / "*.parquet"
    return Path(params.mission_mapper[mission]) / sub_basin / "*.parquet"


def get_grid_cell_dataframe(params: argparse.Namespace, sub_basin: str = "None") -> pl.LazyFrame:
    """
    Load and concatenate epoch-averaged altimetry data from multiple missions.

    For each mission in params.mission_mapper, load the corresponding Parquet files, concatinate
    into a LazyFrame.

    Args:
        params (argparse.Namespace): Parsed command line arguments.
            Includes :
            - mission_mapper (dict): Maps mission names to base directories
            - missions_sorted (list): List of mission identifiers to process
            - epoch_column (str): Epoch number column name
            - time_column (str): Epoch time column. (Fractional midpoint year)
            - dh_column (str): Elevation change column name
        sub_basin (str, optional): Sub-basin directory name to process. If 'None',
                                 processes root-level data. Defaults to 'None'.

    Returns:
        pl.LazyFrame: Concatenated LazyFrame
    """
    lf = []
    for mission in params.missions_sorted:
        path = get_path(params, mission, sub_basin)
        lf.append(
            pl.scan_parquet(path).select(
                [
                    "x_bin",
                    "y_bin",
                    pl.col(params.epoch_column),
                    pl.col(params.time_column),
                    pl.col(params.dh_column),
                    pl.lit(mission).alias("mission"),
                ]
            )
        )

    return pl.concat(lf)


def get_time_absolute_years(data: pl.LazyFrame, time_column: str, base_year=1991.0) -> pl.LazyFrame:
    """
    Convert time column to absolute fractional years.

    If datetime: year + (ordinal_day - 1) / 365.25
    If not datetime, assume time is years since base_year (default 1991.0)

    Args:
        data (pl.LazyFrame): Input dataset containing time column.
        time_column (str): Name of the time column
    Returns:
        pl.LazyFrame: Data with additional column 'time_absolute_years'
    """
    time_col_dtype = data.collect_schema()[time_column]

    # If datetime, convert to absolute fractional year
    if time_col_dtype == pl.Datetime:
        time_expr = (
            pl.col(time_column).dt.year().cast(pl.Float64)
            + (pl.col(time_column).dt.ordinal_day() - 1).cast(pl.Float64) / 365.25
        ).alias("time_absolute_years")
    else:
        time_expr = (pl.col(time_column).cast(pl.Float64) + base_year).alias("time_absolute_years")

    return data.with_columns([time_expr])


def get_valid_cells(data: pl.LazyFrame, params: argparse.Namespace) -> pl.LazyFrame:
    """
    Filter out grid cells with insufficient data for cross-calibration.
        - Less than the minimum number of observations (params.min_observations)
    and convert time to absolute years.

    Args:
        data (pl.LazyFrame): Input mission data containing columns:
            - x_bin, y_bin: Grid cell coordinates
            - mission: Mission identifier
            - dh: Elevation change values
            - epoch_number, epoch_time: Temporal information
        params (argparse.Namespace): Command line parameters.
            Includes :
            - min_observations (int): Minimum total observations required per cell
            - missions_sorted (list): List of mission identifiers

    Returns:
        pl.LazyFrame: Filtered LazyFrame with valid cells
    """
    data = get_time_absolute_years(data, params.time_column)

    data = data.drop_nans(subset=[params.dh_column])

    valid_cells = (
        data.group_by(["x_bin", "y_bin"])
        .agg(
            [
                pl.len().alias("n_obs"),
                pl.col("mission").n_unique().alias("n_missions"),
                pl.col("mission").alias("missions"),
                pl.col(params.dh_column),
                pl.col(params.epoch_column),
                pl.col(params.time_column),
                pl.col("time_absolute_years"),
            ]
        )
        .filter(
            (pl.col("n_obs") >= params.min_observations)
            & (pl.col("n_missions") == len(params.missions_sorted))
        )
        .drop(["n_obs", "n_missions"])
    )
    return valid_cells


def fit_cross_calibrated_model_gridcell(
    missions: np.ndarray, dh: np.ndarray, t: np.ndarray, missions_sorted: list, degree: int = 3
) -> tuple:
    """
    Fit a cross-calibrated regression model for a single grid cell.
    Capture the time trend of elevation change (dh) and estimate a bias for each mission.

    The model includes :
        - A polynomial trend in time (of specified degree)
        - A per-mission bias term, the first mission in missions_sorted is used as the reference
        and given a bias of 0.

    Args:
        missions (np.ndarray): Array of mission values for each data point.
        dh (np.ndarray): Array of elevation change values for each data point.
        t (np.ndarray): Array of epoch times (fractional years) for each data point.
        missions_sorted (list): Ordered list of missions.
        degree (int, optional): Degree of polynomial temporal trend. Defaults to 3.

    Returns:
        tuple:
            - fitted_curve (np.ndarray or None): Fitted modelled time trend curve.
            - biases (dict or None): Bias for each mission.
                                Reference mission and missions not present in the data : bias = 0.
                                Fit fails : bias = None.

            - model (LinearRegression or None): Fitted sklearn LinearRegression model.
    """
    # Check that all arrays have the same length
    if len(missions) != len(dh) or len(missions) != len(t):
        return None, None, None

    missions_in_cell = list(np.unique(missions))
    if len(missions_in_cell) < 2:
        return None, None, None

    # Build a regression design matrix
    x = np.hstack([np.ones((len(t), 1)), np.vstack([t**i for i in range(1, degree + 1)]).T])

    for m in missions_sorted[1:]:
        if m in missions_in_cell:
            flag = (missions == m).astype(int).reshape(-1, 1)
            x = np.hstack([x, flag])
    try:
        model = LinearRegression().fit(x, dh)
        # Construct fitted curve using absolute years
        fitted_curve = model.intercept_
        for i in range(degree):
            fitted_curve += model.coef_[i + 1] * t ** (i + 1)

        # Extract biases
        biases = {}
        biases[missions_sorted[0]] = 0  # Reference mission always has bias = 0
        bias_idx = degree + 1
        for m in missions_sorted[1:]:
            if m in missions_in_cell:
                biases[m] = model.coef_[bias_idx]
                bias_idx += 1
            else:
                biases[m] = 0

        return fitted_curve, biases, model
    except Exception:  # pylint: disable=broad-except
        return None, None, None


def fit_cell_biases(data: pl.DataFrame, params: argparse.Namespace) -> pl.DataFrame:
    """
    Get cross-calibration bias for a single grid cell.

    Steps:
    1. Fit temporal trend to get mission biases 'get_cross_calibrated_model_gridcell()'
    2. Get standard error for cell

    Args:
        data (pl.DataFrame): Single grid cell DataFrame from group_by().map_groups().
            Contains columns: x_bin, y_bin, missions (list), dh_column (list),
            time_absolute_years (list)
        params (argparse.Namespace): Command line parameters.
            Includes :
            - missions_sorted (list): Mission identifiers in processing order
            - dh_column (str): Name of elevation change column

    Returns:
        pl.DataFrame: Single-row DataFrame with:
            - x_bin, y_bin: Grid cell coordinates
            - success: Boolean indicating successful model fitting
            - {mission}_bias: Bias correction for each mission
            - stderr: Standard error of regression model fit
    """
    # Extract arrays from grouped DataFrame
    missions_col = data.select(pl.col("missions")).to_series()[0]
    dh_col = data.select(pl.col(params.dh_column)).to_series()[0]
    time_col = data.select(pl.col("time_absolute_years")).to_series()[0]

    dh = np.array(dh_col)
    _, biases, model = fit_cross_calibrated_model_gridcell(
        np.array(missions_col),
        dh,
        np.array(time_col),
        missions_sorted=params.missions_sorted,
        degree=3,
    )

    # Initialize bias row with coordinates
    x_bin = data.select(pl.col("x_bin")).to_series()[0]
    y_bin = data.select(pl.col("y_bin")).to_series()[0]
    bias_row = {
        "x_bin": x_bin,
        "y_bin": y_bin,
        "success": biases is not None and model is not None,
    }

    # Set biases - None if regression failed, otherwise use computed values
    # Note: Reference mission always gets bias=0
    if biases is not None and model is not None:
        for mission in params.missions_sorted:
            bias_row[f"{mission}_bias"] = biases.get(mission, 0)

        # Compute standard error from residuals
        y_pred = model.predict(
            model.feature_names_in_
            if hasattr(model, "feature_names_in_")
            else np.arange(len(model.coef_)).reshape(1, -1)
        )
        bias_row["stderr"] = np.sqrt(np.mean((dh - y_pred) ** 2))
    else:
        # Regression failed - set all biases and stderr to None
        # These will be filled via spatial interpolation
        for mission in params.missions_sorted:
            bias_row[f"{mission}_bias"] = None
        bias_row["stderr"] = None

    return pl.DataFrame([bias_row])


def apply_bias_thresholds(
    bias_tbl: pl.LazyFrame, params: argparse.Namespace
) -> tuple[pl.DataFrame, dict]:
    """
    Filter out unrealistic bias values: values that > params.bias_threshold.
    Set outlier biases to None.

    Args:
        bias_tbl (pl.DataFrame): DataFrame containing bias corrections for each mission.
        params (argparse.Namespace): Command line parameters.
            Includes :
            - bias_threshold (float): Maximum absolute bias value (meters)
            - missions_sorted (list): List of mission identifiers

    Returns:
        tuple:
            - cleaned_bias_tbl (pl.DataFrame): Filtered bias table.
            - bias_tbl_stats (dict): Statistics about outlier removal.
    """

    # Set outlier biases to None - they'll be interpolated later
    cleaned = bias_tbl.with_columns(
        [
            pl.when(pl.col(f"{m}_bias").abs() > params.bias_threshold)
            .then(None)
            .otherwise(pl.col(f"{m}_bias"))
            .alias(f"{m}_bias")
            for m in params.missions_sorted
        ]
    )

    cleaned_bias_tbl = cleaned.collect()

    # Compute statistics on outliers removed
    bias_tbl_stats = {}
    for m in params.missions_sorted:
        bias_col = f"{m}_bias"
        n_null = cleaned_bias_tbl[bias_col].is_null().sum()
        n_valid = cleaned_bias_tbl[bias_col].is_not_null().sum()

        bias_tbl_stats[m] = {
            "n_outlier_bias": n_null,
            "n_total_bias": n_valid,
            "cleaned_min_bias": cleaned_bias_tbl[bias_col].min(),
            "cleaned_max_bias": cleaned_bias_tbl[bias_col].max(),
        }

    return cleaned_bias_tbl, bias_tbl_stats


def interpolate_biases_spatially(
    epoch_data: pl.LazyFrame, grid_cell_data: pl.DataFrame, missions_sorted: list[str]
) -> pl.DataFrame:
    """
    Spatially interpolate bias corrections using nearest neighbor method using a KDTree.

    Fills missing bias values in two scenarios:
        1. Grid cells where cross-calibration regression failed (bias = None)
        2. Grid cells where bias was an outlier and set to None by threshold filtering

    Args:
        epoch_data (pl.LazyFrame): Complete grid of epoch-averaged data (lazy).
                                   Used to get all grid cell coordinates.
        grid_cell_data (pl.DataFrame): Bias correction dataframe with columns:
            - y_bin, x_bin: Grid cell coordinates
            - {mission}_bias: Bias values for each mission (None where failed/outlier)
            - stderr: Standard error values (None where failed)
        missions_sorted (list): List of missions

    Returns:
        pl.DataFrame: Complete bias grid with all missing values filled by interpolation.
                     Has same grid coverage as epoch_data input.
    """

    all_points = epoch_data.select(["y_bin", "x_bin"]).unique().collect().to_numpy()
    df_data = {"y_bin": all_points[:, 0], "x_bin": all_points[:, 1]}

    # Interpolate each mission's bias column and stderr
    for col in [f"{mission}_bias" for mission in missions_sorted] + ["stderr"]:
        # Get grid cells with valid (non-null) values for this column
        # This excludes:
        # - Cells where regression failed (bias=None)
        # - Cells where bias exceeded threshold (bias set to None)
        valid_data = grid_cell_data.filter(pl.col(col).is_not_null()).select(
            ["y_bin", "x_bin", col]
        )

        if valid_data.height > 0:
            # Nearest neighbor interpolation
            valid_points = valid_data.select(["y_bin", "x_bin"]).to_numpy()
            valid_values = valid_data.select(col).to_numpy().flatten()

            tree = KDTree(valid_points)
            _, nearest_idx = tree.query(all_points)
            interpolated_values = valid_values[nearest_idx]
            df_data[col] = interpolated_values
        else:
            # No valid data for this mission
            df_data[col] = np.full(len(all_points), np.nan)

    # Create the filled bias DataFrame
    filled_bias_df = pl.DataFrame(df_data)
    return filled_bias_df


def get_output_dataframe(
    params: argparse.Namespace, filled_df: pl.DataFrame, sub_basin: str = "None"
) -> tuple[pl.LazyFrame, pl.LazyFrame]:
    """
    Create bias-corrected elevation-change data for all missions
    and produce a corrected time series.

    Apply mission-specific bias corrections to the original data,
    merges them with the bias grid, and outputs both the corrected per-cell dataset
    and an averaged time series.

    Args:
        params (argparse.Namespace): Command line parameters containing:
            - missions_sorted (list): Mission identifiers to process
            - mission_mapper (dict): Maps missions to data directories
        filled_df (pl.DataFrame): Complete bias correction grid from interpolation:
            - y_bin, x_bin: Grid cell coordinates
            - {mission}_bias: Bias correction for each mission
            - stderr: Uncertainty estimates
        sub_basin (str, optional): Sub-basin name for data loading.
                                 If None, processes root-level data. Defaults to "None".

    Returns:
        tuple: Two-element tuple containing:
            - final_df (pl.LazyFrame): Bias-corrected data for all missions and grid cells.
            - timeseries_df (pl.LazyFrame):  Bias-corrected epoch-averaged time series.

    """

    all_corrected_data = []

    # Convert filled_df to LazyFrame once for reuse
    filled_lf = filled_df.lazy()

    for mission in params.missions_sorted:
        path = get_path(params, mission, sub_basin)

        # Build lazy query for mission data - convert time to absolute years right after loading
        # Convert time to absolute years (to match old_cross_cal behavior)
        df_mission_lf = get_time_absolute_years(pl.scan_parquet(path), params.time_column)

        df_mission_lf = (
            df_mission_lf.join(filled_lf, on=["y_bin", "x_bin"], how="left")
            .with_columns(
                [
                    # Apply bias correction
                    (pl.col("dh_ave") - pl.col(f"{mission}_bias")).alias("biased_dh"),
                    pl.col("stderr").alias("biased_dh_xcal_stderr"),
                    # Add mission identifier
                    pl.lit(mission).alias("mission"),
                ]
            )
            .filter(  # Calculating xcal_ok
                pl.any_horizontal(
                    [pl.col(f"{m}_bias").is_not_null() for m in params.missions_sorted]
                )  # Check we have valid bias correction for at least one mission
                &
                # Has actual data
                pl.col("dh_ave").is_not_null()
            )
        )
        all_corrected_data.append(df_mission_lf)

    # Combine all missions into single LazyFrame
    final_lf = pl.concat(all_corrected_data)

    # Time is already in absolute years from the conversion done when loading each mission
    timeseries_df = get_dh_timeseries(
        final_lf, dh_column="biased_dh", epoch_time_column="time_absolute_years"
    )
    return final_lf, timeseries_df


def get_dh_timeseries(df: pl.LazyFrame, dh_column: str, epoch_time_column: str) -> pl.LazyFrame:
    """
    Get time series of elevation.
    Average dh values across grid cells for each mission and epoch.

    Args:
        df (pl.LazyFrame): Input LazyFrame containing elevation-change data.
        dh_column (str): Name of the elevation change column.
        epoch_time_column (str): Name of the epoch time column.
    Returns:
        pl.LazyFrame: LazyFrame with columns:
            - mission
            - epoch_midpoint_fractional_yr
            - mean_dh
            - std_dh
            - n_cells
            - error_dh
    """
    timeseries_df = (
        df.group_by(["mission", epoch_time_column])
        .agg(
            [
                pl.col(dh_column).mean().alias("mean_dh"),
                pl.col(dh_column).std().alias("std_dh"),
                pl.col(dh_column).count().alias("n_cells"),
                (pl.col(dh_column).std() / pl.col(dh_column).count().sqrt()).alias("error_dh"),
                # add xcal_ok
            ]
        )
        .with_columns(
            [
                pl.col(epoch_time_column).alias(
                    "time"
                )  # Add 'time' column to match old_cross_cal output
            ]
        )
        .sort(["mission", epoch_time_column])
    )

    return timeseries_df


def get_metadata_json(
    params: argparse.Namespace,
    outlier_stats: dict,
    output_dir: Path,
    start_time: float,
    logger: logging.Logger,
) -> None:
    """
    Create and save crosscal_metadata.json in the output directory.
    Metadata includes:
        - Command line parameters
        - Outlier statistics
        - Execution time
    Args:
        params (argparse.Namespace): Command line parameters.
        outlier_stats (dict): Outlier statistics from bias correction step.
        start_time (float): Processing start time
        logger (logging.Logger): Logger
    """
    hours, remainder = divmod(int(time.time() - start_time), 3600)
    minutes, seconds = divmod(remainder, 60)

    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f_meta:
        json.dump(
            {
                **vars(params),
                **outlier_stats,
                "execution_time": f"{hours:02}:{minutes:02}:{seconds:02}",
            },
            f_meta,
            indent=2,
        )
    logger.info("Wrote data_set metadata to %s", output_dir / "crosscal_metadata.json")


def plot(
    params: argparse.Namespace,
    input_lf: pl.LazyFrame,
    bias_df: pl.DataFrame,
    output_dir: Path,
    suffix: str,
) -> None:
    """
    Get plot comparing the original time series to the cross-calibrated time series.

    Args:
        params (argparse.Namespace): Command line parameters.
        input_lf (pl.LazyFrame): Original elevation change data.
        bias_df (pl.DataFrame): Table of grid cells with bias corrections.
        output_dir (Path): Directory to save plots.
        suffix (str): Suffix to add to the output plot filenames.
    """

    plt.figure(figsize=(12, 8))

    # Filter original data to only include cells where we have valid bias corrections
    input_with_bias = input_lf.join(
        bias_df.lazy().select(["x_bin", "y_bin"]), on=["x_bin", "y_bin"], how="inner"
    )

    # Compute original basin averages
    input_with_abs_time = get_time_absolute_years(input_with_bias, params.time_column)
    original_time_series = get_dh_timeseries(
        input_with_abs_time, dh_column=params.dh_column, epoch_time_column="time_absolute_years"
    )

    # Reload timeseries for plotting since sink_parquet consumes the LazyFrame
    timeseries_df = pl.scan_parquet(output_dir / f"{suffix}cross_calibrated_timeseries.parquet")

    # Plot original time series
    plt.subplot(2, 1, 1)
    plt.title("Original Time Series (No Cross-Calibration)")
    for m in params.missions_sorted:
        subset = original_time_series.filter(pl.col("mission") == m).collect()
        if subset.height > 0:  # Check if subset has data
            plt.errorbar(
                subset["time"],
                subset["mean_dh"],
                yerr=subset["error_dh"],
                label=f"{m} original",
                fmt="o",
            )
    plt.xlabel("Year")
    plt.ylabel("Δh (m)")
    plt.legend()
    plt.grid()
    ax = plt.gca()
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.25))

    # Plot cross-calibrated time series
    plt.subplot(2, 1, 2)
    plt.title("Cross-Calibrated Time Series")

    for m in params.missions_sorted:
        subset = timeseries_df.filter(pl.col("mission") == m).collect()
        if subset.height > 0:  # Check if subset has data
            plt.errorbar(
                subset["time"],
                subset["mean_dh"],
                yerr=subset["error_dh"],
                label=f"{m} xcal",
                fmt="o",
            )
    plt.xlabel("Year")
    plt.ylabel("Δh (m)")
    plt.legend()
    plt.grid()
    ax = plt.gca()
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.25))

    plt.tight_layout()
    plt.savefig(output_dir / f"{suffix}_cross_calibration_comparison.png")
    plt.close()


# pylint: disable=too-many-arguments
def write_results(
    params: argparse.Namespace,
    input_lf: pl.LazyFrame,
    cross_calibrated_lf: pl.LazyFrame,
    timeseries_lf: pl.LazyFrame,
    bias_df: pl.DataFrame,
    sub_basin: str = "None",
) -> Path:
    """
    Save cross-calibrated results to parquet files and generate plots.

    Writes:
        - Cross-calibrated grid cell data to :
          <out_dir>/<sub_basin>/<sub_basin>_corrected_epoch_average.parquet
        - Cross-calibrated time series to :
          <out_dir>/<sub_basin>/<sub_basin>_cross_calibrated_timeseries.parquet
        - Comparison plots if params.plot_results is True

    Args:
        params (argparse.Namespace): Command line parameters.
        input_lf (pl.LazyFrame): Original input LazyFrame before cross-calibration.
        cross_calibrated_lf (pl.LazyFrame): Cross-calibrated grid cell data.
        bias_df (pl.DataFrame): Bias corrections table.
        timeseries_lf (pl.LazyFrame): Cross-calibrated time series data.
        sub_basin (str, optional): Sub-basin name for output directory.
                                 If None, saves to root-level output. Defaults to "None".
    Returns:
        Path: The output directory where results are saved.
    """

    # Set output directory and suffix based on sub_basin
    if sub_basin == "None":
        output_dir = Path(params.out_dir)
        suffix = ""
    else:
        output_dir = Path(params.out_dir) / sub_basin
        suffix = f"{sub_basin}_"
    os.makedirs(output_dir, exist_ok=True)

    # Write Parquet files
    cross_calibrated_lf.sink_parquet(output_dir / f"{suffix}corrected_epoch_average.parquet")
    timeseries_lf.sink_parquet(output_dir / f"{suffix}cross_calibrated_timeseries.parquet")

    if params.plot_results:
        plot(params, input_lf, bias_df, output_dir, suffix)

    return output_dir


def main(args):
    """
    Main script for multi-mission cross-calibration.

    Loads and validates command line arguments, sets up logging,
    loads input data, estimate mission biases, applies corrections,
    saves results.

    Modes :
        - params.sub_basins = ['basin1', 'basin2'] : Process specified basins
        - params.sub_basins = ['all'] : Auto-discover and process all basins
        - params.sub_basins = ['None']: Process root-level data

    Steps :
        1. Parse and validate command line arguments.
        2. Get logger.
        3. Get sub-basins to process.
        4. For each sub-basin:
            a. Load grid cell data from all missions.
            b. Filter to valid cells.
            c. Perform cross-calibration to estimate mission biases.
            d. Interpolate biases spatially to fill missing values.
            e. Apply bias corrections to original data.
            f. Save corrected data and metadata.

    """

    parser = argparse.ArgumentParser(
        description="Cross-calibrate multiple altimetry missions across grid cells",
    )

    parser.add_argument(
        "--mission_mapper",
        type=json.loads,  # Parse JSON string to dict
        required=True,
        help="JSON string mapping mission names to parquet file directories",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        help="Output directory for results. If not provided, uses current directory.",
    )
    parser.add_argument(
        "--missions_sorted",
        nargs="+",
        required=True,
        default=["e1", "e2", "env", "cs2"],
        help="List of missions to cross-calibrate, in order.",
    )
    parser.add_argument(
        "--bias_threshold", type=float, default=100.0, help="Maximum allowable bias correction"
    )
    parser.add_argument(
        "--epoch_column",
        help="Name of the epoch column in the input data",
        default="epoch_number",
    )
    parser.add_argument(
        "--time_column",
        help="Name of the fractional time column in the input data "
        "(will be converted to absolute years)",
        default="epoch_midpoint_fractional_yr",
    )
    parser.add_argument(
        "--dh_column",
        help="Name of the dh column in the input data",
        default="dh_ave",
    )
    parser.add_argument(
        "--sub_basins",
        nargs="+",
        default=["all"],
        help="Specific list of sub-basins to process."
        "If not provided : Will calculate for entire directory. "
        "If set to 'all' will auto-discover sub-basins."
        "To process root-level data, set to [None].",
    )
    parser.add_argument(
        "--min_observations",
        type=int,
        default=5,
        help="Minimum number of observations required per grid cell",
    )
    parser.add_argument(
        "--plot_results",
        action="store_false",
        help="Generate comparison plots of original vs cross-calibrated time series",
    )

    start_time = time.time()
    params = parser.parse_args(args)
    os.makedirs(params.out_dir, exist_ok=True)
    logger = set_loggers(
        log_file_info=Path(params.out_dir) / "info.log",
        log_file_error=Path(params.out_dir) / "errors.log",
    )

    sub_basins_to_process = get_sub_basins(params, logger)

    # Process each sub-basin
    for sub_basin in sub_basins_to_process:
        logger.info("Processing sub-basin: %s", sub_basin)
        input_lf = get_grid_cell_dataframe(params, sub_basin=sub_basin)

        if input_lf.select(pl.len()).collect().item() == 0:
            logger.info("No data found for sub-basin: %s", sub_basin)
            continue

        valid_cells = get_valid_cells(input_lf, params)
        if valid_cells.select(pl.len()).collect().item() == 0:
            logger.info("No valid cells found for sub-basin: %s", sub_basin)
            continue

        bias_tbl = valid_cells.group_by(["x_bin", "y_bin"]).map_groups(
            lambda df: fit_cell_biases(df, params),
            schema={
                "x_bin": pl.Float64,
                "y_bin": pl.Float64,
                "success": pl.Boolean,
                "stderr": pl.Float64,
                **{f"{m}_bias": pl.Float64 for m in params.missions_sorted},
            },
        )

        bias_tbl, outlier_stats = apply_bias_thresholds(bias_tbl, params)
        filled_df = interpolate_biases_spatially(input_lf, bias_tbl, params.missions_sorted)
        cross_calibrated_lf, timeseries_lf = get_output_dataframe(
            params, filled_df, sub_basin=sub_basin
        )
        output_dir = write_results(
            params, input_lf, cross_calibrated_lf, timeseries_lf, filled_df, sub_basin
        )
        get_metadata_json(params, outlier_stats, output_dir, start_time, logger)


if __name__ == "__main__":
    main(sys.argv[1:])
