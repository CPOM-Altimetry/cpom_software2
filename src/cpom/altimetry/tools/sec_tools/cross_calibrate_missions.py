"""
cpom.altimetry.tools.sec_tools.cross_calibrate_missions

Purpose:
    Cross-calibrate multiple altimetry missions to remove systematic inter-mission biases.

    This tool aligns elevation change measurements from multiple satellite altimetry missions
    by estimating and removing relative biases between missions. It uses the first mission
    in the sorted mission list as a reference (bias = 0) and estimates biases for all other
    missions relative to this reference.

    Supports both ice sheet-wide processing and basin-level processing with independent
    calibration per basin.

Output Files:
    - Corrected grid data: <out_dir>/<basin>/corrected_epoch_average.parquet
    - Time series: <out_dir>/<basin>/cross_calibrated_timeseries.parquet
    - Outlier statistics: <out_dir>/<basin>/outlier_stats.json
    - Central metadata: <out_dir>/cross_calibration_metadata.json
    - Comparison plots: <out_dir>/<basin>/*_cross_calibration_comparison.png
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

from cpom.altimetry.tools.sec_tools.basin_selection_helper import (
    add_basin_selection_arguments,
    get_basins_to_process,
)
from cpom.logging_funcs.logging import set_loggers


def parse_arguments(args: list[str]) -> argparse.Namespace:
    """
    Parse command line arguments for cross-calibration.

    Args:
        args: List of command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Cross-calibrate multiple altimetry missions to remove inter-mission biases",
    )

    # I/O Arguments
    parser.add_argument(
        "--mission_mapper",
        type=json.loads,  # Parse JSON string to dict
        required=True,
        help="JSON string mapping mission identifiers to their data directories. "
        "Each directory should contain Parquet files with epoch-averaged elevation data.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Output directory for cross-calibrated results, statistics, and metadata. ",
    )

    # Mission Configuration
    parser.add_argument(
        "--missions_sorted",
        nargs="+",
        required=True,
        default=["e1", "e2", "env", "cs2"],
        help="Ordered list of mission identifiers to cross-calibrate. "
        "The first mission is used as the reference (bias = 0). "
        "All other missions will be adjusted relative to this reference. ",
    )

    # Quality Control Parameters
    parser.add_argument(
        "--bias_threshold",
        type=float,
        default=100.0,
        help="Maximum allowable absolute bias value in meters. "
        "Bias corrections exceeding this threshold are considered outliers, "
        "set to None, and filled via spatial interpolation. Default: 100.0m",
    )
    parser.add_argument(
        "--min_observations",
        type=int,
        default=5,
        help="Minimum total number of observations required per grid cell across all missions. "
        "Cells with fewer observations are excluded from cross-calibration. Default: 5",
    )

    # Column Name Arguments
    parser.add_argument(
        "--epoch_column",
        type=str,
        help="Name of the epoch number column in input Parquet files.",
        default="epoch_number",
    )
    parser.add_argument(
        "--time_column",
        type=str,
        help="Name of the epoch time column in input Parquet files. "
        "For datetime columns: converted to fractional years (year + day/365.25). "
        "For numeric columns: interpreted as years since 1991.0. ",
        default="epoch_midpoint_fractional_yr",
    )
    parser.add_argument(
        "--dh_column",
        type=str,
        help="Name of the elevation change (dh) column in input Parquet files.",
        default="dh_ave",
    )

    # Output Options
    parser.add_argument(
        "--plot_results",
        action="store_false",
        help="Boolean flag to enable/disable generation of comparison plots ",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable DEBUG level logging",
    )
    # Shared basin/region selection arguments
    add_basin_selection_arguments(parser)
    return parser.parse_args(args)


def get_basins_for_all_missions(
    params: argparse.Namespace, basins_to_process: list[str], logger: logging.Logger
) -> list[str]:
    """
    Filter basin list to include only basins with data available for all missions.

    Validates that each basin subdirectory exists and contains Parquet files for every
    mission specified in params.mission_mapper. Basins missing data for any mission
    are excluded.

    Args:
        params (argparse.Namespace): Command line arguments containing:
            - mission_mapper (dict): Maps mission identifiers to base directories
        basins_to_process (list[str]): List of basin names to validate
        logger (logging.Logger): Logger

    Returns:
        list[str]: Validated basin names present in all missions
    """
    valid_sub_basins = []
    for sub_basin in basins_to_process:
        all_missions_have_basin = True
        for mission, mission_dir in params.mission_mapper.items():
            path = Path(mission_dir) / sub_basin
            if not (path.exists() and list(path.glob("*.parquet"))):
                logger.warning("Basin %s not found for mission %s at %s", sub_basin, mission, path)
                all_missions_have_basin = False
                break
        if all_missions_have_basin:
            valid_sub_basins.append(sub_basin)
    logger.info("Processing valid sub-basins across all missions: %s", valid_sub_basins)
    return valid_sub_basins


def get_path(params: argparse.Namespace, mission: str, sub_basin: str = "None") -> Path:
    """
    Construct file path pattern for locating mission Parquet files.
    Path patterns:
        - With basin: <mission_root>/<sub_basin>/*.parquet
        - Root level: <mission_root>/*.parquet

    Args:
        params (argparse.Namespace): Command line arguments containing:
            - mission_mapper (dict): Maps mission identifiers to base directory paths
        mission (str): Mission identifier to look up in mission_mapper
        sub_basin (str, optional): Basin subdirectory name. If "None" or None,
            returns root-level pattern. Defaults to "None".

    Returns:
        Path: Glob pattern (*.parquet) for matching mission data files
    """
    if sub_basin in ["None", ["None"], None]:
        return Path(params.mission_mapper[mission]) / "*.parquet"
    return Path(params.mission_mapper[mission]) / sub_basin / "*.parquet"


def get_grid_cell_dataframe(params: argparse.Namespace, sub_basin: str = "None") -> pl.LazyFrame:
    """
    Load and concatenate epoch-averaged elevation data from all missions into a single LazyFrame.

    Scans Parquet files for each mission specified in params.missions_sorted, selects required
    columns (grid coordinates, epoch info, elevation, time), adds a mission identifier column,
    and concatenates all missions into a unified dataset for cross-calibration.

    Args:
        params (argparse.Namespace): Parsed command line arguments containing:
            - mission_mapper (dict): Maps mission identifiers to data directories
            - missions_sorted (list[str]): Ordered list of missions to process
            - epoch_column (str): Name of epoch number column
            - time_column (str): Name of epoch time column (fractional year)
            - dh_column (str): Name of elevation change column
        sub_basin (str, optional): Basin subdirectory name. If 'None',
            processes root-level data. Defaults to 'None'.

    Returns:
        pl.LazyFrame: Concatenated lazy dataframe with columns:
            x_bin, y_bin, epoch_column, time_column, dh_column, mission
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


def get_time_absolute_years(
    data: pl.LazyFrame, time_column: str, base_year: float = 1991.0
) -> pl.LazyFrame:
    """
    Convert epoch time column to absolute fractional years.

    Handles two time formats:
        - Datetime: Converts to fractional year using year + (ordinal_day - 1) / 365.25
        - Numeric: Assumes values are years relative to base_year

    Args:
        data (pl.LazyFrame): Input dataset containing time column
        time_column (str): Name of the time column to convert
        base_year (float, optional): Reference year for numeric time values.
            Defaults to 1991.0.

    Returns:
        pl.LazyFrame: Input data with added 'time_absolute_years' column containing
            absolute fractional years
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
    Filter grid cells to retain only those suitable for cross-calibration.

    Applies quality control filters to ensure each grid cell has:
        1. Time values converted to absolute fractional years
        2. Non-null elevation change values
        3. At least params.min_observations total observations
        4. Data from all missions in params.missions_sorted

    Cells failing any criterion are excluded from cross-calibration.

    Args:
        data (pl.LazyFrame): Input elevation data with columns:
            - x_bin, y_bin: Grid cell coordinates
            - mission: Mission identifier
            - params.dh_column: Elevation change values
            - params.epoch_column: Epoch numbers
            - params.time_column: Epoch time values
        params (argparse.Namespace): Command line parameters containing:
            - min_observations (int): Minimum total observations per cell
            - missions_sorted (list[str]): Required missions for each cell
            - dh_column (str): Name of elevation change column
            - epoch_column (str): Name of epoch column
            - time_column (str): Name of time column

    Returns:
        pl.LazyFrame: Filtered data containing only valid grid cells with grouped
            observations per cell
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
    missions: np.ndarray, dh: np.ndarray, t: np.ndarray, missions_sorted: list[str], degree: int = 3
) -> tuple[np.ndarray | None, dict | None, LinearRegression | None]:
    """
    Fit cross-calibration regression model to estimate temporal trend
    and inter-mission biases.

    Args:
        missions (np.ndarray): Mission identifier for each observation
        dh (np.ndarray): Elevation change values in meters
        t (np.ndarray): Epoch times in absolute fractional years
        missions_sorted (list[str]): Ordered mission identifiers. First mission is reference.
        degree (int, optional): Degree of polynomial temporal trend. Defaults to 3

    Returns:
        tuple: Three-element tuple containing:
            - fitted_curve (np.ndarray | None): Modeled temporal trend (no bias terms).
                Returns None if regression fails.
            - biases (dict | None): Mission bias estimates in meters.
                Reference mission has bias = 0. Missions not in data have bias = 0.
                Returns None if regression fails or <2 missions present.
            - model (LinearRegression | None): Fitted sklearn model object.
                Returns None if regression fails.
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
    Estimate cross-calibration biases for a single grid cell.

    Steps:
        1. Fit temporal trend to estimate per-mission bias corrections
            (fit_cross_calibrated_model_gridcell)
        2. Calculate regression standard error from model residuals

    Failed fit's are set to None.

    Args:
        data (pl.DataFrame): Single grid cell data with columns:
            - x_bin, y_bin: Grid cell coordinates
            - missions: List of mission identifiers for each observation
            - params.dh_column: List of elevation change values
            - time_absolute_years: List of absolute fractional years
        params (argparse.Namespace): Command line parameters containing:
            - missions_sorted (list[str]): Mission identifiers in processing order
            - dh_column (str): Name of elevation change column

    Returns:
        pl.DataFrame: Single-row DataFrame with columns:
            - x_bin, y_bin: Grid cell coordinates
            - success (bool): True if regression succeeded, False otherwise
            - {mission}_bias (float | None): Bias correction for each mission in meters
            - stderr (float | None): Regression model standard error in meters
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
    Apply quality control to filter unrealistic bias values exceeding threshold.
    Remove bias estimates that exceed params.bias_threshold. Outlier biases are set to None.

    Args:
        bias_tbl (pl.LazyFrame): Bias correction table with columns:
            - x_bin, y_bin: Grid cell coordinates
            - {mission}_bias: Bias value for each mission in meters
        params (argparse.Namespace): Command line parameters containing:
            - bias_threshold (float): Maximum absolute bias value in meters
            - missions_sorted (list[str]): Mission identifiers

    Returns:
        tuple: Two-element tuple containing:
            - cleaned_bias_tbl (pl.DataFrame): Bias table with outliers set to None
            - bias_tbl_stats (dict): Per-mission statistics with keys:
                - n_outlier_bias: Count of bias values set to None
                - n_total_bias: Count of valid (non-None) bias values
                - cleaned_min_bias: Minimum valid bias value
                - cleaned_max_bias: Maximum valid bias value
    """

    # Set outlier biases to None
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
    Fill missing bias values using k-nearest neighbour spatial interpolation.

    Where:
        1. Regression failed due to insufficient data (bias = None)
        2. Bias exceeded threshold and was flagged as outlier (bias set to None)

    Args:
        epoch_data (pl.LazyFrame): Epoch averaged data.
        grid_cell_data (pl.DataFrame): Bias correction dataframe with columns:
            - y_bin, x_bin: Grid cell coordinates
            - {mission}_bias: Bias values (None where missing)
            - stderr: Standard error values (None where missing)
        missions_sorted (list[str]): Mission identifiers for bias column names

    Returns:
        pl.DataFrame: Complete bias grid covering all cells in epoch_data with columns:
            - y_bin, x_bin: Grid cell coordinates
            - {mission}_bias: Interpolated bias for each mission
            - stderr: Interpolated standard error
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
            - missions_sorted (list[str]): Mission identifiers to process
            - mission_mapper (dict): Maps missions to data directories
            - dh_column (str): Original elevation column name
            - time_column (str): Time column name
        filled_df (pl.DataFrame): Complete bias correction grid with columns:
            - y_bin, x_bin: Grid cell coordinates
            - {mission}_bias: Bias correction for each mission
            - stderr: Cross-calibration uncertainty estimate
        sub_basin (str, optional): Basin subdirectory name. If "None",
            processes root-level data. Defaults to "None".

    Returns:
        tuple: Two-element tuple containing:
            - final_lf (pl.LazyFrame): Bias-corrected grid cell data with columns:
                x_bin, y_bin, mission, biased_dh, biased_dh_xcal_stderr, time_absolute_years
            - timeseries_lf (pl.LazyFrame): Basin-averaged time series with columns:
                mission, time, mean_dh, std_dh, n_cells, error_dh
    """

    all_corrected_data = []
    filled_lf = filled_df.lazy()

    for mission in params.missions_sorted:
        path = get_path(params, mission, sub_basin)

        # Convert time to absolute years
        df_mission_lf = get_time_absolute_years(pl.scan_parquet(path), params.time_column)

        df_mission_lf = (
            df_mission_lf.join(filled_lf, on=["y_bin", "x_bin"], how="left")
            .with_columns(
                [
                    # Apply bias correction
                    (pl.col("dh_ave") - pl.col(f"{mission}_bias")).alias("biased_dh"),
                    pl.col("stderr").alias("biased_dh_xcal_stderr"),
                    pl.lit(mission).alias("mission"),
                ]
            )
            .filter(  # Calculating xcal_ok
                pl.any_horizontal(
                    [pl.col(f"{m}_bias").is_not_null() for m in params.missions_sorted]
                )  # Check we have valid bias correction for at least one mission
                & pl.col("dh_ave").is_not_null()
            )
        )
        all_corrected_data.append(df_mission_lf)

    # Combine all missions into single LazyFrame
    final_lf = pl.concat(all_corrected_data)
    timeseries_df = get_dh_timeseries(
        final_lf, dh_column="biased_dh", epoch_time_column="time_absolute_years"
    )
    return final_lf, timeseries_df


def get_dh_timeseries(df: pl.LazyFrame, dh_column: str, epoch_time_column: str) -> pl.LazyFrame:
    """
    Aggregate elevation change data into mission-specific time series.

    Groups data by mission and epoch time, then computes spatial statistics across all
    grid cells for each time step. Standard error is calculated as std / sqrt(n_cells).

    Args:
        df (pl.LazyFrame): Elevation change data with columns:
            - mission: Mission identifier
            - epoch_time_column: Absolute fractional year
            - dh_column: Elevation change values
        dh_column (str): Name of the elevation change column to aggregate
        epoch_time_column (str): Name of the time column for grouping

    Returns:
        pl.LazyFrame: Time series with columns:
            - mission: Mission identifier
            - epoch_time_column: Time value (also aliased as 'time')
            - mean_dh: Mean elevation change across grid cells
            - std_dh: Standard deviation of elevation change
            - n_cells: Number of grid cells with data
            - error_dh: Standard error (std / sqrt(n_cells))
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


def write_outlier_stats(
    outlier_stats: dict,
    output_dir: Path,
    basin_name: str,
    logger: logging.Logger,
) -> None:
    """
    Write outlier statistics for a specific basin to its output directory.

    Args:
        outlier_stats (dict): Outlier statistics from bias correction step.
        output_dir (Path): Basin-specific output directory.
        basin_name (str): Name of the basin being processed.
        logger (logging.Logger): Logger
    """
    outlier_file = output_dir / "outlier_stats.json"
    with open(outlier_file, "w", encoding="utf-8") as f:
        json.dump(outlier_stats, f, indent=2)
    logger.info("Wrote outlier statistics for %s to %s", basin_name, outlier_file)


def write_central_metadata(
    params: argparse.Namespace,
    start_time: float,
    processed_basins: list[str],
    logger: logging.Logger,
) -> None:
    """
    Write central metadata file with processing parameters, timing, and basin list.

    Args:
        params (argparse.Namespace): All command line parameters including:
            mission_mapper, missions_sorted, bias_threshold, min_observations, etc.
        start_time (float): Processing start timestamp from time.time()
        processed_basins (list[str]): Basin names successfully processed
        logger (logging.Logger): Logger instance

    Output:
        Writes JSON file: <params.out_dir>/cross_calibration_metadata.json
        Contains: all params + processed_basins + total_basins_processed + execution_time
    """
    hours, remainder = divmod(int(time.time() - start_time), 3600)
    minutes, seconds = divmod(remainder, 60)

    metadata_file = Path(params.out_dir) / "cross_calibration_metadata.json"
    with open(metadata_file, "w", encoding="utf-8") as f_meta:
        json.dump(
            {
                **vars(params),
                "processed_basins": processed_basins,
                "total_basins_processed": len(processed_basins),
                "execution_time": f"{hours:02}:{minutes:02}:{seconds:02}",
            },
            f_meta,
            indent=2,
        )
    logger.info("Wrote central metadata to %s", metadata_file)


def plot(
    params: argparse.Namespace,
    input_lf: pl.LazyFrame,
    bias_df: pl.DataFrame,
    output_dir: Path,
    suffix: str,
) -> None:
    """
    Generate two-panel comparison plot of original vs cross-calibrated time series.

    Args:
        params (argparse.Namespace): Command line parameters containing:
            - missions_sorted (list[str]): Mission identifiers
            - dh_column (str): Original elevation column name
            - time_column (str): Time column name
        input_lf (pl.LazyFrame): Original uncorrected elevation data
        bias_df (pl.DataFrame): Bias correction grid (used to filter cells)
        output_dir (Path): Directory to save plot file
        suffix (str): Filename prefix (typically basin name)

    Output:
        Saves plot as: <output_dir>/<suffix>_cross_calibration_comparison.png
    """

    plt.figure(figsize=(12, 8))

    # Filter original data to data with valid bias corrections
    input_with_bias = input_lf.join(
        bias_df.lazy().select(["x_bin", "y_bin"]), on=["x_bin", "y_bin"], how="inner"
    )

    # Compute original basin averages
    input_with_abs_time = get_time_absolute_years(input_with_bias, params.time_column)
    original_time_series = get_dh_timeseries(
        input_with_abs_time, dh_column=params.dh_column, epoch_time_column="time_absolute_years"
    )

    # Reload timeseries for plotting
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

    plt.tight_layout()
    plt.savefig(output_dir / f"{suffix}_cross_calibration_comparison.png")
    plt.close()


def process_basin(
    basin_path: str,
    params: argparse.Namespace,
    logger: logging.Logger,
) -> tuple[Path | None, dict | None]:
    """
    Execute complete cross-calibration workflow for a single basin.

    Processing steps:
        1. Load grid cell data from all missions
        2. Filter to cells meeting quality requirements (valid data from all missions)
        3. Fit regression models to estimate mission-specific biases per grid cell
        4. Apply threshold filtering to remove unrealistic bias outliers
        5. Spatially interpolate bias values for cells with missing/outlier biases
        6. Apply bias corrections to original elevation data
        7. Generate basin-averaged time series
        8. Write results to Parquet files
        9. Optionally generate comparison plots

    Args:
        basin_path (str): Basin subdirectory name, or "None" for root-level processing
        params (argparse.Namespace): Command line parameters
        logger (logging.Logger): Logger

    Returns:
        tuple: Two-element tuple:
            - output_dir (Path | None): Basin output directory path if successful, None if failed
            - outlier_stats (dict | None): Bias outlier statistics if successful, None if failed

    Output Files (when successful):
        - <output_dir>/<basin>_corrected_epoch_average.parquet: Bias-corrected grid data
        - <output_dir>/<basin>_cross_calibrated_timeseries.parquet: Aggregated time series
        - <output_dir>/<basin>_cross_calibration_comparison.png: Comparison plot (if enabled)

    Returns (None, None) if:
        - No data found for basin
        - No cells meet quality requirements
    """
    logger.info("Processing: %s", basin_path)
    input_lf = get_grid_cell_dataframe(params, sub_basin=basin_path)

    if input_lf.select(pl.len()).collect().item() == 0:
        logger.info("No data found for: %s", basin_path)
        return None, None

    valid_cells = get_valid_cells(input_lf, params)
    if valid_cells.select(pl.len()).collect().item() == 0:
        logger.info("No valid cells found for: %s", basin_path)
        return None, None

    bias_tbl_lf = valid_cells.group_by(["x_bin", "y_bin"]).map_groups(
        lambda df: fit_cell_biases(df, params),
        schema={
            "x_bin": pl.Float64,
            "y_bin": pl.Float64,
            "success": pl.Boolean,
            "stderr": pl.Float64,
            **{f"{m}_bias": pl.Float64 for m in params.missions_sorted},
        },
    )

    bias_tbl, outlier_stats = apply_bias_thresholds(bias_tbl_lf, params)
    filled_df = interpolate_biases_spatially(input_lf, bias_tbl, params.missions_sorted)
    cross_calibrated_lf, timeseries_lf = get_output_dataframe(
        params, filled_df, sub_basin=basin_path
    )

    if basin_path == "None":
        output_dir = Path(params.out_dir)
        suffix = ""
    else:
        output_dir = Path(params.out_dir) / basin_path
        suffix = f"{Path(basin_path).name}_"
    os.makedirs(output_dir, exist_ok=True)

    # Write Parquet files
    cross_calibrated_lf.sink_parquet(output_dir / f"{suffix}corrected_epoch_average.parquet")
    timeseries_lf.sink_parquet(output_dir / f"{suffix}cross_calibrated_timeseries.parquet")

    if params.plot_results:
        plot(params, input_lf, bias_tbl, output_dir, suffix)

    return output_dir, outlier_stats


def main(args: list[str]) -> None:
    """
    Main entry point for multi-mission cross-calibration processing.

    Orchestrates the complete cross-calibration workflow including argument parsing,
    logging setup, basin discovery, individual basin processing, and metadata generation.

    Processing Modes:
        - Ice sheet-wide (--basin_structure False):
            Processes all data in mission root directories as a single unit

        - Basin-structured (--basin_structure True):
            Processes each basin subdirectory independently, with separate bias
            estimates per basin. Only processes basins present in ALL missions.

    Workflow:
        1. Parse and validate command line arguments
        2. Create output directory and configure logging
        3. Determine processing mode and discover basins
        4. For each basin/root:
            a. Load multi-mission grid cell data
            b. Filter cells
            c. Estimate per-cell mission biases
            d. Apply quality control and spatial interpolation
            e. Generate bias-corrected outputs
            f. Write results and statistics
        5. Write central metadata file

    Args:
        args: Command line arguments list
    """

    start_time = time.time()
    params = parse_arguments(args)
    os.makedirs(params.out_dir, exist_ok=True)
    logger = set_loggers(
        log_dir=params.out_dir,
        default_log_level=logging.DEBUG if params.debug else logging.INFO,
    )

    processed_basins = []

    if params.basin_structure is False:
        # Process root-level data without subdirectories
        logger.info("Processing root-level data")
        output_dir, outlier_stats = process_basin("None", params, logger)

        if output_dir is not None and outlier_stats is not None:
            write_outlier_stats(outlier_stats, output_dir, "root", logger)
            processed_basins.append("root")
    else:
        # Get a the list of basins in the first missions directory
        basins_to_process = get_basins_to_process(
            params, Path(params.mission_mapper[list(params.mission_mapper.keys())[0]]), logger
        )
        # Filter sub-basins to those present in all missions
        valid_sub_basins = get_basins_for_all_missions(params, basins_to_process, logger)

        # Process each basin/region/subregion
        for basin_path in valid_sub_basins:
            output_dir, outlier_stats = process_basin(basin_path, params, logger)

            if output_dir is not None and outlier_stats is not None:
                write_outlier_stats(outlier_stats, output_dir, basin_path, logger)
                processed_basins.append(basin_path)

    # Write central metadata file after all processing
    write_central_metadata(params, start_time, processed_basins, logger)


if __name__ == "__main__":
    main(sys.argv[1:])
