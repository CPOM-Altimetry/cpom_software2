"""
cpom.altimetry.tools.sec_tools.cross_calibrate_missions

Purpose:
Cross-calibrate multiple altimetry missions to remove systematic inter-mission biases.

Aligns elevation change measurements from multiple satellite altimetry missions by estimating
and removing relative biases. Uses the first mission in missions_sorted as reference (bias = 0).
Supports both ice sheet-wide and basin-level processing with independent calibration per basin.

Output Files:
    - Corrected grid data: <out_dir>/<basin>/corrected_epoch_average.parquet
    - Time series: <out_dir>/<basin>/cross_calibrated_timeseries.parquet
    - Outlier statistics: <out_dir>/<basin>/outlier_stats.json
    - Shared metadata: <out_dir>/cross_calibrate_missions_meta.json
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
from cpom.altimetry.tools.sec_tools.metadata_helper import (
    elapsed,
    get_algo_name,
    write_metadata,
)
from cpom.logging_funcs.logging import set_loggers


def parse_arguments(args: list[str]) -> argparse.Namespace:
    """Parse command line arguments for cross-calibration."""
    parser = argparse.ArgumentParser(
        description="Cross-calibrate multiple altimetry missions to remove inter-mission biases",
    )
    parser.add_argument(
        "--mission_mapper",
        type=json.loads,
        required=True,
        help="JSON string mapping mission identifiers to their data directories.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Output directory for cross-calibrated results, statistics, and metadata.",
    )
    parser.add_argument(
        "--missions_sorted",
        nargs="+",
        required=True,
        default=["e1", "e2", "env", "cs2"],
        help="Ordered list of missions. First is reference (bias=0).",
    )
    parser.add_argument(
        "--bias_threshold",
        type=float,
        default=100.0,
        help="Maximum allowable absolute bias (m)."
        "Outliers are spatially interpolated. Default: 100.0",
    )
    parser.add_argument(
        "--min_observations",
        type=int,
        default=5,
        help="Minimum observations per grid cell across all missions. Default: 5",
    )
    parser.add_argument("--epoch_column", type=str, default="epoch_number")
    parser.add_argument(
        "--time_column",
        type=str,
        default="epoch_midpoint_fractional_yr",
        help="Datetime columns are converted to fractional years"
        "numeric columns are treated as years since 1991.0.",
    )
    parser.add_argument("--dh_column", type=str, default="dh_ave")
    parser.add_argument(
        "--plot_results", action="store_false", help="Disable generation of comparison plots."
    )
    parser.add_argument("--debug", action="store_true")
    add_basin_selection_arguments(parser)
    return parser.parse_args(args)


def get_basins_for_all_missions(
    params: argparse.Namespace, basins_to_process: list[str], logger: logging.Logger
) -> list[str]:
    """
    Return basins that have Parquet data available for every mission in params.mission_mapper.

    Args:
        params (argparse.Namespace): Command line parameters (includes : mission_mapper).
        basins_to_process (list[str]): List of basin names to validate
        logger (logging.Logger): Logger Object.

    Returns:
        list[str]: Validated basin names present in all missions
    """
    valid_sub_basins = []
    for sub_basin in basins_to_process:
        if all(
            (Path(d) / sub_basin).exists() and list((Path(d) / sub_basin).glob("*.parquet"))
            for d in params.mission_mapper.values()
        ):
            valid_sub_basins.append(sub_basin)
        else:
            logger.warning("Basin %s missing for one or more missions — skipping", sub_basin)
    logger.info("Valid basins across all missions: %s", valid_sub_basins)
    return valid_sub_basins


def get_path(params: argparse.Namespace, mission: str, sub_basin: str = "None") -> Path:
    """
    Get file glob pattern for a given mission and basin. <mission_root>/<sub_basin>/*.parquet or
    <mission_root>/*.parquet.
    Args:
        params (argparse.Namespace): Command line Parameters.
        mission (str): Mission identifier
        sub_basin (str, optional): Basin subdirectory name or None. Defaults to "None".

    Returns:
        Path: Glob pattern (*.parquet) for matching mission data files
    """
    base = Path(params.mission_mapper[mission])
    return (
        base / "*.parquet"
        if sub_basin in ("None", ["None"], None)
        else base / sub_basin / "*.parquet"
    )


def get_grid_cell_dataframe(params: argparse.Namespace, sub_basin: str = "None") -> pl.LazyFrame:
    """
    Load and concatenate epoch-averaged elevation data from all missions.

    For each mission in params.missions_sorted, load LazyFrame, select columns, and concatinate
    into a single LazyFrame.

    Args:
        params (argparse.Namespace): Command line parameters (includes: mission_mapper,
        missions_sorted, epoch_column, time_column, dh_column)
        sub_basin (str, optional): Basin name or None for root-level processing. Defaults to "None".

    Returns:
        pl.LazyFrame: Concatenated lazy dataframe with columns:
            x_bin, y_bin, epoch_column, time_column, dh_column, mission
    """
    return pl.concat(
        [
            pl.scan_parquet(get_path(params, mission, sub_basin)).select(
                [
                    "x_bin",
                    "y_bin",
                    pl.col(params.epoch_column),
                    pl.col(params.time_column),
                    pl.col(params.dh_column),
                    pl.lit(mission).alias("mission"),
                ]
            )
            for mission in params.missions_sorted
        ]
    )


def get_time_absolute_years(
    data: pl.LazyFrame, time_column: str, base_year: float = 1991.0
) -> pl.LazyFrame:
    """
    Add 'time_absolute_years' column converting epoch times to absolute fractional years.

    Handles two time formats:
        - Datetime: Converts to fractional year using year + (ordinal_day - 1) / 365.25
        - Numeric: Assumes values are years relative to base_year

    Args:
        data (pl.LazyFrame): Input dataset containing time column
        time_column (str): Name of the time column to convert
        base_year (float, optional): Reference year for numeric time values.
            Defaults to 1991.0.

    Returns:
        pl.LazyFrame: Input data with added 'time_absolute_years' column.

    """
    # If datetime, convert to absolute fractional year
    if data.collect_schema()[time_column] == pl.Datetime:
        time_expr = (
            pl.col(time_column).dt.year().cast(pl.Float64)
            + (pl.col(time_column).dt.ordinal_day() - 1).cast(pl.Float64) / 365.25
        ).alias("time_absolute_years")
    else:
        time_expr = (pl.col(time_column).cast(pl.Float64) + base_year).alias("time_absolute_years")

    return data.with_columns([time_expr])


def get_valid_cells(data: pl.LazyFrame, params: argparse.Namespace) -> pl.LazyFrame:
    """
    Filter to grid cells with sufficient observations from all missions.

    Removes none-null elevation change values, converts time to absolute fractional years,
    filters to cells with at least params.min_observations total observations and
    data from all missions in params.missions_sorted.

    Args:
        data (pl.LazyFrame): Input elevation data.
        params (argparse.Namespace): Command line parameters (includes
        min_observations, missions_sorted, dh_column, epoch_column, time_column)

    Returns:
        pl.LazyFrame: Filtered data containing only valid grid cells with grouped
            observations per cell
    """
    data = get_time_absolute_years(data, params.time_column)
    data = data.filter(
        pl.col(params.dh_column).is_not_null() & pl.col(params.dh_column).is_not_nan()
    )

    return (
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


def fit_cross_calibrated_model_gridcell(
    missions: np.ndarray, dh: np.ndarray, t: np.ndarray, missions_sorted: list[str], degree: int = 3
) -> tuple[dict | None, LinearRegression | None, np.ndarray | None]:
    """
    Fit a regression model estimating a polynomial temporal trend and per-mission biases.
    Reference mission (first in missions_sorted) has bias = 0.

    Args:
        missions (np.ndarray): Mission identifier for each observation
        dh (np.ndarray): Elevation change values in meters
        t (np.ndarray): Epoch times in absolute fractional years
        missions_sorted (list[str]): Ordered mission identifiers. First mission is reference.
        degree (int, optional): Degree of polynomial temporal trend. Defaults to 3

    Returns:
        tuple:
            - biases (dict | None): Mission bias estimates in meters.
                Reference mission has bias = 0. Missions not in data have bias = 0.
                Returns None if regression fails or <2 missions present.
            - model (LinearRegression | None): Fitted sklearn model object or None.
            - X (np.ndarray | None): Design matrix used for regression or None
    """
    # Check that all arrays have the same length
    if len(missions) != len(dh) or len(missions) != len(t):
        return None, None, None

    missions_in_cell = list(np.unique(missions))
    if len(missions_in_cell) < 2:
        return None, None, None

    poly_cols = np.vstack([t**i for i in range(1, degree + 1)]).T
    x = np.hstack([np.ones((len(t), 1)), poly_cols])

    non_ref = [m for m in missions_sorted[1:] if m in missions_in_cell]
    for m in non_ref:
        x = np.hstack([x, (missions == m).astype(int).reshape(-1, 1)])

    try:
        model = LinearRegression().fit(x, dh)
        # fitted_curve = model.intercept_ +
        # sum(model.coef_[i] * t ** (i + 1) for i in range(degree))
        biases = {missions_sorted[0]: 0.0}
        bias_idx = degree + 1
        for m in missions_sorted[1:]:
            if m in missions_in_cell:
                biases[m] = model.coef_[bias_idx]
                bias_idx += 1
            else:
                biases[m] = 0

        return biases, model, x
    except Exception:  # pylint: disable=broad-except
        return None, None, None


def fit_cell_biases(data: pl.DataFrame, params: argparse.Namespace) -> pl.DataFrame:
    """
    Estimate cross-calibration biases for a single grid cell.

    Steps:
        1. Fit temporal trend to estimate per-mission bias corrections
            (fit_cross_calibrated_model_gridcell)
        2. Calculate regression standard error from model residuals

    Args:
        data (pl.DataFrame): Single grid cell rows.
        params (argparse.Namespace): Command line parameters
            (includes: missions_sorted, dh_column, time_column)

    Returns:
        pl.DataFrame: Single-row DataFrame (includes: x_bin, y_bin, success,
        {mission}_bias for each mission, stderr)
    """
    # Extract arrays from grouped DataFrame
    missions_arr = np.array(data.select(pl.col("missions")).to_series()[0])
    dh_arr = np.array(data.select(pl.col(params.dh_column)).to_series()[0])
    time_arr = np.array(data.select(pl.col("time_absolute_years")).to_series()[0])

    biases, model, x = fit_cross_calibrated_model_gridcell(
        missions_arr, dh_arr, time_arr, missions_sorted=params.missions_sorted
    )

    # Initialize bias row with coordinates
    bias_row = {
        "x_bin": data["x_bin"][0],
        "y_bin": data["y_bin"][0],
        "success": biases is not None and model is not None,
    }

    # Set biases - None if regression failed, otherwise use computed values
    # Note: Reference mission always gets bias=0
    if biases is not None and model is not None:
        for mission in params.missions_sorted:
            bias_row[f"{mission}_bias"] = biases.get(mission, 0)

        # Compute standard error from residuals
        y_pred = model.predict(x)
        bias_row["stderr"] = np.sqrt(np.mean((dh_arr - y_pred) ** 2))
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
    Remove bias estimates that exceed params.bias_threshold. Outlier biases are set to None.

    Args:
        bias_tbl (pl.LazyFrame): Bias correction table
        params (argparse.Namespace): Command line parameters.
    Returns:
        tuple:
            - cleaned_bias_tbl (pl.DataFrame): Bias table with outliers set to None
            - bias_tbl_stats (dict): Per-mission statistics (includes n_outlier_bias, n_total_bias,
            cleaned_min_bias, cleaned_max_bias):
    """
    bias_df = bias_tbl.collect()
    cleaned = bias_df.with_columns(
        [
            pl.when(pl.col(f"{m}_bias").abs() > params.bias_threshold)
            .then(None)
            .otherwise(pl.col(f"{m}_bias"))
            .alias(f"{m}_bias")
            for m in params.missions_sorted
        ]
    )

    stats = {
        m: {
            "n_outlier_bias": cleaned[f"{m}_bias"].is_null().sum(),
            "n_total_bias": cleaned[f"{m}_bias"].is_not_null().sum(),
            "cleaned_min_bias": cleaned[f"{m}_bias"].min(),
            "cleaned_max_bias": cleaned[f"{m}_bias"].max(),
        }
        for m in params.missions_sorted
    }
    return cleaned, stats


def interpolate_biases_spatially(
    epoch_data: pl.LazyFrame, grid_cell_data: pl.DataFrame, missions_sorted: list[str]
) -> pl.DataFrame:
    """
    Fill missing bias values using k-nearest neighbour spatial interpolation.

    Where regression failed due to insufficient data or the
    bias exceeded threshold and was flagged as outlier.

    Args:
        epoch_data (pl.LazyFrame): Epoch averaged data.
        grid_cell_data (pl.DataFrame): Bias correction dataframe.
        missions_sorted (list[str]): Mission identifiers for bias column names

    Returns:
        pl.DataFrame: Complete bias grid covering all cells in epoch_data. (Includes:
        y_bin, x_bin, {mission}_bias for each mission, stderr.
    """

    all_points = epoch_data.select(["y_bin", "x_bin"]).unique().collect().to_numpy()
    df_data: dict = {"y_bin": all_points[:, 0], "x_bin": all_points[:, 1]}

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
            _, nearest_idx = KDTree(valid_data.select(["y_bin", "x_bin"]).to_numpy()).query(
                all_points
            )
            df_data[col] = valid_data.select(col).to_numpy().flatten()[nearest_idx]
        else:
            # No valid data for this mission
            df_data[col] = np.full(len(all_points), np.nan)

    # Create the filled bias DataFrame
    return pl.DataFrame(df_data)


def get_output_dataframe(
    params: argparse.Namespace, filled_df: pl.DataFrame, sub_basin: str = "None"
) -> tuple[pl.LazyFrame, pl.LazyFrame]:
    """
    Apply bias corrections to all missions and return corrected data + averaged time series.

    Args:
        params (argparse.Namespace): Command line parameters
        filled_df (pl.DataFrame): Bias grid with columns:
            y_bin, x_bin, {mission}_bias for each mission, stderr
        sub_basin (str, optional): Basin name or None for root-level processing. Defaults to "None".

    Returns:
        tuple:
            - final_lf (pl.LazyFrame): Bias-corrected grid cell data
                (Includes: x_bin, y_bin, mission, biased_dh, biased_dh_xcal_stderr,
                time_absolute_years)
            - timeseries_lf (pl.LazyFrame): Basin-averaged time series.
                (Includes: mission, time, mean_dh, std_dh, n_cells, error_dh)
    """

    all_corrected_data = []
    filled_lf = filled_df.lazy()

    for mission in params.missions_sorted:
        # Convert time to absolute years
        df_mission_lf = get_time_absolute_years(
            pl.scan_parquet(get_path(params, mission, sub_basin)), params.time_column
        )

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
    Aggregate elevation change into mission-specific time series with spatial statistics.

    Args:
        df (pl.LazyFrame): Elevation change data.
        dh_column (str): Name of elevation column.
        epoch_time_column (str): Name of the time column to group by.
    Returns:
        pl.LazyFrame: Time series of elevation change. (Includes: mission, time, mean_dh,
        std_dh, n_cells, error_dh)

    """
    return (
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
        params (argparse.Namespace): All command line parameters.
        start_time (float): Processing start timestamp.
        processed_basins (list[str]): Basin names successfully processed
        logger (logging.Logger): Logger Object

    Output:
        Writes <params.out_dir>/cross_calibrate_missions_meta.json
    """
    metadata = {
        **vars(params),
        "processed_basins": processed_basins,
        "total_basins_processed": len(processed_basins),
        "execution_time": elapsed(start_time),
        # Output field definitions for downstream consumers (e.g. calculate_dhdt).
        "output_dh_column": "biased_dh",
        "output_xcal_stderr_column": "biased_dh_xcal_stderr",
        "output_time_fractional_column": "time_absolute_years",
    }

    write_metadata(
        params=params,
        algo_name=get_algo_name(__file__),
        out_meta_path=Path(params.out_dir),
        metadata=metadata,
        logger=logger,
    )
    logger.info(
        "Wrote shared metadata to %s",
        Path(params.out_dir) / f"{get_algo_name(__file__)}_meta.json",
    )


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
        params (argparse.Namespace): Command line parameters.
        input_lf (pl.LazyFrame): Original uncorrected elevation data
        bias_df (pl.DataFrame): Bias correction grid.
        output_dir (Path): Directory to save plot file
        suffix (str): Filename suffix for plot (e.g. basin name or "ice_sheet_wide")

    Output:
        Saves plot as: <output_dir>/<suffix>_cross_calibration_comparison.png
    """

    plt.figure(figsize=(12, 8))

    input_with_bias = input_lf.join(
        bias_df.lazy().select(["x_bin", "y_bin"]), on=["x_bin", "y_bin"], how="inner"
    )
    original_ts = get_dh_timeseries(
        get_time_absolute_years(input_with_bias, params.time_column),
        dh_column=params.dh_column,
        epoch_time_column="time_absolute_years",
    )
    calibrated_ts = pl.scan_parquet(output_dir / f"{suffix}cross_calibrated_timeseries.parquet")

    for i, (ts, title, label_suffix) in enumerate(
        [
            (original_ts, "Original Time Series (No Cross-Calibration)", "original"),
            (calibrated_ts, "Cross-Calibrated Time Series", "xcal"),
        ],
        1,
    ):
        plt.subplot(2, 1, i)
        plt.title(title)
        for m in params.missions_sorted:
            subset = ts.filter(pl.col("mission") == m).collect()
            if subset.height > 0:
                plt.errorbar(
                    subset["time"],
                    subset["mean_dh"],
                    yerr=subset["error_dh"],
                    label=f"{m} {label_suffix}",
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

    Load grid cell data from all missions, filter to cells meeting quality requirements,
    estimate biases, remove unrealistic bias outliers, spatially interpolate missing biases,
    apply bias corrections to elevations, and generate outputs.

    Writes :
        - <output_dir>/<basin>_corrected_epoch_average.parquet: Bias-corrected grid data
        - <output_dir>/<basin>_cross_calibrated_timeseries.parquet: Aggregated time series
        - <output_dir>/<basin>_cross_calibration_comparison.png: Comparison plot (if enabled)

    Args:
        basin_path (str): Basin subdirectory name, or "None" for root-level processing
        params (argparse.Namespace): Command line parameters
        logger (logging.Logger): Logger Object
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
        schema=None,
    )

    bias_tbl, outlier_stats = apply_bias_thresholds(bias_tbl_lf, params)
    filled_df = interpolate_biases_spatially(input_lf, bias_tbl, params.missions_sorted)
    cross_calibrated_lf, timeseries_lf = get_output_dataframe(
        params, filled_df, sub_basin=basin_path
    )

    if basin_path == "None":
        output_dir, suffix = Path(params.out_dir), ""
    else:
        output_dir, suffix = Path(params.out_dir) / basin_path, f"{Path(basin_path).name}_"
    os.makedirs(output_dir, exist_ok=True)

    # Write Parquet files
    cross_calibrated_lf.sink_parquet(output_dir / f"{suffix}corrected_epoch_average.parquet")
    timeseries_lf.sink_parquet(output_dir / f"{suffix}cross_calibrated_timeseries.parquet")

    if params.plot_results:
        plot(params, input_lf, bias_tbl, output_dir, suffix)

    return output_dir, outlier_stats


def cross_calibrate_missions(args: list[str]) -> None:
    """
    Main entry point for multi-mission cross-calibration processing.

    Parses arguments, discovers basins, runs per-basin cross-calibration,
    and writes central metadata.

    Supports basins and root-level processing modes, through --basin_structure flag.

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
        first_mission_dir = Path(params.mission_mapper[list(params.mission_mapper.keys())[0]])
        basins = get_basins_to_process(params, first_mission_dir, logger)
        valid_basins = get_basins_for_all_missions(params, basins, logger)

        # Process each basin/region/subregion
        for basin_path in valid_basins:
            output_dir, outlier_stats = process_basin(basin_path, params, logger)

            if output_dir is not None and outlier_stats is not None:
                write_outlier_stats(outlier_stats, output_dir, basin_path, logger)
                processed_basins.append(basin_path)

    # Write central metadata file after all processing
    write_central_metadata(params, start_time, processed_basins, logger)


if __name__ == "__main__":
    cross_calibrate_missions(sys.argv[1:])
