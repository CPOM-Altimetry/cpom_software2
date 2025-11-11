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


def discover_sub_basins(mission_dirs: dict) -> list:
    """
    Discover available sub-basins by scanning the directory structure.
    Used when no specific sub-basins are provided but sub-basins is set to "all".
    Validates the basin exists for all missions.

    Args:
        mission_dirs (dict): Dictionary mapping mission names to their base directories.
                           Example: {"e1": "/path/to/e1", "e2": "/path/to/e2"}

    Returns:
        list: Sorted list of sub-basin directory names that exist across all missions.
               Returns empty list if no common sub-basins found.
    """
    all_sub_basins = set()

    # Get subdirectories from the first mission directory
    first_mission = list(mission_dirs.keys())[0]
    first_dir = Path(mission_dirs[first_mission])

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
        for _, mission_dir in mission_dirs.items():
            sub_basin_path = Path(mission_dir) / sub_basin
            if not sub_basin_path.exists() or not list(sub_basin_path.glob("*.parquet")):
                exists_in_all = False
                break

        if exists_in_all:
            valid_sub_basins.append(sub_basin)

    return sorted(valid_sub_basins)


def get_path(params, mission: str, sub_basin: str = "None") -> Path:
    """
    Construct the file path for parquet files for a given mission (and optional sub-basin).

    Args:
        params (argparse.Namespace): Command line arguments.
                        Includes : mission_mapper
        mission (str): Mission identifier (e.g., 'e1', 'e2', 'env', 'cs2').
        sub_basin (str): Sub-basin directory name. If 'None', searches at root level.
                            Defaults to 'None'.
    Returns:
        str: File path pattern to parquet files for the specified mission (and sub-basin).

    Raises:
        KeyError: If mission not found in params.mission_mapper
    """
    if sub_basin == ["None"]:
        return Path(params.mission_mapper[mission]) / "*.parquet"
    return Path(params.mission_mapper[mission]) / sub_basin / "*.parquet"


def get_grid_cell_dataframe(params: argparse.Namespace, sub_basin: str = "None") -> pl.DataFrame:
    """
    Load and concatenate epoch-averaged altimetry data from all missions into a single DataFrame.

    This function loads parquet files for each mission specified in params.missions_sorted,
    and concatenates all mission data.

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
        pl.DataFrame: Concatenated DataFrame
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

    return pl.concat(lf).collect()


def get_time_absolute_years(data: pl.DataFrame, time_column: str) -> pl.DataFrame:
    """
    Convert time column to absolute fractional years.
    Args:
        data (pl.DataFrame): Input DataFrame
        params (argparse.Namespace): Command line parameters.
            Includes :
            - time_column (str): Name of the time column
    Returns:
        pl.DataFrame: DataFrame with additional column 'time_absolute_years'
    """
    time_col_dtype = data.schema[time_column]

    # If datetime, convert to absolute fractional year
    if time_col_dtype == pl.Datetime:
        time_expr = (
            pl.col(time_column).dt.year().cast(pl.Float64)
            + (pl.col(time_column).dt.ordinal_day() - 1).cast(pl.Float64) / 365.25
        ).alias("time_absolute_years")
    else:
        time_expr = (pl.col(time_column).cast(pl.Float64) + 1991.0).alias("time_absolute_years")

    return data.with_columns([time_expr])


def get_valid_cells(data: pl.DataFrame, params: argparse.Namespace) -> pl.DataFrame:
    """
    Filter out grid cells with insufficient data for cross-calibration.
    - Less than the minimum number of observations (params.min_observations)
    - Less than the minimum number of unique missions (params.min_missions)
    - Convert time to absolute years.
    Args:
        data (pl.DataFrame): Input DataFrame with mission data containing columns:
            - x_bin, y_bin: Grid cell coordinates
            - mission: Mission identifier
            - dh: Elevation change values
            - epoch_number, epoch_time: Temporal information
        params (argparse.Namespace): Command line parameters.
            Includes :
            - min_observations (int): Minimum total observations required per cell
            - min_missions (int): Minimum unique missions required per cell

    Returns:
        Filtered DataFrame with valid cells
    """
    data = get_time_absolute_years(data, params.time_column)

    valid_cells = (
        data.group_by(["x_bin", "y_bin"])
        .agg(
            [
                pl.len().alias("n_obs"),
                pl.col("mission").n_unique().alias("n_missions"),
                pl.col("mission").alias("missions"),
                pl.col(params.dh_column),
                pl.col(params.epoch_column),
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


def process_grid_cell_cross_calibration(
    data: pl.DataFrame, params: argparse.Namespace
) -> tuple[pl.DataFrame, dict]:
    """
    Perform cross-calibration for each grid cell to estimate mission biases.
    1. Extracting time series data for all missions in the cell
    2. Fit polynomial regression to get mission bias for cell
    3. Computing standard error from model residuals
    4. Applying bias threshold filters to remove outliers

    Args:
        data (pl.DataFrame): Valid grid cells DataFrame from get_valid_cells().
        params (argparse.Namespace): Command line parameters.
            Includes :
            - missions_sorted (list): Mission identifiers in processing order
            - bias_threshold (float): Maximum allowable bias magnitude

    Returns:
        tuple
            - cleaned_bias_tbl (pl.DataFrame): Processed bias corrections with columns:
                - x_bin, y_bin: Grid cell coordinates
                - success: Boolean indicating successful model fitting
                - {mission}_bias: Bias correction for each mission (outliers set to null)
                - stderr: Standard error of regression model fit
            - bias_tbl_stats (dict): Stats on the number of outliers removed.
    """

    def _fit_cross_calibrated_model_gridcell(
        missions: np.ndarray, dh: np.ndarray, t: np.ndarray, missions_sorted: list, degree: int = 3
    ) -> tuple:
        """
        Fit a polynomial regression model to get a mission bias per grid cell.
        The model separates temporal trends from mission biases.

        Args:
            missions (np.ndarray): Array of mission values.
            dh (np.ndarray): Array of elevation change values.
            t (np.ndarray): Array of epoch times (fractional years).
            degree (int, optional): Degree of polynomial temporal trend. Defaults to 3.
            missions_sorted (list): Ordered list of all missions.
        Returns:
            tuple:
                - fitted_curve (np.ndarray or None): Fitted polynomial curve.
                - biases (dict or None): Dictionary mapping mission names to bias values.
                                    First mission always has bias=0 as reference.
                - model (LinearRegression or None): Fitted sklearn LinearRegression model.
        """

        # Check that all arrays have the same length
        if len(missions) != len(dh) or len(missions) != len(t):
            return None, None, None

        missions_in_cell = list(set(missions))

        # Build a regression design matrix
        x = np.hstack([np.ones((len(t), 1)), np.vstack([t**i for i in range(1, degree + 1)]).T])

        # Add dummy variables for mission offsets (biases)
        for m in missions_sorted[1:]:
            if m in missions_in_cell:
                flag = (missions == m).astype(int).reshape(-1, 1)
                x = np.hstack([x, flag])
        try:
            model = LinearRegression().fit(x, dh)
            # Construct fitted curve
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
        except Exception as e:  # pylint: disable=broad-except
            print(f"Regression failed: {e}")
            return None, None, None

    def _apply_bias_thresholds(
        bias_tbl: pl.DataFrame, params: argparse.Namespace
    ) -> tuple[pl.DataFrame, dict]:
        """
        Filter out bias values greater than the threshold (params.bias_threshold).

        Args:
            bias_tbl (pl.DataFrame): DataFrame containing bias corrections for each mission.
            params (argparse.Namespace): Command line parameters.
                Includes:
                - bias_threshold (float): Maximum allowable absolute bias value (meters)
                - missions_sorted (list): List of mission identifiers

        Returns:
            tuple:
                - cleaned_bias_tbl (pl.DataFrame): Input DataFrame with outlier biases set to null.
                - bias_tbl_stats (dict): Statistics about outlier removal with structure:
                    {mission: {
                        'n_outlier_bias': int,      # Number of outliers removed
                        'n_total_bias': int,        # Total number of valid biases
                        'original_min_bias': float, # Minimum bias before cleaning
                        'original_max_bias': float, # Maximum bias before cleaning
                        'cleaned_min_bias': float,  # Minimum bias after cleaning
                        'cleaned_max_bias': float   # Maximum bias after cleaning
                    }}

        """

        # Filter out biases exceeding threshold
        cleaned_bias_tbl = bias_tbl.with_columns(
            [
                pl.when(pl.col(bias_col).abs() > params.bias_threshold)
                .then(None)  # If the bias exceeds the threshold, set to null
                .otherwise(pl.col(bias_col))
                .alias(bias_col)
                for bias_col in [f"{m}_bias" for m in params.missions_sorted]
            ]
        )

        # Get statistics on outliers
        bias_tbl_stats = {}
        for m in params.missions_sorted:
            bias_col = f"{m}_bias"
            bias_tbl_stats[m] = {
                "n_outlier_bias": cleaned_bias_tbl[bias_col].is_null().sum(),
                "n_total_bias": cleaned_bias_tbl[bias_col].is_not_null().sum(),
                "original_min_bias": bias_tbl[bias_col].min(),
                "original_max_bias": bias_tbl[bias_col].max(),
                "cleaned_min_bias": cleaned_bias_tbl[bias_col].min(),
                "cleaned_max_bias": cleaned_bias_tbl[bias_col].max(),
            }

        return cleaned_bias_tbl, bias_tbl_stats

    bias_tbl = []
    # Loop over every grid cell
    for row in data.iter_rows(named=True):
        # Filter to cell data
        dh = np.array(row[params.dh_column])
        # Fit a model to get mission biases per grid cell
        _, biases, model = _fit_cross_calibrated_model_gridcell(
            np.array(row["missions"]),
            dh,
            np.array(row["time_absolute_years"]),
            missions_sorted=params.missions_sorted,
            degree=3,
        )

        bias_row = {
            "x_bin": row["x_bin"],
            "y_bin": row["y_bin"],
            "success": biases is not None and model is not None,
        }

        stderr = None  # Initialize stderr

        if biases is not None and model is not None:
            # Store biases for each mission
            for mission in params.missions_sorted:
                bias_row[f"{mission}_bias"] = biases.get(
                    mission, 0
                )  # Default to 0 if mission not present

            # Compute standard errors from model residuals
            y_pred = model.predict(
                model.feature_names_in_
                if hasattr(model, "feature_names_in_")
                else np.arange(len(model.coef_)).reshape(1, -1)
            )
            residuals = dh - y_pred
            stderr = np.sqrt(np.mean(residuals**2))  # sqrt of mse

        # Store standard error for all missions in cell
        bias_row["stderr"] = stderr
        bias_tbl.append(bias_row)

    cleaned_bias_tbl, bias_tbl_stats = _apply_bias_thresholds(pl.DataFrame(bias_tbl), params)
    return cleaned_bias_tbl, bias_tbl_stats


def interpolate_biases_spatially(
    epoch_data: pl.DataFrame, grid_cell_data: pl.DataFrame, missions_sorted: list[str]
) -> pl.DataFrame:
    """
    Spatially interpolate bias corrections, using nearest neighbour method using a KDTree.
    Interpolates bias in grid cells where cross-calibration failed or was filtered out as outliers.

    Args:
        epoch_data (pl.DataFrame): Complete grid of epoch-averaged data.
        grid_cell_data (pl.DataFrame): Bias correction dataframe:
            - y_bin, x_bin: Grid cell coordinates with successful calibration
            - {mission}_bias: Bias values for each mission (may contain nulls)
            - stderr: Standard error values (may contain nulls)
        missions_sorted (list): List of missions.

    Returns:
        pl.DataFrame: Bias grid with missing values filled.
    """

    all_points = epoch_data.select(["y_bin", "x_bin"]).to_numpy()
    df_data = {"y_bin": all_points[:, 0], "x_bin": all_points[:, 1]}

    for col in [f"{mission}_bias" for mission in missions_sorted] + ["stderr"]:
        valid_data = grid_cell_data.filter(
            pl.col(col).is_not_null()
        ).select(  # & (pl.col(col) != 0))  # Filter out 0 values
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
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Apply bias corrections to mission data and generate corrected time series.

    Apply bias corrections to the original epoch-averaged data for all missions.
    Output the bias-corrected elevation change time series and basin-wide averages.

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
            - final_df (pl.DataFrame): Bias-corrected data for all grid cells.
            - timeseries_df (pl.DataFrame): Epoch-averaged time series.

    """

    all_corrected_data = []

    for mission in params.missions_sorted:
        path = get_path(params, mission, sub_basin)
        # Load mission data
        df_mission = (
            pl.scan_parquet(path)
            .join(filled_df.lazy(), on=["y_bin", "x_bin"], how="left")
            .with_columns(
                [
                    # Apply bias correction
                    (pl.col("dh_ave") - pl.col(f"{mission}_bias")).alias("biased_dh"),
                    pl.col("stderr").alias("biased_dh_xcal_stderr"),
                    # Add mission identifier
                    pl.lit(mission).alias("mission"),
                ]
            )
            .collect()
        ).filter(  # Calculating xcal_ok
            pl.any_horizontal(
                [pl.col(f"{m}_bias").is_not_null() for m in params.missions_sorted]
            )  # Check with have valid bias correction for at least one mission
            &
            # Has actual data
            pl.col("dh_ave").is_not_null()
        )

        all_corrected_data.append(df_mission)

    # Combine all missions into single DataFrame
    final_df = pl.concat(all_corrected_data)

    timeseries_df = get_dh_timeseries(
        final_df, dh_column="biased_dh", epoch_time_column=params.output_time_column
    )

    return final_df, timeseries_df


def get_dh_timeseries(df: pl.DataFrame, dh_column: str, epoch_time_column: str) -> pl.DataFrame:
    """
    Compute mean, standard deviation, and error of dh time series from a grid.

    Args:
        df (pl.DataFrame): Input DataFrame.
        dh_column (str): Name of the elevation change column.
        epoch_time_column (str): Name of the epoch time column.
    Returns:
        pl.DataFrame: DataFrame with columns:
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

    try:
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

    except OSError as e:
        logger.error("Failed to write crosscal_metadata.json with %s", e)


def plot(
    params: argparse.Namespace,
    input_df: pl.DataFrame,
    timeseries_df: pl.DataFrame,
    output_dir: Path,
    sub_basin: str = "None",
) -> None:
    """
    Generate comparison plots of original vs cross-calibrated time series.
    Args:
        params (argparse.Namespace): Command line parameters.
        input_df (pl.DataFrame): Original input DataFrame before cross-calibration.
        timeseries_df (pl.DataFrame): Cross-calibrated time series DataFrame.
        sub_basin (str, optional): Sub-basin name for plot titles. Defaults to None.
        output_dir (Path, optional): Directory to save plots.
    Returns:
        None
    """

    plt.figure(figsize=(12, 8))

    # Compute original basin averages for comparison (use original time column)
    original_time_series = get_dh_timeseries(
        input_df, dh_column=params.dh_column, epoch_time_column=params.time_column
    )
    original_time_series = get_time_absolute_years(original_time_series, params.time_column)

    # Plot original time series
    plt.subplot(2, 1, 1)
    plt.title("Original Time Series (No Cross-Calibration)")
    for m in params.missions_sorted:
        subset = original_time_series.filter(pl.col("mission") == m)
        if subset.height > 0:  # Check if subset has data
            plt.errorbar(
                subset["time_absolute_years"],
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
    plt.title("Cross-Calibrated Time Series (Bias Corrected)")

    # Convert timeseries_df to absolute years for plotting
    timeseries_df_plot = get_time_absolute_years(timeseries_df, params.output_time_column)

    for m in params.missions_sorted:
        subset = timeseries_df_plot.filter(pl.col("mission") == m)
        if subset.height > 0:  # Check if subset has data
            plt.errorbar(
                subset["time_absolute_years"],
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
    plt.savefig(output_dir / f"{sub_basin}_cross_calibration_comparison.png")
    plt.close()


def multi_mission_cross_calibrate(params: argparse.Namespace, logger: logging.Logger) -> None:
    """
    Execute the complete multi-mission cross-calibration workflow for specified sub-basins.

    Modes:
    - params.sub_basins = ['basin1', 'basin2']: Process specified basins
    - params.sub_basins = 'all': Auto-discover and process all basins
    - params.sub_basins = None: Process root-level data

    Steps:
    1. Determine sub-basins to process based on params.sub_basins
    2. For each sub-basin:
        a. Load grid cell data from all missions
        b. Filter to valid cells
        c. Perform cross-calibration to estimate mission biases
        d. Interpolate biases spatially to fill missing values
        e. Apply bias corrections to original data
        f. Save corrected data and metadata

    Args:
        params (argparse.Namespace): Command line parameters.
        logger (logging.Logger): Logger
    """
    logger.info("Starting multi-mission cross-calibration")
    start_time = time.time()

    # Determine which sub-basins to proces
    if params.sub_basins is None or params.sub_basins == ["None"]:
        sub_basins_to_process = ["None"]
        logger.info("Processing root level data")
    elif params.sub_basins in ("all", ["all"]):
        sub_basins_to_process = discover_sub_basins(params.mission_mapper)
        logger.info(f"Processing sub-basins: {sub_basins_to_process}")
    else:
        sub_basins_to_process = params.sub_basins
        logger.info(f"Processing sub-basins: {sub_basins_to_process}")

    # Process each sub-basin
    for sub_basin in sub_basins_to_process:
        logger.info(f"Processing sub-basin: {sub_basin}")
        input_df = get_grid_cell_dataframe(params, sub_basin=sub_basin)

        if input_df.height == 0:
            logger.warning(f"No data found for sub-basin: {sub_basin}")
            continue

        valid_cells = get_valid_cells(input_df, params)
        logger.info(f"Found {valid_cells.height} valid cells for cross-calibration")

        if valid_cells.height == 0:
            logger.info(f"No valid cells found for sub-basin: {sub_basin}")
            continue

        bias_tbl, outlier_stats = process_grid_cell_cross_calibration(valid_cells, params)
        filled_df = interpolate_biases_spatially(input_df, bias_tbl, params.missions_sorted)
        output, timeseries_df = get_output_dataframe(params, filled_df, sub_basin=sub_basin)

        output_dir = (
            Path(params.out_dir) / sub_basin if sub_basin is not None else Path(params.out_dir)
        )
        os.makedirs(output_dir, exist_ok=True)
        suffix = f"_{sub_basin}" if sub_basin else ""

        # Write Parquet files
        output.write_parquet(output_dir / f"corrected_epoch_average{suffix}.parquet")
        timeseries_df.write_parquet(output_dir / f"cross_calibrated_timeseries{suffix}.parquet")

        if params.plot_results:
            plot(params, input_df, timeseries_df, output_dir, sub_basin)

        get_metadata_json(params, outlier_stats, output_dir, start_time, logger)


def main(args):
    """
    Main script for multi-mission cross-calibration of altimetry data.
    Steps :
        1. Parse and validate command line arguments.
        2. Get logger.
        3. Validate input data directories.
        4. Run multi-mission cross-calibration workflow.
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
        help="Name of the fractional time column in the input data",
        default="epoch_midpoint_fractional_yr",
    )
    parser.add_argument(
        "--output_time_column",
        help="Name of the time column to use for output (can be datetime or fractional year)",
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
        "--min_missions",
        type=int,
        default=4,
        help="Minimum number of missions required per grid cell",
    )
    parser.add_argument(
        "--plot_results",
        action="store_false",
        help="Generate comparison plots of original vs cross-calibrated time series",
    )

    params = parser.parse_args(args)

    os.makedirs(params.out_dir, exist_ok=True)

    logger = set_loggers(
        log_file_info=Path(params.out_dir) / "info.log",
        log_file_error=Path(params.out_dir) / "errors.log",
    )

    multi_mission_cross_calibrate(params, logger)


if __name__ == "__main__":
    main(sys.argv[1:])
