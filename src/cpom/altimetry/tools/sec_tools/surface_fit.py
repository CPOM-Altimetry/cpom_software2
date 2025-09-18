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
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import duckdb
import numpy as np
import polars as pl
import statsmodels.api as sm
from scipy import stats

log = logging.getLogger(__name__)


def get_min_max_time(
    epoch_time: str, parquet_glob: str, mintime: str | None = None, maxtime: str | None = None
) -> tuple:
    """
    Get the min/max time as a timestamp and in seconds.
    Calculated from :
        mintime and maxtime date strings
        or
        The date range in the data if not provided.
    Args:
        mintime (str: Optional): Start time in (YYYY/MM/DD or YYYY.MM.DD).
        maxtime (str: Optional): End time in (YYYY/MM/DD or YYYY.MM.DD).
        epoch_time: Epoch time is referenced to.
        parquet_glob (str): Directionary glob for parquet files
    """

    def get_date(epoch_time, timedt):
        """
        Convert a date string to a datetime object
        and calculate the seconds from the epoch.
        """
        if "/" in timedt:
            time_dt = datetime.strptime(timedt, "%Y/%m/%d")
        elif "." in timedt:
            time_dt = datetime.strptime(timedt, "%Y.%m.%d")
        else:
            raise ValueError(
                f"Unrecognized date format: {timedt}, pass as YYYY/MM/DD or YYYY.MM.DD "
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
) -> pl.LazyFrame:
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
    Returns:
        pl.DataFrame: Filtered Polars DataFrame
    """
    # Get parquet files and apply time filter
    lf = pl.scan_parquet(parquet_glob).filter(
        (pl.col("time") >= min_secs) & (pl.col("time") <= max_secs)
    )

    # Get cells with valid elevations
    valid_cells = lf.filter(pl.col("elevation").is_not_null()).select(["x_bin", "y_bin"]).unique()
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
            (pl.col("time") / (365.25 * 24 * 3600)).alias("time_years"),
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
    if params.weight_name in schema:
        columns.append(params.weight_name)

    # Filter each gridcell to a singular (most common mode)
    if "mode" in schema:
        mode_counts = (
            lf.group_by(["x_bin", "y_bin", "mode"])
            .len()
            .with_columns(
                pl.col("len").rank("dense", descending=True).over(["x_bin", "y_bin"]).alias("rank")
            )
            .filter(pl.col("rank") == 1)
            .select(["x_bin", "y_bin", "mode"])
        )
        lf = lf.join(mode_counts, on=["x_bin", "y_bin", "mode"], how="inner")

    return lf.select([pl.col(col) for col in columns])


# pylint: disable=R0911, R0914
def _fit_surface_model_per_group(
    params: argparse.Namespace,
    group_np: np.ndarray,
    mintime: float,
    maxtime: float,
    logger: logging.Logger,
) -> tuple | str:
    """
    Fits an OLS model to the surface function:
    z(x,y,t,heading) = a0 + a1x + a2y + a3x^2 + a4y^2 + a5xy + a6heading+ a7t
    using multiple regression from statsmodel, for each grid cell.

    Iteratively filters out outliers beyond n_sigma standard deviations until convergence
    or maximum iterations reached.
    Compute dH removing fitted surface components from elevations.

    Args:
        params (argparse.Namespace): Configuration parameters.
        group_np (np.ndarray): Grid cell data containing:
            "height", "time", "time_years", "x", "y", "x2", "y2", "xy", "heading"
        mintime (float): Minimum time for surface fit (seconds)
        maxtime (float): Maximum time for surface fit (seconds)
        logger (logging.Logger): Logger Object
    Returns:
        tuple | str: On success, returns (statsmodels.OLS.Results, filtered_data_dict).
                    On failure, returns error status string.
    """
    # --------------------------------------------------------------------------
    # Iterate a surface model fit to the cell
    # --------------------------------------------------------------------------
    for i in range(params.max_surfacefit_iterations):

        if group_np["height"].size <= 8 | group_np["height"].size < params.min_vals_in_cell:
            return "n_cells_with_too_few_measurements"

        if (group_np["time"].max() - group_np["time"].min()) < params.min_timespan_in_cell_in_secs:
            return "n_cells_with_timespan_too_short"

        # --------------------------------------------------------------------------
        # Convert time to reference time in centre of timerange
        # --------------------------------------------------------------------------
        ref_time = group_np["time_years"] - ((mintime + maxtime) / (2.0 * 31557600))
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
        if params.weighted_surface_fit is True:

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
                    return "n_fit_failed"
            # pylint: disable=W0718
            except Exception as e:
                logger.warning(f"WLS failed with: {e}")
                return "wls_failure"
            fit_params = res.params
        else:
            # ----------------------------------------------------
            # Least Squares fit to a surface function
            # -----------------------------------------------------
            fit_params, _, _, _ = np.linalg.lstsq(iv, group_np["height"], rcond=None)

        # --------------------------------------------------------------------------
        # Calculate the resulting modelled heights
        # --------------------------------------------------------------------------
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
        # --------------------------------------------------------------------------
        # Calculate the standard deviation (sigma) of the absolute differences between heights and
        # modelled heights. Filter where differences >  n_sigma * sigma
        # --------------------------------------------------------------------------
        differences = modelled_heights - group_np["height"]
        sigma = np.std(differences, ddof=1)  # this ddof matches IDL
        mask = np.abs(differences) <= params.n_sigma * sigma

        if np.count_nonzero(mask) < params.min_vals_in_cell:
            return "n_cells_with_too_few_measurements_after_fit_sigma_filter"

        if (
            params.weighted_surface_fit is False
        ):  # Do not check for convergence with Weighted surface fit
            if np.count_nonzero(~mask) == 0:  # If the filtering converged (No outliers)
                # ----------------------------------------------------
                # Remove modelled heights from measured elevation
                # This leave the temporal change + residuals
                # ----------------------------------------------------
                group_np["dH"] = group_np["height"] - (
                    fit_params[0]
                    + fit_params[1] * group_np["x"]
                    + fit_params[2] * group_np["y"]
                    + fit_params[3] * group_np["x2"]
                    + fit_params[4] * group_np["y2"]
                    + fit_params[5] * group_np["xy"]
                    + fit_params[6] * group_np["heading"]
                )

                return fit_params, {
                    key: value
                    for key, value in group_np.items()
                    if key not in ["x", "y", "x2", "y2", "xy"]
                }
            if np.count_nonzero(~mask) > 0 and i == (params.max_surfacefit_iterations) - 1:
                return "n_cells_didnot_converge"

        for key in group_np:  # Apply mask to all arrays in group dictionary
            group_np[key] = group_np[key][mask]

    group_np["dH"] = group_np["height"] - (
        fit_params[0]
        + fit_params[1] * group_np["x"]
        + fit_params[2] * group_np["y"]
        + fit_params[3] * group_np["x2"]
        + fit_params[4] * group_np["y2"]
        + fit_params[5] * group_np["xy"]
        + fit_params[6] * group_np["heading"]
    )

    return fit_params, {
        key: value
        for key, value in group_np.items()
        if key not in ["x", "y", "x2", "y2", "xy", "height"]
    }


# pylint: disable=R0917, R0913 , R0914
def _fit_power_correction_per_group(
    params: argparse.Namespace,
    group_np: np.ndarray,
    pc_min_secs: float,
    pc_max_secs: float,
    mintime: float,
    maxtime: float,
    logger: logging.Logger,
) -> dict | str:
    """
    Apply power (backscatter) correction to elevation residuals.
    Fits an OLS model to power as a function of time and heading (p = a + bt + ch),
    then removes height components correlated with time-dependent power changes.
    The correction is applied as: dH = dH - (dH/dP) * dP, where dH/dP is the
    gradient calculated over the power correction time window.

    Args:
        params (argparse.Namespace): Configuration parameters
        group_np (np.ndarray): Grid cell containing:
        "heading", "time", "time_years", "power", "dH"
        pc_min_secs (float): Minimum time for power correction period (seconds)
        pc_max_secs (float): Maximum time for power correction period (seconds)
        mintime (float): Minimum time for surface fit reference (seconds)
        maxtime (float): Maximum time for surface fit reference (seconds)
        logger (logging.Logger): Logger Object
    Returns:
        dict | str: Dictionary with corrected data ("time", "time_years", "dH")
                   or error status string.
    """

    pc_timeok_indices = np.where(
        (group_np["time"] >= pc_min_secs) & (group_np["time"] <= pc_max_secs)
    )[0]
    ref_time = group_np["time_years"] - ((mintime + maxtime) / (2.0 * 31557600))

    if pc_timeok_indices.size >= params.min_vals_in_cell:
        # ----------------------------------------------------------------------------------------
        # Fit a model to power as a function of time and heading, p = a + bt + ch
        # ----------------------------------------------------------------------------------------
        iv = np.zeros((ref_time.size, 2))
        iv[:, 0] = ref_time
        iv[:, 1] = group_np["heading"]
        iv = sm.add_constant(iv, has_constant="add")

        weights = group_np[params.weight_name]
        w = np.where(weights > 0, 1.0 / (weights**2), 0.0)
        wt = w / np.nansum(w)

        if params.weighted_power_fit:
            power_res = sm.WLS(group_np["power"], iv, weight=wt, missing="drop").fit()
            if power_res.model.data.param_names != ["const", "x1", "x2"]:
                return "n_fit_failed"
        else:
            try:
                power_res = sm.OLS(group_np["power"], iv, missing="drop").fit()
                if power_res.model.data.param_names != ["const", "x1", "x2"]:
                    return "n_fit_failed"
            except Exception as e:  # pylint: disable=W0718
                logger.warning(f"OLS failed with: {e}")
                return "ols_failure"
        # -----------------------------------------------------------------
        # Get the time dependent component of power is: bt = p -(a  + ch)
        # ------------------------------------------------------------------
        dp = group_np["power"] - (power_res.params[0] + power_res.params[2] * group_np["heading"])
        # --------------------------------------------------------------
        # Remove component of dH due to correlated power change
        # Find the gradient of dH/dP over time interval (pcmintime,pcmaxtime)
        # --------------------------------------------------------------
        m, _ = np.polyfit(dp[pc_timeok_indices], group_np["dH"][pc_timeok_indices], 1)  # m=dH/dP
        group_np["dH"] -= m * dp

        return {
            key: value
            for key, value in group_np.items()
            if key in ["time", "time_years", "dH", params.weight_name]
        }
    return "too_few_vals_after_pctime_filter"


def _fit_linear_fit_per_group(params: argparse.Namespace, group_np: np.ndarray):
    """
    Calculate (dH/dt), from heights and time :  h = a + bt
    Iterate until the difference between the dH and modelled dH are
    are within +/-2 * rms or the max_linearfit_iterations limit is reached.

    Args:
        params (argparse.Namespace): Configuration parameters
        group_np (np.ndarray): Grid cell data dictionary with keys:
            "time_years", "dH" (elevation residuals after surface/power correction)

    Returns:
        tuple | str: On success, returns (slope, rms, sigma, mask, filtered_data).
                    On failure, returns error status string.
    """
    for _ in range(params.max_linearfit_iterations):

        # Check if all the time values in the cell are the same.
        if params.weighted_surface_fit is False:
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
        if np.count_nonzero(~mask) == 0:
            return {
                "m": m,
                "rms": rms,
                "sigma": sigma,
                "mask": mask,
                "group_np": group_np,
                "std_err": std_err,
            }

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


# pylint: disable=R0914, R0915
def surface_fit(params, parquet_glob, grid_meta):
    """
    Perform surface fit on grid data.
    1. Get the min/max time period for surface fit and power correction
    2. Get the minimum time span in a cell in seconds
    3. Get unique chunks (x_part, y_part) in the dataset
    4. Loop though chunks and load data using 'get_grid_data'
    5. Loop through grid cells :
        - Convert gridcell data to a numpy array
        - Perform surface fit 'fit_surface_model_per_group'
        - Perform power correction 'fit_power_correction_per_group'
        - Perform linear fit 'fit_linear_fit_per_group'
        - Construct output dictionaries
    6. Write parquet files
        - A parquet of grid cells metadata
        - A timeseries grid

    Args:
        params (argparse.Namespace): Configuration parameters
        parquet_glob (str): Path to the input parquet files
        grid_meta (dict): Metadata for the grid
        log (logging.Logger, optional): Logger

    Returns:
        dict: Status dictionary containing result counts
    """

    # 1. Get the min/max time period for surface fit and power correction
    mintime, maxtime, min_secs, max_secs = get_min_max_time(
        datetime.fromisoformat(grid_meta["standard_epoch"]),
        parquet_glob,
        params.mintime,
        params.maxtime,
    )

    # 2. Get the minimum time span in a cell in seconds
    # Defaults to 70% of the timespan in seconds
    params.min_timespan_in_cell_in_secs = (maxtime - mintime) * (
        params.min_percent_timespan_in_cell / 100.0
        if params.min_percent_timespan_in_cell
        else (maxtime - mintime) * 0.7
    )  # 70% of required time span

    status = {
        k: 0
        for k in [
            "n_cells_with_timespan_too_short",
            "n_cells_with_time_outside_minmax",
            "n_cells_with_too_few_measurements",
            "n_cells_with_too_few_measurements_after_fit_sigma_filter",
            "n_cells_fit_failed",
            "n_cells_didnot_converge",
            "n_cells_power_corrected",
            "n_cells_ols_failure",
            "n_cells_too_few_values_in_linear_fit",
            "n_cells_too_few_vals_after_pctime_filter",
            "n_cells_fitted",
        ]
    }

    with warnings.catch_warnings():
        warnings.simplefilter("error")

    # 3. Get unique chunks in the dataset
    conn = duckdb.connect()
    part_df = conn.execute(
        f"""
        SELECT DISTINCT x_part, y_part
        FROM read_parquet('{params.griddir}/*/x_part=*/y_part=*/*.parquet', hive_partitioning=1);
        """
    ).pl()
    conn.close()

    i = 0
    grid_records = []
    for row in part_df.iter_rows(named=True):
        timeseries_records = []

        i += 1
        print(f"Processing partition {i} / {len(part_df)}")

        # 4. Get grid data for chunk
        gridcell_lazy = get_grid_data(
            f"{params.griddir}/*/x_part={row['x_part']}/y_part={row['y_part']}/*.parquet",
            params,
            min_secs,
            max_secs,
        ).collect()
        grouped = gridcell_lazy.group_by(["x_bin", "y_bin", "x_part", "y_part"])

        # 5. Execute surface fit process
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
            result = _fit_surface_model_per_group(
                params=params, group_np=gridcell_np, mintime=mintime, maxtime=maxtime, logger=log
            )

            if isinstance(result, str):
                status[result] += 1
                continue
            res, group_np = result
            # --------------------------------------------------------------------
            #  Apply a Power Correction if --apply_power_correction is set
            #  power correction can be performed using the whole time period, or
            #  a short period (ie 60 months) set by pcmintime,pcmaxtime
            # --------------------------------------------------------------------
            if params.powercorrect:
                _, _, pc_min_secs, pc_max_secs = get_min_max_time(
                    datetime.fromisoformat(grid_meta["standard_epoch"]),
                    parquet_glob,
                    params.pcmintime,
                    params.pcmaxtime,
                )

                result = _fit_power_correction_per_group(
                    params,
                    group_np,
                    pc_min_secs=pc_min_secs,
                    pc_max_secs=pc_max_secs,
                    mintime=mintime,
                    maxtime=maxtime,
                    logger=log,
                )

                if isinstance(result, str):
                    status[result] += 1
                    continue
                status["n_cells_power_corrected"] += 1
                group_np = result

            # ------------------------------------------------
            # Get modelled dh/dt
            # ------------------------------------------------
            result = _fit_linear_fit_per_group(params, group_np)
            if isinstance(result, str):
                status[result] += 1
                continue

            # Unpack results
            m = result["m"]
            rms = result["rms"]
            sigma = result["sigma"]
            mask = result["mask"]
            group_np = result["group_np"]
            std_err = result.get("std_err")

            if len(mask) > 3:
                grid_record = {
                    "x_part": x_part,
                    "y_part": y_part,
                    "x_bin": x_bin,
                    "y_bin": y_bin,
                    "dhdt": m,
                    "slope": (180.0 / np.pi) * np.sqrt(res[1] ** 2 + res[2] ** 2),
                    "sigma": sigma,
                    "rms": rms,
                }
                if std_err is not None:
                    grid_record["std_err"] = std_err
                grid_records.append(grid_record)

                for idx, (ts_val, ty_val, dh_val) in enumerate(
                    zip(group_np["time"], group_np["time_years"], group_np["dH"])
                ):
                    ts_record = {
                        "x_part": x_part,
                        "y_part": y_part,
                        "x_bin": x_bin,
                        "y_bin": y_bin,
                        "time_years": ty_val,
                        "time": ts_val,
                        "dh": dh_val,
                    }
                    if params.weight_name in group_np:
                        ts_record[params.weight_name] = group_np[params.weight_name][idx]
                    timeseries_records.append(ts_record)

                status["n_cells_fitted"] += 1
            else:
                status["n_too_few_values_in_linear_fit"] += 1

        # 6. Write Parquet Files per chunk
        chunk_outdir = Path(params.sf_dir) / f"x_part={row['x_part']}" / f"y_part={row['y_part']}"
        os.makedirs(chunk_outdir, exist_ok=True)
        pl.DataFrame(timeseries_records).lazy().sink_parquet(
            chunk_outdir / "dh_time_grid.parquet", compression="zstd"
        )

    pl.DataFrame(grid_records).lazy().sink_parquet(
        Path(params.sf_dir) / "grid_data.parquet", compression="zstd"
    )
    return status


def main(args):
    """
    Main function to parse arguments and execute surface fit.
    1. Load command line arguments
    2. Load grid metadata from the grid_dir
    3. Set output directory for surface fit results
    4. Perform surface fit
    5. Write metadata json file
    """

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
        "--griddir",
        help=("Path of the grid dir containing parquet files"),
        type=str,
        required=True,
    )
    parser.add_argument(
        "--sf_dir",
        help="Path of output directory for surface fit results",
        type=str,
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
        default=30,
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
        help="(optional) start time for plane fit (YYYY/MM/DD or YYYY.MM.DD) \
            | min time in the data if not provided.",
    )
    parser.add_argument(
        "--maxtime",
        type=str,
        help="(optional) End time for plane fit (YYYY/MM/DD or YYYY.MM.DD) \
            | max time in the grid if not provided.",
    )
    parser.add_argument(
        "--powercorrect",
        help="(optional) if set apply a power (backscatter) correction",
        type=bool,
        default=False,
    )
    parser.add_argument(
        "--pcmintime",
        type=str,
        help="(optional) Start date of power correction (YYYY/MM/DD or YYYY.MM.DD) \
        | mintime if not provided.",
    )
    parser.add_argument(
        "--pcmaxtime",
        type=str,
        help="(optional) End date of power correction (YYYY/MM/DD or YYYY.MM.DD) \
              | maxtime if not provided.",
    )
    parser.add_argument(
        "--weighted_surface_fit",
        type=bool,
        default=False,
        help="Use weighted surface fit for is2 data",
    )
    parser.add_argument(
        "--weight_name",
        type=str,
        help="(optional) Name of the weight column to use for weighted surface fit",
    )
    start_time = time.time()

    # -----------------------------------------#
    # 2. Load grid metadata from the grid_dir #
    # -----------------------------------------#
    params = parser.parse_args(args)

    if params.weighted_surface_fit is True:
        if params.weight_name is None:
            sys.exit("Weight name must be provided for weighted surface fit")

    parquet_glob = f"{params.griddir}/**/*.parquet"
    with open(params.griddir + "grid_meta.json", "r", encoding="utf-8") as f:
        grid_meta = json.load(f)
    # ------------------------------------------------#
    # 3. Set output directory for surface fit results #
    # ------------------------------------------------#

    params.sf_dir = os.path.join(
        params.sf_dir,
        grid_meta["mission"],
        f'{grid_meta["gridarea"]}_{int(grid_meta["binsize"]/1000)}km_{grid_meta["mission"]}',
    )
    if os.path.exists(params.sf_dir):
        if params.sf_dir != "/" and grid_meta["mission"] in params.sf_dir:  # safety check
            log.info("Removing previous surface fit dir: %s ...", params.sf_dir)
            response = (
                input("Confirm removal of previous surface fit archive? (y/n): ").strip().lower()
            )
            if response == "y":
                shutil.rmtree(params.sf_dir)
            else:
                print("Exiting as user requested not to overwrite surface fit archive")
                sys.exit(0)
        else:
            sys.exit(1)
        os.makedirs(params.sf_dir, exist_ok=True)

    # -----------------------------------------#
    # 4. Perform surface fit
    # -----------------------------------------#

    status = surface_fit(params, parquet_glob=parquet_glob, grid_meta=grid_meta)
    hours, remainder = divmod(int(time.time() - start_time), 3600)
    minutes, seconds = divmod(remainder, 60)

    # ----------------------------#
    # 5. Write metadata json file #
    # ----------------------------#
    try:
        with open(Path(params.sf_dir) / "surface_fit_meta.json", "w", encoding="utf-8") as f_meta:
            json.dump(
                {
                    "gridarea": grid_meta["gridarea"],
                    "dataset": grid_meta["dataset"],
                    **vars(params),
                    **status,
                    "execution_time": f"{hours:02}:{minutes:02}:{seconds:02}",
                },
                f_meta,
                indent=2,
            )
    except OSError as e:
        log.error("Failed to write surface_fit_meta.json with ")
        sys.exit(e)


if __name__ == "__main__":
    main(sys.argv[1:])
