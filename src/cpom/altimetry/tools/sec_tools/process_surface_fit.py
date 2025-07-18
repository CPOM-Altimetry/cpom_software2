import argparse
import json
import logging
import os
import shutil
import sys
import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import duckdb
import numpy as np
import polars as pl
import statsmodels.api as sm

from cpom.gridding.gridareas import GridArea
from cpom.logging_funcs.logging import set_loggers

log = logging.getLogger(__name__)


def get_min_max_time(mintime, maxtime, epoch_time, parquet_glob):
    """
    Process command line options --mintime, --maxtime into datetime and seconds objects.
    If no options provided, get min and max from parquet grid files with duckdb.

    Args:
        mintime:
        maxtime:
        epoch_time:
        parquet_glob (str): Directionary glob for parquet files
    """

    def get_date(epoch_time, timedt=None):
        if timedt is not None:
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


def filter_mode_and_time(parquet_glob, min_secs, max_secs):
    query = f"""
        WITH
            raw AS (
                SELECT *
                FROM parquet_scan('{parquet_glob}')
                WHERE time >= {min_secs} AND time <= {max_secs}
            ),
            mode_filter AS (
                SELECT x_bin, y_bin, mode
                FROM raw
                GROUP BY x_bin, y_bin, mode
                QUALIFY ROW_NUMBER() OVER (PARTITION BY x_bin, y_bin ORDER BY COUNT(*) DESC) = 1
                ORDER BY x_bin, y_bin
            ),
            valid_cells AS (
                SELECT DISTINCT x_bin, y_bin
                FROM raw
                WHERE elevation IS NOT NULL
            ),
            tbl AS (
                SELECT
                    x_part,
                    y_part,
                    x_bin,
                    y_bin,
                    x_cell_offset as x,
                    y_cell_offset as y,
                    x ** 2 as x2,
                    y ** 2 as y2,
                    x * y as xy,
                    elevation as height,
                    power,
                    CAST(ascending AS INTEGER) AS heading,
                    mode,
                    time,
                    time / 31557600 as time_years
                FROM raw
            )
        SELECT tbl.*
        FROM tbl
        JOIN mode_filter
            ON tbl.x_bin = mode_filter.x_bin
           AND tbl.y_bin = mode_filter.y_bin
           AND tbl.mode = mode_filter.mode
        JOIN valid_cells
            ON tbl.x_bin = valid_cells.x_bin
           AND tbl.y_bin = valid_cells.y_bin
    """

    conn = duckdb.connect()

    # # Set memory limit (e.g., 1 GB)
    # conn.execute("SET memory_limit='100GB';")

    # # # Set number of threads (e.g., 4)
    # conn.execute("SET threads=20;")
    tbl = conn.execute(query)
    df = tbl.pl()

    return df


def fit_surface_model_per_group(params, group_np, mintime, maxtime, logger):
    # --------------------------------------------------------------------------
    # Iterate a surface model fit to the cell
    # --------------------------------------------------------------------------
    for n_iter in range(params.max_surfacefit_iterations):

        if group_np["height"].size <= 8:
            return "n_cells_with_too_few_measurements"

        if group_np["height"].size < params.min_vals_in_cell:
            return "n_cells_with_too_few_measurements"

        if (group_np["time"].max() - group_np["time"].min()) < params.min_timespan_in_cell_in_secs:
            return "n_cells_with_timespan_too_short"

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
        # -------------------------------------------------------------------------
        # Perform a Least Squares fit to a surface function
        # --------------------------------------------------------------------------
        try:
            res = sm.OLS(group_np["height"], iv, missing="drop").fit()
            if res.model.data.param_names != ["const", "x1", "x2", "x3", "x4", "x5", "x6", "x7"]:
                return "n_fit_failed"
        except Exception as e:
            logger.warning(f"OLS failed with: {e}")
            return "ols_failure"
        # --------------------------------------------------------------------------
        # Calculate the resulting modelled heights
        # --------------------------------------------------------------------------
        modelled_heights = (
            res.params[0]
            + res.params[1] * group_np["x"]
            + res.params[2] * group_np["y"]
            + res.params[3] * group_np["x2"]
            + res.params[4] * group_np["y2"]
            + res.params[5] * group_np["xy"]
            + res.params[6] * group_np["heading"]
            + res.params[7] * ref_time
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

        if np.count_nonzero(~mask) == 0:
            # ----------------------------------------------------------------------------------------
            # Remove surface components of the modelled elevation from measured elevation
            # This leave the temporal change + residuals
            # ----------------------------------------------------------------------------------------
            group_np["dH"] = group_np["height"] - modelled_heights
            return res, {
                key: value
                for key, value in group_np.items()
                if key not in ["x", "y", "x2", "y2", "xy"]
            }

        for key in group_np:  # Apply mask to all arrays in group dictionary
            group_np[key] = group_np[key][mask]

    return "n_didnot_converge"


def fit_power_correction_per_group(
    params, group_np, pc_min_secs, pc_max_secs, mintime, maxtime, logger
):

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

        try:
            res = sm.OLS(group_np["power"], iv, missing="drop").fit()
            if res.model.data.param_names != ["const", "x1", "x2"]:
                return "n_fit_failed"
        except Exception as e:
            logger.warning(f"OLS failed with: {e}")
            return "ols_failure"

        # time dependent component of power is: bt = p -(a  + ch)
        dP = group_np["power"] - (res.params[0] + res.params[2] * group_np["heading"])
        # -------------------------------------------------
        # Apply a power (backscatter correction)
        #  Correlate power with height in order to perform power correction
        #   dh = dh - dp * (dh/dP), where the correlation
        #       of dh&dP is > min_correlation (ie typically 0.5)
        #   Note power correction gradient calculated between 2 dates (typically 18 months or 36 months apart)
        #   as specificed in min_pc_time, max_pc_time (in years).
        # -------------------------------------------------

        # remove component of dH due to correlated power change
        # find the gradient of dH/dP, but only over a specified time interval (pcmintime,pcmaxtime)
        m, _ = np.polyfit(dP[pc_timeok_indices], group_np["dH"][pc_timeok_indices], 1)  # m=dH/dP
        group_np["dH"] -= m * dP

        return {
            key: value for key, value in group_np.items() if key in ["time", "time_years", "dH"]
        }
    else:
        return "too_few_vals_after_pctime_filter"


def fit_linear_fit_per_group(params, group_np):
    for n_iter in range(params.max_linearfit_iterations):

        m, c = np.polyfit(group_np["time_years"], group_np["dH"], 1)
        poly1d_fn = np.poly1d((m, c))
        modelled_dH = poly1d_fn(group_np["time_years"])
        differences = group_np["dH"] - modelled_dH
        rms = np.sqrt(np.mean(differences**2))
        sigma = np.std(differences, ddof=1)  # this ddof matches IDL

        mask = np.absolute(differences) <= rms * 2.0

        if np.count_nonzero(mask) <= 3:
            return "n_too_few_values_in_linear_fit"

        if np.count_nonzero(~mask) == 0:
            return m, rms, sigma, mask, group_np

        for key in group_np:  # Apply mask to all arrays in group dictionary
            group_np[key] = group_np[key][mask]

    return m, rms, sigma, mask, group_np


def surface_fit(params, parquet_glob, grid_meta, log=None):

    standard_epoch = datetime.fromisoformat(grid_meta["standard_epoch"])

    mintime, maxtime, min_secs, max_secs = get_min_max_time(
        params.mintime, params.maxtime, standard_epoch, parquet_glob
    )
    _, _, pc_min_secs, pc_max_secs = get_min_max_time(
        params.pcmintime, params.pcmaxtime, standard_epoch, parquet_glob
    )

    if params.min_percent_timespan_in_cell:
        params.min_timespan_in_cell_in_secs = (maxtime - mintime) * (
            params.min_percent_timespan_in_cell / 100.0
        )
    else:
        params.min_timespan_in_cell_in_secs = (maxtime - mintime) * 0.7  # 70% of required time span

    df = filter_mode_and_time(parquet_glob, min_secs, max_secs)

    grid_records, timeseries_records = [], []
    status = {
        "n_cells_with_timespan_too_short": 0,
        "n_cells_with_time_outside_minmax": 0,
        "n_cells_with_too_few_measurements": 0,
        "n_cells_with_too_few_measurements_after_fit_sigma_filter": 0,
        "n_fit_failed": 0,
        "n_didnot_converge": 0,
        "n_power_corrected": 0,
        "ols_failure": 0,
        "n_too_few_values_in_linear_fit": 0,
        "too_few_vals_after_pctime_filter": 0,
        "n_cells_fitted": 0,
    }

    with warnings.catch_warnings():
        warnings.simplefilter("error")

    unique_counts = (
        df.select(["x_bin", "y_bin", "x_part", "y_part"])
        .unique()
        .group_by(["x_bin", "y_bin"])
        .count()
    )
    if unique_counts["count"].n_unique() == 1:
        grouped = df.group_by(["x_bin", "y_bin", "x_part", "y_part"])
    i = 0
    for (x_bin, y_bin, x_part, y_part), group in grouped:
        i += 1
        if i % 1000 == 0:
            print(f"Processing cell {i} at {time.time()}")

        group_np = {
            col: group[col].to_numpy()
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
            ]
        }

        # --------------------------------------------------------------------------
        # Perform a Least Squares fit to a surface function
        # --------------------------------------------------------------------------
        result = fit_surface_model_per_group(
            params=params, group_np=group_np, mintime=mintime, maxtime=maxtime, logger=log
        )

        if isinstance(result, str):
            status[result] += 1
            continue
        else:
            res, group_np = result
            # ----------------------------------------------------------------------------------------
            #  Apply a Power Correction if --apply_power_correction is set
            #  power correction can be performed using the whole time period, or
            #  a short period (ie 60 months) set by pcmintime,pcmaxtime
            # ----------------------------------------------------------------------------------------
            if params.powercorrect:

                result = fit_power_correction_per_group(
                    params,
                    group_np,
                    pc_min_secs=pc_min_secs,
                    pc_max_secs=pc_max_secs,
                    mintime=mintime,
                    maxtime=maxtime,
                    logger=log,
                )

                if type(result) is str:
                    status[result] += 1
                    continue
                else:
                    status["n_power_corrected"] += 1
                    group_np = result

            result = fit_linear_fit_per_group(params, group_np)
            if isinstance(result, str):
                status[result] += 1
            else:
                m, rms, sigma, mask, group_np = result

                if len(mask) > 3:

                    grid_records.append(
                        {
                            "x_part": x_part,
                            "y_part": y_part,
                            "x_bin": x_bin,
                            "y_bin": y_bin,
                            "dhdt": m,
                            "slope": (180.0 / np.pi)
                            * np.sqrt(res.params[1] ** 2 + res.params[2] ** 2),
                            "sigma": sigma,
                            "rms": rms,
                        }
                    )

                    # Save time series as flat table
                    for ts_val, ty_val, dh_val in zip(
                        group_np["time"], group_np["time_years"], group_np["dH"]
                    ):
                        timeseries_records.append(
                            {
                                "x_part": x_part,
                                "y_part": y_part,
                                "x_bin": x_bin,
                                "y_bin": y_bin,
                                "time_years": ty_val,
                                "time": ts_val,
                                "dh": dh_val,
                            }
                        )

                    status["n_cells_fitted"] += 1

                else:
                    status["n_too_few_values_in_linear_fit"] += 1

    con = duckdb.connect()
    con.register("grid_df", pl.DataFrame(grid_records))
    con.register("timeseries_df", pl.DataFrame(timeseries_records))

    combos = con.execute(
        f"""
        SELECT DISTINCT x_part, y_part FROM grid_df
    """
    ).fetchall()
    os.makedirs(params.surface_fit_dir, exist_ok=True)
    con.execute(
        f"""
        COPY (
            SELECT * FROM grid_df
        ) TO '{Path(params.surface_fit_dir) / "grid_data.parquet"}'
        (FORMAT PARQUET, COMPRESSION zstd)
    """
    )

    for xp, yp in sorted(combos):
        # Create directory for partition
        chunk_outdir = Path(params.surface_fit_dir) / f"x_part={xp}" / f"y_part={yp}"
        os.makedirs(chunk_outdir, exist_ok=True)

        time_path = Path(chunk_outdir) / "dh_time_grid.parquet"

        # Now df has the entire set of rows for that chunk, across all years
        con.execute(
            f"""
            COPY (
                SELECT * FROM timeseries_df
                WHERE x_part = {xp} AND y_part = {yp}
            ) TO '{time_path}'
            (FORMAT PARQUET, COMPRESSION zstd)
        """
        )

    return status


def get_surface_fit_json(params, grid_meta, status, time):

    json_data = {
        "grid_name": grid_meta["grid_name"],
        "dataset": grid_meta["dataset"],
        **vars(params),
        **status,
        "execution_time": time,
    }

    json_metadata_path = Path(params.surface_fit_dir) / "surface_fit_meta.json"
    try:
        with open(json_metadata_path, "w", encoding="utf-8") as f_meta:
            json.dump(json_data, f_meta, indent=2)
    except OSError as exc:
        pass
        # log.error("Failed to write surface_fit_meta.json")


def main(args):

    parser = argparse.ArgumentParser(
        description=("Compute fitted elevation per grid cell from a 'compacted' Parquet dataset, ")
    )

    parser.add_argument(
        "--debug",
        "-d",
        help="Output debug log messages to console",
        required=False,
        action="store_true",
    )
    parser.add_argument(
        "--griddir",
        "-gd",
        help=(
            "Path of the 'compacted' grid dir containing parquet files under x_part=NN/y_part=MM."
        ),
        type=str,
        required=True,
    )
    parser.add_argument(
        "--surface_fit_dir",
        "-o",
        help="path of output directory for output files. If not specified use directory path of input grid file for output",
    )
    parser.add_argument(
        "--min_vals_in_cell",
        "-m",
        default=20,
        type=int,
        help="(integer, default=20), minimum number of measurements in cell to perform plane fit",
    )
    parser.add_argument(
        "--min_percent_timespan_in_cell",
        "-t",
        default=70.0,
        type=float,
        help="(optional), minimum time span in cell to perform plane fit as percent of (maxtime-mintime), default is 70%%",
    )
    parser.add_argument(
        "--n_sigma",
        "-s",
        default=2.0,
        type=float,
        help="n : (float, default=2.0) recursively filter out measurements where (modelfit-height) > n * stdev(modelfit-height)",
    )
    parser.add_argument(
        "--max_surfacefit_iterations",
        "-ms",
        default=30,
        type=int,
        help="(integer, default=50) maximum number of iterations of surface modelfit",
    )
    parser.add_argument(
        "--max_linearfit_iterations",
        "-ml",
        default=3,
        type=int,
        help="(integer, default=3) maximum number of iterations of linear fit",
    )
    parser.add_argument(
        "--mintime",
        "-t1",
        help="(optional) start time to use for  plane fit in decimal years YYYY.yy or DD/MM/YYYY format. If not provided then earliest time in grid is used.",
    )
    parser.add_argument(
        "--maxtime",
        "-t2",
        help="(optional) end time to use for plane fit in decimal years YYYY.yy or DD/MM/YYYY format. If not provided then latest time in grid is used.",
    )
    parser.add_argument(
        "--powercorrect",
        "-pc",
        help="(optional) if set apply a power (backscatter) correction",
        action="store_true",
    )
    parser.add_argument(
        "--pcmintime",
        "-pt1",
        help="(optional)start date of power correction period, in decimal years YYYY.yy or DD/MM/YYYY format. If not provided use mintime",
    )
    parser.add_argument(
        "--pcmaxtime",
        "-pt2",
        help="(optional) end date of power correction period, in decimal years YYYY.yy or DD/MM/YYYY format. If not provided use maxtime ",
    )
    start_time = time.time()

    params = parser.parse_args()
    parquet_glob = f"{params.griddir}/**/*.parquet"
    with open(params.griddir + "grid_meta.json", "r") as f:
        grid_meta = json.load(f)

    params.surface_fit_dir = os.path.join(
        params.surface_fit_dir,
        grid_meta["mission"],
        f'{grid_meta["grid_name"]}_{int(grid_meta["bin_size"]/1000)}km_{grid_meta["dataset"]}',
    )

    if os.path.exists(params.surface_fit_dir):
        if (
            params.surface_fit_dir != "/" and grid_meta["mission"] in params.surface_fit_dir
        ):  # safety check
            log.info("Removing previous grid dir: %s ...", params.surface_fit_dir)
            response = input("Confirm removal of previous grid archive? (y/n): ").strip().lower()
            if response == "y":
                shutil.rmtree(params.surface_fit_dir)
            else:
                print("Exiting as user requested not to overwrite grid archive")
                sys.exit(0)
        else:
            sys.exit(1)
        os.makedirs(params.surface_fit_dir, exist_ok=True)

    status = surface_fit(params, parquet_glob=parquet_glob, grid_meta=grid_meta)
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, remainder = divmod(int(elapsed_time), 3600)
    minutes, seconds = divmod(remainder, 60)
    get_surface_fit_json(params, grid_meta, status, f"{hours:02}:{minutes:02}:{seconds:02}")


if __name__ == "__main__":
    main(sys.argv[1:])
