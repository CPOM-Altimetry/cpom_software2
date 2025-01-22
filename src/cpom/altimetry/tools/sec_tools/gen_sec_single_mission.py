#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""cpom.altimetry.tools.sec_tools.gen_sec_single_mission.py

# Purpose

Controller for generating single mission surface elevation change (SEC) data.

Uses a run control file (rcf) to configure the following steps

    1. gridding of L2 altimetry data for a mission as per grid_altimetry_data.py

# Examples

Grid CryoTEMPO Baseline C001 over GIS for all 14 years (2010-2024)

``` 
gen_sec_single_mission.py --rcf_filename tests/rcfs/grn_cs2_cryotempo_c.rcf --regrid_mission
```
This took: 15mins (MSSLXBD server)

Update a single year in the grid, keeping other years

``` 
gen_sec_single_mission.py --rcf_filename tests/rcfs/grn_cs2_cryotempo_c.rcf --update_year 2015
```

"""

import argparse
import logging
import os
import sys

import yaml

from cpom.altimetry.tools.sec_tools.compute_stats_per_gridcell import (
    attach_xy_latlon,
    compute_stats_per_cell,
    plot_variable,
)
from cpom.altimetry.tools.sec_tools.grid_altimetry_data import grid_dataset
from cpom.logging_funcs.logging import set_loggers

log = logging.getLogger(__name__)


def main(args):
    """
    Main entry point for the script.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Controller for generating single mission surface elevation change (SEC) data. "
        )
    )

    parser.add_argument(
        "--debug",
        "-d",
        help="Output debug log messages to console",
        required=False,
        action="store_true",
    )

    parser.add_argument(
        "--max_files",
        "-mf",
        help="Restrict number of input L2 files (int)",
        required=False,
        type=int,
    )

    parser.add_argument(
        "--no_confirm",
        "-nc",
        help=(
            "when regridding, do not ask for user confirmation to delete previous grid archive."
            "By default the user will be prompted before grid archive for this scenario is "
            "overwritten."
        ),
        action="store_true",
    )

    parser.add_argument(
        "--regrid_mission",
        "-rg",
        help="regrid whole mission: removes previously gridded data",
        action="store_true",
    )

    parser.add_argument("--rcf_filename", "-r", help="path of run control file", required=True)

    parser.add_argument(
        "--update_year",
        "-y",
        help=(
            "YYYY : update the gridded data archive with a specific year YYYY."
            "Example: --update_year 2015"
        ),
        type=int,
        default=None,
    )

    args = parser.parse_args(args)

    # ----------------------------------------------------------------------------------------------
    # Create a logger for this tool to output to console
    # ----------------------------------------------------------------------------------------------

    default_log_level = logging.INFO
    if args.debug:
        default_log_level = logging.DEBUG
    logfile = "/tmp/grid.log"
    set_loggers(
        log_file_info=logfile[:-3] + "info.log",
        log_file_warning=logfile[:-3] + "warning.log",
        log_file_error=logfile[:-3] + "errors.log",
        log_file_debug=logfile[:-3] + "debug.log",
        log_format="%(levelname)s : %(asctime)s %(name)s : %(message)s",
        default_log_level=default_log_level,
    )

    # ----------------------------------------------------------------------------------------------
    # Read run control file (rcf)
    # ----------------------------------------------------------------------------------------------
    # Read the raw YAML file as a string
    with open(args.rcf_filename, "r", encoding="utf-8") as f:
        raw_yaml = f.read()

    # 2) Substitute environment variables in the string
    #   we denote env vars with a simple syntax like: ${MY_VAR}
    for key, value in os.environ.items():
        placeholder = f"${{{key}}}"
        raw_yaml = raw_yaml.replace(placeholder, value)

    # 3) Now load the substituted YAML string
    config = yaml.safe_load(raw_yaml)

    if args.max_files:
        config["max_files"] = args.max_files

    # ----------------------------------------------------------------------------------------------
    # Optional Gridding Stage
    # ----------------------------------------------------------------------------------------------

    if args.regrid_mission or args.update_year:

        grid_dict = grid_dataset(
            config,
            regrid=args.regrid_mission,
            update_year=args.update_year,
            confirm_regrid=not args.no_confirm,
        )

        log.info("Elapsed time for gridding: %s", grid_dict["gridding_time"])

        # 1) Compute stats (mean_elev, coverage_yrs) for each cell in parallel

        df, grid_obj, grid_meta = compute_stats_per_cell(grid_dict["grid_path"], max_workers=8)

        if df.empty:
            log.error("No data returned by compute_stats_per_cell")
            sys.exit(1)

        if grid_obj is not None:
            df = attach_xy_latlon(df, grid_obj)
            plot_variable(
                df,
                grid_meta,
                var_name="mean_elev",
                plot_file=os.path.join(grid_dict["grid_path"], "mean_elev.png"),
            )
            plot_variable(
                df,
                grid_meta,
                var_name="coverage_yrs",
                plot_file=os.path.join(grid_dict["grid_path"], "coverage_yrs.png"),
            )
            plot_variable(
                df,
                grid_meta,
                var_name="n_elev",
                plot_file=os.path.join(grid_dict["grid_path"], "measurements_per_gridcell.png"),
            )
        else:
            log.warning("No grid info (grid_obj) available, cannot do lat/lon. Skipping plot.")

    else:
        log.info("No gridding options chosen")


if __name__ == "__main__":
    main(sys.argv[1:])
