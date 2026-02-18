#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Update sea ice portal timeseries

Input:

/cpnet/altimetry/seaice/<CS2>/<arco>/archive/\
    timeseries_<basin 0:17>_<thickness,volume,mass>.txt

Output:

</Users/alanmuir/Sites/seaice/sidata>/<cs2>/<arco>/\
    timeseries_<basin 0:17>_<thickness,volume,mass>.csv


Author: Alan Muir (MSSL)
Date: 2024
Copyright: UCL/MSSL/CPOM

"""

import os
import sys
from os.path import exists


def update_timeseries(
    sidata_dir: str, mission: str, area: str, select_basin: str | int | None
) -> None:
    """Update sea ice portal timeseries.

    Args:
        sidata_dir (str): Path to the output directory for sea ice data.
        mission (str): Mission name (e.g., 'cs2').
        area (str): Area name (e.g., 'arco').
        select_basin (str | int | None): Specific basin index to update. If None, all basins are
        updated.
    """
    for basin in range(0, 18):
        if select_basin is not None:
            if basin != int(select_basin):
                continue
        print("Updating timeseries for basin ", basin)

        # Read thickness timeseries file

        timeseries_types = ["thickness", "volume", "mass"]

        for ts_type in timeseries_types:
            # <Date YYYYMMDD> <Mean Sea Ice Thickness in Metres of Basin>
            # 20101115    1.0747     2046.47
            # 20101215    1.2638   154510.51

            filename = (
                f"/cpnet/altimetry/seaice/{mission.upper()}/{area}/archive/"
                f"timeseries_{basin:02d}_{ts_type}.txt"
            )
            print(f"Reading: {filename}")
            if not exists(filename):
                print(filename, " not found")
                continue

            days = []
            months = []
            years = []
            vals = []

            # read current year list
            try:
                with open(filename, "r", encoding="utf-8") as file:
                    content = file.readlines()
                    for entry in content:
                        item = entry.split()
                        if ts_type == "thickness":  # 3 columns: Date thickness extent
                            if len(item) != 3:
                                continue
                        if ts_type in ("volume", "mass"):  # 2 columns: Date volume/mass
                            if len(item) != 2:
                                continue
                        datestr = item[0]
                        y = datestr[0:4]
                        m = datestr[4:6]
                        d = datestr[6:8]
                        thickness = float(item[1])

                        days.append(d)
                        months.append(m)
                        years.append(y)
                        vals.append(thickness)
            except IOError:
                pass

            for i, day in enumerate(days):
                print(f"{day}/{months[i]}/{years[i]} {vals[i]}")

            filename = f"{sidata_dir}/{mission}/{area}/timeseries_{basin}_{ts_type}.csv"
            if not os.path.isdir(f"{sidata_dir}/{mission}/{area}"):
                try:
                    os.makedirs(f"{sidata_dir}/{mission}/{area}")
                except OSError:
                    sys.exit(f"Can not create dir: {sidata_dir}/{mission}/{area}")

            with open(filename, "w", encoding="utf-8") as file:
                file.write(f"Date,{ts_type},January_{ts_type},October_{ts_type},April_{ts_type}\n")
                season = 1
                last_month = None
                for i, d in enumerate(days):
                    if last_month is not None:
                        if (int(months[i]) - int(last_month)) > 3:
                            season += 1
                            file.write(
                                f"{years[i]}-{int(months[i]):02d}-{int(d):02d},"
                                "null,null,null,null\n"
                            )

                jan_val = "null"
                oct_val = "null"
                apr_val = "null"
                if int(months[i]) == 1:
                    jan_val = f"{vals[i]}"
                if int(months[i]) == 10:
                    oct_val = f"{vals[i]}"
                if int(months[i]) == 4:
                    apr_val = f"{vals[i]}"
                file.write(
                    f"{years[i]}-{int(months[i]):02d}-{int(d):02d},{vals[i]},"
                    f"{jan_val},{oct_val},{apr_val}\n"
                )
                last_month = months[i]

            print(filename, " written")


# ---------------------------------------------------------------------
#  Module Unit tests
# ---------------------------------------------------------------------
if __name__ == "__main__":
    import argparse  # for command line arguments

    # initiate the command line parser
    parser = argparse.ArgumentParser()

    # add long and short command line arguments
    parser.add_argument("--mission", "-m", help="mission")
    parser.add_argument("--area", "-a", help="arco or anto")
    parser.add_argument("--basin", "-b", help="basin number to process (0..17), def=process all")

    parser.add_argument(
        "--outdir",
        "-od",
        help=(
            "directory to store output .csv files"
            " as <mission=cs2>/<arco,anto>/timeseries_<basin 0:17>_<thickness,volume,mass>.csv"
        ),
        default="/Users/alanmuir/Sites/seaice/sidata",
    )

    # read arguments from the command line
    args = parser.parse_args()

    if not args.mission:
        sys.exit("--mission missing")
    if not args.area:
        sys.exit("--area missing, = arco, anto")

    update_timeseries(args.outdir, args.mission, args.area, args.basin)
