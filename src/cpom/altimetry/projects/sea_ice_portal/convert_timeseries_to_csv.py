#!/usr/bin/env python3
"""Convert timeseries txt files to csv files.

Requirements:

Input txt files:

/cpnet/altimetry/seaice/<CS2>/<arco>/archive/\
    timeseries_<basin 0:17>_<thickness,volume,mass>.txt

format:
<YYYYMMDD> <value of thickness or volume or mass> <Sea ice extent in km2>
20101115    1.2567  9238672.63
20101215    1.2944  11382515.17
...


Output csv files:

<outdir>/<cs2>/<arco>/\
    timeseries_<basin 0:17>_<thickness,volume,mass>.csv

format of csv files:

Date,Value,Sea_ice_extent_km2
2010-11-15,1.2567,9238672.63
2010-12-15,1.2944,11382515.17
"""

import argparse
import os
import sys
from typing import Optional


def convert_timeseries(
    outdir: str, mission: str, area: str, select_basin: Optional[int | str] = None
) -> None:
    """Convert sea ice portal txt timeseries files to csv format.

    Args:
        outdir (str): Path to the output directory for sea ice data.
        mission (str): Mission name (e.g., 'cs2').
        area (str): Area name (e.g., 'arco' or 'anto').
        select_basin (Optional[int | str]): Specific basin index (0-17)
            to process, or None to process all basins.
    """
    timeseries_types = ["thickness", "volume", "mass"]
    mission_upper = mission.upper()

    for basin in range(0, 18):
        if select_basin is not None and basin != int(select_basin):
            continue

        for ts_type in timeseries_types:
            filename = (
                f"/cpnet/altimetry/seaice/{mission_upper}/{area}/archive/"
                f"timeseries_{basin:02d}_{ts_type}.txt"
            )

            if not os.path.exists(filename):
                print(f"{filename} not found")
                continue

            print(f"Reading: {filename}")
            output_dir = os.path.join(outdir, mission, area)
            os.makedirs(output_dir, exist_ok=True)
            output_filename = os.path.join(output_dir, f"timeseries_{basin}_{ts_type}.csv")

            try:
                with open(filename, "r", encoding="utf-8") as file:
                    lines = file.readlines()
            except IOError as error:
                print(f"Error reading {filename}: {error}")
                continue

            try:
                with open(output_filename, "w", encoding="utf-8") as out_file:
                    out_file.write("Date,Value,Sea_ice_extent_km2\n")
                    count = 0
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            datestr = parts[0]
                            value = parts[1]
                            extent = parts[2] if len(parts) >= 3 else "NaN"

                            if not datestr.isdigit() or len(datestr) != 8:
                                continue

                            year = datestr[0:4]
                            month = datestr[4:6]
                            day = datestr[6:8]
                            formatted_date = f"{year}-{month}-{day}"

                            out_file.write(f"{formatted_date},{value},{extent}\n")
                            count += 1
                print(f"Wrote {count} records to {output_filename}")
            except IOError as error:
                print(f"Error writing {output_filename}: {error}")


if __name__ == "__main__":
    DESC = "Convert timeseries txt files to csv files."
    parser = argparse.ArgumentParser(description=DESC)
    parser.add_argument("--mission", "-m", help="mission name, e.g. cs2")
    parser.add_argument("--area", "-a", help="area name, e.g. arco or anto")
    parser.add_argument(
        "--basin",
        "-b",
        help="basin number to process (0..17), def=process all",
    )
    parser.add_argument(
        "--outdir",
        "-o",
        "-od",
        help="directory to store output .csv files",
        default="/Users/alanmuir/Sites/seaice/sidata",
    )

    args = parser.parse_args()

    if not args.mission:
        sys.exit("--mission missing")
    if not args.area:
        sys.exit("--area missing, = arco, anto")

    convert_timeseries(args.outdir, args.mission, args.area, args.basin)
