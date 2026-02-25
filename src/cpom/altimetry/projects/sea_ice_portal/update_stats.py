#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Update sea ice portal stats

Author: Alan Muir (MSSL/UCL)
Date: 2022
Copyright: UCL/MSSL/CPOM. Not to be used outside CPOM/MSSL without permission of author

"""

import logging
import os
import sys
from datetime import datetime

log = logging.getLogger(__name__)


def update_availability_database(sidata_dir, mission, area, year, month, count):
    """
    save database as <sidata_dir>/mission/ntc/area/availability.csv
    """
    this_dir = f"{sidata_dir}/{mission}/ntc/{area}"
    if not os.path.exists(this_dir):
        print(f"Creating directory: {this_dir}")
        os.makedirs(this_dir)
    filename = f"{this_dir}/availability.csv"

    if mission == "cs2":
        start_year = 2010
    elif mission == "s3a":
        start_year = 2016
    elif mission == "s3b":
        start_year = 2018
    elif mission == "env":
        start_year = 2002
    else:
        sys.exit(f"Unknown mission: {mission}")

    # Get current year
    current_year = datetime.now().year

    year_list = [[0 for m in range(1, 13)] for i in range(start_year, current_year + 1)]

    # read current year list
    try:
        with open(filename, "r", encoding="utf-8") as file:
            content = file.readlines()
            for entry in content:
                item = entry.split(",")
                if item[0] == "Date":
                    continue
                if len(item) == 2:
                    y = int(item[0][0:4])
                    m = int(item[0][5:7])
                    c = int(item[1])
                    if start_year <= y <= current_year:
                        year_list[y - start_year][m - 1] = c
    except IOError:
        pass

    # Add new entry to year_list

    year_list[int(year) - start_year][int(month) - 1] = int(count)

    # Write year list
    with open(filename, "w", encoding="utf-8") as file:
        file.write("Date,nvals\n")
        for y in range(start_year, current_year + 1):
            for m in range(0, 12):
                if year_list[y - start_year][m] > 0:
                    file.write(f"{y}-{m + 1:02d}-01,{year_list[y - start_year][m]}\n")
                else:
                    file.write(f"{y}-{m + 1:02d}-01,{0}\n")

    log.info("Updated availability list : %s", filename)
    print(f"Updated availability list : {filename}")


# -----------------------------------------------------------------------------
#  Module Unit tests
# -----------------------------------------------------------------------------
if __name__ == "__main__":

    import argparse  # for command line arguments

    # initiate the command line parser
    parser = argparse.ArgumentParser()

    # add long and short command line arguments
    parser.add_argument("--mission", "-m", help="mission")
    parser.add_argument("--year", "-y", help="year")
    parser.add_argument("--month", "-mo", help="month")
    parser.add_argument("--area", "-a", help="area")
    parser.add_argument("--count", "-c", help="count")

    # read arguments from the command line
    args = parser.parse_args()

    if not args.mission:
        sys.exit("--mission missing")
    if not args.area:
        sys.exit("--area missing")
    if not args.month:
        sys.exit("--month missing")
    if not args.count:
        sys.exit("--count missing")

    if args.year and args.month:
        update_availability_database(
            "/Users/alanmuir/Sites/seaice/sidata",
            args.mission,
            args.area,
            int(args.year),
            int(args.month),
            int(args.count),
        )
