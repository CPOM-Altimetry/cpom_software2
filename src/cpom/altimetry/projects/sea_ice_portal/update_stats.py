#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Update sea ice portal stats

Author: Alan Muir (MSSL/UCL)
Date: 2022
Copyright: UCL/MSSL/CPOM. Not to be used outside CPOM/MSSL without permission of author

"""

import logging
import sys

log = logging.getLogger(__name__)


def update_availability_database(sidata_dir, mission, area, year, month, count):
    """
    save database as <sidata_dir>/mission/availability.csv
    """
    filename = f"{sidata_dir}/{mission}/{area}/availability.csv"

    year_list = [[0 for m in range(1, 13)] for i in range(1990, 2030)]

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
                    if 1990 <= y <= 2030:
                        year_list[y - 1990][m - 1] = c
    except IOError:
        pass

    # Add new entry to year_list

    year_list[year - 1990][month - 1] = count

    # Write year list
    with open(filename, "w", encoding="utf-8") as file:
        file.write("Date,nvals\n")
        for y in range(1990, 2030):
            for m in range(0, 12):
                if year_list[y - 1990][m] > 0:
                    file.write(f"{y}-{m + 1:02d}-01,{year_list[y - 1990][m]}\n")
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
