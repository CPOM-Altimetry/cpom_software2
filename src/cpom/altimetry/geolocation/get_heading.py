"""
# cpom.altimetry.geolocation.get_heading.py

utility functions for slope correction
get_heading()       :   Function to get heading angles from [x1,y1,x2,y2,..]
                        (requires at least 2 records)

"""

# ----------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------

from typing import List

import numpy as np


def get_heading(
    x_locs: List[float],
    y_locs: List[float],
) -> np.ndarray:
    """
    Function to get track headings (requires at least 2 records)

    Args:
        x_locs (List[float]): list of nadir x locations
        y_locs (List[float]): list of nadir y locations

    Returns:
        headings (np.ndarray) : list of heading angles in degrees (0..360)

    """

    # ----------------------------------------------------------------------
    # Define variables
    # ----------------------------------------------------------------------

    num_records = len(x_locs)
    if num_records < 2:
        return np.array([np.nan])

    heading = np.full(num_records, np.nan)

    # ----------------------------------------------------------------------
    # Calculate heading for each record
    # ----------------------------------------------------------------------

    for record in range(num_records):
        if (
            record == num_records - 1
        ):  # if it's the final record, copy the preceding dx and dy values
            dx = x_locs[record] - x_locs[record - 1]
            dy = y_locs[record] - y_locs[record - 1]

        else:  # else, compute delta x and delta y to next record
            dx = x_locs[record + 1] - x_locs[record]
            dy = y_locs[record + 1] - y_locs[record]

        heading_inner = np.rad2deg(np.arctan(dx / dy))  # compute heading inner angle

        if dy < 0:  # if heading vector orientated into Q2 or Q3 then add 180 degrees
            heading[record] = heading_inner + 180

        elif (
            dy > 0 > dx
        ):  # if heading vector orientated into Q4 then add 360 degrees to make positive
            heading[record] = heading_inner + 360

        else:
            heading[record] = heading_inner

    return heading
