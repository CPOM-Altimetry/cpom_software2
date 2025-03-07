"""
pytest of cpom.altimetry.geolocation.get_heading.py
"""

import numpy as np

from cpom.altimetry.geolocation.get_heading import get_heading


def test_get_heading():
    """test get_heading() function"""

    # Define some test x,y locations to test all quadrants
    x_locs = [0.0, 1.0, 2.0, 1, 0]
    y_locs = [0.0, 1.0, 0.0, -1, 0]

    headings = get_heading(x_locs, y_locs)

    # Test that the same number of headings as points are returned
    assert len(headings) == len(x_locs)

    # Test that the expected heading angles are returned
    assert np.all(headings == np.asarray([45.0, 135.0, 225.0, 315.0, 315.0]))

    # Test behaviour when only one point is input (no headings possible, so
    # should return np.nan)
    x_locs = [0.0]
    y_locs = [0.0]

    headings = get_heading(x_locs, y_locs)

    assert np.isnan(headings[0])
