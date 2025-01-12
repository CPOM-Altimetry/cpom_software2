"""pytests for clev2er.utils.areas.areas.py
"""

import numpy as np
import pytest

from cpom.areas.areas import Area, list_all_area_definition_names_only

# As loading some areas requires reading large masks in $CPDATA_DIR we need to
# mark these tests as not runnable on GitHub actions
pytestmark = pytest.mark.requires_external_data


def test_bad_area_name():
    """pytest to check for handling of invalid area names"""
    with pytest.raises(ImportError):
        Area("badname")


def test_good_area_name():
    """pytest to check for handling of valid area names"""

    area_list = list_all_area_definition_names_only()

    for area in area_list:
        thisarea = area
        Area(thisarea)


def test_area_funcs():
    """pytest to check for Area class functions"""

    thisarea = Area("antarctica_is")

    lats = np.array([-75.17, -82.63])
    lons = np.array([-45.21, -44.85]) % 360

    bool_mask, n_inside = thisarea.inside_area(lats, lons)

    assert n_inside == 1, "one point is inside AIS"

    assert bool_mask[1], "second point is inside AIS"

    thisarea = Area("greenland_is")

    lats = np.array([-75.17, -82.63, 70.13])
    lons = np.array([-45.21, -44.85, -41.91]) % 360

    bool_mask, n_inside = thisarea.inside_area(lats, lons)

    assert n_inside == 1, "one point is inside greenland ice sheet"
    assert bool_mask[2], "3rd point should be True as inside Greenland"

    thisarea = Area("vostok")
    lats = np.array([-75.17, -82.63, 70.13, -78.46])
    lons = np.array([-45.21, -44.85, -41.91, 106.83]) % 360

    bool_mask, n_inside = thisarea.inside_area(lats, lons)
    assert n_inside == 1, "no points inside Vostok"
    assert bool_mask[3], "4th point should inside Vostok"
