"""pytests for clev2er.utils.areas.areas.py
"""


import pytest

from cpom.areas.areas import Area, list_all_area_definition_names_only

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
