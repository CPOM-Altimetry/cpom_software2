"""pytests of cpom.areas.area_plot3d.py"""

import pytest

from cpom.areas.area_plot3d import plot_3d_area

pytestmark = pytest.mark.plots  # test is blocking with GUI


def test_area_plot3d():
    """pytest of cpom.areas.area_plot3d.py"""

    data_set = {}

    # plot_3d_area('antarctica',data_set)
    #
    # plot_3d_area('ant_rema',data_set)
    plot_3d_area("vostok", data_set, area_overrides={})
