"""
test of cpom.slopes.slopes
"""

import numpy as np
import pytest

from cpom.areas.area_plot import Polarplot
from cpom.areas.areas import Area
from cpom.slopes.slopes import Slopes, slope_list

pytestmark = pytest.mark.requires_external_data


def test_slopes():
    """test loading all slope scenarios"""
    # Try loading all slope scenarios
    for scenario in slope_list:
        _ = Slopes(scenario)


def test_slopes_ant():
    """test Antarctic slop scenarios"""
    this_slope = Slopes("rema_100m_900ws_slopes_zarr")

    # Test the slope for locations in Lake Vostok: -77.5, 106

    lats = np.array([-77.5])
    lons = np.array([106])

    # slopes = this_slope.interp_slope_from_lat_lon(lats, lons)
    slopes = this_slope.interp_slopes(lats, lons, method="linear", xy_is_latlon=True)

    # Test values returned are not nan
    assert np.count_nonzero(~np.isnan(slopes)) == len(lats), "Nan values returned"

    assert (slopes > 0.01).sum() == 0, " Should not have slopes > 0.01 at this Vostok location"
    assert (slopes < 0.0).sum() == 0, " Should not have slopes < 0"

    this_slope = Slopes("cpom_ant_2018_1km_slopes")

    # Test the slope for locations in Lake Vostok: -77.5, 106

    lats = np.array([-77.5])
    lons = np.array([106])

    # slopes = this_slope.interp_slope_from_lat_lon(lats, lons)
    slopes = this_slope.interp_slopes(lats, lons, method="linear", xy_is_latlon=True)

    # Test values returned are not nan
    assert np.count_nonzero(~np.isnan(slopes)) == len(lats), "Nan values returned"

    assert (slopes > 0.01).sum() == 0, " Should not have slopes > 0.01 at this Vostok location"
    assert (slopes < 0.0).sum() == 0, " Should not have slopes < 0"


@pytest.mark.parametrize(
    "slope_name", ["arcticdem_100m_900ws_slopes_zarr", "awi_grn_2013_1km_slopes"]
)
def test_slopes_grn(slope_name):
    """test Greenland slope scenarios"""
    this_slope = Slopes(slope_name)

    # Test the slope for locations in Greenland: -77.5, 106

    lats = np.array([76.41])
    lons = np.array([-39.59])

    slopes = this_slope.interp_slopes(lats, lons, method="linear", xy_is_latlon=True)

    # Test values returned are not nan
    assert np.count_nonzero(~np.isnan(slopes)) == len(lats), "Nan values returned"

    assert (slopes > 0.2).sum() == 0, " Should not have slopes > 0.2 at this Greenland location"
    assert (slopes < 0.0).sum() == 0, " Should not have slopes < 0"


@pytest.mark.plots  # only run if 'pytest -m plots' is used
def test_slope_map_vostok():
    """test Antarctic slope scenarios"""
    this_slope = Slopes("rema_100m_900ws_slopes_zarr")

    thisarea = Area("vostok")
    lon_step = 0.01
    lat_step = 0.01
    # Generate the grid of points
    lon_values = np.arange(thisarea.minlon, thisarea.maxlon + lon_step, lon_step)
    lat_values = np.arange(thisarea.minlat, thisarea.maxlat + lat_step, lat_step)

    # Create the mesh grid
    lons, lats = np.meshgrid(lon_values, lat_values)

    # slopes = this_slope.interp_slope_from_lat_lon(lats, lons)
    slopes = this_slope.interp_slopes(lats, lons, method="linear", xy_is_latlon=True)

    dataset = {
        "name": "slope",
        "units": "degs",
        "lats": lats,
        "lons": lons,
        "vals": slopes,
        "apply_area_mask_to_data": False,
        "min_plot_range": 0.0,
        "max_plot_range": 0.3,
    }
    Polarplot(thisarea.name).plot_points(dataset)


@pytest.mark.parametrize("slope_name", ["cpom_ant_2018_1km_slopes", "rema_100m_900ws_slopes_zarr"])
@pytest.mark.plots  # only run if 'pytest -m plots' is used
def test_slope_map_ant(slope_name):
    """test Antarctic slope scenarios"""

    this_slope = Slopes(slope_name)

    thisarea = Area("antarctica_hs")
    lon_step = 0.1
    lat_step = 0.1
    # Generate the grid of points
    lon_values = np.arange(thisarea.minlon, thisarea.maxlon + lon_step, lon_step)
    lat_values = np.arange(thisarea.minlat, thisarea.maxlat + lat_step, lat_step)

    # Create the mesh grid
    lons, lats = np.meshgrid(lon_values, lat_values)

    # slopes = this_slope.interp_slope_from_lat_lon(lats, lons)
    slopes = this_slope.interp_slopes(lats, lons, method="linear", xy_is_latlon=True)

    dataset = {
        "name": "slope",
        "units": "degs",
        "lats": lats,
        "lons": lons,
        "vals": slopes,
        "apply_area_mask_to_data": True,
        "min_plot_range": 0.0,
        "max_plot_range": 1.0,
    }
    Polarplot(thisarea.name).plot_points(dataset)


@pytest.mark.parametrize("slope_name", ["cpom_ant_2018_1km_slopes", "rema_100m_900ws_slopes_zarr"])
@pytest.mark.plots  # only run if 'pytest -m plots' is used
def test_slope_map_spirit(slope_name):
    """test Antarctic slope scenarios"""

    this_slope = Slopes(slope_name)

    thisarea = Area("spirit")
    lon_step = 0.02
    lat_step = 0.02
    # Generate the grid of points
    lon_values = np.arange(thisarea.minlon, thisarea.maxlon + lon_step, lon_step)
    lat_values = np.arange(thisarea.minlat, thisarea.maxlat + lat_step, lat_step)

    # Create the mesh grid
    lons, lats = np.meshgrid(lon_values, lat_values)

    # slopes = this_slope.interp_slope_from_lat_lon(lats, lons)
    slopes = this_slope.interp_slopes(lats, lons, method="linear", xy_is_latlon=True)

    dataset = {
        "name": slope_name,
        "units": "degs",
        "lats": lats,
        "lons": lons,
        "vals": slopes,
        "apply_area_mask_to_data": True,
        "min_plot_range": 0.0,
        "max_plot_range": 1.5,
        "plot_size_scale_factor": 0.02,
    }
    Polarplot(thisarea.name).plot_points(dataset)


@pytest.mark.parametrize(
    "slope_name", ["arcticdem_100m_900ws_slopes_zarr", "awi_grn_2013_1km_slopes"]
)
@pytest.mark.plots  # only run if 'pytest -m plots' is used
def test_slope_map_arctic(slope_name):
    """test Antarctic slop scenarios"""
    this_slope = Slopes(slope_name)

    thisarea = Area("greenland_hs_is")
    lon_step = 0.05
    lat_step = 0.05
    # Generate the grid of points
    lon_values = np.arange(thisarea.minlon, thisarea.maxlon + lon_step, lon_step)
    lat_values = np.arange(thisarea.minlat, thisarea.maxlat + lat_step, lat_step)

    # Create the mesh grid
    lons, lats = np.meshgrid(lon_values, lat_values)

    # slopes = this_slope.interp_slope_from_lat_lon(lats, lons)
    slopes = this_slope.interp_slopes(lats, lons, method="linear", xy_is_latlon=True)

    dataset = {
        "name": slope_name,
        "units": "degs",
        "lats": lats,
        "lons": lons,
        "vals": slopes,
        "apply_area_mask_to_data": True,
        "min_plot_range": 0.0,
        "max_plot_range": 1.0,
    }
    Polarplot(thisarea.name).plot_points(dataset)
