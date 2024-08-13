"""
test of cpom.roughness.roughness
"""

import numpy as np
import pytest

from cpom.areas.area_plot import Polarplot
from cpom.areas.areas import Area
from cpom.roughness.roughness import Roughness, roughness_list

pytestmark = pytest.mark.requires_external_data


def test_roughness():
    """test loading all roughness scenarios"""
    # Try loading all roughness scenarios
    for scenario in roughness_list:
        try:
            _ = Roughness(scenario)
        except IOError:
            assert False, f"{scenario} could not be initialized"


def test_roughness_ant():
    """test Antarctic roughness scenarios"""
    this_roughness = Roughness("rema_100m_900ws_roughness_zarr")

    # Test the roughness for locations in Lake Vostok: -77.5, 106

    lats = np.array([-77.5])
    lons = np.array([106])

    # roughness = this_roughness.interp_slope_from_lat_lon(lats, lons)
    roughness = this_roughness.interp_roughness(lats, lons, method="linear", xy_is_latlon=True)

    # Test values returned are not nan
    assert np.count_nonzero(~np.isnan(roughness)) == len(lats), "Nan values returned"

    assert (
        roughness > 0.01
    ).sum() == 0, " Should not have roughness > 0.01 at this Vostok location"
    assert (roughness < 0.0).sum() == 0, " Should not have roughness < 0"


@pytest.mark.parametrize("slope_name", ["arcticdem_100m_900ws_roughness_zarr"])
def test_roughness_grn(slope_name):
    """test Greenland roughness scenarios"""
    this_roughness = Roughness(slope_name)

    # Test the roughness for locations in Greenland: -77.5, 106

    lats = np.array([76.41])
    lons = np.array([-39.59])

    roughness = this_roughness.interp_roughness(lats, lons, method="linear", xy_is_latlon=True)

    # Test values returned are not nan
    assert np.count_nonzero(~np.isnan(roughness)) == len(lats), "Nan values returned"

    assert (
        roughness > 0.2
    ).sum() == 0, " Should not have roughness > 0.2 at this Greenland location"
    assert (roughness < 0.0).sum() == 0, " Should not have roughness < 0"


@pytest.mark.plots  # only run if 'pytest -m plots' is used
def test_slope_map_vostok():
    """test Antarctic roughness scenarios"""
    this_roughness = Roughness("rema_100m_900ws_roughness_zarr")

    thisarea = Area("vostok")
    lon_step = 0.01
    lat_step = 0.01
    # Generate the grid of points
    lon_values = np.arange(thisarea.minlon, thisarea.maxlon + lon_step, lon_step)
    lat_values = np.arange(thisarea.minlat, thisarea.maxlat + lat_step, lat_step)

    # Create the mesh grid
    lons, lats = np.meshgrid(lon_values, lat_values)

    # roughness = this_roughness.interp_slope_from_lat_lon(lats, lons)
    roughness = this_roughness.interp_roughness(lats, lons, method="linear", xy_is_latlon=True)

    dataset = {
        "name": "roughness",
        "units": "m",
        "lats": lats,
        "lons": lons,
        "vals": roughness,
        "apply_area_mask_to_data": False,
        "min_plot_range": 0.0,
        "max_plot_range": 0.3,
    }
    Polarplot(thisarea.name).plot_points(dataset)


@pytest.mark.parametrize("slope_name", ["rema_100m_900ws_roughness_zarr"])
@pytest.mark.plots  # only run if 'pytest -m plots' is used
def test_slope_map_ant(slope_name):
    """test Antarctic roughness scenarios"""

    this_roughness = Roughness(slope_name)

    thisarea = Area("antarctica_hs")
    lon_step = 0.1
    lat_step = 0.1
    # Generate the grid of points
    lon_values = np.arange(thisarea.minlon, thisarea.maxlon + lon_step, lon_step)
    lat_values = np.arange(thisarea.minlat, thisarea.maxlat + lat_step, lat_step)

    # Create the mesh grid
    lons, lats = np.meshgrid(lon_values, lat_values)

    # roughness = this_roughness.interp_slope_from_lat_lon(lats, lons)
    roughness = this_roughness.interp_roughness(lats, lons, method="linear", xy_is_latlon=True)

    dataset = {
        "name": "roughness",
        "units": "m",
        "lats": lats,
        "lons": lons,
        "vals": roughness,
        "apply_area_mask_to_data": True,
        "min_plot_range": 0.0,
        "max_plot_range": 1.0,
    }
    Polarplot(thisarea.name).plot_points(dataset)


@pytest.mark.parametrize("slope_name", ["rema_100m_900ws_roughness_zarr"])
@pytest.mark.plots  # only run if 'pytest -m plots' is used
def test_slope_map_spirit(slope_name):
    """test Antarctic roughness scenarios"""

    this_roughness = Roughness(slope_name)

    thisarea = Area("spirit")
    lon_step = 0.02
    lat_step = 0.02
    # Generate the grid of points
    lon_values = np.arange(thisarea.minlon, thisarea.maxlon + lon_step, lon_step)
    lat_values = np.arange(thisarea.minlat, thisarea.maxlat + lat_step, lat_step)

    # Create the mesh grid
    lons, lats = np.meshgrid(lon_values, lat_values)

    # roughness = this_roughness.interp_slope_from_lat_lon(lats, lons)
    roughness = this_roughness.interp_roughness(lats, lons, method="linear", xy_is_latlon=True)

    dataset = {
        "name": slope_name,
        "units": "m",
        "lats": lats,
        "lons": lons,
        "vals": roughness,
        "apply_area_mask_to_data": True,
        "min_plot_range": 0.0,
        "max_plot_range": 1.5,
        "plot_size_scale_factor": 0.02,
    }
    Polarplot(thisarea.name).plot_points(dataset)


@pytest.mark.parametrize(
    "slope_name",
    [
        "arcticdem_100m_900ws_roughness_zarr",
    ],
)
@pytest.mark.plots  # only run if 'pytest -m plots' is used
def test_slope_map_arctic(slope_name):
    """test Antarctic slop scenarios"""
    this_roughness = Roughness(slope_name)

    thisarea = Area("greenland_hs_is")
    lon_step = 0.05
    lat_step = 0.05
    # Generate the grid of points
    lon_values = np.arange(thisarea.minlon, thisarea.maxlon + lon_step, lon_step)
    lat_values = np.arange(thisarea.minlat, thisarea.maxlat + lat_step, lat_step)

    # Create the mesh grid
    lons, lats = np.meshgrid(lon_values, lat_values)

    # roughness = this_roughness.interp_slope_from_lat_lon(lats, lons)
    roughness = this_roughness.interp_roughness(lats, lons, method="linear", xy_is_latlon=True)

    dataset = {
        "name": slope_name,
        "units": "m",
        "lats": lats,
        "lons": lons,
        "vals": roughness,
        "apply_area_mask_to_data": True,
        "min_plot_range": 0.0,
        "max_plot_range": 1.0,
    }
    Polarplot(thisarea.name).plot_points(dataset)
