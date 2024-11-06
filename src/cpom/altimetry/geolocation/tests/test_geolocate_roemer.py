""" cpom.altimetry.geolocation.tests.test_geolocate_roemer.py

# pytest functions for geolocate_roemer.py
"""

import logging

import numpy as np

from cpom.altimetry.geolocation.geolocate_roemer import geolocate_roemer
from cpom.dems.dems import Dem

log = logging.getLogger(__name__)


def test_geolocate_roemer():
    """pytest function for geolocate_roemer()

    This function inputs 3 nadir locations from a CS2 L1b track over Greenland
    (CS_LTA__SIR_LRM_1B_20200930T235609_20200930T235758_E001.nc) and checks that the
    calculated POCA location is as expected when compared to those calculated by
    CryoTEMPO Baseline-D.

    """

    # Define the inputs required for the Roemer slope correction

    # CS2 L1b derived inputs from a track over Greenland
    # CS_LTA__SIR_LRM_1B_20200930T235609_20200930T235758_E001.nc (first 3 measurements)
    lats = np.array([79.6516444, 79.6488579, 79.6460713])
    lons = np.array([315.179219, 315.1759333, 315.1726492])
    altitudes = np.array([732731.089, 732730.655, 732730.22])
    surface_type = np.array([1, 1, 1])
    geo_corrected_tracker_range = np.array([730515.98046547, 730515.24852219, 730515.97791724])
    retracker_correction = np.array([-8.57509287, -9.14679988, -11.11756228])

    # mask which can be used to exclude points that have failed previous quality tests
    # in this case allow all points
    points_to_include = np.array([True, True, True])

    # Define the DEM used for the Roemer slope correction
    # This is a 100m resolution DEM
    thisdem = Dem("arcticdem_100m_greenland_v4.1_zarr")

    # Configuration dictionary
    config = {
        "roemer_geolocation": {
            "fine_grid_sampling": 10,
            "max_poca_reloc_distance": 6600,
            "range_window_lower_trim": 0,
            "range_window_upper_trim": 0,
            "median_filter": False,
            "median_filter_width": 7,
            "reject_outside_range_window": True,
            "use_sliding_window": False,
        },
        "instrument": {
            "across_track_beam_width": 15000,  # meters
            "pulse_limited_footprint_size": 1600,  # meters
            "ref_bin_index": 64,  # reference bin index. Example from CS2 LRM
            "range_bin_size": 0.468425715625,  # c/(2*chirp_bandwidth), in meters for CS2
            "num_range_bins": 128,  # CS2 LRM
        },
    }

    # Run Roemer slope correction

    (height, lat_poca, lon_poca, slope_ok, relocation_distance) = geolocate_roemer(
        lats,  # ndarray of latitude values in degs N
        lons,  # ndarray of longitude values in degs E
        altitudes,  # ndarray of altitude values in m
        thisdem,  # Standard resolution DEM class object for initial POCA search
        thisdem,  # Fine resolution DEM for final POCA search (best to use same DEM as standard)
        # for speed and memory. This will be automatically sub-sampled to a finer res.
        config,  # config dictionary of this algorithm
        surface_type,  # surface type of each measurement (1=grounded ice)
        geo_corrected_tracker_range,  # tracker range (m) corrected for geophysical corrections
        # but NOT retracked
        retracker_correction,  # retracker correction to range (m)
        points_to_include,  # boolean array of points to include. ie exclude (==False) performing
        # the slope correction on points that have previously failed QC steps
        # or retracking
    )

    log.info("slope_ok %s", str(slope_ok))
    log.info("heights %s", str(height))
    log.info("lat_poca %s", str(lat_poca))
    log.info("lon_poca %s", str(lon_poca))
    log.info("relocation_distance %s", str(relocation_distance))

    # Check that Roemer slope correction outputs are as expected, when compared
    # to those output by CryoTEMPO Baseline-D
    assert np.all(slope_ok), "slope_ok flag should be True for all locations in this test"
    assert np.all(
        np.isclose(lat_poca, [79.63436486, 79.63162079, 79.62870034])
    ), "lat_poca values do not match expected POCA"
    assert np.all(
        np.isclose(lon_poca, [-44.76884955, -44.77399924, -44.78932317])
    ), "lon_poca values do not match expected POCA"
    assert np.all(
        np.isclose(relocation_distance, [2144.479, 2123.346, 2038.408])
    ), "relocation_distance values do not match expected POCA"
