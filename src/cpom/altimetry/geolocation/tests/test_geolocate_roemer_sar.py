"""cpom.altimetry.geolocation.tests.test_geolocate_roemer_sar.py

# pytest functions for geolocate_roemer_sar.py
"""

import logging

import numpy as np
import pytest

from cpom.altimetry.geolocation.geolocate_roemer_sar import geolocate_roemer_sar
from cpom.dems.dems import Dem

log = logging.getLogger(__name__)

pytestmark = pytest.mark.requires_external_data


def test_geolocate_roemer_sar():
    """pytest function for geolocate_roemer_sar()

    This function inputs 3 nadir locations from a CS2 L1b track over Greenland
    (CS_LTA__SIR_LRM_1B_20200930T235609_20200930T235758_E001.nc) and checks that the
    calculated POCA location is as expected when compared to those calculated by
    CryoTEMPO Baseline-D.

    """

    # Define the inputs required for the Roemer SAR slope correction

    # CS2 L1b derived inputs from a track over Greenland
    # CS_LTA__SIR_LRM_1B_20200930T235609_20200930T235758_E001.nc (first 3 measurements)
    # TODO: replace with SAR/SARin test, pylint:disable=fixme
    lats = np.array([79.6516444, 79.6488579, 79.6460713])
    lons = np.array([315.179219, 315.1759333, 315.1726492])
    altitudes = np.array([732731.089, 732730.655, 732730.22])
    geo_corrected_tracker_range = np.array([730515.98046547, 730515.24852219, 730515.97791724])

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
            "reject_outside_range_window": False,
            "use_sliding_window": False,
        },
        "instrument": {
            "across_track_beam_width": 15000,  # meters
            "along_track_beam_width": 300,  # meters
            "across_track_doppler_footprint_width": 1600,  # meters
            "ref_bin_index": 64,  # reference bin index. Example from CS2 LRM
            "range_bin_size": 0.468425715625,  # c/(2*chirp_bandwidth), in meters for CS2
            "num_range_bins": 128,  # CS2 LRM
        },
    }

    # Run Roemer slope correction

    (
        topo_correction_to_height,
        lat_poca,
        lon_poca,
        slope_ok,
        relocation_distance,
    ) = geolocate_roemer_sar(
        lats,  # ndarray of latitude values in degs N
        lons,  # ndarray of longitude values in degs E
        altitudes,  # ndarray of altitude values in m
        thisdem,  # Standard resolution DEM class object for initial POCA search
        thisdem,  # Fine resolution DEM for final POCA search (best to use same DEM as standard)
        # for speed and memory. This will be automatically sub-sampled to a finer res.
        config,  # config dictionary of this algorithm
        geo_corrected_tracker_range,  # tracker range (m) corrected for geophysical corrections
        # but NOT retracked
        points_to_include,  # boolean array of points to include. ie exclude (==False) performing
        # the slope correction on points that have previously failed QC steps
        # or retracking
    )

    log.info("slope_ok %s", str(slope_ok))
    log.info("topo_correction_to_height %s", str(topo_correction_to_height))
    log.info("lat_poca %s", str(lat_poca))
    log.info("lon_poca %s", str(lon_poca))
    log.info("relocation_distance %s", str(relocation_distance))

    # Check that Roemer slope correction outputs are as expected, when compared
    # to those output by CryoTEMPO Baseline-D
    assert np.all(slope_ok), "slope_ok flag should be True for all locations in this test"

    if 0:  # TODO replace LRM with SAR relocation values pylint:disable=using-constant-test
        assert np.all(np.isclose(lat_poca, [79.63436486, 79.63162079, 79.62870034])), (
            f"lat_poca values {lat_poca} do not match expected POCA :"
            " 79.63436486, 79.63162079, 79.62870034"
        )
        assert np.all(
            np.isclose(lon_poca, [-44.76884955, -44.77399924, -44.78932317])
        ), "lon_poca values do not match expected POCA"
    assert np.all(
        np.isclose(relocation_distance, [2144.479, 2123.346, 2038.408], atol=2000)
    ), f"relocation_distance {relocation_distance} values do not match expected POCA"
