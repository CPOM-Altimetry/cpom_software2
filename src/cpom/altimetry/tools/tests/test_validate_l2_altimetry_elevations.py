"""pytest for cpom.altimetry.tools.validate_l2_altimetry_elevations.py"""

import argparse
import os
import tempfile
from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from netCDF4 import Dataset  # pylint: disable=E0611

# import h5py

# from cpom.dems.dems import Dem
from cpom.altimetry.tools.validate_l2_altimetry_elevations import (
    ProcessData,
    get_default_variables,
    get_elev_differences,
    get_files_in_dir,
    get_variable,
    correct_elevation_using_slope,
)


# ------------------#
# Test get_variable #
# -------------------#
@pytest.fixture(name="mock_args")
def mock_args_fixture():
    "Fixture to mock command line arguments"
    return argparse.Namespace(
        radius=20.0,
        maxdiff=10.0,
        compare_to_self=False,
        cryotempo_modes=None,
        beams=["beam1", "beam2"],
        dem="rema_ant_200m",
    )


@pytest.fixture(name="netcdf_file")
def netcdf_file_fixture():
    "Mocked NetCDF file"
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".nc")  # pylint: disable=R1732

    with Dataset(temp_file.name, "w", format="NETCDF4") as nc:
        nc.createDimension("id", 5)

        var1 = nc.createVariable("elev", "f8", ("id",))
        var1[:] = np.array([1.1, 2.2, 3.3, 4.4, 5.5], dtype=np.float64)

        group = nc.createGroup("group1")
        var2 = group.createVariable("elevation", "f8", ("id",))
        var2[:] = np.array([6.6, 7.7, 8.8, 9.9, 10.0], dtype=np.float64)

    yield temp_file.name


@pytest.mark.parametrize(
    "variable_path, expected_array",
    [
        ("elev", np.array([1.1, 2.2, 3.3, 4.4, 5.5])),
        ("group1/elevation", np.array([6.6, 7.7, 8.8, 9.9, 10.0])),
    ],
)
def test_get_variable_happy_path(netcdf_file, variable_path, expected_array):
    """Test get_variable: Valid returned array when variable exist"""

    with Dataset(netcdf_file, "r") as nc:
        array = get_variable(nc, variable_path)
        assert np.allclose(array, expected_array)


def test_get_variable_exceptions(netcdf_file):
    """Test get_variable: Raises Index error when variable or group doesn't exist"""
    with Dataset(netcdf_file, "r") as nc:
        with pytest.raises(IndexError, match="NetCDF parameter or group missing_var not found"):
            get_variable(nc, "missing_var")
        with pytest.raises(IndexError, match="NetCDF parameter or group wrong not found"):
            get_variable(nc, "wrong/group/var")


# -----------------------#
# Test get_files_in_dir #
# -----------------------#


@pytest.fixture(name="test_directory")
def test_directory_fixture():
    """Pytest Fixture : Selection of valid and invalid datetime path strings"""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        (temp_path / "20240111T.nc").touch()  # YYYYMMDDT
        (temp_path / "20240111_.nc").touch()  # YYYYMMDD_
        (temp_path / "20240111_1111111111.nc").touch()  # YYYYMMDD_
        (temp_path / "240111_.nc").touch()  # YYMMDD_
        (temp_path / "20240111120000.nc").touch()  # YYYYMMDDHHMMSS

        # Invalid file paths
        (temp_path / "20240111T120000_20240113T130000.h5").touch()  # Different file extension
        (temp_path / "20240101.nc").touch()  # YYYYMMDD without _or T
        (temp_path / "202401011_20240102.nc").touch()  # YYYYMMDD!_YYYYMMDD
        (temp_path / "20240111T120000.eg.nc").touch()  # extra prefix
        (temp_path / "2024011.example.h5").touch()  # extra prefix
        (temp_path / "2024.01.11.nc").touch()  # YYYY.MM.DD
        (temp_path / "2024-01-11.nc").touch()  # YYYY-MM-DD
        yield temp_path


def test_get_files_in_dir(test_directory):
    """Test get_files_in_dir : Returns only path strings for with valid datetime
    formats in there basepath (YYYYMMDDT | YYYYMMDD_ | YYMMDD_ |YYYYMMDDHHMMSS)"""
    files = get_files_in_dir(str(test_directory), "2024", "01", "nc")

    expected_files = {
        str(test_directory / "20240111T.nc"),
        str(test_directory / "20240111_.nc"),
        str(test_directory / "20240111_1111111111.nc"),
        str(test_directory / "240111_.nc"),
        str(test_directory / "20240111120000.nc"),
    }

    assert set(map(str, files)) == expected_files


# ---------------------------#
# Test get_default_variables#
# ---------------------------#
@pytest.mark.parametrize(
    "test_basename, expected_config",
    [
        (
            f"{os.environ['CPDATA_DIR']}/SATS/RA/CRY/L2I/SIN/2020/01/01/ \
            CS_OFFL_SIR_SINI2__20200131T235329_20200131T235346_D001.nc",
            {
                "lat_nadir": "lat_20_ku",
                "lon_nadir": "lon_20_ku",
                "lat": "lat_poca_20_ku",
                "lon": "lon_poca_20_ku",
                "elev": "height_1_20_ku",
            },
        ),
        (
            f"{os.environ['CPDATA_DIR']}/SATS/RA/CRY/L2I/LRM/ 2020/01/01/ \
                CS_OFFL_SIR_LRMI2__20200131T235859_20200201T001028_D001.nc",
            {
                "lat_nadir": "lat_20_ku",
                "lon_nadir": "lon_20_ku",
                "lat": "lat_poca_20_ku",
                "lon": "lon_poca_20_ku",
                "elev": "height_3_20_ku",
            },
        ),
        (
            f"{os.environ['CPDATA_DIR']}/SATS/RA/ENV/L2/FDR4ALT/TDP/greenland/land_ice/Cycle_105/ \
            EN1_F4A_ALT_TDP_LI_GREENL_105_0860_20110823T190810_20110823T190837_V01.nc",
            {
                "lat_nadir": "expert/ice_sheet_lat_poca",
                "lon_nadir": "expert/ice_sheet_lon_poca",
                "lat": "expert/latitude",
                "lon": "expert/longitude",
                "elev": "expert/ice_sheet_elevation_ice1_roemer",
            },
        ),
    ],
)
def test_get_default_variables(test_basename, expected_config):
    """Test get_default_variables : Returns the expected dictionary config
    for passed input data directory path"""
    config = get_default_variables(test_basename)
    assert config == expected_config


# --------------------------#
# Test get_elev_differences #
# --------------------------#
def test_get_elev_differences(mock_args):
    """Test get_elev_differences : Returns valid elevation, dh, and seperation distance
    values, within the expected radius and maximum height"""
    dtype = [("x", "f8"), ("y", "f8"), ("h", "f8")]

    laser_points = np.array(
        [
            (0.0, 0.0, 100.0),  # Point A
            (10.0, 0.0, 110.0),  # Point B
            (0.0, 10.0, 120.0),  # Point C
        ],
        dtype=dtype,
    )

    altimeter_points = np.array(
        [
            (0.0, 0.0, 105.0),
            (10.0, 0.0, 125.0),
            (100.0, 100.0, 130.0),
            (0.0, 10.0, 115.0),
            (0.0, 10.0, 111.0),
        ],
        dtype=dtype,
    )

    result = get_elev_differences(mock_args, laser_points, altimeter_points)
    for i, dh in enumerate(result["dh"]):
        ref_x, ref_y = result["reference_x"][i], result["reference_y"][i]
        alt_x, alt_y = result["x"][i], result["y"][i]

        distance = np.sqrt((ref_x - alt_x) ** 2 + (ref_y - alt_y) ** 2)
        assert distance <= mock_args.radius
        assert dh <= mock_args.maxdiff

    more_complex_sep_dist = np.sqrt((10 - 0) ** 2 + (0 - 10) ** 2)
    expected_result = {
        "dh": [5.0, -5.0, 5.0, 5.0, -5.0, 1.0, -9.0],
        "sep_dist": [
            0.0,
            10.0,
            more_complex_sep_dist,
            more_complex_sep_dist,
            0.0,
            more_complex_sep_dist,
            0.0,
        ],
        "x": [0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0],
        "y": [0.0, 0.0, 0.0, 10.0, 10.0, 10.0, 10.0],
        "h": [105.0, 105.0, 125.0, 115.0, 115.0, 111.0, 111.0],
        "reference_x": [0.0, 10.0, 0.0, 10.0, 0.0, 10.0, 0.0],
        "reference_y": [0.0, 0.0, 10.0, 0.0, 10.0, 0.0, 10.0],
        "reference_h": [100.0, 110.0, 120.0, 110.0, 120.0, 110.0, 120.0],
    }

    assert result == expected_result


# -----------------------------------#
# Test correct_elevation_using_slope #
# -----------------------------------#
class MockDem:
    """Mocked Dem class that returns None"""

    def __init__(self, dem_path):
        self.dem_path = dem_path

    def interp_dem(self, x, y):  # pylint: disable=W0613
        """Mocked interp_dem function return empty array"""
        return np.array([])

    def empty(self, a):
        """Pointless, to pass pylint"""
        return a


def test_slope_correction(mock_args):
    """Test slope_correction : Returns elevations and dh values that have been
    slope corrected to resolve differences in recording location. Test validates
    correction in the right direction"""
    test_data = {
        "x": np.array([1, 2, 3, 4]),
        "y": np.array([10, 20, 30, 40]),
        "h": np.array([515, 610, 695, 400]),
        "reference_x": np.array([5, 6, 7, 8]),
        "reference_y": np.array([50, 60, 70, 80]),
        "reference_h": np.array([500, 600, 700, 410]),
        "dh": np.array([15, 10, -5, -10]),
    }
    dem1 = np.array([520, 615, 699, 402])
    dem2 = np.array([512, 605, 698, 407])

    with patch(
        "cpom.altimetry.tools.validate_l2_altimetry_elevations.Dem", MockDem
    ):  # Replace Dem with MockDem
        with patch.object(MockDem, "interp_dem", side_effect=[dem1, dem2]):  # Mock iterp_dem method
            result = correct_elevation_using_slope(test_data, mock_args, MagicMock(), "")

    assert np.all(result["reference_h"] == [508, 610, 701, 405])
    assert np.all(result["dh"] == [7, 0, -6, -5])
    for col in ["x", "y", "h", "reference_x", "reference_y"]:
        assert np.all(result[col] == test_data[col])


def test_slope_correction_missing_data(mock_args):
    """Test slope correction : Masks out data where NaN values introduced through
    dem interpolation"""
    test_data = {
        "x": np.array([1, 2, 3, 4]),
        "y": np.array([10, 20, 30, 40]),
        "h": np.array([515, 610, 695, 400]),
        "reference_x": np.array([5, 6, 7, 8]),
        "reference_y": np.array([50, 60, 70, 80]),
        "reference_h": np.array([500, 600, 700, 410]),
        "dh": np.array([15, 10, -5, -10]),
    }
    dem1 = np.array([520, 615, 699, 402])
    dem2 = np.array([512, 605, 698, np.NaN])

    with patch(
        "cpom.altimetry.tools.validate_l2_altimetry_elevations.Dem", MockDem
    ):  # Replace Dem with MockDem
        with patch.object(MockDem, "interp_dem", side_effect=[dem1, dem2]):  # Mock iterp_dem method
            result = correct_elevation_using_slope(test_data, mock_args, MagicMock(), "")

    assert np.all(result["reference_h"] == [508, 610, 701])
    assert np.all(result["dh"] == [7, 0, -6])
    for col in ["x", "y", "h", "reference_x", "reference_y"]:
        assert np.all(result[col] == test_data[col][0:3])


# -----------------------------#
# Test Compute elevation stats #
# -----------------------------#
class DummyArea:
    """Mocked area class"""

    hemisphere = "north"

    def inside_latlon_bounds(self, lats, lons):
        """mocked inside_latlon_bounds returns all points"""
        indices = np.arange(len(lats))
        return lats, lons, indices, None  # Keep all points

    def latlon_to_xy(self, lats, lons):
        """mocked latlon_to_xy simplified conversion to numeric lat/lon * 10"""
        # Simpllified conversion to x,y
        return lats * 10, lons * 10

    def inside_mask(self, x, y):  # pylint: disable=W0613
        """mocked inside mask returns all points"""
        return np.arange(len(x)), None  # Keep all points


@pytest.fixture(name="mock_process_data")
def process_data_fixture(mock_args):
    """Mocked instance of the ProcessData Class"""
    area = DummyArea()
    logger = MagicMock()
    mock_args = Namespace(**vars(mock_args))  # Create a copy that allows modifications
    return ProcessData(mock_args, area, logger)


@pytest.fixture(name="mock_config")
def mock_config_fixture():
    "Mocked standard configuration from get_default_variables"
    return {
        "lat": "lat",
        "lon": "lon",
        "elev": "elev",
        "lat_nadir": "lat_nadir",
        "lon_nadir": "lon_nadir",
    }


# -------------------------------------------#
# Test ProcessData._get_altimetry_data_array #
# --------------------------------------------#
def mock_get_variable(nc, varname):  # pylint: disable=W0613
    """Function to return standard data of an input nc file"""
    dummy_data = {
        "lat": np.array([10, 20, 30, 40]),
        "lon": np.array([100, 110, 120, 130]),
        "elev": np.array([500, 600, 700, 800]),
        "lat_nadir": np.array([11, 21, 31, 41]),
        "lon_nadir": np.array([101, 111, 121, 131]),
        "instrument_mode": np.array([1, 2, 3, 1]),
        "extra_var": np.array([1, 2, 3, 4]),
    }
    return dummy_data.get(varname, np.array([]))


# -----------------------------------------#
# Test ProcessData.get_cryotempo_filters #
# -----------------------------------------#
@pytest.mark.parametrize(
    "cryotempo_modes, expected_mask",
    [
        (["lrm"], np.array([True, False, False, True])),
        (["sar"], np.array([False, True, False, False])),
        (["sin"], np.array([False, False, True, False])),
        (["lrm", "sar"], np.array([True, True, False, True])),
        (["lrm", "sin"], np.array([True, False, True, True])),
        (["sar", "sin"], np.array([False, True, True, False])),
    ],
)
def test_get_cryotempo_filters(mock_process_data, cryotempo_modes, expected_mask):
    """Test get_cryotempo filters : Returns a masked array containing only
    the passed cryotempo modes"""
    mock_process_data.args.cryotempo_modes = cryotempo_modes
    dummy_nc = MagicMock()
    with patch(
        "cpom.altimetry.tools.validate_l2_altimetry_elevations.get_variable",
        side_effect=mock_get_variable,
    ):
        mask = mock_process_data.get_cryotempo_filters(dummy_nc, mock_process_data.args)
    assert np.all(mask == expected_mask)


# -----------------------------------------------#
# Test ProcessData.fill_empty_latlon_with_nadir
# -----------------------------------------------#
@pytest.mark.parametrize(
    "lat, lon, expected_lat, expected_lon",
    [
        (  # Case 1 : Single NaN valuea
            np.array([10, 20, 30, np.nan]),
            np.array([100, 110, 120, np.nan]),
            np.array([10, 20, 30, 41]),  # NaN replaced by lat_nadir
            np.array([100, 110, 120, 131]),  # NaN replaced by lon_nadir
        ),
        (  # Case 2 : Multiple NaN values
            np.array([np.nan, 20, 30, np.nan]),
            np.array([np.nan, 110, 120, np.nan]),
            np.array([11, 20, 30, 41]),
            np.array([101, 110, 120, 131]),
        ),
        (  # Case 3 : No NaN values
            np.array([10, 20, 30, 40]),
            np.array([100, 110, 120, 130]),
            np.array([10, 20, 30, 40]),  # Unchanged
            np.array([100, 110, 120, 130]),  # Unchanged
        ),
    ],
)
def test_fill_empty_latlon_with_nadir(mock_process_data, lat, lon, expected_lat, expected_lon):
    """
    Test fill_empty_latlon_with_nadir : Fills empty lat/lon values with provided Nadir values.
    """
    dummy_nc = MagicMock()
    config = {"lat_nadir": "lat_nadir", "lon_nadir": "lon_nadir"}

    with patch(
        "cpom.altimetry.tools.validate_l2_altimetry_elevations.get_variable",
        side_effect=mock_get_variable,
    ):
        filled_lat, filled_lon = mock_process_data.fill_empty_latlon_with_nadir(
            dummy_nc, lat, lon, config
        )

    assert np.allclose(filled_lat, expected_lat, equal_nan=True)
    assert np.allclose(filled_lon, expected_lon, equal_nan=True)


def test_get_altimetry_data_array_unsupported_file(mock_process_data):
    """Test get_altimetry_data_array : Returns None when file is not supported
    and logs an error"""
    dummy_filename = "unsupported.nc"
    with (
        patch(
            "cpom.altimetry.tools.validate_l2_altimetry_elevations.get_default_variables",
            return_value=None,
        ),
        patch("cpom.altimetry.tools.validate_l2_altimetry_elevations.Dataset"),
    ):
        result = mock_process_data.get_altimetry_data_array(dummy_filename)
        assert result is None
        mock_process_data.log.error.assert_called_with(
            "Unsupported file basename %s for file", dummy_filename
        )


def test_get_altimetry_data_array_empty_inside_mask(mock_process_data, mock_config):
    """Test get_altimetry_data array : Returns None when no valid points found in masked
    object"""
    dummy_filename = "dummy.nc"

    with (
        patch(
            "cpom.altimetry.tools.validate_l2_altimetry_elevations.get_default_variables",
            return_value=mock_config,
        ),
        patch("cpom.altimetry.tools.validate_l2_altimetry_elevations.Dataset"),
        patch(
            "cpom.altimetry.tools.validate_l2_altimetry_elevations.get_variable",
            return_value=np.array([1, 2, 3, 4]),
        ),
    ):
        mock_process_data.area.inside_mask = MagicMock(return_value=(np.array([]), None))
        result = mock_process_data.get_altimetry_data_array(dummy_filename)
        assert result is None


# Edge Case: Mismatch in lengths for x, y, and elevation -> raises ValueError
def test_get_altimetry_data_array_mismatch_lengths(mock_process_data, mock_config):
    """Test get_altimetry_data_array : Exists with a ValueError when x,y,h have different
    lengths"""
    dummy_filename = "example.nc"
    mock_process_data.args.add_vars = []

    with (
        patch(
            "cpom.altimetry.tools.validate_l2_altimetry_elevations.get_default_variables",
            return_value=mock_config,
        ),
        patch("cpom.altimetry.tools.validate_l2_altimetry_elevations.Dataset"),
    ):
        # Patch get_variable: for elev, return an array with only 3 elements.
        def side_effect(nc, var):  # pylint: disable=W0613
            if var == "elev":
                return np.array([500, 600, 700])
            return np.array([10, 20, 30, 40])

        with patch(
            "cpom.altimetry.tools.validate_l2_altimetry_elevations.get_variable",
            side_effect=side_effect,
        ):
            with pytest.raises(ValueError, match="Mismatch in variable lengths for x,y,h"):
                mock_process_data.get_altimetry_data_array(dummy_filename)


def test_get_altimetry_data_array_additional_var_length_mismatch(mock_process_data, mock_config):
    """Test get_altimetry_data_array : Exists with a ValueError when the additional
    variables have a different length"""
    mock_process_data.args.add_vars = ["extra_var", "extra_var1"]  # Add extra args
    with (
        patch(
            "cpom.altimetry.tools.validate_l2_altimetry_elevations.get_default_variables",
            return_value=mock_config,
        ),
        patch(
            "cpom.altimetry.tools.validate_l2_altimetry_elevations.Dataset", MagicMock()
        ) as mock_nc,
    ):
        mock_nc.return_value.__enter__.return_value.variables = {
            "extra_var": MagicMock(),
            "extra_var1": MagicMock(),
        }

        def side_effect(nc, var):  # pylint: disable=W0613
            return np.array([1, 2, 3]) if var == "extra_var" else np.array([1, 2, 3, 4])

        with patch(
            "cpom.altimetry.tools.validate_l2_altimetry_elevations.get_variable",
            side_effect=side_effect,
        ):
            with pytest.raises(ValueError, match="Mismatch in additional variable lengths."):
                mock_process_data.get_altimetry_data_array("example.nc")


def test_get_altimetry_data_array_missing_vars_dropped(mock_process_data, mock_config):
    """Test get_altimetry_data_arrray : logs any difference in length of additional variables"""
    mock_process_data.args.add_vars = ["extra_var", "extra_var1"]
    with (
        patch(
            "cpom.altimetry.tools.validate_l2_altimetry_elevations.get_default_variables",
            return_value=mock_config,
        ),
        patch(
            "cpom.altimetry.tools.validate_l2_altimetry_elevations.Dataset", MagicMock()
        ) as mock_nc,
    ):
        mock_nc.return_value.__enter__.return_value.variables = {"extra_var": MagicMock()}

        with patch(
            "cpom.altimetry.tools.validate_l2_altimetry_elevations.get_variable",
            return_value=np.array([1, 2, 3, 4]),
        ):
            mock_process_data.get_altimetry_data_array("example.nc")
            mock_process_data.log.info.assert_called_with(
                "Variable(s) %s missing from netcdf", {"extra_var1"}
            )


def test_get_altimetry_data_array_happy_path(mock_process_data, mock_config):
    """Test get_altimetry_data_array : Returns a structured array , with x,y, h + any
    additional variables when a valid path is provided."""
    dummy_filename = "dummy.nc"
    # Set an additional variable.
    mock_process_data.args.add_vars = ["extra_var"]

    with (
        patch(
            "cpom.altimetry.tools.validate_l2_altimetry_elevations.get_default_variables",
            return_value=mock_config,
        ),
        patch(
            "cpom.altimetry.tools.validate_l2_altimetry_elevations.get_variable",
            side_effect=mock_get_variable,
        ),
        patch(
            "cpom.altimetry.tools.validate_l2_altimetry_elevations.Dataset", MagicMock()
        ) as mock_nc,
    ):
        # Simulate context manager behavior for Dataset.
        mock_nc.return_value.__enter__.return_value.variables = {"extra_var": MagicMock()}
        result = mock_process_data.get_altimetry_data_array(dummy_filename)

    expected_dtype = [
        ("x", "float64"),
        ("y", "float64"),
        ("h", "float64"),
        ("extra_var", "int64"),
    ]
    assert result.dtype.names == tuple(name for name, _ in expected_dtype)
    assert np.all(result["x"] == np.array([10, 20, 30, 40]) * 10)
    assert np.all(result["y"] == np.array([100, 110, 120, 130]) * 10)
    assert np.all(result["h"] == np.array([500, 600, 700, 800]))
    assert np.all(result["extra_var"] == np.array([1, 2, 3, 4]))


# ---------------------------#
# Test : get_is2_data_array #
# ---------------------------#
def test_get_is2_data_array_unsupported_file(mock_process_data):
    """Test get_is2_data_array : Returns None when file is not supported / doesn't exist
    and logs an error"""
    dummy_filename = "unsupported.h5"
    with (
        patch(
            "cpom.altimetry.tools.validate_l2_altimetry_elevations.get_default_variables",
            return_value=None,
        ),
        patch(
            "cpom.altimetry.tools.validate_l2_altimetry_elevations.h5py.File",
            side_effect=OSError("File not found"),
        ),
    ):
        result = mock_process_data.get_is2_data_array(dummy_filename)
        assert result.size == 0
    mock_process_data.log.error.assert_called_with(
        f"Error loading atl06 data file {dummy_filename} failed with : File not found"
    )


@pytest.mark.parametrize(
    "area_hemisphere, measured_lat",
    [
        ("north", np.array([-10, -10, -10, -10])),
        ("south", np.array([10, 10, 10, 10])),
    ],
)
def test_get_is2_data_array_hemisphere(
    mock_process_data, mock_config, area_hemisphere, measured_lat
):
    """Test get_is2_data_array : Returns None and exists loop early if data does not
    match the area's hemisphere"""

    mock_process_data.area.hemisphere = area_hemisphere

    with (
        patch(
            "cpom.altimetry.tools.validate_l2_altimetry_elevations.get_default_variables",
            return_value=mock_config,
        ),
        patch("cpom.altimetry.tools.validate_l2_altimetry_elevations.h5py.File", MagicMock()),
    ):

        def side_effect(nc, var):  # pylint: disable=W0613
            if var == mock_config["lat"]:
                return measured_lat  # Negative hemisphere
            return np.array([0, 0, 0, 0])  # Dummy for other vars

        with patch(
            "cpom.altimetry.tools.validate_l2_altimetry_elevations.get_variable",
            side_effect=side_effect,
        ):
            result = mock_process_data.get_is2_data_array("example.h5")

    # No beams processed, exit early , empty data
    expected_dtype = [("x", "float64"), ("y", "float64"), ("h", "float64")]
    assert result.dtype == np.dtype(expected_dtype)
    assert result.size == 0


def test_get_is2_data_array_empty_inside_mask(mock_process_data, mock_config):
    """Test get_is2_data_array : Returns None when no valid points found in masked
    object"""
    with (
        patch(
            "cpom.altimetry.tools.validate_l2_altimetry_elevations.get_default_variables",
            return_value=mock_config,
        ),
        patch("cpom.altimetry.tools.validate_l2_altimetry_elevations.h5py.File", MagicMock()),
        patch(
            "cpom.altimetry.tools.validate_l2_altimetry_elevations.get_variable",
            return_value=np.array([1, 2, 3, 4]),
        ),
    ):
        result = mock_process_data.get_is2_data_array("example.h5")
        assert result.size == 0


def test_get_is2_data_array_filtering(mock_process_data, mock_config):
    """Test get_is2_data_array : Correctly filters data using atl06_quality_summary and elevation."""

    mock_process_data.args.beams = ["gt1l"]  # Simulate a single beam

    with (
        patch(
            "cpom.altimetry.tools.validate_l2_altimetry_elevations.get_default_variables",
            return_value=mock_config,
        ),
        patch("cpom.altimetry.tools.validate_l2_altimetry_elevations.h5py.File", MagicMock()),
    ):

        def side_effect(nc, var):  # pylint: disable=W0613
            """Mock variable retrieval behavior."""
            if var == mock_config["elev"]:
                return np.array([500, 600, 11000, 800])  # Third value >10e3 should be removed
            if var.endswith("atl06_quality_summary"):
                return np.array([0, 1, 0, 0])  # Second value should be removed (not 0)
            return np.array([10, 20, 30, 40])  # Dummy data

        with patch(
            "cpom.altimetry.tools.validate_l2_altimetry_elevations.get_variable",
            side_effect=side_effect,
        ):
            result = mock_process_data.get_is2_data_array("example.h5")
    # Expected result: Only indices 0 and 3 should pass filtering
    assert result.size == 2


def test_get_is2_data_array_happy_path_one_beam(
    mock_process_data,
    mock_config,
):
    """Test get_is2_data_array : Returns a structured array with x,y,h arguments"""
    mock_process_data.args.beams = ["beam1"]
    mock_process_data.area.hemisphere = "north"

    with (
        patch(
            "cpom.altimetry.tools.validate_l2_altimetry_elevations.get_default_variables",
            return_value=mock_config,
        ),
        patch("cpom.altimetry.tools.validate_l2_altimetry_elevations.h5py.File", MagicMock()),
    ):

        def side_effect(nc, var):
            """Mock variable retrieval behavior."""
            if var.endswith("atl06_quality_summary"):
                return np.array([0, 0, 0, 0])  # Keep all in filter
            return mock_get_variable(nc, var)  # Dummy data

        with patch(
            "cpom.altimetry.tools.validate_l2_altimetry_elevations.get_variable",
            side_effect=side_effect,
        ):
            result = mock_process_data.get_is2_data_array("example.h5")

    # Validate the structured array result.
    expected_dtype = [("x", "float64"), ("y", "float64"), ("h", "float64")]
    assert result.dtype == expected_dtype
    # assert result.dtype.names == tuple(name for name, _ in expected_dtype)
    np.all(result["x"] == np.array([10, 20, 30, 40]) * 10)
    np.all(result["y"] == np.array([100, 110, 120, 130]) * 10)
    np.all(result["h"] == np.array([500, 600, 700, 800]))


def test_get_is2_data_array_happy_path_multiple_beam(
    mock_process_data,
    mock_config,
):
    """Test get_is2_data_array : Returns a structured array with x,y,h concatinating
    multiple beams"""

    mock_process_data.args.beams = ["beam1", "beam2"]
    mock_process_data.area.hemisphere = "north"

    with (
        patch(
            "cpom.altimetry.tools.validate_l2_altimetry_elevations.get_default_variables",
            return_value=mock_config,
        ),
        patch("cpom.altimetry.tools.validate_l2_altimetry_elevations.h5py.File", MagicMock()),
    ):

        def side_effect(nc, var):
            """Mock variable retrieval behavior."""
            if var.endswith("atl06_quality_summary"):
                return np.array([0, 0, 0, 0])  # Keep all in filter
            return mock_get_variable(nc, var)  # Dummy data

        with patch(
            "cpom.altimetry.tools.validate_l2_altimetry_elevations.get_variable",
            side_effect=side_effect,
        ):
            result = mock_process_data.get_is2_data_array("example.h5")

    # Validate the structured array result.
    expected_dtype = [("x", "float64"), ("y", "float64"), ("h", "float64")]
    assert result.dtype == expected_dtype

    expected_x = np.tile(np.array([10, 20, 30, 40]) * 10, 2)
    expected_y = np.tile(np.array([100, 110, 120, 130]) * 10, 2)
    expected_h = np.tile(np.array([500, 600, 700, 800]), 2)
    np.all(result["x"] == expected_x)
    np.all(result["y"] == expected_y)
    np.all(result["h"] == expected_h)
