"""pytest for cpom.altimetry.tools.sec_tools.grid_for_elev_change.py"""

# pylint: disable=redefined-outer-name,unused-argument

import argparse
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from cpom.altimetry.tools.sec_tools.grid_for_elev_change import (
    _fill_missing_poca_with_nadir_fdr4alt,
    clean_directory,
    get_coordinates_from_file,
    get_grid_cells,
    get_spatial_filter,
    get_variables_and_mask,
    parse_arguments,
)


# Global Fixtures
@pytest.fixture
def mock_ds():
    """Fixture providing mock dataset."""
    mock_ds = mock.MagicMock()
    mock_ds.mission = "my_mission"
    mock_ds.latitude_param = "latitude"
    mock_ds.longitude_param = "longitude"
    mock_ds.elevation_param = "elevation"
    mock_ds.time_param = "time"
    mock_ds.power_param = None
    mock_ds.quality_param = None
    mock_ds.uncertainty_param = None
    mock_ds.latitude_nadir_param = None
    return mock_ds


@pytest.fixture
def mock_nc():
    """Fixture providing mock NetCDF file."""
    return mock.MagicMock()


@pytest.fixture
def mock_dict():
    """Fixture providing test coordinate arrays."""
    return {
        "latitude": np.array([10.0, 20.0, 30.0, 40.0]),
        "longitude": np.array([1.0, 2.0, 3.0, 4.0]),
        "latitude_nadir": np.array([-10.1, -20.1, -30.1, -40.1]),
        "longitude_nadir": np.array([1.1, 2.1, 3.1, 4.1]),
        "beams": np.array(["gt1l", "gt1r", "gt2l", "gt2r"]),
        "elevation": np.array([100.0, 200.0, 300.0, 400.0]),
        "time": np.array([1.0, 2.0, 3.0, 4.0]),
        "quality_flag": np.array([0, 1, 0, 1]),
        "power": np.array([10.0, 20.0, 30.0, 40.0]),
        "uncertainty": np.array([0.1, 0.2, 0.3, 0.4]),
    }


@pytest.fixture
def mock_params(tmp_path):
    """Fixture to mock command-line parameters with temp directory."""
    params = argparse.Namespace()
    params.out_dir = str(tmp_path / "my_mission_output")
    params.gridarea = "antarctica"
    params.binsize = 25
    params.area = "antarctica"
    params.mask_name = None
    return params


def get_variable_side_effect(mock_dict):
    """mock get_variable dataset call"""

    def side_effect(nc, var_name, replace_fill=True, return_beams=False):
        if var_name == "latitude":
            if return_beams:
                return mock_dict["latitude"], mock_dict["beams"]
            return mock_dict["latitude"]

        if var_name in mock_dict:
            return mock_dict[var_name]

        # If missing, raise same error your real implementation would
        raise KeyError(f"Variable {var_name} not found")

    return side_effect


class TestParseArguments:
    """Tests for parse_arguments function."""

    def test_required_fields_missing(self):
        """Test that all required arguments must be supplied."""
        with pytest.raises(SystemExit):
            parse_arguments([])

    def test_partition_columns_multiple_values(self):
        """Test nargs='+' behavior for partition_columns."""
        args = parse_arguments(
            [
                "--data_input_dir",
                "/data",
                "--dataset",
                "dataset.yml",
                "--out_dir",
                "/out",
                "--area",
                "antarctica",
                "--gridarea",
                "antarctica",
                "--binsize",
                "5000",
                "--partition_columns",
                "year",
                "month",
                "x_part",
                "y_part",
            ]
        )
        assert args.partition_columns == ["year", "month", "x_part", "y_part"]


class TestCleanDirectory:
    """Tests for clean_directory function."""

    def test_removes_directory_without_prompt(self, mock_params, mock_ds):
        """Test that the directory is removed when confirm_regrid=False."""
        out_dir = Path(mock_params.out_dir)
        out_dir.mkdir(parents=True)
        test_file = out_dir / "test_file.txt"
        test_file.write_text("test content")

        assert out_dir.exists()
        assert test_file.exists()

        clean_directory(mock_params, mock_ds, confirm_regrid=False)

        assert out_dir.exists()
        assert not test_file.exists()

    @mock.patch("builtins.input", return_value="y")
    def test_removes_directory_with_prompt_yes(self, mock_input, mock_params, mock_ds):
        """Test that directory is removed when confirm_regrid=True and user confirms."""
        out_dir = Path(mock_params.out_dir)
        out_dir.mkdir(parents=True)
        test_file = out_dir / "test_file.txt"
        test_file.write_text("test content")

        clean_directory(mock_params, mock_ds, confirm_regrid=True)

        mock_input.assert_called_once()
        assert "Confirm removal" in mock_input.call_args[0][0]
        assert out_dir.exists()
        assert not test_file.exists()

    @mock.patch("builtins.input", return_value="n")
    def test_does_not_remove_directory_with_prompt_no(self, mock_input, mock_params, mock_ds):
        """Test that the directory is not removed when user declines."""
        out_dir = Path(mock_params.out_dir)
        out_dir.mkdir(parents=True)
        test_file = out_dir / "test_file.txt"
        test_file.write_text("test content")

        with pytest.raises(SystemExit) as exc_info:
            clean_directory(mock_params, mock_ds, confirm_regrid=True)

        assert exc_info.value.code == 0
        mock_input.assert_called_once()
        assert out_dir.exists()
        assert test_file.exists()

    def test_safety_check_mission_name(self, mock_params, mock_ds):
        """Test that safety check exits when mission name not in output directory path."""
        mock_ds.mission = "different_mission"
        out_dir = Path(mock_params.out_dir)
        out_dir.mkdir(parents=True)

        with pytest.raises(SystemExit) as exc_info:
            clean_directory(mock_params, mock_ds)
        assert exc_info.value.code == 1


class TestFillMissingPocaWithNadirFdr4alt:
    """Tests for _fill_missing_poca_with_nadir_fdr4alt function."""

    def test_replaces_poca_with_nadir_values(self, mock_dict, mock_ds, mock_nc):
        """Test that missing POCA lat/lon are replaced with nadir values."""
        mock_ds.latitude_nadir_param = "latitude_nadir"
        mock_ds.longitude_nadir_param = "longitude_nadir"

        def side_effect(nc, var_name, replace_fill=True):
            if var_name == "expert/ice_sheet_qual_relocation":
                return mock_dict["quality_flag"]
            if var_name == "latitude_nadir":
                return mock_dict["latitude_nadir"]
            if var_name == "longitude_nadir":
                return mock_dict["longitude_nadir"]
            raise KeyError(f"Variable {var_name} not found")

        mock_ds.get_variable.side_effect = side_effect

        result_lats, result_lons = _fill_missing_poca_with_nadir_fdr4alt(
            mock_ds, mock_nc, mock_dict["latitude"], mock_dict["longitude"]
        )

        # Verify indices 1 and 3 are replaced (where quality_flag > 0)
        assert result_lats[1] == mock_dict["latitude_nadir"][1]
        assert result_lons[1] == mock_dict["longitude_nadir"][1]
        assert result_lats[3] == mock_dict["latitude_nadir"][3]
        assert result_lons[3] == mock_dict["longitude_nadir"][3]

        # Verify indices 0 and 2 remain unchanged (where quality_flag == 0)
        assert result_lats[0] == mock_dict["latitude"][0]
        assert result_lons[0] == mock_dict["longitude"][0]
        assert result_lats[2] == mock_dict["latitude"][2]
        assert result_lons[2] == mock_dict["longitude"][2]

    def test_returns_unchanged_on_key_error(self, mock_dict, mock_ds, mock_nc):
        """Test that lat/lon are returned unchanged when KeyError is raised."""
        mock_ds.get_variable.side_effect = KeyError("missing variable")

        result_lats, result_lons = _fill_missing_poca_with_nadir_fdr4alt(
            mock_ds, mock_nc, mock_dict["latitude"], mock_dict["longitude"]
        )

        assert np.array_equal(result_lats, mock_dict["latitude"])
        assert np.array_equal(result_lons, mock_dict["longitude"])


class TestGetCoordinatesFromFile:
    """Tests for get_coordinates_from_file function."""

    def test_no_beams(self, mock_dict, mock_ds, mock_nc):
        """Test that coordinates are returned without beams."""
        mock_ds.beams = []
        mock_ds.get_variable.side_effect = get_variable_side_effect(mock_dict)

        result = get_coordinates_from_file(mock_ds, mock_nc)

        assert np.array_equal(result["latitude"], mock_dict["latitude"])
        assert np.array_equal(result["longitude"], mock_dict["longitude"])
        assert result["beams"] is None

    def test_with_beams(self, mock_dict, mock_ds, mock_nc):
        """Test that coordinates and beams are returned."""
        mock_ds.beams = ["gt1l", "gt1r", "gt2l", "gt2r"]
        mock_ds.get_variable.side_effect = get_variable_side_effect(mock_dict)

        result = get_coordinates_from_file(mock_ds, mock_nc)

        assert np.array_equal(result["latitude"], mock_dict["latitude"])
        assert np.array_equal(result["longitude"], mock_dict["longitude"])
        assert np.array_equal(result["beams"], mock_dict["beams"])

    def test_fill_missing_poca_called(self, mock_dict, mock_ds, mock_nc):
        """Test that _fill_missing_poca_with_nadir_fdr4alt is called when fill_missing_poca=True."""
        modified_lats = np.array([10.0, -2.0, 30.0, -4.0])
        modified_lons = np.array([1.0, -20.0, 3.0, -40.0])

        mock_ds.beams = []
        mock_ds.get_variable.side_effect = get_variable_side_effect(mock_dict)

        with mock.patch(
            "cpom.altimetry.tools."
            "sec_tools.grid_for_elev_change._fill_missing_poca_with_nadir_fdr4alt",
            return_value=(modified_lats, modified_lons),
        ) as mock_fill_poca:
            result = get_coordinates_from_file(mock_ds, mock_nc, fill_missing_poca=True)

            # Verify _fill_missing_poca_with_nadir_fdr4alt was called once
            mock_fill_poca.assert_called_once()
            assert np.array_equal(result["latitude"], modified_lats)
            assert np.array_equal(result["longitude"], modified_lons)
            assert result["beams"] is None


class TestGetSpatialFilter:
    """Tests for get_spatial_filter function."""

    @pytest.fixture
    def mock_area(self):
        """Fixture providing mock Area object."""
        area = mock.MagicMock()
        area.inside_latlon_bounds.return_value = (
            np.array([10.0, 20.0, 30.0]),  # bounded_lat
            np.array([1.0, 2.0, 3.0]),  # bounded_lon
            np.array([True, True, True, False]),  # bounded_mask
            None,  # fourth return value
        )
        area.inside_area.return_value = (
            np.array([True, False, True]),  # area_mask_valid
            2,  # n_inside
        )
        area.latlon_to_xy.return_value = (
            np.array([100.0, 200.0, 300.0, 400.0]),  # x_coords
            np.array([1000.0, 2000.0, 3000.0, 4000.0]),  # y_coords
        )
        return area

    @pytest.fixture
    def mock_mask(self):
        """Fixture providing mock Mask object."""
        mask = mock.MagicMock()
        mask.points_inside.return_value = (
            np.array([False, False, True]),  # area_mask_valid
            1,  # n_inside
        )
        mask.latlon_to_xy.return_value = (
            np.array([100.0, 200.0, 300.0, 400.0]),  # x_coords
            np.array([1000.0, 2000.0, 3000.0, 4000.0]),  # y_coords
        )
        return mask

    def test_uses_area_when_mask_none(self, mock_dict, mock_area, mock_mask):
        """Test that area methods are used when mask is None."""
        _, n_inside, _ = get_spatial_filter(mock_dict, mock_area, this_mask=None)
        mock_area.inside_latlon_bounds.assert_called_once()

        mock_area.inside_area.assert_called_once()
        call_args = mock_area.inside_area.call_args[0]
        assert np.array_equal(call_args[0], np.array([10.0, 20.0, 30.0]))
        assert np.array_equal(call_args[1], np.array([1.0, 2.0, 3.0]))
        mock_mask.inside_area.assert_not_called()
        assert n_inside == 2

    def test_uses_mask_when_provided(self, mock_dict, mock_area, mock_mask):
        """Test that mask methods are used when mask is provided."""
        _, n_inside, _ = get_spatial_filter(mock_dict, mock_area, this_mask=mock_mask)

        mock_area.inside_latlon_bounds.assert_called_once()

        # Verify Mask methods were called instead of Area methods
        mock_mask.points_inside.assert_called_once()
        call_args = mock_mask.points_inside.call_args[0]
        assert np.array_equal(call_args[0], np.array([10.0, 20.0, 30.0]))
        assert np.array_equal(call_args[1], np.array([1.0, 2.0, 3.0]))

        mock_area.inside_area.assert_not_called()
        assert n_inside == 1

    def test_returned_dict_has_correct_keys(self, mock_dict, mock_area):
        """Test that returned dictionary contains all expected keys."""
        result_dict, _, _ = get_spatial_filter(mock_dict, mock_area, this_mask=None)
        assert {"x", "y"}.issubset(set(result_dict.keys()))

    def test_mask_reconstruction_correct_length(self, mock_dict, mock_area):
        """Test that reconstructed mask has same length as input latitude array."""
        _, _, bounded_mask = get_spatial_filter(mock_dict, mock_area, this_mask=None)
        assert len(bounded_mask) == len(mock_dict["latitude"])
        assert len(bounded_mask) == 4

    def test_mask_reconstruction_correct_items_masked(self, mock_area):
        """Test that mask reconstruction correctly identifies masked and unmasked items."""
        variable_dict = {
            "latitude": np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0]),
            "longitude": np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        }
        bounded_mask = np.array([True, True, True, False, True, False])
        area_mask_valid = np.array([True, False, True, False])
        expected_output_mask = np.array([True, False, True, False, False, False])

        mock_area.inside_latlon_bounds.return_value = (
            np.array([10.0, 20.0, 30.0, 50.0]),  # bounded_lat (4 points)
            np.array([1.0, 2.0, 3.0, 5.0]),  # bounded_lon (4 points)
            bounded_mask,
            None,
        )
        mock_area.inside_area.return_value = (
            area_mask_valid,
            2,  # n_inside
        )
        mock_area.latlon_to_xy.return_value = (
            np.array([100.0, 200.0, 300.0, 400.0, 500.0, 600.0]),
            np.array([1000.0, 2000.0, 3000.0, 4000.0, 5000.0, 6000.0]),
        )

        _, n_inside, returned_bounded_mask = get_spatial_filter(
            variable_dict, mock_area, this_mask=None
        )

        assert np.array_equal(returned_bounded_mask, expected_output_mask)
        assert n_inside == 2


class TestGetVariablesAndMask:
    """Tests for get_variables_and_mask function."""

    def test_loads_required_variables(self, mock_dict, mock_ds, mock_nc):
        """Test that required variables (elevation, time) are loaded correctly."""
        area_mask = np.array([True, True, True, False])

        mock_ds.get_variable.side_effect = get_variable_side_effect(mock_dict)
        mock_ds.get_file_orbital_direction.return_value = True

        result_dict, _ = get_variables_and_mask(mock_ds, mock_nc, mock_dict.copy(), 0.0, area_mask)

        assert "elevation" in result_dict
        assert "time" in result_dict
        assert np.array_equal(result_dict["elevation"], mock_dict["elevation"])
        assert np.array_equal(result_dict["time"], mock_dict["time"])

    def test_applies_time_offset(self, mock_dict, mock_ds, mock_nc):
        """Test that time offset is correctly applied to time data."""
        area_mask = np.array([True, True, True, True])
        offset = 50.0

        mock_ds.get_variable.side_effect = get_variable_side_effect(mock_dict)
        mock_ds.get_file_orbital_direction.return_value = False

        result_dict, _ = get_variables_and_mask(
            mock_ds, mock_nc, mock_dict.copy(), offset, area_mask
        )

        expected_time = mock_dict["time"] + offset
        assert np.array_equal(result_dict["time"], expected_time)

    def test_combines_masks_with_finite_checks(self, mock_dict, mock_ds, mock_nc):
        """Test that masks are combined with finite value checks."""
        area_mask = np.array([True, True, True, True])
        offset = 0.0

        # Create data with some non-finite values
        test_dict = mock_dict.copy()
        test_dict["elevation"] = np.array([100.0, np.nan, 300.0, 400.0])
        test_dict["time"] = np.array([1.0, 2.0, np.inf, 4.0])

        mock_ds.get_variable.side_effect = get_variable_side_effect(test_dict)
        mock_ds.get_file_orbital_direction.return_value = True

        _, bool_mask = get_variables_and_mask(mock_ds, mock_nc, test_dict.copy(), offset, area_mask)

        expected_mask = np.array([True, False, False, True])
        assert np.array_equal(bool_mask, expected_mask)

    def test_returns_none_when_insufficient_valid_points(self, mock_dict, mock_ds, mock_nc):
        """Test that None is returned when bool_mask.sum() < 2."""
        # Only one valid point
        area_mask = np.array([True, False, False, False])
        offset = 0.0

        test_dict = mock_dict.copy()
        test_dict["elevation"] = np.array([100.0, np.nan, np.nan, np.nan])
        mock_ds.get_variable.side_effect = get_variable_side_effect(test_dict)

        result_dict, bool_mask = get_variables_and_mask(
            mock_ds, mock_nc, test_dict.copy(), offset, area_mask
        )

        assert result_dict is None
        assert bool_mask is None

    def test_loads_optional_param(self, mock_dict, mock_ds, mock_nc):
        """Test that power and uncertainty parameters are loaded when available."""
        area_mask = np.array([True, True, True, True])
        offset = 0.0

        mock_ds.power_param = "power"
        mock_ds.uncertainty_param = "uncertainty"
        mock_ds.get_variable.side_effect = get_variable_side_effect(mock_dict)
        mock_ds.get_file_orbital_direction.return_value = True

        result_dict, _ = get_variables_and_mask(
            mock_ds, mock_nc, mock_dict.copy(), offset, area_mask
        )

        assert "power" in result_dict
        assert np.array_equal(result_dict["power"], mock_dict["power"])
        assert "uncertainty" in result_dict
        assert np.array_equal(result_dict["uncertainty"], mock_dict["uncertainty"])

    def test_applies_quality_mask(self, mock_dict, mock_ds, mock_nc):
        """Test that quality mask is applied when quality_param is not None."""
        area_mask = np.array([True, True, True, True])
        offset = 0.0

        # quality_flag: 0=good, 1=bad
        test_dict = mock_dict.copy()
        test_dict["quality_flag"] = np.array([0, 1, 0, 1])

        mock_ds.quality_param = "quality_flag"
        mock_ds.get_variable.side_effect = get_variable_side_effect(test_dict)
        mock_ds.get_file_orbital_direction.return_value = True

        _, bool_mask = get_variables_and_mask(mock_ds, mock_nc, test_dict.copy(), offset, area_mask)

        expected_mask = np.array([True, False, True, False])
        assert np.array_equal(bool_mask, expected_mask)


class TestGetGridCells:
    """Tests for get_grid_cells function."""

    @pytest.fixture
    def mock_grid(self):
        """Fixture providing mock GridArea object."""
        grid = mock.MagicMock()
        grid.get_col_row_from_x_y.return_value = (
            np.array([10, 20, 30, 40]),  # x_bin
            np.array([100, 200, 300, 400]),  # y_bin
        )
        grid.get_xy_relative_to_cellcentre.return_value = (
            np.array([0.5, 1.5, 2.5, 3.5]),  # x_cell_offset
            np.array([5.0, 6.0, 7.0, 8.0]),  # y_cell_offset
        )
        return grid

    def test_adds_grid_cell_indices(self, mock_dict, mock_grid):
        """Test that x_bin and y_bin are added to variable_dict."""

        mock_dict["x"] = np.array([1000.0, 2000.0, 3000.0, 4000.0])
        mock_dict["y"] = np.array([10000.0, 20000.0, 30000.0, 40000.0])
        result = get_grid_cells(mock_dict, mock_grid)

        assert "x_bin" in result
        assert "y_bin" in result
        assert np.array_equal(result["x_bin"], np.array([10, 20, 30, 40]))
        assert np.array_equal(result["y_bin"], np.array([100, 200, 300, 400]))
        assert "x_cell_offset" in result
        assert "y_cell_offset" in result
        assert np.array_equal(result["x_cell_offset"], np.array([0.5, 1.5, 2.5, 3.5]))
        assert np.array_equal(result["y_cell_offset"], np.array([5.0, 6.0, 7.0, 8.0]))
