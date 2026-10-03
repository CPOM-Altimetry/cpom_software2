"""pytests of cpom.altimetry.projects.csqa.gridding"""

import numpy as np
import pytest

from cpom.altimetry.projects.csqa.gridding import (
    cell_centres_latlon,
    get_grid_area,
    grid_image,
    grid_measurements,
)

STATS = ["median", "mean", "max", "min", "std", "count"]


def _cell_points(grid, col: int, row: int, n: int) -> tuple[np.ndarray, np.ndarray]:
    """latitudes and longitudes of n points inside a grid cell (around its centre)"""
    x, y = grid.get_cellcentre_x_y_from_col_row(col, row)
    offsets = np.linspace(-0.4, 0.4, n) * grid.binsize
    lons, lats = grid.xy_to_lonlat_transformer.transform(x + offsets, y - offsets)
    return np.asarray(lats), np.asarray(lons)


def test_grid_statistics():
    """per cell statistics of the measurements, ignoring NaN values and points off the grid"""
    grid = get_grid_area("arctic", 10000)
    ncols = grid.get_ncols_nrows()[0]
    cell_a, cell_b = (380, 560), (381, 560)
    lats_a, lons_a = _cell_points(grid, *cell_a, 5)
    lats_b, lons_b = _cell_points(grid, *cell_b, 2)
    vals_a = np.array([0.4, 0.1, np.nan, 0.3, 0.2])
    vals_b = np.array([1.0, 2.0])
    # a measurement far outside the grid (south pole) is ignored
    lats = np.concatenate((lats_a, lats_b, [-89.0]))
    lons = np.concatenate((lons_a, lons_b, [0.0]))
    vals = np.concatenate((vals_a, vals_b, [5.0])).astype(np.float32)

    gridded = grid_measurements("arctic", 10000, lats, lons, vals, STATS)
    assert gridded.n_records == 6
    assert list(gridded.cells) == [c[1] * ncols + c[0] for c in (cell_a, cell_b)]
    assert list(gridded.counts) == [4, 2]
    assert np.allclose(gridded.values["median"], [0.25, 1.5])
    assert np.allclose(gridded.values["mean"], [0.25, 1.5])
    assert np.allclose(gridded.values["max"], [0.4, 2.0])
    assert np.allclose(gridded.values["min"], [0.1, 1.0])
    assert np.allclose(gridded.values["std"], [np.std([0.4, 0.1, 0.3, 0.2]), 0.5])
    assert np.allclose(gridded.values["count"], [4, 2])

    # odd number of measurements: the median is the middle value
    odd = grid_measurements(
        "arctic", 10000, lats_a[:3], lons_a[:3], np.array([3.0, 1.0, 2.0]), ["median"]
    )
    assert odd.values["median"][0] == 2.0

    # cells with fewer than min_count measurements are excluded
    gridded = grid_measurements("arctic", 10000, lats, lons, vals, ["mean"], min_count=3)
    assert list(gridded.counts) == [4]
    assert np.allclose(gridded.values["mean"], [0.25])

    with pytest.raises(ValueError):
        grid_measurements("arctic", 10000, lats, lons, vals, ["mode"])


def test_no_measurements():
    """a selection without valid measurements has no cells"""
    gridded = grid_measurements(
        "antarctic_ocean", 10000, np.array([-70.0]), np.array([0.0]), np.array([np.nan]), STATS
    )
    assert gridded.n_records == 0 and gridded.cells.size == 0
    assert all(v.size == 0 for v in gridded.values.values())


def test_grid_image_and_cell_centres():
    """the grid image is cropped to the cells with data, and cell centres are in their cells"""
    grid = get_grid_area("antarctic_ocean", 10000)
    lats_a, lons_a = _cell_points(grid, 100, 200, 3)
    lats_b, lons_b = _cell_points(grid, 102, 205, 1)
    gridded = grid_measurements(
        "antarctic_ocean",
        10000,
        np.concatenate((lats_a, lats_b)),
        np.concatenate((lons_a, lons_b)),
        np.array([1.0, 2.0, 3.0, 7.0]),
        ["mean"],
    )
    image = grid_image(gridded, "mean")
    assert image["epsg"] == 3031
    assert image["values"].shape == (6, 3)
    assert image["values"][0, 0] == 2.0 and image["values"][5, 2] == 7.0
    assert np.count_nonzero(np.isfinite(image["values"])) == 2
    assert image["x_edges"][0] == grid.minxm + 100 * 10000
    assert image["y_edges"][-1] == grid.minym + 206 * 10000

    lats, lons = cell_centres_latlon(gridded)
    assert np.all((lons >= 0) & (lons < 360))
    x, y = grid.transform_lat_lon_to_x_y(lats, lons)
    cols, rows = grid.get_col_row_from_x_y(np.asarray(x), np.asarray(y))
    assert list(cols) == [100, 102] and list(rows) == [200, 205]
