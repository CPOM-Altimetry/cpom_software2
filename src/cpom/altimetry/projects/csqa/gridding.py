"""cpom.altimetry.projects.csqa.gridding

Grid the measurements of a CSQA parameter selection into the cells of a polar stereographic
grid (ie 10 km cells), giving per cell statistics of the measurement values (count, mean,
median, maximum..).

The grids are cpom.gridding.gridareas.GridArea grids (ie 'arctic' in EPSG:3413 and
'antarctic_ocean' in EPSG:3031), with the cell size configured per parameter. The binning is
vectorised (sorting the measurements by cell), so cycles of millions of measurements are
gridded in a few seconds.
"""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from cpom.gridding.gridareas import GridArea

# statistics available for grids smoothed over a radius (sums of the cells within the radius)
SMOOTHED_STATISTICS = ("mean", "std", "count")

# statistics of the measurements in each grid cell: id -> display name
GRID_STATISTICS = {
    "median": "Median",
    "mean": "Mean",
    "max": "Maximum",
    "min": "Minimum",
    "std": "Std Dev",
    "count": "Count",  # number of measurements in the cell
}


@lru_cache(maxsize=8)
def get_grid_area(name: str, binsize_m: int) -> GridArea:
    """A cpom GridArea (cached, as building its coordinate mesh takes a moment)

    Args:
        name (str): GridArea name, ie 'arctic'
        binsize_m (int): cell size in m

    Returns:
        GridArea
    """
    grid = GridArea(name, binsize_m)
    if not hasattr(grid, "minxm"):
        raise ValueError(f"unknown cpom grid area {name}")
    return grid


def grid_epsg(grid: GridArea) -> int:
    """EPSG number of a grid's projection"""
    return int(str(grid.coordinate_reference_system).lower().replace("epsg:", ""))


@dataclass
class GriddedSelection:
    """Measurements of a selection gridded into the cells of a grid"""

    grid_name: str
    binsize_m: int
    n_records: int  # valid measurements inside the grid
    cells: np.ndarray  # flat indices (row * ncols + col) of the cells with data, ascending
    counts: np.ndarray  # number of measurements in each cell with data
    values: dict[str, np.ndarray]  # statistic -> value of each cell with data

    @property
    def grid(self) -> GridArea:
        """the grid"""
        return get_grid_area(self.grid_name, self.binsize_m)


def grid_measurements(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    grid_name: str,
    binsize_m: int,
    lats: np.ndarray,
    lons: np.ndarray,
    vals: np.ndarray,
    statistics: list[str],
    min_count: int = 1,
    smooth_radius_m: float = 0.0,
) -> GriddedSelection:
    """Grid measurements, calculating statistics of the values in each cell

    Args:
        grid_name (str): cpom GridArea name, ie 'arctic'
        binsize_m (int): cell size in m
        lats (np.ndarray): measurement latitudes
        lons (np.ndarray): measurement longitudes
        vals (np.ndarray): measurement values (NaN for missing)
        statistics (list[str]): statistics to calculate (keys of GRID_STATISTICS)
        min_count (int): minimum number of measurements in a cell for it to have values
        smooth_radius_m (float): if > 0, the statistics of each cell are of the measurements
                                 of all cells within this distance of it (centre to centre),
                                 so sparse measurements (ie crossovers) give a continuous map.
                                 Only the SMOOTHED_STATISTICS are available

    Returns:
        GriddedSelection: only cells with at least min_count measurements
    """
    for stat in statistics:
        if stat not in GRID_STATISTICS:
            raise ValueError(f"unknown grid statistic {stat}")
    grid = get_grid_area(grid_name, binsize_m)
    ncols, nrows = grid.get_ncols_nrows()

    valid = np.isfinite(vals) & np.isfinite(lats) & np.isfinite(lons)
    x, y = grid.transform_lat_lon_to_x_y(
        lats[valid].astype(np.float64), lons[valid].astype(np.float64)
    )
    col, row = grid.get_col_row_from_x_y(np.asarray(x), np.asarray(y))
    col, row = col.astype(np.int64), row.astype(np.int64)
    inside = (col >= 0) & (col < ncols) & (row >= 0) & (row < nrows)
    cell = row[inside] * ncols + col[inside]
    v = vals[valid][inside].astype(np.float64)

    if smooth_radius_m > 0:
        return _smoothed_grid(grid_name, binsize_m, cell, v, statistics, min_count, smooth_radius_m)

    # measurements sorted by cell, then value (for the medians)
    order = np.lexsort((v, cell))
    cell, v = cell[order], v[order]
    cells, starts, counts = np.unique(cell, return_index=True, return_counts=True)

    keep = counts >= max(int(min_count), 1)
    values: dict[str, np.ndarray] = {}
    if cells.size:
        ends = starts + counts - 1
        sums = np.add.reduceat(v, starts)
        for stat in statistics:
            if stat == "count":
                result = counts.astype(np.float64)
            elif stat == "mean":
                result = sums / counts
            elif stat == "median":
                result = 0.5 * (v[starts + (counts - 1) // 2] + v[starts + counts // 2])
            elif stat == "min":
                result = v[starts]
            elif stat == "max":
                result = v[ends]
            else:  # std (population)
                mean = sums / counts
                sq_dev = (v - np.repeat(mean, counts)) ** 2
                result = np.sqrt(np.add.reduceat(sq_dev, starts) / counts)
            values[stat] = result[keep]
    else:
        values = {stat: np.array([], dtype=np.float64) for stat in statistics}

    return GriddedSelection(
        grid_name=grid_name,
        binsize_m=binsize_m,
        n_records=int(v.size),
        cells=cells[keep],
        counts=counts[keep],
        values=values,
    )


def _smoothed_grid(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    grid_name: str,
    binsize_m: int,
    cell: np.ndarray,
    v: np.ndarray,
    statistics: list[str],
    min_count: int,
    smooth_radius_m: float,
) -> GriddedSelection:
    """grid statistics of the measurements within a radius of each cell (see grid_measurements)

    Args:
        grid_name (str): cpom GridArea name
        binsize_m (int): cell size in m
        cell (np.ndarray): flat cell index of each measurement
        v (np.ndarray): value of each measurement
        statistics (list[str]): statistics (of SMOOTHED_STATISTICS)
        min_count (int): minimum number of measurements within the radius of a cell
        smooth_radius_m (float): radius (m)

    Returns:
        GriddedSelection
    """
    # pylint: disable=too-many-locals
    for stat in statistics:
        if stat not in SMOOTHED_STATISTICS:
            raise ValueError(f"grid statistic {stat} can not be smoothed")
    empty = {stat: np.array([], dtype=np.float64) for stat in statistics}
    if cell.size == 0:
        return GriddedSelection(
            grid_name, binsize_m, 0, np.array([], np.int64), np.array([]), empty
        )

    grid = get_grid_area(grid_name, binsize_m)
    ncols, nrows = grid.get_ncols_nrows()
    radius = int(np.floor(smooth_radius_m / binsize_m))
    # cells within the radius (centre to centre)
    offsets = np.arange(-radius, radius + 1)
    disc = (offsets[:, None] ** 2 + offsets[None, :] ** 2) <= (smooth_radius_m / binsize_m) ** 2
    d_rows, d_cols = np.nonzero(disc)
    d_rows, d_cols = d_rows - radius, d_cols - radius

    # per cell sums, in the bounding box of the cells with measurements (+ radius)
    rows, cols = np.divmod(cell, ncols)
    row0, col0 = max(int(rows.min()) - radius, 0), max(int(cols.min()) - radius, 0)
    row1, col1 = min(int(rows.max()) + radius, nrows - 1), min(int(cols.max()) + radius, ncols - 1)
    shape = (row1 - row0 + 1, col1 - col0 + 1)
    flat = (rows - row0) * shape[1] + (cols - col0)
    sums = {
        "n": np.bincount(flat, minlength=shape[0] * shape[1]).reshape(shape).astype(np.float64),
        "s": np.bincount(flat, weights=v, minlength=shape[0] * shape[1]).reshape(shape),
        "ss": np.bincount(flat, weights=v * v, minlength=shape[0] * shape[1]).reshape(shape),
    }
    # sum over the disc of cells around each cell
    smoothed = {}
    for key, arr in sums.items():
        padded = np.pad(arr, radius)
        total = np.zeros(shape)
        for d_row, d_col in zip(d_rows, d_cols):
            total += padded[
                radius + d_row : radius + d_row + shape[0],
                radius + d_col : radius + d_col + shape[1],
            ]
        smoothed[key] = total

    n = smoothed["n"]
    keep = n >= max(int(min_count), 1)
    k_rows, k_cols = np.nonzero(keep)
    cells = (k_rows + row0) * ncols + (k_cols + col0)
    counts, mean = n[keep], smoothed["s"][keep] / n[keep]
    values: dict[str, np.ndarray] = {}
    for stat in statistics:
        if stat == "count":
            values[stat] = counts
        elif stat == "mean":
            values[stat] = mean
        else:  # std (population)
            values[stat] = np.sqrt(np.maximum(smoothed["ss"][keep] / counts - mean * mean, 0.0))
    order = np.argsort(cells)
    return GriddedSelection(
        grid_name=grid_name,
        binsize_m=binsize_m,
        n_records=int(v.size),
        cells=cells[order],
        counts=counts[order],
        values={stat: vals[order] for stat, vals in values.items()},
    )


def cell_centres_latlon(gridded: GriddedSelection) -> tuple[np.ndarray, np.ndarray]:
    """latitudes and longitudes of the centres of the cells with data"""
    grid = gridded.grid
    rows, cols = np.divmod(gridded.cells, grid.get_ncols_nrows()[0])
    lats, lons = grid.get_cellcentre_lat_lon_from_col_row(cols, rows)  # longitudes 0..360
    return np.asarray(lats), np.asarray(lons)


def grid_image(gridded: GriddedSelection, stat: str) -> dict:
    """A statistic of the cells with data as a 2-d grid, cropped to the cells with data

    Args:
        gridded (GriddedSelection): gridded measurements (with at least one cell)
        stat (str): statistic

    Returns:
        dict: {"x_edges": (nx+1), "y_edges": (ny+1), "values": (ny, nx) float32, NaN where a
               cell has no data, "epsg": projection EPSG number}, as used by
               cpom.areas.area_plot.Polarplot data sets
    """
    grid = gridded.grid
    rows, cols = np.divmod(gridded.cells, grid.get_ncols_nrows()[0])
    row0, col0 = int(rows.min()), int(cols.min())
    image = np.full(
        (int(rows.max()) - row0 + 1, int(cols.max()) - col0 + 1), np.nan, dtype=np.float32
    )
    image[rows - row0, cols - col0] = gridded.values[stat]
    return {
        "x_edges": grid.minxm + (col0 + np.arange(image.shape[1] + 1)) * grid.binsize,
        "y_edges": grid.minym + (row0 + np.arange(image.shape[0] + 1)) * grid.binsize,
        "values": image,
        "epsg": grid_epsg(grid),
    }
