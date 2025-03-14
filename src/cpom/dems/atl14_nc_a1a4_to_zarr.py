#!/usr/bin/env python3
"""
Utility to convert NetCDF format DEM files to Zarr format using a
chunk size equivalent to the CS2 beamwidth or a selectable chunk size.
"""
import os
import shutil

import numpy as np
import zarr
from netCDF4 import Dataset  # pylint:disable=E0611


class DEMConverter:
    """Class to convert NetCDF DEM files to Zarr"""

    def __init__(self, void_value: float | None = None):
        self.void_value = void_value

    def get_netcdf_extent(self, x, y):
        """
        Get DEM extent and resolution .

        Reads the coordinate arrays 'x' and 'y' and computes:
         - ncols, nrows,
         - binsize (assumed equal in x and y),
         - corner coordinates (approximated using the cell centers and binsize).

        Args:
            fname (str): path to the NetCDF file.

        Raises:
            ValueError: if the resolutions in x and y are not equal.

        Returns:
            tuple: (ncols, nrows, top_left, top_right, bottom_left, bottom_right, binsize)
        """

        ncols = int(len(x))
        nrows = int(len(y))
        # Compute pixel resolutions from the coordinates
        dx = float(abs(x[1] - x[0]))
        dy = float(abs(y[1] - y[0]))
        if int(dx) != int(dy):
            raise ValueError(f"resolution_x ({dx}) != resolution_y ({dy})")
        binsize = dx

        # Compute approximate corner coordinates (as native Python floats)
        top_left = (float(x.min()), float(y.max()))
        top_right = (float(x.max()), float(y.max()))
        bottom_left = (float(x.min()), float(y.min()))
        bottom_right = (float(x.max()), float(y.min()))

        print(f"bottom_left {bottom_left} ncols {ncols} nrows {nrows}")

        return ncols, nrows, top_left, top_right, bottom_left, bottom_right, binsize

    def convert_netcdf_to_zarr(
        self,
        x,
        y,
        h,
        zarrfile: str,
        flipped_zarrfile: str,
        chunk_width: int = 20000,
    ):
        """
        Convert a NetCDF DEM file to Zarr format and create a flipped version.

        Args:
            ncfile (str): Path of the NetCDF file.
            zarrfile (str): Path where the original Zarr file will be saved.
            flipped_zarrfile (str): Path where the flipped Zarr file will be saved.
            chunk_width (int, optional): chunk size in meters (default is 20000).
        """
        (
            ncols,
            nrows,
            top_left,
            top_right,
            bottom_left,
            _,  # bottom_right
            binsize,
        ) = self.get_netcdf_extent(x, y)

        print(f"binsize: {binsize}, ncols: {ncols}, nrows: {nrows}")

        # Compute chunk size in grid cells so that each chunk is ~chunk_width meters.
        chunk_y = int(nrows * binsize / chunk_width)
        chunk_x = int(ncols * binsize / chunk_width)
        chunk_y = max(chunk_y, 1)
        chunk_x = max(chunk_x, 1)
        chunk_size = (chunk_y, chunk_x)
        print(f"Chunk size chosen (rows, cols): {chunk_size}")

        # --- NEW STEP ---
        # Flip the DEM data vertically so that row 0 corresponds to the top.
        # This makes the NetCDF DEM match the GeoTIFF orientation.
        h_data = h[::-1, :]

        # Replace void values with np.nan (if specified)
        if self.void_value is not None:
            h_data = np.where(h_data == self.void_value, np.nan, h_data)

        # Create Zarr arrays
        compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=zarr.Blosc.SHUFFLE)
        zarr_array = zarr.open_array(
            zarrfile,
            mode="w",
            shape=(nrows, ncols),
            chunks=chunk_size,
            dtype=h_data.dtype,
            compressor=compressor,
        )
        flipped_zarr_array = zarr.open_array(
            flipped_zarrfile,
            mode="w",
            shape=(nrows, ncols),
            chunks=chunk_size,
            dtype=h_data.dtype,
            compressor=compressor,
        )

        # Write data in chunks
        for i in range(0, nrows, chunk_size[0]):
            for j in range(0, ncols, chunk_size[1]):
                data_chunk = h_data[i : i + chunk_size[0], j : j + chunk_size[1]]
                actual_chunk_shape = data_chunk.shape

                zarr_array[i : i + actual_chunk_shape[0], j : j + actual_chunk_shape[1]] = (
                    data_chunk
                )
                flipped_zarr_array[
                    (nrows - i - actual_chunk_shape[0]) : (nrows - i),
                    j : j + actual_chunk_shape[1],
                ] = data_chunk[::-1, :]

        # Prepare metadata ensuring all values are JSON serializable (native Python types).
        geo_metadata = {
            "transform": [-2670000.0, 100.0, 0.0, 2630000.0, 0.0, -100.0],
            "crs": "EPSG:3031",
            "void_value": float(self.void_value) if self.void_value is not None else None,
        }
        grid_metadata = {
            "ncols": ncols,
            "nrows": nrows,
            "top_l": (float(top_left[0]), float(top_left[1])),
            "top_r": (float(top_right[0]), float(top_right[1])),
            "bottom_l": (float(bottom_left[0]), float(bottom_left[1])),
            "binsize": binsize,
        }
        # Update attributes in two steps (as in your original GeoTIFF converter)
        zarr_array.attrs.update(geo_metadata)
        flipped_zarr_array.attrs.update(geo_metadata)
        zarr_array.attrs.update(grid_metadata)
        flipped_zarr_array.attrs.update(grid_metadata)

        print(f"Created {zarrfile}")
        print(f"Created {flipped_zarrfile}")


def combine_nc_files(a1_file, a2_file, a3_file, a4_file):
    """combine x,y,h(y,x) from 4 netcdf files
       each file contains x,y values stepped by 100.00 but
       in different segments of cartesian space.
       we want the final x,y,h(y,x) to include the maximum extent
       of the contributing data.

    Args:
        a1_file (str): file name of netcdf file 1
        a2_file (str): file name of netcdf file 2
        a3_file (str): file name of netcdf file 3
        a4_file (str): file name of netcdf file 4

    Returns:
        h,x,y: h(y,x), x, y are all numpy arrays. h is a 2d array
    """
    # Read each file
    print("Reading A2...")
    nc2 = Dataset(a2_file)
    h2 = nc2["h"][:]
    x2 = nc2["x"][:]
    y2 = nc2["y"][:]
    print(
        f"A2: xmin {np.nanmin(x2)} xmax {np.nanmax(x2)} ymin {np.nanmin(y2)} ymax {np.nanmax(y2)}"
    )

    print("Reading A1...")
    nc1 = Dataset(a1_file)
    h1 = nc1["h"][:]
    x1 = nc1["x"][:]
    y1 = nc1["y"][:]
    print(
        f"A1: xmin {np.nanmin(x1)} xmax {np.nanmax(x1)} ymin {np.nanmin(y1)} ymax {np.nanmax(y1)}"
    )

    print("Reading A3...")
    nc3 = Dataset(a3_file)
    h3 = nc3["h"][:]
    x3 = nc3["x"][:]
    y3 = nc3["y"][:]
    print(
        f"A3: xmin {np.nanmin(x3)} xmax {np.nanmax(x3)} ymin {np.nanmin(y3)} ymax {np.nanmax(y3)}"
    )

    print("Reading A4...")
    nc4 = Dataset(a4_file)
    h4 = nc4["h"][:]
    x4 = nc4["x"][:]
    y4 = nc4["y"][:]
    print(
        f"A4: xmin {np.nanmin(x4)} xmax {np.nanmax(x4)} ymin {np.nanmin(y4)} ymax {np.nanmax(y4)}"
    )

    minx1 = np.nanmin(x1)
    minx2 = np.nanmin(x2)
    minx3 = np.nanmin(x3)
    minx4 = np.nanmin(x4)
    maxx1 = np.nanmax(x1)
    maxx2 = np.nanmax(x2)
    maxx3 = np.nanmax(x3)
    maxx4 = np.nanmax(x4)

    miny1 = np.nanmin(y1)
    miny2 = np.nanmin(y2)
    miny3 = np.nanmin(y3)
    miny4 = np.nanmin(y4)
    maxy1 = np.nanmax(y1)
    maxy2 = np.nanmax(y2)
    maxy3 = np.nanmax(y3)
    maxy4 = np.nanmax(y4)

    print(f"A1_x {minx1}  {maxx1}")
    print(f"A1_y {miny1}  {maxy1}")

    # Define global extents (with step size 100)
    step = 100.0
    global_xmin = np.nanmin((minx1, minx2, minx3, minx4))
    global_xmax = np.nanmax((maxx1, maxx2, maxx3, maxx4))
    global_ymin = np.nanmin((miny1, miny2, miny3, miny4))
    global_ymax = np.nanmax((maxy1, maxy2, maxy3, maxy4))

    print(f"global_x {global_xmin}  {global_xmax}")
    print(f"global_y {global_ymin}  {global_ymax}")

    # Create global coordinate arrays.
    global_x = np.arange(global_xmin, global_xmax + step, step)
    global_y = np.arange(global_ymin, global_ymax + step, step)

    # Initialize final global h grid with FillValue.
    fv = 3.402823e38
    h_grid = np.full((global_y.size, global_x.size), fv)

    for iy, y in enumerate(global_y):
        for ix, x in enumerate(global_x):
            if minx1 <= x <= maxx1 and miny1 <= y <= maxy1:
                # find index in to h1
                hx = int((x - minx1) / step)
                hy = int((y - miny1) / step)

                h_grid[iy, ix] = h1[hy, hx]

            if minx2 <= x <= maxx2 and miny2 <= y <= maxy2:
                # find index in to h1
                hx = int((x - minx2) / step)
                hy = int((y - miny2) / step)

                h_grid[iy, ix] = h2[hy, hx]

            if minx3 <= x <= maxx3 and miny3 <= y <= maxy3:
                # find index in to h1
                hx = int((x - minx3) / step)
                hy = int((y - miny3) / step)

                h_grid[iy, ix] = h3[hy, hx]

            if minx4 <= x <= maxx4 and miny4 <= y <= maxy4:
                # find index in to h1
                hx = int((x - minx4) / step)
                hy = int((y - miny4) / step)

                h_grid[iy, ix] = h4[hy, hx]

    return h_grid, global_x, global_y


def combine_nc_files2(a1_file, a2_file, a3_file, a4_file):
    """Combine x, y, h(y,x) from 4 netCDF files into a single grid."""

    # Read each file and collect data along with their extents
    datasets = []
    file_order = [a1_file, a2_file, a3_file, a4_file]
    for i, file in enumerate(file_order):
        print(f"Reading A{i+1}...")
        with Dataset(file) as nc:
            h = nc["h"][:]
            x = nc["x"][:]
            y = nc["y"][:]
            minx, maxx = np.nanmin(x), np.nanmax(x)
            miny, maxy = np.nanmin(y), np.nanmax(y)
            print(f"A{i+1}: xmin {minx} xmax {maxx} ymin {miny} ymax {maxy}")
            datasets.append((h, minx, maxx, miny, maxy))

    # Determine global extents
    all_minx = [d[1] for d in datasets]
    all_maxx = [d[2] for d in datasets]
    all_miny = [d[3] for d in datasets]
    all_maxy = [d[4] for d in datasets]

    global_xmin = np.nanmin(all_minx)
    global_xmax = np.nanmax(all_maxx)
    global_ymin = np.nanmin(all_miny)
    global_ymax = np.nanmax(all_maxy)

    print(f"Global x: {global_xmin} to {global_xmax}")
    print(f"Global y: {global_ymin} to {global_ymax}")

    # Create global coordinate arrays with step 100.0
    step = 100.0
    global_x = np.arange(global_xmin, global_xmax + step, step)
    global_y = np.arange(global_ymin, global_ymax + step, step)

    # Initialize output grid with fill value
    fv = 3.402823e38
    h_grid = np.full((global_y.size, global_x.size), fv)

    # Process each dataset to fill the global grid
    for h, minx, maxx, miny, maxy in datasets[:1]:
        # Determine coverage in global grid
        x_mask = (global_x >= minx) & (global_x <= maxx)
        y_mask = (global_y >= miny) & (global_y <= maxy)
        ix = np.where(x_mask)[0]
        iy = np.where(y_mask)[0]

        if not ix.size or not iy.size:
            continue

        # Calculate indices in the dataset's grid
        x_vals = global_x[x_mask]
        y_vals = global_y[y_mask]
        hx = np.round((x_vals - minx) / step).astype(int)
        hy = np.round((y_vals - miny) / step).astype(int)

        # Clip indices to prevent out-of-bounds errors
        hx = np.clip(hx, 0, h.shape[1] - 1)
        hy = np.clip(hy, 0, h.shape[0] - 1)

        # Update the global grid using vectorized assignment
        h_grid[np.ix_(iy, ix)] = h[np.ix_(hy, hx)]

    return h_grid, global_x, global_y


def combine_nc_files3(a1_file, a2_file, a3_file, a4_file):
    """Combine x, y, h(y,x) from 4 netCDF files with a uniform grid spacing.

    Each file contains x, y values stepped by 100.0, covering different segments
    of Cartesian space. This function creates a global grid covering the maximum extent
    of the data and fills in the values from each file.

    Args:
        a1_file, a2_file, a3_file, a4_file (str): File names of the netCDF files.

    Returns:
        tuple: (h_grid, global_x, global_y)
            - h_grid (2D numpy array): Combined h values over the global grid.
            - global_x (1D numpy array): Global x coordinates.
            - global_y (1D numpy array): Global y coordinates.
    """
    # List of file names
    files = [a1_file, a2_file, a3_file, a4_file]

    # Read all datasets and extract x, y, h arrays
    datasets = []
    for fname in files:
        with Dataset(fname) as nc:
            x = nc["x"][:]
            y = nc["y"][:]
            h = nc["h"][:]
            datasets.append({"x": x, "y": y, "h": h})

    # Determine global extents using the file boundaries
    step = 100.0
    global_xmin = min(data["x"][0] for data in datasets)
    global_xmax = max(data["x"][-1] for data in datasets)
    global_ymin = min(data["y"][0] for data in datasets)
    global_ymax = max(data["y"][-1] for data in datasets)

    global_x = np.arange(global_xmin, global_xmax + step, step)
    global_y = np.arange(global_ymin, global_ymax + step, step)

    # Initialize the global h grid with the FillValue.
    fv = 3.402823e38
    h_grid = np.full((global_y.size, global_x.size), fv)

    # Insert data from each file into the global grid
    for data in datasets:
        x = data["x"]
        y = data["y"]
        h = data["h"]

        # Compute the starting indices in the global grid for this dataset.
        ix0 = int((x[0] - global_xmin) / step)
        iy0 = int((y[0] - global_ymin) / step)

        nx = x.size
        ny = y.size

        # Directly assign the block from h into the corresponding slice of h_grid.
        h_grid[iy0 : iy0 + ny, ix0 : ix0 + nx] = h

    return h_grid, global_x, global_y


def main():
    """Main function for the command-line tool."""

    a1_file = "/cpdata/SATS/RA/DEMS/ATL14/ATL14_A1_0324_100m_004_04.nc"  # NE
    a2_file = "/cpdata/SATS/RA/DEMS/ATL14/ATL14_A2_0324_100m_004_04.nc"  # NW
    a3_file = "/cpdata/SATS/RA/DEMS/ATL14/ATL14_A3_0324_100m_004_04.nc"  # SW
    a4_file = "/cpdata/SATS/RA/DEMS/ATL14/ATL14_A4_0324_100m_004_04.nc"  # SE

    zarr_file = "/cpdata/SATS/RA/DEMS/ATL14/ATL14_0324_100m_004_04.zarr"
    zarr_flipped_file = zarr_file.replace(".zarr", "_flipped.zarr")

    # Remove existing Zarr directories if they exist.
    for zfile in (zarr_file, zarr_flipped_file):
        if os.path.isdir(zfile):
            print(f"{zfile} already exists")
            try:
                shutil.rmtree(zfile)
                print(f"Directory {zfile} and all its contents have been removed.")
            except Exception as e:  # pylint: disable=broad-exception-caught
                print(f"Error removing {zfile}: {e}")

    h, x, y = combine_nc_files(a1_file, a2_file, a3_file, a4_file)

    print(f"combined h {np.shape(h)}")
    print(f"combined x {np.shape(x)}")
    print(f"combined y {np.shape(y)}")

    DEMConverter(void_value=3.402823e38).convert_netcdf_to_zarr(
        x,
        y,
        h,
        zarr_file,
        zarr_flipped_file,
        chunk_width=20000,
    )

    print(f"Created {zarr_file}")
    print(f"Created {zarr_flipped_file}")


if __name__ == "__main__":
    main()
