#!/usr/bin/env python3
"""
Utility to convert NetCDF format DEM files to Zarr format using a
chunk size equivalent to the CS2 beamwidth or a selectable chunk size.
"""
import argparse
import os
import shutil
import sys

import numpy as np
import xarray as xr
import zarr


class DEMConverter:
    """Class to convert NetCDF DEM files to Zarr"""

    def __init__(self, void_value: float | None = None):
        self.void_value = void_value

    def get_netcdf_extent(self, fname: str):
        """
        Get DEM extent and resolution from the NetCDF file.

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
        ds = xr.open_dataset(fname)
        # Assume the file contains 1D coordinate variables 'x' and 'y'
        x = ds["x"].values
        y = ds["y"].values
        ds.close()

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
        ncfile: str,
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
            _,  # bottom_right,
            binsize,
        ) = self.get_netcdf_extent(ncfile)

        print(f"Converting {ncfile} to Zarr")
        print(f"binsize: {binsize}, ncols: {ncols}, nrows: {nrows}")

        # Compute chunk size in grid cells so that each chunk is ~chunk_width meters.
        chunk_y = int(nrows * binsize / chunk_width)
        chunk_x = int(ncols * binsize / chunk_width)
        chunk_y = max(chunk_y, 1)
        chunk_x = max(chunk_x, 1)
        chunk_size = (chunk_y, chunk_x)
        print(f"Chunk size chosen (rows, cols): {chunk_size}")

        # Open the NetCDF file and extract georeferencing metadata from "Polar_Stereographic"
        ds = xr.open_dataset(ncfile)
        if "Polar_Stereographic" in ds.variables:
            ps_attr = ds["Polar_Stereographic"].attrs
            transform = ps_attr.get("GeoTransform", None)
            if transform is not None:
                if hasattr(transform, "tolist"):
                    transform = transform.tolist()
                elif isinstance(transform, (list, tuple)):
                    transform = [float(val) for val in transform]
                else:
                    transform = float(transform)
            crs = ps_attr.get("spatial_ref", None)
            if crs is not None:
                crs = str(crs)
        else:
            transform = None
            crs = None

        # Read the DEM variable 'h'
        h_data = ds["h"].values
        ds.close()

        # --- NEW STEP ---
        # Flip the DEM data vertically so that row 0 corresponds to the top.
        # This makes the NetCDF DEM match the GeoTIFF orientation.
        h_data = h_data[::-1, :]

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
            "transform": transform,
            "crs": crs,
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


def main():
    """Main function for the command-line tool."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--nc_file",
        "-f",
        help="Path of the NetCDF DEM file to convert to Zarr",
        required=True,
    )
    parser.add_argument(
        "--void_value",
        "-v",
        type=float,
        help="Void value in DEM NetCDF file (e.g. 3.402823e+38)",
        required=False,
        default=3.402823e38,
    )
    parser.add_argument(
        "--outdir",
        "-o",
        type=str,
        help="Directory to write Zarr output (not the Zarr directory itself)",
        required=False,
    )
    parser.add_argument(
        "--chunk_size",
        "-c",
        type=int,
        help="Chunk size in meters, default is 20000 (i.e. 20km)",
        required=False,
        default=20000,
    )
    args = parser.parse_args()

    if not os.path.isfile(args.nc_file):
        sys.exit(f"{args.nc_file} could not be found")
    if not args.nc_file.endswith(".nc"):
        sys.exit("NetCDF file must end in .nc")

    nc_file = args.nc_file
    zarr_file = nc_file.replace(".nc", ".zarr")
    zarr_flipped_file = nc_file.replace(".nc", "_flipped.zarr")
    if args.outdir:
        zarr_file = os.path.join(args.outdir, os.path.basename(zarr_file))
        zarr_flipped_file = os.path.join(args.outdir, os.path.basename(zarr_flipped_file))

    # Remove existing Zarr directories if they exist.
    for zfile in (zarr_file, zarr_flipped_file):
        if os.path.isdir(zfile):
            print(f"{zfile} already exists")
            try:
                shutil.rmtree(zfile)
                print(f"Directory {zfile} and all its contents have been removed.")
            except Exception as e:  # pylint: disable=broad-exception-caught
                print(f"Error removing {zfile}: {e}")

    DEMConverter(void_value=args.void_value).convert_netcdf_to_zarr(
        nc_file,
        zarr_file,
        zarr_flipped_file,
        chunk_width=args.chunk_size,
    )

    print(f"Source NetCDF file: {nc_file}")
    print(f"Created {zarr_file}")
    print(f"Created {zarr_flipped_file}")


if __name__ == "__main__":
    main()
