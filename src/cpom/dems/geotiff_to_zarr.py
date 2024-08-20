"""Utility to convert Geotiff format DEM files to Zarr format using a
chunk size equivalent to the CS2 beamwidth
"""

import argparse
import os
import shutil
import sys

import numpy as np
import rasterio
import zarr

# pylint: disable=too-many-locals


class DEMConverter:
    """Class to convert Geotiff format DEMs to zarr"""

    def __init__(self, void_value: float | None = None):
        self.void_value = void_value

    def get_geotiff_extent(self, fname: str):
        """Get info from GeoTIFF on its extent

        Args:
            fname (str): path of GeoTIFF file

        Raises:
            ValueError: _description_
            IOError: _description_

        Returns:
            tuple(int,int,int,int,int,int,int): width,height,top_left,top_right,bottom_left,
            bottom_right,pixel_width
        """
        try:
            with rasterio.open(fname) as dataset:
                transform = dataset.transform
                width = dataset.width
                height = dataset.height

                top_left = transform * (0, 0)
                top_right = transform * (width, 0)
                bottom_left = transform * (0, height)
                bottom_right = transform * (width, height)

                pixel_width = transform[0]
                pixel_height = -transform[4]  # Negative because the height is
                # typically negative in GeoTIFFs
                if int(pixel_width) != int(pixel_height):
                    raise ValueError(f"pixel_width {pixel_width} != pixel_height {pixel_height}")
        except rasterio.errors.RasterioIOError as exc:
            raise IOError(f"Could not read GeoTIFF: {exc}") from exc
        return (
            width,
            height,
            top_left,
            top_right,
            bottom_left,
            bottom_right,
            pixel_width,
        )

    def convert_geotiff_to_zarr(
        self,
        demfile: str,
        zarrfile: str,
        flipped_zarrfile: str,
    ):
        """Convert a GeoTIFF file to Zarr format and create a flipped version.

        Args:
            demfile (str): Path of GeoTIFF file.
            zarrfile (str): Path where the original Zarr file will be saved.
            flipped_zarrfile (str): Path where the flipped Zarr file will be saved.
        """

        (
            ncols,
            nrows,
            top_l,
            top_r,
            bottom_l,
            _,
            binsize,
        ) = self.get_geotiff_extent(demfile)

        print(f"Converting {demfile} to zarr")
        print(f"binsize {binsize}, ncols {ncols}, nrows {nrows}")

        chunk_x = int(ncols * binsize / 20000)  # set chunk size to ~20km
        chunk_y = int(nrows * binsize / 20000)

        print(f"chunk size chosen {chunk_x} {chunk_y}")
        chunk_size = (chunk_x, chunk_y)

        try:
            with rasterio.open(demfile) as src:
                ncols, nrows = src.width, src.height
                dtype = src.dtypes[0]

                # Create Zarr arrays
                compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=zarr.Blosc.SHUFFLE)
                zarr_array = zarr.open_array(
                    zarrfile,
                    mode="w",
                    shape=(nrows, ncols),
                    chunks=chunk_size,
                    dtype=dtype,
                    compressor=compressor,
                )
                flipped_zarr_array = zarr.open_array(
                    flipped_zarrfile,
                    mode="w",
                    shape=(nrows, ncols),
                    chunks=chunk_size,
                    dtype=dtype,
                    compressor=compressor,
                )

                # Read data in chunks and write to Zarr
                for i in range(0, nrows, chunk_size[0]):
                    for j in range(0, ncols, chunk_size[1]):
                        window = rasterio.windows.Window(j, i, chunk_size[1], chunk_size[0])
                        data = src.read(1, window=window, masked=True)

                        # Handle void values
                        if self.void_value is not None:
                            data = data.filled(np.nan)
                            data[data == self.void_value] = np.nan

                        # Determine the actual size of the chunk
                        actual_chunk_size = data.shape

                        zarr_array[
                            i : i + actual_chunk_size[0], j : j + actual_chunk_size[1]
                        ] = data
                        flipped_zarr_array[
                            (nrows - i - actual_chunk_size[0]) : (nrows - i),
                            j : j + actual_chunk_size[1],
                        ] = data[::-1, :]

                # Save metadata
                zarr_array.attrs.update(
                    {
                        "transform": src.transform.to_gdal(),
                        "crs": src.crs.to_string(),
                        "void_value": self.void_value,
                    }
                )
                flipped_zarr_array.attrs.update(
                    {
                        "transform": src.transform.to_gdal(),
                        "crs": src.crs.to_string(),
                        "void_value": self.void_value,
                    }
                )

                zarr_array.attrs.update(
                    {
                        "ncols": ncols,
                        "nrows": nrows,
                        "top_l": top_l,
                        "top_r": top_r,
                        "bottom_l": bottom_l,
                        "binsize": binsize,
                    }
                )
                flipped_zarr_array.attrs.update(
                    {
                        "ncols": ncols,
                        "nrows": nrows,
                        "top_l": top_l,
                        "top_r": top_r,
                        "bottom_l": bottom_l,
                        "binsize": binsize,
                    }
                )

        except Exception as exc:
            raise IOError(f"Failed to convert GeoTIFF to Zarr: {exc}") from exc


def main():
    """main function for command line tool"""

    # ----------------------------------------------------------------------
    # Process Command Line Arguments for tool
    # ----------------------------------------------------------------------

    # initiate the command line parser
    parser = argparse.ArgumentParser()

    # add each argument

    parser.add_argument(
        "--tiff_file",
        "-f",
        help=("path of tiff DEM file to convert to Zarr"),
        required=True,
    )

    parser.add_argument(
        "--void_value",
        "-v",
        type=int,
        help=("void value in DEM tiff file (ie -9999)"),
        required=True,
    )

    parser.add_argument(
        "--outdir",
        "-o",
        type=str,
        help=("directory to write Zarr to (not the Zarr dir itself)"),
        required=False,
    )

    # read arguments from the command line
    args = parser.parse_args()

    if not os.path.isfile(args.tiff_file):
        sys.exit(f"{args.tiff_file} could not be found")

    if args.tiff_file.endswith(".tif"):
        ends = ".tif"
    elif args.tiff_file.endswith(".tiff"):
        ends = ".tiff"
    else:
        sys.exit("Must end in .tif or .tiff")

    tiff_file = args.tiff_file
    zarr_file = args.tiff_file.replace(ends, ".zarr")
    zarr_flipped_file = args.tiff_file.replace(ends, "_flipped.zarr")
    if args.outdir:
        zarr_file = f"{args.outdir}/{os.path.basename(zarr_file)}"
        zarr_flipped_file = f"{args.outdir}/{os.path.basename(zarr_flipped_file)}"

    if os.path.isdir(zarr_file):
        print(f"{zarr_file} already exists")
        try:
            shutil.rmtree(zarr_file)
            print(f"Directory {zarr_file} and " "all its contents have been removed.")
        except FileNotFoundError:
            print(f"Directory {zarr_file} does not exist.")
        except Exception as e:  # pylint: disable=broad-exception-caught
            print(f"Error: {e}")

    if os.path.isdir(zarr_flipped_file):
        print(f"{zarr_flipped_file} already exists")
        try:
            shutil.rmtree(zarr_flipped_file)
            print(f"Directory {zarr_flipped_file} and " "all its contents have been removed.")
        except FileNotFoundError:
            print(f"Directory {zarr_flipped_file} does not exist.")
        except Exception as e:  # pylint: disable=broad-exception-caught
            print(f"Error: {e}")

    DEMConverter(void_value=args.void_value).convert_geotiff_to_zarr(
        tiff_file,
        zarr_file,
        zarr_flipped_file,
    )

    print(f"Source TIFF file: {tiff_file}")
    print(f"Created {zarr_file}")
    print(f"Created {zarr_flipped_file}")


if __name__ == "__main__":
    main()

# # Example usage
# DEMConverter(void_value=-9999).convert_geotiff_to_zarr(
#     "/cpdata/SATS/RA/DEMS/ant_rema_200m_dem/REMA_200m_dem_filled.tif",
#     "/cpdata/SATS/RA/DEMS/ant_rema_200m_dem/REMA_200m_dem_filled.zarr",
#     "/cpdata/SATS/RA/DEMS/ant_rema_200m_dem/REMA_200m_dem_filled_flipped.zarr",
# )

# sys.exit(1)

# DEMConverter(void_value=-9999).convert_geotiff_to_zarr(
#     "/cpdata/SATS/RA/DEMS/arctic_dem_1km/arcticdem_mosaic_1km_v3.0.tif",
#     "/cpdata/SATS/RA/DEMS/arctic_dem_1km/arcticdem_mosaic_1km_v3.0.zarr",
#     "/cpdata/SATS/RA/DEMS/arctic_dem_1km/arcticdem_mosaic_1km_v3.0_flipped.zarr",
# )


# # Example usage
# DEMConverter(void_value=-9999).convert_geotiff_to_zarr(
#     "/cpdata/SATS/RA/DEMS/rema_1km_dem/REMA_1km_dem_filled.tif",
#     "/cpdata/SATS/RA/DEMS/rema_1km_dem/REMA_1km_dem_filled.zarr",
#     "/cpdata/SATS/RA/DEMS/rema_1km_dem/REMA_1km_dem_filled_flipped.zarr",
# )
# DEMConverter(void_value=-9999).convert_geotiff_to_zarr(
#     "/cpdata/SATS/RA/DEMS/rema_1km_dem_v2/rema_mosaic_1km_v2.0_filled_cop30_dem.tif",
#     "/cpdata/SATS/RA/DEMS/rema_1km_dem_v2/rema_mosaic_1km_v2.0_filled_cop30_dem.zarr",
#     "/cpdata/SATS/RA/DEMS/rema_1km_dem_v2/rema_mosaic_1km_v2.0_filled_cop30_dem_flipped.zarr",
# )
# DEMConverter(void_value=-32767).convert_geotiff_to_zarr(
#     "/cpdata/SATS/RA/DEMS/rema_gapless_100m/GaplessREMA100.tif",
#     "/cpdata/SATS/RA/DEMS/rema_gapless_100m/GaplessREMA100.zarr",
#     "/cpdata/SATS/RA/DEMS/rema_gapless_100m/GaplessREMA100_flipped.zarr",
# )
# DEMConverter(void_value=-32767).convert_geotiff_to_zarr(
#     "/cpdata/SATS/RA/DEMS/rema_gapless_100m/GaplessREMA1km.tif",
#     "/cpdata/SATS/RA/DEMS/rema_gapless_100m/GaplessREMA1km.zarr",
#     "/cpdata/SATS/RA/DEMS/rema_gapless_100m/GaplessREMA1km_flipped.zarr",
# )
# DEMConverter(void_value=-9999).convert_geotiff_to_zarr(
#     "/cpdata/SATS/RA/DEMS/arctic_dem_100m_v4.1/arcticdem_mosaic_100m_v4.1_subarea_greenland.tif",
#     "/cpdata/SATS/RA/DEMS/arctic_dem_100m_v4.1/arcticdem_mosaic_100m_v4.1_subarea_greenland.zarr",
#     (
#         "/cpdata/SATS/RA/DEMS/arctic_dem_100m_v4.1/"
#         "arcticdem_mosaic_100m_v4.1_subarea_greenland_flipped.zarr"
#     ),
# )
# DEMConverter(void_value=-9999).convert_geotiff_to_zarr(
#     "/cpdata/SATS/RA/DEMS/arctic_dem_1km/arcticdem_mosaic_1km_v3.0.tif",
#     "/cpdata/SATS/RA/DEMS/arctic_dem_1km/arcticdem_mosaic_1km_v3.0.zarr",
#     "/cpdata/SATS/RA/DEMS/arctic_dem_1km/arcticdem_mosaic_1km_v3.0_flipped.zarr",
# )
# DEMConverter(void_value=-9999).convert_geotiff_to_zarr(
#     "/cpdata/SATS/RA/DEMS/arctic_dem_1km/arcticdem_mosaic_1km_v3.0_subarea_greenland.tif",
#     "/cpdata/SATS/RA/DEMS/arctic_dem_1km/arcticdem_mosaic_1km_v3.0_subarea_greenland.zarr",
#     "/cpdata/SATS/RA/DEMS/arctic_dem_1km/"
#     "arcticdem_mosaic_1km_v3.0_subarea_greenland_flipped.zarr",
# )
# DEMConverter(void_value=-9999).convert_geotiff_to_zarr(
#     "/cpdata/SATS/RA/DEMS/arctic_dem_1km_v4.1/arcticdem_mosaic_1km_v4.1_dem.tif",
#     "/cpdata/SATS/RA/DEMS/arctic_dem_1km_v4.1/arcticdem_mosaic_1km_v4.1_dem.zarr",
#     "/cpdata/SATS/RA/DEMS/arctic_dem_1km_v4.1/arcticdem_mosaic_1km_v4.1_dem_flipped.zarr",
# )
# DEMConverter(void_value=-9999).convert_geotiff_to_zarr(
#     "/cpdata/SATS/RA/DEMS/arctic_dem_1km_v4.1/arcticdem_mosaic_1km_v4.1_subarea_greenland.tif",
#     "/cpdata/SATS/RA/DEMS/arctic_dem_1km_v4.1/arcticdem_mosaic_1km_v4.1_subarea_greenland.zarr",
#     (
#         "/cpdata/SATS/RA/DEMS/arctic_dem_1km_v4.1/"
#         "arcticdem_mosaic_1km_v4.1_subarea_greenland_flipped.zarr"
#     ),
# )
