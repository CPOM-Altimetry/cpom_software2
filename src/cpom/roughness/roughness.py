"""cpom.roughness.roughness

Roughness class to read and interpolate surface Roughness files
"""

from __future__ import annotations

import logging
import os
from multiprocessing.shared_memory import SharedMemory
from typing import Any

import numpy as np
import rasterio  # to extract GeoTIFF extents
import zarr
from pyproj import CRS  # coordinate reference system
from pyproj import Transformer  # transforms
from rasterio.errors import RasterioIOError
from scipy.interpolate import RegularGridInterpolator, interpn
from scipy.ndimage import gaussian_filter
from tifffile import imread  # to support large TIFF files

# pylint: disable=too-many-arguments
# pylint: disable=too-many-statements
# pylint: disable=too-many-branches
# pylint: disable=too-many-locals
# pylint: disable=R0801
# pylint: disable=too-many-lines

log = logging.getLogger(__name__)

# List of supported roughness scenarios
#   - add to this list if you add a new roughness in the roughness class
roughness_list = [
    "rema_100m_900ws_roughness_zarr",  # Roughness calculated from REMA 100m
    # (900m width) [J.Phillips,Lancs]
    "arcticdem_100m_900ws_roughness_zarr",  # Roughness calculated from ArcticDEM 100m
]


class Roughness:
    """class to load and interpolate Polar roughness"""

    # pylint: disable=too-many-instance-attributes

    def __init__(
        self,
        name: str,
        filled: bool = True,
        config: None | dict = None,
        roughness_dir: str | None = None,
        store_in_shared_memory: bool = False,
        thislog: logging.Logger | None = None,
    ):
        """class initialization function

        Args:
            name (str): roughness name id, must be in global roughness_list
            filled (bool, optional): Use filled version of roughness if True. Defaults to True.
            config (dict, optional): configuration dictionary, defaults to None
            roughness_dir (str, optional): path of directory containing roughness. Defaults to None
            store_in_shared_memory (bool, optional): stores/accesses zroughness array in
            SharedMemory
            thislog (logging.Logger|None, optional): attach to a different log instance
        Raises:
            ValueError: when name not in global roughness_list
        """
        self.name = name
        self.crs_wgs = CRS("epsg:4326")  # we are using WGS84 for all roughnesss
        self.config = config
        self.name = name
        self.roughness_dir = roughness_dir
        self.filled = filled
        self.reference_year = 0  # YYYY, the year the roughness's elevations are referenced to
        self.xroughness = np.array([])
        self.yroughness = np.array([])
        self.zroughness = np.array([])
        self.zroughness_flip = np.array([])
        self.minroughnessx = None
        self.minroughnessy = None
        self.binsize = 0
        self.store_in_shared_memory = store_in_shared_memory
        self.shape = ()
        self.dtype = np.float32
        self.shared_mem: Any = None
        self.shared_mem_child = False  # set to True if a child process
        self.npz_type = False  # set to True when using .npz roughness file
        self.zarr_type = False  # set to True when using .zarr roughness file
        self.nc_type = False  # set to True when using NetCDF Roughness files

        # is accessing the roughness's shared memory
        # default is False (parent process which allocates
        # the shared memory). Necessary for tracking who
        # unlinks (parent) or closes (child) the shared
        # memory at the end
        if thislog is not None:
            self.log = thislog  # optionally attach to a different log instance
        else:
            self.log = log

        if name not in roughness_list:
            self.log.error("roughness %s not in allowed list", name)
            raise ValueError(f"roughness name {name} not in allowed list")

        self.load()

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
        except RasterioIOError as exc:
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

    def get_filename(self, default_dir: str, filename: str, filled_filename: str) -> str:
        """Find the path of the roughness file from dir and file names :
        For the directory, it is chosen in order of preference:
        a) self.config["roughness_dirs"][self.name], or
        b) supplied self.roughness_dir, or
        c) default_dir
        The file name is:
        filename: is self.filled use filled_filename

        Args:
            default_dir (str): default dir to find roughness file names
            filename (str): file name of roughness (not filled)
            filled_filename (str): file name of roughness (not filled)
        Returns:
            str : path of roughness file
        Raises:
            OSError : directory or file not found
        """
        this_roughness_dir = None
        if self.config:
            if "roughness_dirs" in self.config and self.name in self.config["roughness_dirs"]:
                this_roughness_dir = self.config["roughness_dirs"][self.name]
        if this_roughness_dir is None and self.roughness_dir:
            this_roughness_dir = self.roughness_dir
        if this_roughness_dir is None:
            this_roughness_dir = default_dir

        if not os.path.isdir(this_roughness_dir):
            raise OSError(f"{this_roughness_dir} not found")
        if self.filled and filled_filename:
            this_path = f"{this_roughness_dir}/{filled_filename}"
        else:
            this_path = f"{this_roughness_dir}/{filename}"

        self.log.info("Loading roughness name: %s", self.name)
        self.log.info("Loading roughness file: %s", this_path)

        if self.zarr_type:
            if not os.path.isdir(this_path):
                raise OSError(f"{this_path} not found")
        elif not os.path.isfile(this_path):
            raise OSError(f"{this_path} not found")

        return this_path

    def clean_up(self):
        """Free up, close or release any shared memory or other resources associated
        with roughness
        """
        if self.store_in_shared_memory:
            try:
                if self.shared_mem is not None:
                    if self.shared_mem_child:
                        self.shared_mem.close()
                        self.log.info("closed shared memory for %s", self.name)
                    else:
                        self.shared_mem.close()
                        self.shared_mem.unlink()
                        self.log.info("unlinked shared memory for %s", self.name)

            except Exception as exc:  # pylint: disable=broad-exception-caught
                self.log.error("Shared memory for %s could not be closed %s", self.name, exc)
        else:
            if self.zarr_type:
                self.zroughness = None
                self.zroughness_flip = None

    def load_npz(self, npz_file: str):
        """Load roughness from npz format file

        Args:
            npz_file (str): path of npz file
        """
        data = np.load(npz_file, allow_pickle=True)
        self.zroughness = data["zroughness"]
        self.xroughness = data["xroughness"]
        self.yroughness = data["yroughness"]
        self.minroughnessx = data["minroughnessx"]
        self.minroughnessy = data["minroughnessy"]
        self.binsize = data["binsize"]

    def load_zarr(self, roughnessfile: str):
        """Load a .zarr file

        Args:
            roughnessfile (str): path of .zarr file
        """

        try:
            zroughness = zarr.open_array(roughnessfile, mode="r")
        except Exception as exc:
            raise IOError(f"Failed to open Zarr file: {roughnessfile} {exc}") from exc

        ncols = zroughness.attrs["ncols"]
        nrows = zroughness.attrs["nrows"]
        top_l = zroughness.attrs["top_l"]
        top_r = zroughness.attrs["top_r"]
        bottom_l = zroughness.attrs["bottom_l"]
        binsize = zroughness.attrs["binsize"]

        self.xroughness = np.linspace(top_l[0], top_r[0], ncols, endpoint=True)
        self.yroughness = np.linspace(bottom_l[1], top_l[1], nrows, endpoint=True)
        self.yroughness = np.flip(self.yroughness)
        self.minroughnessx = self.xroughness.min()
        self.minroughnessy = self.yroughness.min()
        self.binsize = binsize  # grid resolution in m

        self.zroughness = zroughness

        try:
            self.zroughness_flip = zarr.open_array(
                roughnessfile.replace(".zarr", "_flipped.zarr"), mode="r"
            )
        except Exception as exc:
            raise IOError(f"Failed to open Zarr file: {roughnessfile} {exc}") from exc

    def load_geotiff(self, roughnessfile: str):
        """Load a GeoTIFF file

        Args:
            roughnessfile (str): path of GeoTIFF
        """
        (
            ncols,
            nrows,
            top_l,
            top_r,
            bottom_l,
            _,
            binsize,
        ) = self.get_geotiff_extent(roughnessfile)

        if self.store_in_shared_memory:
            # First try attaching to an existing shared memory buffer if it
            # exists with the roughnesss name. If that is unavailable, create the shared memory
            try:
                self.shared_mem = SharedMemory(name=self.name, create=False)
                self.zroughness = np.ndarray(
                    shape=(nrows, ncols), dtype=self.dtype, buffer=self.shared_mem.buf
                )
                self.shared_mem_child = True

                self.log.info("attached to existing shared memory for %s ", self.name)

            except FileNotFoundError as exc:
                zroughness = imread(roughnessfile)

                if not isinstance(zroughness, np.ndarray):
                    raise TypeError(
                        f"roughness image type not supported : {type(zroughness)}"
                    ) from exc

                # Create the shared memory with the appropriate size
                self.shared_mem = SharedMemory(name=self.name, create=True, size=zroughness.nbytes)

                # Link the shared memory to the zroughness data
                self.zroughness = np.ndarray(
                    zroughness.shape, dtype=zroughness.dtype, buffer=self.shared_mem.buf
                )

                # Copy the data from zroughness to the shared_np_array
                self.zroughness[:] = zroughness[:]

                self.log.info("created shared memory for %s", self.name)
        else:
            zroughness = imread(roughnessfile)
            if not isinstance(zroughness, np.ndarray):
                raise TypeError(f"roughness image type not supported : {type(zroughness)}")
            self.zroughness = zroughness

        # Set void data to Nan
        if self.void_value:
            void_data = np.where(self.zroughness == self.void_value)
            if np.any(void_data):
                self.zroughness[void_data] = np.nan

        self.xroughness = np.linspace(top_l[0], top_r[0], ncols, endpoint=True)
        self.yroughness = np.linspace(bottom_l[1], top_l[1], nrows, endpoint=True)
        self.yroughness = np.flip(self.yroughness)
        self.minroughnessx = self.xroughness.min()
        self.minroughnessy = self.yroughness.min()
        self.binsize = binsize  # grid resolution in m

    def load(self) -> bool:
        """load the roughness

        Returns:
            bool: roughness loaded ok (True), failed (False)
        """

        # --------------------------------------------------------------------------------
        if self.name == "rema_100m_900ws_roughness_zarr":
            # roughness calculated from REMA DEM by J.Phillips (CPOM/Lancs),
            # converted to Zarr (A.Muir)
            filename = "REMA_Roughness_100m_900ws.zarr"
            filled_filename = "REMA_Roughness_100m_900ws.zarr"
            # default_dir can be modified in class init
            default_dir = f'{os.environ["CPDATA_DIR"]}/SATS/RA/DEMS/slope_and_rougness'
            self.src_url = "TBD"  # Add REMA src URL
            self.src_url_filled = "TBD"  # Add REMA src URL
            self.roughness_version = "1.1"
            self.src_institute = "CPOM/PGC"
            self.long_name = "Surface Roughness at 100m from REMA"
            self.crs_bng = CRS("epsg:3031")  # Polar Stereo - South -71S
            self.southern_hemisphere = True
            self.void_value = -9999
            self.dtype = np.float32
            self.reference_year = (
                2010  # YYYY, the year the roughness's elevations are referenced to
            )
            self.zarr_type = True  # from a Zarr file type
        elif self.name == "arcticdem_100m_900ws_roughness_zarr":
            # roughness calculated from ArcticDEM by J.Phillips (CPOM/Lancs),
            # converted to Zarr (A.Muir)
            filename = "ArcticDEM_Roughness_100m_900ws.zarr"
            filled_filename = "ArcticDEM_Roughness_100m_900ws.zarr"
            # default_dir can be modified in class init
            default_dir = f'{os.environ["CPDATA_DIR"]}/SATS/RA/DEMS/slope_and_rougness'
            self.src_url = "TBD"  # Add ArcticDEM src URL
            self.src_url_filled = "TBD"  # Add ArcticDEM src URL
            self.roughness_version = "1.1"
            self.src_institute = "PGC"
            self.long_name = "roughness from ArcticDEM"
            self.crs_bng = CRS("epsg:3413")  # Polar Stereo - North -lat of origin 70N, 45
            self.southern_hemisphere = False
            self.void_value = -9999
            self.dtype = np.float32
            self.reference_year = (
                2010  # YYYY, the year the roughness's elevations are referenced to
            )
            self.zarr_type = True

        else:
            raise ValueError(f"{self.name} does not have load support")

        # Form the roughness file name and load the roughness
        # For the directory, it is chosen in order of preference:
        # a) self.config["roughness_dirs"][self.name], or
        # b) supplied self.roughness_dir, or
        # c) default_dir
        try:
            roughnessfile = self.get_filename(default_dir, filename, filled_filename)
        except OSError as exc:
            self.log.error("Could not form roughness path for %s : %s", self.name, exc)
            raise exc

        try:
            if self.npz_type:
                self.load_npz(roughnessfile)
            elif self.zarr_type:
                self.load_zarr(roughnessfile)
            else:
                self.load_geotiff(roughnessfile)
        except IOError as exc:
            self.log.error("Could not load roughness file for %s : %s", roughnessfile, exc)
            return False

        # Setup the Transforms
        self.xy_to_lonlat_transformer = Transformer.from_proj(
            self.crs_bng, self.crs_wgs, always_xy=True
        )
        self.lonlat_to_xy_transformer = Transformer.from_proj(
            self.crs_wgs, self.crs_bng, always_xy=True
        )

        return True

    def get_segment(
        self, segment_bounds: list, grid_xy: bool = True, flatten: bool = False
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """return a cropped segment of the roughness, flattened or as a grid

        Args:
            segment_bounds (List): [(minx,maxx),(miny,maxy)]
            grid_xy (bool, optional): return segment as a grid. Defaults to True.
            flatten (bool, optional): return segment as flattened list. Defaults to False.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: (xroughness,yroughness,zroughness)
        """

        # ----------------------------------------------------------------------
        # Get coord bounds as index bounds
        # ----------------------------------------------------------------------

        minx_ind = (np.absolute(segment_bounds[0][0] - self.xroughness)).argmin()
        maxx_ind = (np.absolute(segment_bounds[0][1] - self.xroughness)).argmin()
        miny_ind = (np.absolute(segment_bounds[1][0] - self.yroughness)).argmin()
        maxy_ind = (np.absolute(segment_bounds[1][1] - self.yroughness)).argmin()

        # ----------------------------------------------------------------------
        # Crop full roughness coords to segment bounds
        # ----------------------------------------------------------------------

        zroughness = self.zroughness[maxy_ind:miny_ind, minx_ind:maxx_ind]
        xroughness = self.xroughness[minx_ind:maxx_ind]
        yroughness = self.yroughness[maxy_ind:miny_ind]

        if grid_xy is True:
            xroughness, yroughness = np.meshgrid(xroughness, yroughness)

            # Set x,y to nan where z is nan
            zroughness_nan = np.isnan(zroughness)
            xroughness[zroughness_nan] = np.nan
            yroughness[zroughness_nan] = np.nan

        # ----------------------------------------------------------------------
        # Return, flattened if requested
        # ----------------------------------------------------------------------

        if flatten is False:
            return (xroughness, yroughness, zroughness)

        return (xroughness.flatten(), yroughness.flatten(), zroughness.flatten())

    def chunked_interpolation(
        self,
        x: np.ndarray,
        y: np.ndarray,
        myyroughness: np.ndarray,
        xroughness: np.ndarray,
        method: str,
    ) -> np.ndarray:
        """Interpolate roughness in chunks to handle large datasets efficiently.

        This function performs interpolation on a roughness stored in a Zarr array by
        extracting relevant chunks and creating a sub-grid for interpolation.

        Args:
            x (np.ndarray): Array of x coordinates in the roughness's projection (in meters).
            y (np.ndarray): Array of y coordinates in the roughness's projection (in meters).
            myyroughness (np.ndarray): Flipped y coordinates corresponding to the roughness grid.
            xroughness (np.ndarray): x coordinates corresponding to the roughness grid.
            method (str): Interpolation method to use ('linear', 'nearest', etc.).

        Returns:
            np.ndarray: Interpolated roughness elevation values at the specified coordinates.
        """
        results = np.full_like(x, np.nan, dtype=np.float64)

        # Identify valid points (where x and y are not NaN)
        valid_mask = ~np.isnan(x) & ~np.isnan(y)

        # Only proceed if there are valid points
        if valid_mask.any():
            x_valid = x[valid_mask]
            y_valid = y[valid_mask]

            # Define the bounding box for valid points
            x_min, x_max = x_valid.min(), x_valid.max()
            y_min, y_max = y_valid.min(), y_valid.max()

            # Determine the indices of the bounding box in the roughness grid
            x_indices = np.searchsorted(xroughness, [x_min, x_max])
            y_indices = np.searchsorted(myyroughness, [y_min, y_max])

            # Expand the indices to ensure we cover the region adequately
            x_indices[0] = max(x_indices[0] - 1, 0)
            x_indices[1] = min(x_indices[1] + 1, len(xroughness) - 1)
            y_indices[0] = max(y_indices[0] - 1, 0)
            y_indices[1] = min(y_indices[1] + 1, len(myyroughness) - 1)

            # Extract the sub-array
            sub_zarr = self.zroughness_flip[
                y_indices[0] : y_indices[1] + 1, x_indices[0] : x_indices[1] + 1
            ]
            sub_zarr = np.array(sub_zarr)

            sub_myyroughness = myyroughness[y_indices[0] : y_indices[1] + 1]
            sub_xroughness = xroughness[x_indices[0] : x_indices[1] + 1]

            # Create an interpolator for the sub-array
            interpolator = RegularGridInterpolator(
                (sub_myyroughness, sub_xroughness),
                sub_zarr,
                method=method,
                bounds_error=False,
                fill_value=np.nan,
            )

            # Perform the interpolation for valid points
            points = np.vstack((y_valid, x_valid)).T
            interpolated_values = interpolator(points)

            # Store the results in the corresponding places
            results[valid_mask] = interpolated_values

        return results

    # ----------------------------------------------------------------------------------------------
    # Interpolate roughness, input x,y can be arrays or single, units m, in projection (epsg:3031")
    # returns the interpolated elevation(s) at x,y
    # x,y : x,y cartesian coordinates in the roughness's projection in m
    # OR, when xy_is_latlon is True:
    # x,y : latitude, longitude values in degs N and E (note the order, not longitude, latitude!)
    #
    # method: string containing the interpolation method. Default is 'linear'. Options are
    # “linear” and “nearest”, and “splinef2d” (see scipy.interpolate.interpn docs).
    #
    # Where your input points are outside the roughness area, then np.nan values will be returned
    # ----------------------------------------------------------------------------------------------

    def interp_roughness(self, x, y, method="linear", xy_is_latlon=False) -> np.ndarray:
        """Interpolate roughness to return elevation values corresponding to
           cartesian x,y in roughness's projection or lat,lon values

        Args:
            x (np.ndarray): x cartesian coordinates in the roughness's projection in m, or lat vals
            y (np.ndarray): x cartesian coordinates in the roughness's projection in m, or lon vals
            method (str, optional): linear, nearest, splinef2d. Defaults to "linear".
            xy_is_latlon (bool, optional): if True, x,y are lat, lon values. Defaults to False.

        Returns:
            np.ndarray: interpolated roughness elevation values
        """

        x = np.array(x)
        y = np.array(y)

        # Transform to x,y if inputs are lat,lon
        if xy_is_latlon:
            x, y = self.lonlat_to_xy_transformer.transform(  # pylint: disable=E0633
                y, x
            )  # transform lon,lat -> x,y
        myyroughness = np.flip(self.yroughness.copy())
        # If zroughness is actually a zarr array instead of a numpy array
        # then we use the pre-flipped zarr version, but need to convert it to
        # a numpy array first (which is slow)
        if self.zarr_type:
            return self.chunked_interpolation(x, y, myyroughness, self.xroughness, method)

        myzroughness = np.flip(self.zroughness.copy(), 0)
        return interpn(
            (myyroughness, self.xroughness),
            myzroughness,
            (y, x),
            method=method,
            bounds_error=False,
            fill_value=np.nan,
        )

    def gaussian_smooth(self, sigma=1.0):
        """
        perform a gaussian smooth on the current loaded roughness
        sigma : degree of smoothing, def=1.0
        """
        # Gaussian smooth roughness
        this_zroughness = self.zroughness.copy()
        this_zroughness[np.isnan(self.zroughness)] = 0
        f_zroughness = gaussian_filter(this_zroughness, sigma=sigma)
        www = 0 * self.zroughness.copy() + 1
        www[np.isnan(self.zroughness)] = 0
        f_www = gaussian_filter(www, sigma=sigma)
        self.zroughness = f_zroughness / f_www

    def hillshade(self, azimuth=225, pitch=45):
        """
        Convert the roughness 'z_roughness' values to a hillshade value between 0..255
        azimuth: angle in degrees (0..360)
        pitch : angle in degrees (0..90)
        """
        azimuth = 360.0 - azimuth

        x, y = np.gradient(self.zroughness)
        roughness = np.pi / 2.0 - np.arctan(np.sqrt(x * x + y * y))
        aspect = np.arctan2(-x, y)
        azimuthrad = azimuth * np.pi / 180.0
        altituderad = pitch * np.pi / 180.0

        shaded = np.sin(altituderad) * np.sin(roughness) + np.cos(altituderad) * np.cos(
            roughness
        ) * np.cos((azimuthrad - np.pi / 2.0) - aspect)

        self.zroughness = 255 * (shaded + 1) / 2
