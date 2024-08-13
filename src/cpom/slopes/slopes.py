"""cpom.slopes.slopes

Slope class to read and interpolate surface slope files
"""

from __future__ import annotations

import logging
import os
from multiprocessing.shared_memory import SharedMemory
from typing import Any

import numpy as np
import rasterio  # to extract GeoTIFF extents
import zarr
from netCDF4 import Dataset  # pylint: disable=no-name-in-module
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

# List of supported slopes
#   - add to this list if you add a new slopes in the slopes class
slope_list = [
    "rema_100m_900ws_slopes_zarr",  # slope calculated from REMA 100m
    # (900m width) [J.Phillips,Lancs]
    "arcticdem_100m_900ws_slopes_zarr",  # slope calculated from ArcticDEM 100m
    # (900m width) [J.Phillips,Lancs]
    "awi_grn_2013_1km_slopes",  # Greenland slopes from AWI/CryoSat at 1km
    "cpom_ant_2018_1km_slopes",  # Antarctic slope at 1km from Slater (2018)/CPOM
]


class Slopes:
    """class to load and interpolate Polar slopes"""

    # pylint: disable=too-many-instance-attributes

    def __init__(
        self,
        name: str,
        filled: bool = True,
        config: None | dict = None,
        slopes_dir: str | None = None,
        store_in_shared_memory: bool = False,
        thislog: logging.Logger | None = None,
    ):
        """class initialization function

        Args:
            name (str): slopes name id, must be in global slope_list
            filled (bool, optional): Use filled version of slopes if True. Defaults to True.
            config (dict, optional): configuration dictionary, defaults to None
            slopes_dir (str, optional): path of directory containing slopes. Defaults to None
            store_in_shared_memory (bool, optional): stores/accesses zslopes array in SharedMemory
            thislog (logging.Logger|None, optional): attach to a different log instance
        Raises:
            ValueError: when name not in global slope_list
        """
        self.name = name
        self.crs_wgs = CRS("epsg:4326")  # we are using WGS84 for all slopess
        self.config = config
        self.name = name
        self.slopes_dir = slopes_dir
        self.filled = filled
        self.reference_year = 0  # YYYY, the year the slopes's elevations are referenced to
        self.xslopes = np.array([])
        self.yslopes = np.array([])
        self.zslopes = np.array([])
        self.zslopes_flip = np.array([])
        self.minslopesx = None
        self.minslopesy = None
        self.binsize = 0
        self.store_in_shared_memory = store_in_shared_memory
        self.shape = ()
        self.dtype = np.float32
        self.shared_mem: Any = None
        self.shared_mem_child = False  # set to True if a child process
        self.npz_type = False  # set to True when using .npz slopes file
        self.zarr_type = False  # set to True when using .zarr slopes file
        self.nc_type = False  # set to True when using NetCDF slope files

        # is accessing the slopes's shared memory
        # default is False (parent process which allocates
        # the shared memory). Necessary for tracking who
        # unlinks (parent) or closes (child) the shared
        # memory at the end
        if thislog is not None:
            self.log = thislog  # optionally attach to a different log instance
        else:
            self.log = log

        if name not in slope_list:
            self.log.error("slopes %s not in allowed list", name)
            raise ValueError(f"slopes name {name} not in allowed list")

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
        """Find the path of the slopes file from dir and file names :
        For the directory, it is chosen in order of preference:
        a) self.config["slopes_dirs"][self.name], or
        b) supplied self.slopes_dir, or
        c) default_dir
        The file name is:
        filename: is self.filled use filled_filename

        Args:
            default_dir (str): default dir to find slopes file names
            filename (str): file name of slopes (not filled)
            filled_filename (str): file name of slopes (not filled)
        Returns:
            str : path of slopes file
        Raises:
            OSError : directory or file not found
        """
        this_slopes_dir = None
        if self.config:
            if "slopes_dirs" in self.config and self.name in self.config["slopes_dirs"]:
                this_slopes_dir = self.config["slopes_dirs"][self.name]
        if this_slopes_dir is None and self.slopes_dir:
            this_slopes_dir = self.slopes_dir
        if this_slopes_dir is None:
            this_slopes_dir = default_dir

        if not os.path.isdir(this_slopes_dir):
            raise OSError(f"{this_slopes_dir} not found")
        if self.filled and filled_filename:
            this_path = f"{this_slopes_dir}/{filled_filename}"
        else:
            this_path = f"{this_slopes_dir}/{filename}"

        self.log.info("Loading slopes name: %s", self.name)
        self.log.info("Loading slopes file: %s", this_path)

        if self.zarr_type:
            if not os.path.isdir(this_path):
                raise OSError(f"{this_path} not found")
        elif not os.path.isfile(this_path):
            raise OSError(f"{this_path} not found")

        return this_path

    def clean_up(self):
        """Free up, close or release any shared memory or other resources associated
        with slopes
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
                self.zslopes = None
                self.zslopes_flip = None

    def load_npz(self, npz_file: str):
        """Load slopes from npz format file

        Args:
            npz_file (str): path of npz file
        """
        data = np.load(npz_file, allow_pickle=True)
        self.zslopes = data["zslopes"]
        self.xslopes = data["xslopes"]
        self.yslopes = data["yslopes"]
        self.minslopesx = data["minslopesx"]
        self.minslopesy = data["minslopesy"]
        self.binsize = data["binsize"]

    def load_zarr(self, slopesfile: str):
        """Load a .zarr file

        Args:
            slopesfile (str): path of .zarr file
        """

        try:
            zslopes = zarr.open_array(slopesfile, mode="r")
        except Exception as exc:
            raise IOError(f"Failed to open Zarr file: {slopesfile} {exc}") from exc

        ncols = zslopes.attrs["ncols"]
        nrows = zslopes.attrs["nrows"]
        top_l = zslopes.attrs["top_l"]
        top_r = zslopes.attrs["top_r"]
        bottom_l = zslopes.attrs["bottom_l"]
        binsize = zslopes.attrs["binsize"]

        self.xslopes = np.linspace(top_l[0], top_r[0], ncols, endpoint=True)
        self.yslopes = np.linspace(bottom_l[1], top_l[1], nrows, endpoint=True)
        self.yslopes = np.flip(self.yslopes)
        self.minslopesx = self.xslopes.min()
        self.minslopesy = self.yslopes.min()
        self.binsize = binsize  # grid resolution in m

        self.zslopes = zslopes

        try:
            self.zslopes_flip = zarr.open_array(
                slopesfile.replace(".zarr", "_flipped.zarr"), mode="r"
            )
        except Exception as exc:
            raise IOError(f"Failed to open Zarr file: {slopesfile} {exc}") from exc

    def load_nc(self, slopesfile: str):
        """load NetCDF slope files

        Args:
            slopesfile (str): path of netcdf slope file
        """
        nc_dem = Dataset(slopesfile)

        self.zslopes = nc_dem.variables["slope"][:]
        # self.slopes = np.flip(self.slopes,0)

        nrows = self.zslopes.shape[0]
        ncols = self.zslopes.shape[1]

        # self.slopes = im
        # self.slopes = im.reshape((nrows, ncols))
        # self.slopes = np.flip(self.slopes,0)
        minx = -1823000.0
        maxx = 1973000.0
        miny = -3441000.0
        maxy = -533000.0

        self.xslopes = np.linspace(minx, maxx, ncols, endpoint=True)
        self.yslopes = np.linspace(miny, maxy, nrows, endpoint=True)
        # self.xmesh, self.ymesh = np.meshgrid(self.x, self.y)

        # Set void data to Nan
        if self.void_value:
            void_data = np.where(self.zslopes == self.void_value)
            if np.any(void_data):
                self.zslopes[void_data] = np.nan

        self.minslopesx = self.xslopes.min()
        self.minslopesy = self.yslopes.min()
        self.binsize = 1000  # grid resolution in m

    def load_geotiff(self, slopesfile: str):
        """Load a GeoTIFF file

        Args:
            slopesfile (str): path of GeoTIFF
        """
        (
            ncols,
            nrows,
            top_l,
            top_r,
            bottom_l,
            _,
            binsize,
        ) = self.get_geotiff_extent(slopesfile)

        if self.store_in_shared_memory:
            # First try attaching to an existing shared memory buffer if it
            # exists with the slopess name. If that is unavailable, create the shared memory
            try:
                self.shared_mem = SharedMemory(name=self.name, create=False)
                self.zslopes = np.ndarray(
                    shape=(nrows, ncols), dtype=self.dtype, buffer=self.shared_mem.buf
                )
                self.shared_mem_child = True

                self.log.info("attached to existing shared memory for %s ", self.name)

            except FileNotFoundError as exc:
                zslopes = imread(slopesfile)

                if not isinstance(zslopes, np.ndarray):
                    raise TypeError(f"slopes image type not supported : {type(zslopes)}") from exc

                # Create the shared memory with the appropriate size
                self.shared_mem = SharedMemory(name=self.name, create=True, size=zslopes.nbytes)

                # Link the shared memory to the zslopes data
                self.zslopes = np.ndarray(
                    zslopes.shape, dtype=zslopes.dtype, buffer=self.shared_mem.buf
                )

                # Copy the data from zslopes to the shared_np_array
                self.zslopes[:] = zslopes[:]

                self.log.info("created shared memory for %s", self.name)
        else:
            zslopes = imread(slopesfile)
            if not isinstance(zslopes, np.ndarray):
                raise TypeError(f"slopes image type not supported : {type(zslopes)}")
            self.zslopes = zslopes

        # Set void data to Nan
        if self.void_value:
            void_data = np.where(self.zslopes == self.void_value)
            if np.any(void_data):
                self.zslopes[void_data] = np.nan

        self.xslopes = np.linspace(top_l[0], top_r[0], ncols, endpoint=True)
        self.yslopes = np.linspace(bottom_l[1], top_l[1], nrows, endpoint=True)
        self.yslopes = np.flip(self.yslopes)
        self.minslopesx = self.xslopes.min()
        self.minslopesy = self.yslopes.min()
        self.binsize = binsize  # grid resolution in m

    def load(self) -> bool:
        """load the slopes

        Returns:
            bool: slopes loaded ok (True), failed (False)
        """

        # --------------------------------------------------------------------------------
        if self.name == "rema_100m_900ws_slopes_zarr":
            # Slopes calculated from REMA DEM by J.Phillips (CPOM/Lancs),
            # converted to Zarr (A.Muir)
            filename = "REMA_Slope_100m_900ws.zarr"
            filled_filename = "REMA_Slope_100m_900ws.zarr"
            # default_dir can be modified in class init
            default_dir = f'{os.environ["CPDATA_DIR"]}/SATS/RA/DEMS/slope_and_rougness'
            self.src_url = "TBD"  # Add REMA src URL
            self.src_url_filled = "TBD"  # Add REMA src URL
            self.slopes_version = "1.1"
            self.src_institute = "CPOM/PGC"
            self.long_name = "Surface slope at 100m from REMA"
            self.crs_bng = CRS("epsg:3031")  # Polar Stereo - South -71S
            self.southern_hemisphere = True
            self.void_value = -9999
            self.dtype = np.float32
            self.reference_year = 2010  # YYYY, the year the slopes's elevations are referenced to
            self.zarr_type = True  # from a Zarr file type
        elif self.name == "arcticdem_100m_900ws_slopes_zarr":
            # Slopes calculated from ArcticDEM by J.Phillips (CPOM/Lancs),
            # converted to Zarr (A.Muir)
            filename = "ArcticDEM_Slope_100m_900ws.zarr"
            filled_filename = "ArcticDEM_Slope_100m_900ws.zarr"
            # default_dir can be modified in class init
            default_dir = f'{os.environ["CPDATA_DIR"]}/SATS/RA/DEMS/slope_and_rougness'
            self.src_url = "TBD"  # Add ArcticDEM src URL
            self.src_url_filled = "TBD"  # Add ArcticDEM src URL
            self.slopes_version = "1.1"
            self.src_institute = "PGC"
            self.long_name = "slopes from ArcticDEM"
            self.crs_bng = CRS("epsg:3413")  # Polar Stereo - North -lat of origin 70N, 45
            self.southern_hemisphere = False
            self.void_value = -9999
            self.dtype = np.float32
            self.reference_year = 2010  # YYYY, the year the slopes's elevations are referenced to
            self.zarr_type = True
        elif self.name == "cpom_ant_2018_1km_slopes":
            filename = "Antarctica_Cryosat2_1km_DEMv1.0_slope.unpacked.tif"
            filled_filename = "Antarctica_Cryosat2_1km_DEMv1.0_slope.unpacked.tif"
            # default_dir can be modified in class init
            default_dir = f'{os.environ["CPDATA_DIR"]}/SATS/RA/DEMS/ant_cpom_cs2_1km'
            self.southern_hemisphere = True
            self.src_url = "TBD"  # Add src URL
            self.src_url_filled = "TBD"  # Add src URL
            self.src_institute = "CPOM"
            self.slopes_version = "1.0"
            self.void_value = -9999
            self.dtype = np.float32
            self.long_name = "Antarctic slopes from CryoSat by Slater(2018)/CPOM"
            self.crs_bng = CRS("epsg:3031")  # Polar Stereo - South -71S
            self.coordinate_reference_system = CRS(
                "epsg:3031"
            )  # WGS 84 / Antarctic Polar Stereographic, lon0=0E, X along 90E, Y along 0E
            self.reference_year = 2016  # YYYY, the year the slopes's elevations are referenced to
            self.zarr_type = False
        elif self.name == "awi_grn_2013_1km_slopes":
            self.zarr_type = False
            self.nc_type = True
            default_dir = f'{os.environ["CPDATA_DIR"]}/SATS/RA/DEMS/grn_awi_2013_dem'
            filename = "grn_awi_2013_dem_slope.nc"
            filled_filename = filename
            self.long_name = "Greenland Slopes from AWI (2013"
            self.src_url = "TBD"  # Add src URL
            self.src_url_filled = "TBD"  # Add src URL
            self.src_institute = "AWI"
            self.slopes_version = "1.0"
            self.southern_hemisphere = False
            self.crs_bng = CRS("epsg:3413")  # Polar Stereo - North -lat of origin 70N, 45
            self.void_value = -9999

        else:
            raise ValueError(f"{self.name} does not have load support")

        # Form the slopes file name and load the slopes
        # For the directory, it is chosen in order of preference:
        # a) self.config["slopes_dirs"][self.name], or
        # b) supplied self.slopes_dir, or
        # c) default_dir
        try:
            slopesfile = self.get_filename(default_dir, filename, filled_filename)
        except OSError as exc:
            self.log.error("Could not form slopes path for %s : %s", self.name, exc)
            return False

        if self.npz_type:
            self.load_npz(slopesfile)
        elif self.zarr_type:
            self.load_zarr(slopesfile)
        elif self.nc_type:
            self.load_nc(slopesfile)
        else:
            self.load_geotiff(slopesfile)

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
        """return a cropped segment of the slopes, flattened or as a grid

        Args:
            segment_bounds (List): [(minx,maxx),(miny,maxy)]
            grid_xy (bool, optional): return segment as a grid. Defaults to True.
            flatten (bool, optional): return segment as flattened list. Defaults to False.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: (xslopes,yslopes,zslopes)
        """

        # ----------------------------------------------------------------------
        # Get coord bounds as index bounds
        # ----------------------------------------------------------------------

        minx_ind = (np.absolute(segment_bounds[0][0] - self.xslopes)).argmin()
        maxx_ind = (np.absolute(segment_bounds[0][1] - self.xslopes)).argmin()
        miny_ind = (np.absolute(segment_bounds[1][0] - self.yslopes)).argmin()
        maxy_ind = (np.absolute(segment_bounds[1][1] - self.yslopes)).argmin()

        # ----------------------------------------------------------------------
        # Crop full slopes coords to segment bounds
        # ----------------------------------------------------------------------

        zslopes = self.zslopes[maxy_ind:miny_ind, minx_ind:maxx_ind]
        xslopes = self.xslopes[minx_ind:maxx_ind]
        yslopes = self.yslopes[maxy_ind:miny_ind]

        if grid_xy is True:
            xslopes, yslopes = np.meshgrid(xslopes, yslopes)

            # Set x,y to nan where z is nan
            zslopes_nan = np.isnan(zslopes)
            xslopes[zslopes_nan] = np.nan
            yslopes[zslopes_nan] = np.nan

        # ----------------------------------------------------------------------
        # Return, flattened if requested
        # ----------------------------------------------------------------------

        if flatten is False:
            return (xslopes, yslopes, zslopes)

        return (xslopes.flatten(), yslopes.flatten(), zslopes.flatten())

    def chunked_interpolation(
        self, x: np.ndarray, y: np.ndarray, myyslopes: np.ndarray, xslopes: np.ndarray, method: str
    ) -> np.ndarray:
        """Interpolate slopes in chunks to handle large datasets efficiently.

        This function performs interpolation on a slopes stored in a Zarr array by
        extracting relevant chunks and creating a sub-grid for interpolation.

        Args:
            x (np.ndarray): Array of x coordinates in the slopes's projection (in meters).
            y (np.ndarray): Array of y coordinates in the slopes's projection (in meters).
            myyslopes (np.ndarray): Flipped y coordinates corresponding to the slopes grid.
            xslopes (np.ndarray): x coordinates corresponding to the slopes grid.
            method (str): Interpolation method to use ('linear', 'nearest', etc.).

        Returns:
            np.ndarray: Interpolated slopes elevation values at the specified coordinates.
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

            # Determine the indices of the bounding box in the slopes grid
            x_indices = np.searchsorted(xslopes, [x_min, x_max])
            y_indices = np.searchsorted(myyslopes, [y_min, y_max])

            # Expand the indices to ensure we cover the region adequately
            x_indices[0] = max(x_indices[0] - 1, 0)
            x_indices[1] = min(x_indices[1] + 1, len(xslopes) - 1)
            y_indices[0] = max(y_indices[0] - 1, 0)
            y_indices[1] = min(y_indices[1] + 1, len(myyslopes) - 1)

            # Extract the sub-array
            sub_zarr = self.zslopes_flip[
                y_indices[0] : y_indices[1] + 1, x_indices[0] : x_indices[1] + 1
            ]
            sub_zarr = np.array(sub_zarr)

            sub_myyslopes = myyslopes[y_indices[0] : y_indices[1] + 1]
            sub_xslopes = xslopes[x_indices[0] : x_indices[1] + 1]

            # Create an interpolator for the sub-array
            interpolator = RegularGridInterpolator(
                (sub_myyslopes, sub_xslopes),
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
    # Interpolate slopes, input x,y can be arrays or single, units m, in projection (epsg:3031")
    # returns the interpolated elevation(s) at x,y
    # x,y : x,y cartesian coordinates in the slopes's projection in m
    # OR, when xy_is_latlon is True:
    # x,y : latitude, longitude values in degs N and E (note the order, not longitude, latitude!)
    #
    # method: string containing the interpolation method. Default is 'linear'. Options are
    # “linear” and “nearest”, and “splinef2d” (see scipy.interpolate.interpn docs).
    #
    # Where your input points are outside the slopes area, then np.nan values will be returned
    # ----------------------------------------------------------------------------------------------

    def interp_slopes(self, x, y, method="linear", xy_is_latlon=False) -> np.ndarray:
        """Interpolate slopes to return elevation values corresponding to
           cartesian x,y in slopes's projection or lat,lon values

        Args:
            x (np.ndarray): x cartesian coordinates in the slopes's projection in m, or lat values
            y (np.ndarray): x cartesian coordinates in the slopes's projection in m, or lon values
            method (str, optional): linear, nearest, splinef2d. Defaults to "linear".
            xy_is_latlon (bool, optional): if True, x,y are lat, lon values. Defaults to False.

        Returns:
            np.ndarray: interpolated slopes elevation values
        """

        x = np.array(x)
        y = np.array(y)

        # Transform to x,y if inputs are lat,lon
        if xy_is_latlon:
            x, y = self.lonlat_to_xy_transformer.transform(  # pylint: disable=E0633
                y, x
            )  # transform lon,lat -> x,y
        myyslopes = np.flip(self.yslopes.copy())
        # If zslopes is actually a zarr array instead of a numpy array
        # then we use the pre-flipped zarr version, but need to convert it to
        # a numpy array first (which is slow)
        if self.zarr_type:
            return self.chunked_interpolation(x, y, myyslopes, self.xslopes, method)

        myzslopes = np.flip(self.zslopes.copy(), 0)
        return interpn(
            (myyslopes, self.xslopes),
            myzslopes,
            (y, x),
            method=method,
            bounds_error=False,
            fill_value=np.nan,
        )

    def gaussian_smooth(self, sigma=1.0):
        """
        perform a gaussian smooth on the current loaded slopes
        sigma : degree of smoothing, def=1.0
        """
        # Gaussian smooth slopes
        this_zslopes = self.zslopes.copy()
        this_zslopes[np.isnan(self.zslopes)] = 0
        f_zslopes = gaussian_filter(this_zslopes, sigma=sigma)
        www = 0 * self.zslopes.copy() + 1
        www[np.isnan(self.zslopes)] = 0
        f_www = gaussian_filter(www, sigma=sigma)
        self.zslopes = f_zslopes / f_www

    def hillshade(self, azimuth=225, pitch=45):
        """
        Convert the slopes 'z_slopes' values to a hillshade value between 0..255
        azimuth: angle in degrees (0..360)
        pitch : angle in degrees (0..90)
        """
        azimuth = 360.0 - azimuth

        x, y = np.gradient(self.zslopes)
        slope = np.pi / 2.0 - np.arctan(np.sqrt(x * x + y * y))
        aspect = np.arctan2(-x, y)
        azimuthrad = azimuth * np.pi / 180.0
        altituderad = pitch * np.pi / 180.0

        shaded = np.sin(altituderad) * np.sin(slope) + np.cos(altituderad) * np.cos(slope) * np.cos(
            (azimuthrad - np.pi / 2.0) - aspect
        )

        self.zslopes = 255 * (shaded + 1) / 2
