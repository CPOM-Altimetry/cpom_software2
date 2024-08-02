"""cpom.dems.dems.py

DEM class to read and interpolate DEMs
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

# List of supported DEMs
#   - add to this list if you add a new DEM in the Dem class
dem_list = [
    "awi_ant_1km",  # AWI (2014) DEM of Antarctica from CS2. Helm et al.
    "awi_grn_1km",  # AWI (2014) DEM of Greenland from CS2. Helm et al.
    "awi_ant_1km_grounded",  # awi_ant_1km but masked for grounded ice only, from Bedmachine v2 mask
    "awi_ant_1km_floating",  # awi_ant_1km but masked for floating ice only from Bedmachine v2 mask
    "rema_ant_1km",  # Antarctic REMA v1.1 at 1km
    "rema_ant_1km_zarr",  # Antarctic REMA v1.1 at 1km
    "rema_ant_1km_v2",  # Antarctic REMA v2.0 at 1km
    "rema_ant_1km_v2_zarr",  # Antarctic REMA v2.0 at 1km in zarr format
    "rema_ant_200m",  # Antarctic REMA v1.1 at 200m
    "rema_ant_200m_zarr",  # Antarctic REMA v1.1 at 200m, zarr format
    "rema_gapless_100m",  # REMA (v1.1)Gapless DEM Antarctica at 100m,:
    "rema_gapless_100m_zarr",  # REMA (v1.1)Gapless DEM Antarctica at 100m, Zarr format:
    # https://doi.org/10.1016/j.isprsjprs.2022.01.024
    "rema_gapless_1km",  # REMA (v1.1)Gapless DEM Antarctica at 1km,:
    "rema_gapless_1km_zarr",  # REMA (v1.1)Gapless DEM Antarctica at 1km,:
    # https://doi.org/10.1016/j.isprsjprs.2022.01.024
    "arcticdem_1km",  # ArcticDEM v3.0 at 1km
    "arcticdem_1km_zarr",  # ArcticDEM v3.0 at 1km, Zarr format
    "arcticdem_1km_v4.1",  # ArcticDEM v4.1 at 1km
    "arcticdem_1km_greenland_v4.1",  # ArcticDEM v4.1 at 1km, subarea greenland
    "arcticdem_1km_greenland_v4.1_zarr",  # ArcticDEM v4.1 at 1km, subarea greenland
    "arcticdem_100m_greenland",  # ArcticDEM v3.0, 100m resolution, subarea greenland
    "arcticdem_100m_greenland_v4.1",  # ArcticDEM v4.1, 100m resolution, subarea greenland
    "arcticdem_100m_greenland_v4.1_zarr",  # Zarr format of ArcticDEM v4.1, 100m resolution, grn
]


class Dem:
    """class to load and interpolate Polar DEMs"""

    # pylint: disable=too-many-instance-attributes

    def __init__(
        self,
        name: str,
        filled: bool = True,
        config: None | dict = None,
        dem_dir: str | None = None,
        store_in_shared_memory: bool = False,
        thislog: logging.Logger | None = None,
    ):
        """class initialization function

        Args:
            name (str): DEM name id, must be in global dem_list
            filled (bool, optional): Use filled version of DEM if True. Defaults to True.
            config (dict, optional): configuration dictionary, defaults to None
            dem_dir (str, optional): path of directory containing DEM. Defaults to None
            store_in_shared_memory (bool, optional): stores/accesses zdem array in SharedMemory
            thislog (logging.Logger|None, optional): attach to a different log instance
        Raises:
            ValueError: when name not in global dem_list
        """
        self.name = name
        self.crs_wgs = CRS("epsg:4326")  # we are using WGS84 for all DEMs
        self.config = config
        self.name = name
        self.dem_dir = dem_dir
        self.filled = filled
        self.reference_year = 0  # YYYY, the year the DEM's elevations are referenced to
        self.xdem = np.array([])
        self.ydem = np.array([])
        self.zdem = np.array([])
        self.zdem_flip = np.array([])
        self.mindemx = None
        self.mindemy = None
        self.binsize = 0
        self.store_in_shared_memory = store_in_shared_memory
        self.shape = ()
        self.dtype = np.float32
        self.shared_mem: Any = None
        self.shared_mem_child = False  # set to True if a child process
        self.npz_type = False  # set to True when using .npz DEM file
        self.zarr_type = False  # set to True when using .zarr DEM file

        # is accessing the Dem's shared memory
        # default is False (parent process which allocates
        # the shared memory). Necessary for tracking who
        # unlinks (parent) or closes (child) the shared
        # memory at the end
        if thislog is not None:
            self.log = thislog  # optionally attach to a different log instance
        else:
            self.log = log

        if name not in dem_list:
            self.log.error("DEM %s not in allowed list", name)
            raise ValueError(f"DEM name {name} not in allowed list")

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
        """Find the path of the DEM file from dir and file names :
        For the directory, it is chosen in order of preference:
        a) self.config["dem_dirs"][self.name], or
        b) supplied self.dem_dir, or
        c) default_dir
        The file name is:
        filename: is self.filled use filled_filename

        Args:
            default_dir (str): default dir to find DEM file names
            filename (str): file name of DEM (not filled)
            filled_filename (str): file name of DEM (not filled)
        Returns:
            str : path of DEM file
        Raises:
            OSError : directory or file not found
        """
        this_dem_dir = None
        if self.config:
            if "dem_dirs" in self.config and self.name in self.config["dem_dirs"]:
                this_dem_dir = self.config["dem_dirs"][self.name]
        if this_dem_dir is None and self.dem_dir:
            this_dem_dir = self.dem_dir
        if this_dem_dir is None:
            this_dem_dir = default_dir

        if not os.path.isdir(this_dem_dir):
            raise OSError(f"{this_dem_dir} not found")
        if self.filled and filled_filename:
            this_path = f"{this_dem_dir}/{filled_filename}"
        else:
            this_path = f"{this_dem_dir}/{filename}"

        self.log.info("Loading dem name: %s", self.name)
        self.log.info("Loading dem file: %s", this_path)

        if self.zarr_type:
            if not os.path.isdir(this_path):
                raise OSError(f"{this_path} not found")
        elif not os.path.isfile(this_path):
            raise OSError(f"{this_path} not found")

        return this_path

    def clean_up(self):
        """Free up, close or release any shared memory or other resources associated
        with DEM
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
                self.zdem = None
                self.zdem_flip = None

    def load_npz(self, npz_file: str):
        """Load DEM from npz format file

        Args:
            npz_file (str): path of npz file
        """
        data = np.load(npz_file, allow_pickle=True)
        self.zdem = data["zdem"]
        self.xdem = data["xdem"]
        self.ydem = data["ydem"]
        self.mindemx = data["mindemx"]
        self.mindemy = data["mindemy"]
        self.binsize = data["binsize"]

    def load_zarr(self, demfile: str):
        """Load a .zarr file

        Args:
            demfile (str): path of .zarr file
        """

        try:
            zdem = zarr.open_array(demfile, mode="r")
        except Exception as exc:
            raise IOError(f"Failed to open Zarr file: {demfile} {exc}") from exc

        ncols = zdem.attrs["ncols"]
        nrows = zdem.attrs["nrows"]
        top_l = zdem.attrs["top_l"]
        top_r = zdem.attrs["top_r"]
        bottom_l = zdem.attrs["bottom_l"]
        binsize = zdem.attrs["binsize"]

        self.xdem = np.linspace(top_l[0], top_r[0], ncols, endpoint=True)
        self.ydem = np.linspace(bottom_l[1], top_l[1], nrows, endpoint=True)
        self.ydem = np.flip(self.ydem)
        self.mindemx = self.xdem.min()
        self.mindemy = self.ydem.min()
        self.binsize = binsize  # grid resolution in m

        self.zdem = zdem

        try:
            self.zdem_flip = zarr.open_array(demfile.replace(".zarr", "_flipped.zarr"), mode="r")
        except Exception as exc:
            raise IOError(f"Failed to open Zarr file: {demfile} {exc}") from exc

    def load_geotiff(self, demfile: str):
        """Load a GeoTIFF file

        Args:
            demfile (str): path of GeoTIFF
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

        if self.store_in_shared_memory:
            # First try attaching to an existing shared memory buffer if it
            # exists with the DEMs name. If that is unavailable, create the shared memory
            try:
                self.shared_mem = SharedMemory(name=self.name, create=False)
                self.zdem = np.ndarray(
                    shape=(nrows, ncols), dtype=self.dtype, buffer=self.shared_mem.buf
                )
                self.shared_mem_child = True

                self.log.info("attached to existing shared memory for %s ", self.name)

            except FileNotFoundError as exc:
                zdem = imread(demfile)

                if not isinstance(zdem, np.ndarray):
                    raise TypeError(f"DEM image type not supported : {type(zdem)}") from exc

                # Create the shared memory with the appropriate size
                self.shared_mem = SharedMemory(name=self.name, create=True, size=zdem.nbytes)

                # Link the shared memory to the zdem data
                self.zdem = np.ndarray(zdem.shape, dtype=zdem.dtype, buffer=self.shared_mem.buf)

                # Copy the data from zdem to the shared_np_array
                self.zdem[:] = zdem[:]

                self.log.info("created shared memory for %s", self.name)
        else:
            zdem = imread(demfile)
            if not isinstance(zdem, np.ndarray):
                raise TypeError(f"DEM image type not supported : {type(zdem)}")
            self.zdem = zdem

        # Set void data to Nan
        if self.void_value:
            void_data = np.where(self.zdem == self.void_value)
            if np.any(void_data):
                self.zdem[void_data] = np.nan

        self.xdem = np.linspace(top_l[0], top_r[0], ncols, endpoint=True)
        self.ydem = np.linspace(bottom_l[1], top_l[1], nrows, endpoint=True)
        self.ydem = np.flip(self.ydem)
        self.mindemx = self.xdem.min()
        self.mindemy = self.ydem.min()
        self.binsize = binsize  # grid resolution in m

    def load(self) -> bool:
        """load the DEM

        Returns:
            bool: DEM loaded ok (True), failed (False)
        """
        # --------------------------------------------------------------------------------
        if self.name == "awi_ant_1km":
            filename = "DEM_ANT_CS_20130901.tif"
            filled_filename = "DEM_ANT_CS_20130901.tif"
            default_dir = f'{os.environ["CPDATA_DIR"]}/SATS/RA/DEMS/ant_awi_2013_dem'
            self.src_url = (
                "http://data.pgc.umn.edu/elev/dem/setsm/REMA/mosaic/v1.1/1km/REMA_1km_dem.tif"
            )
            self.src_url_filled = (
                "http://data.pgc.umn.edu/elev/dem/setsm/REMA/mosaic/v1.1/1km/"
                "REMA_1km_dem_filled.tif"
            )
            self.dem_version = "1.0"
            self.src_institute = "AWI"
            self.long_name = "AWI Antarctic DEM (2013)"
            self.crs_bng = CRS("epsg:3031")  # Polar Stereo - South -71S
            self.southern_hemisphere = True
            self.void_value = -9999
            self.dtype = np.float32
            self.reference_year = 0  # YYYY, the year the DEM's elevations are referenced to

        # --------------------------------------------------------------------------------
        # awi_ant_1km but masked for grounded ice only, using Bedmachine v2 mask
        elif self.name == "awi_ant_1km_grounded":
            filename = "ant_awi_2013_dem_grounded.npz"
            filled_filename = "ant_awi_2013_dem_grounded.npz"
            default_dir = f'{os.environ["CPDATA_DIR"]}/SATS/RA/DEMS/ant_awi_2013_dem'
            self.src_url = ""
            self.src_url_filled = ""
            self.dem_version = "1.0"
            self.src_institute = "AWI"
            self.long_name = "AWI Antarctic DEM (2013), Grounded Ice only"
            self.crs_bng = CRS("epsg:3031")  # Polar Stereo - South -71S
            self.southern_hemisphere = True
            self.void_value = -9999
            self.dtype = np.float32
            self.reference_year = 0  # YYYY, the year the DEM's elevations are referenced to
            self.npz_type = True

        # --------------------------------------------------------------------------------
        # awi_ant_1km but masked for floating ice only, using Bedmachine v2 mask
        elif self.name == "awi_ant_1km_floating":
            filename = "ant_awi_2013_dem_floating.npz"
            filled_filename = "ant_awi_2013_dem_floating.npz"
            default_dir = f'{os.environ["CPDATA_DIR"]}/SATS/RA/DEMS/ant_awi_2013_dem'
            self.src_url = ""
            self.src_url_filled = ""
            self.dem_version = "1.0"
            self.src_institute = "AWI"
            self.long_name = "AWI Antarctic DEM (2013), Floating Ice only"
            self.crs_bng = CRS("epsg:3031")  # Polar Stereo - South -71S
            self.southern_hemisphere = True
            self.void_value = -9999
            self.dtype = np.float32
            self.reference_year = 0  # YYYY, the year the DEM's elevations are referenced to
            self.npz_type = True
        # --------------------------------------------------------------------------------
        elif self.name == "awi_grn_1km":
            filename = "DEM_GRE_CS_20130826.tif"
            filled_filename = "DEM_GRE_CS_20130826.tif"
            default_dir = f'{os.environ["CPDATA_DIR"]}/SATS/RA/DEMS/grn_awi_2013_dem'
            self.src_url = ""
            self.src_url_filled = ""
            self.dem_version = "1.0"
            self.src_institute = "AWI"
            self.long_name = "AWI Greenlabd DEM (2013)"
            self.crs_bng = CRS("epsg:3413")  # Polar Stereo - South -71S
            self.southern_hemisphere = False
            self.void_value = -9999
            self.dtype = np.float32
            self.reference_year = 0  # YYYY, the year the DEM's elevations are referenced to

        # --------------------------------------------------------------------------------
        elif self.name == "arcticdem_1km":
            # Arctic DEM at 1km resolution

            filename = "arcticdem_mosaic_1km_v3.0.tif"
            filled_filename = ""
            default_dir = f'{os.environ["CPDATA_DIR"]}/SATS/RA/DEMS/arctic_dem_1km'
            self.src_url = (
                "http://data.pgc.umn.edu/elev/dem/setsm/ArcticDEM/mosaic/"
                "v3.0/1km/arcticdem_mosaic_1km_v3.0.tif"
            )
            self.reference_year = 2010  # YYYY, the year the DEM's elevations are referenced to
            self.src_url_filled = ""
            self.dem_version = "3.0"
            self.src_institute = "PGC"
            self.long_name = "ArcticDEM 1km"
            self.crs_bng = CRS("epsg:3413")  # Polar Stereo - North -lat of origin 70N, 45
            self.southern_hemisphere = False
            self.void_value = -9999
            self.dtype = np.float32

        # --------------------------------------------------------------------------------
        elif self.name == "arcticdem_1km_zarr":
            # Arctic DEM at 1km resolution

            filename = "arcticdem_mosaic_1km_v3.0.zarr"
            filled_filename = ""
            default_dir = f'{os.environ["CPDATA_DIR"]}/SATS/RA/DEMS/arctic_dem_1km'
            self.src_url = (
                "http://data.pgc.umn.edu/elev/dem/setsm/ArcticDEM/mosaic/"
                "v3.0/1km/arcticdem_mosaic_1km_v3.0.tif"
            )
            self.reference_year = 2010  # YYYY, the year the DEM's elevations are referenced to
            self.src_url_filled = ""
            self.dem_version = "3.0"
            self.src_institute = "PGC"
            self.long_name = "ArcticDEM 1km"
            self.crs_bng = CRS("epsg:3413")  # Polar Stereo - North -lat of origin 70N, 45
            self.southern_hemisphere = False
            self.void_value = -9999
            self.dtype = np.float32
            self.zarr_type = True

        # --------------------------------------------------------------------------------
        elif self.name == "arcticdem_1km_v4.1":
            # Arctic DEM v4.1 at 1km resolution

            filename = "arcticdem_mosaic_1km_v4.1_dem.tif"
            filled_filename = "arcticdem_mosaic_1km_v4.1_dem.tif"
            default_dir = f'{os.environ["CPDATA_DIR"]}/SATS/RA/DEMS/arctic_dem_1km_v4.1'
            self.src_url = (
                "https://data.pgc.umn.edu/elev/dem/setsm/ArcticDEM/mosaic"
                "/latest/1km/arcticdem_mosaic_1km_v4.1_dem.tif"
            )
            self.reference_year = 2010  # YYYY, the year the DEM's elevations are referenced to
            self.src_url_filled = ""
            self.dem_version = "4.1"
            self.src_institute = "PGC"
            self.long_name = "ArcticDEM 1km (4.1)"
            self.crs_bng = CRS("epsg:3413")  # Polar Stereo - North -lat of origin 70N, 45
            self.southern_hemisphere = False
            self.void_value = -9999
            self.dtype = np.float32

        # --------------------------------------------------------------------------------
        elif self.name == "rema_ant_1km":
            # REMA Antarctic 1km DEM  v1.1 (PGC 2018)
            # The void areas will contain null values (-9999) in lieu of the terrain elevations.

            filename = "REMA_1km_dem.tif"
            filled_filename = "REMA_1km_dem_filled.tif"
            default_dir = f'{os.environ["CPDATA_DIR"]}/SATS/RA/DEMS/rema_1km_dem'
            self.src_url = (
                "http://data.pgc.umn.edu/elev/dem/setsm/REMA/mosaic/v1.1/1km/REMA_1km_dem.tif"
            )
            self.src_url_filled = (
                "http://data.pgc.umn.edu/elev/dem/setsm/REMA/mosaic/v1.1/1km/"
                "REMA_1km_dem_filled.tif"
            )
            self.dem_version = "1.1"
            self.src_institute = "PGC"
            self.long_name = "REMA"
            self.crs_bng = CRS("epsg:3031")  # Polar Stereo - South -71S
            self.southern_hemisphere = True
            self.void_value = -9999
            self.dtype = np.float32
            self.reference_year = 2010  # YYYY, the year the DEM's elevations are referenced to

        # --------------------------------------------------------------------------------
        elif self.name == "rema_ant_1km_zarr":
            # REMA Antarctic 1km DEM  v1.1 (PGC 2018)
            # The void areas will contain null values (-9999) in lieu of the terrain elevations.

            filename = "REMA_1km_dem.zarr"
            filled_filename = "REMA_1km_dem_filled.zarr"
            default_dir = f'{os.environ["CPDATA_DIR"]}/SATS/RA/DEMS/rema_1km_dem'
            self.src_url = (
                "http://data.pgc.umn.edu/elev/dem/setsm/REMA/mosaic/v1.1/1km/REMA_1km_dem.tif"
            )
            self.src_url_filled = (
                "http://data.pgc.umn.edu/elev/dem/setsm/REMA/mosaic/v1.1/1km/"
                "REMA_1km_dem_filled.tif"
            )
            self.dem_version = "1.1"
            self.src_institute = "PGC"
            self.long_name = "REMA"
            self.crs_bng = CRS("epsg:3031")  # Polar Stereo - South -71S
            self.southern_hemisphere = True
            self.void_value = -9999
            self.dtype = np.float32
            self.reference_year = 2010  # YYYY, the year the DEM's elevations are referenced to
            self.zarr_type = True
        # --------------------------------------------------------------------------------
        elif self.name == "rema_ant_1km_v2":
            # REMA Antarctic 1km DEM  v2.0 (PGC 2022)
            # Acknowledgment:
            # Howat, Ian, et al., 2022, The Reference Elevation Model of Antarctica – Mosaics,
            # Version 2, https://doi.org/10.7910/DVN/EBW8UC, Harvard Dataverse, V1.
            # The void areas will contain null values (-9999) in lieu of the terrain elevations.
            #
            filename = "rema_mosaic_1km_v2.0_dem.tif"
            filled_filename = "rema_mosaic_1km_v2.0_filled_cop30_dem.tif"
            default_dir = f'{os.environ["CPDATA_DIR"]}/SATS/RA/DEMS/rema_1km_dem_v2'
            self.src_url = (
                "https://data.pgc.umn.edu/elev/dem/setsm/REMA/mosaic/v2.0/1km"
                "rema_mosaic_1km_v2.0_dem.tif"
            )
            self.src_url_filled = (
                "http://data.pgc.umn.edu/elev/dem/setsm/REMA/mosaic/v2.0/1km/"
                "rema_mosaic_1km_v2.0_filled_cop30_dem.tif"
            )
            self.dem_version = "2.0"
            self.src_institute = "PGC"
            self.long_name = "REMA v2"
            self.crs_bng = CRS("epsg:3031")  # Polar Stereo - South -71S
            self.southern_hemisphere = True
            self.void_value = -9999
            self.dtype = np.float32

        # --------------------------------------------------------------------------------
        elif self.name == "rema_ant_1km_v2_zarr":
            # REMA Antarctic 1km DEM  v2.0 (PGC 2022) in Zarr format
            # Acknowledgment:
            # Howat, Ian, et al., 2022, The Reference Elevation Model of Antarctica – Mosaics,
            # Version 2, https://doi.org/10.7910/DVN/EBW8UC, Harvard Dataverse, V1.
            # The void areas will contain null values (-9999) in lieu of the terrain elevations.
            #
            filename = "rema_mosaic_1km_v2.0_dem.zarr"
            filled_filename = "rema_mosaic_1km_v2.0_filled_cop30_dem.zarr"
            default_dir = f'{os.environ["CPDATA_DIR"]}/SATS/RA/DEMS/rema_1km_dem_v2'
            self.src_url = (
                "https://data.pgc.umn.edu/elev/dem/setsm/REMA/mosaic/v2.0/1km"
                "rema_mosaic_1km_v2.0_dem.tif"
            )
            self.src_url_filled = (
                "http://data.pgc.umn.edu/elev/dem/setsm/REMA/mosaic/v2.0/1km/"
                "rema_mosaic_1km_v2.0_filled_cop30_dem.tif"
            )
            self.dem_version = "2.0"
            self.src_institute = "PGC"
            self.long_name = "REMA v2"
            self.crs_bng = CRS("epsg:3031")  # Polar Stereo - South -71S
            self.southern_hemisphere = True
            self.void_value = -9999
            self.dtype = np.float32
            self.zarr_type = True

        # --------------------------------------------------------------------------------
        elif self.name == "rema_ant_200m":
            # REMA Antarctic 200m DEM  v1.1 (PGC 2018)
            # The void areas will contain null values (-9999) in lieu of the terrain elevations.

            filename = "REMA_200m_dem.tif"
            filled_filename = "REMA_200m_dem_filled.tif"
            default_dir = f'{os.environ["CPDATA_DIR"]}/SATS/RA/DEMS/ant_rema_200m_dem'
            self.src_url = (
                "https://data.pgc.umn.edu/elev/dem/setsm/REMA/mosaic/v1.1/200m/REMA_200m_dem.tif"
            )
            self.src_url_filled = (
                "https://data.pgc.umn.edu/elev/dem/setsm/REMA/mosaic/v1.1/200m/"
                "REMA_200m_dem_filled.tif"
            )
            self.dem_version = "1.1"
            self.src_institute = "PGC"
            self.long_name = "REMA-1.1-Gapless-1km"
            self.crs_bng = CRS("epsg:3031")  # Polar Stereo - South -71S
            self.southern_hemisphere = True
            self.void_value = -9999
            self.dtype = np.float32
            self.reference_year = 2010  # YYYY, the year the DEM's elevations are referenced to

        # --------------------------------------------------------------------------------
        elif self.name == "rema_ant_200m_zarr":
            # REMA Antarctic 200m DEM  v1.1 (PGC 2018), zarr format
            # The void areas will contain null values (-9999) in lieu of the terrain elevations.

            filename = "REMA_200m_dem.zarr"
            filled_filename = "REMA_200m_dem_filled.zarr"
            default_dir = f'{os.environ["CPDATA_DIR"]}/SATS/RA/DEMS/ant_rema_200m_dem'
            self.src_url = (
                "https://data.pgc.umn.edu/elev/dem/setsm/REMA/mosaic/v1.1/200m/REMA_200m_dem.tif"
            )
            self.src_url_filled = (
                "https://data.pgc.umn.edu/elev/dem/setsm/REMA/mosaic/v1.1/200m/"
                "REMA_200m_dem_filled.tif"
            )
            self.dem_version = "1.1"
            self.src_institute = "PGC"
            self.long_name = "REMA-1.1-200m"
            self.crs_bng = CRS("epsg:3031")  # Polar Stereo - South -71S
            self.southern_hemisphere = True
            self.void_value = -9999
            self.dtype = np.float32
            self.reference_year = 2010  # YYYY, the year the DEM's elevations are referenced to
            self.zarr_type = True

        # --------------------------------------------------------------------------------
        elif self.name == "rema_gapless_100m":
            # REMA v1.1 at 100m with Gapless post-processing as per
            # https://doi.org/10.1016/j.isprsjprs.2022.01.024

            filename = "GaplessREMA100.tif"
            filled_filename = "GaplessREMA100.tif"
            default_dir = f'{os.environ["CPDATA_DIR"]}/SATS/RA/DEMS/rema_gapless_100m'
            self.src_url = "https://figshare.com/articles/dataset/Gapless-REMA100/19122212"
            self.src_url_filled = "https://figshare.com/articles/dataset/Gapless-REMA100/19122212"
            self.dem_version = "1.1(REMA)/2.0(Gapless)"
            self.src_institute = "PGC/Dong(2022)"
            self.long_name = "REMA-1.1-Gapless-100m"
            self.crs_bng = CRS("epsg:3031")  # Polar Stereo - South -71S
            self.southern_hemisphere = True
            self.void_value = -32767
            self.dtype = np.float32
            self.reference_year = 2010  # YYYY, the year the DEM's elevations are referenced to

        # --------------------------------------------------------------------------------
        elif self.name == "rema_gapless_100m_zarr":
            # REMA v1.1 at 100m with Gapless post-processing (v2.0) as per Dong at al,
            # https://doi.org/10.1016/j.isprsjprs.2022.01.024

            filename = "GaplessREMA100.zarr"
            filled_filename = "GaplessREMA100.zarr"
            default_dir = (
                f'{os.environ["CLEV2ER_BASE_DIR"]}/testdata_external/adf/landice/dems/antarctica/'
                "rema_gapless_100m"
            )
            self.src_url = "https://figshare.com/articles/dataset/Gapless-REMA100/19122212"
            self.src_url_filled = "https://figshare.com/articles/dataset/Gapless-REMA100/19122212"
            self.dem_version = "1.1(REMA)/2.0(Gapless)"
            self.src_institute = "PGC/Dong(2022)"
            self.long_name = "REMA-1.1-Gapless-100m"
            self.crs_bng = CRS("epsg:3031")  # Polar Stereo - South -71S
            self.southern_hemisphere = True
            self.void_value = -32767
            self.dtype = np.float32
            self.reference_year = 2010  # YYYY, the year the DEM's elevations are referenced to
            self.zarr_type = True
        # --------------------------------------------------------------------------------
        elif self.name == "rema_gapless_1km":
            # REMA v1.1 at 100m with Gapless post-processing as per
            # https://doi.org/10.1016/j.isprsjprs.2022.01.024

            filename = "GaplessREMA1km.tif"
            filled_filename = "GaplessREMA1km.tif"
            default_dir = f'{os.environ["CPDATA_DIR"]}/SATS/RA/DEMS/rema_gapless_100m'
            self.src_url = "https://figshare.com/articles/dataset/Gapless-REMA100/19122212"
            self.src_url_filled = "https://figshare.com/articles/dataset/Gapless-REMA100/19122212"
            self.dem_version = "1.1(REMA)/2.0(Gapless)"
            self.src_institute = "PGC/Dong(2022)"
            self.long_name = "REMA-1.1-Gapless-100m"
            self.crs_bng = CRS("epsg:3031")  # Polar Stereo - South -71S
            self.southern_hemisphere = True
            self.void_value = -32767
            self.dtype = np.float32
            self.reference_year = 2010  # YYYY, the year the DEM's elevations are referenced to
        elif self.name == "rema_gapless_1km_zarr":
            # REMA v1.1 at 100m with Gapless post-processing as per
            # https://doi.org/10.1016/j.isprsjprs.2022.01.024

            filename = "GaplessREMA1km.zarr"
            filled_filename = "GaplessREMA1km.zarr"
            default_dir = (
                f'{os.environ["CLEV2ER_BASE_DIR"]}/testdata_external/adf/landice/dems/antarctica/'
                "rema_gapless_1km"
            )
            self.src_url = "https://figshare.com/articles/dataset/Gapless-REMA100/19122212"
            self.src_url_filled = "https://figshare.com/articles/dataset/Gapless-REMA100/19122212"
            self.dem_version = "1.1(REMA)/2.0(Gapless)"
            self.src_institute = "PGC/Dong(2022)"
            self.long_name = "REMA-1.1-Gapless-100m"
            self.crs_bng = CRS("epsg:3031")  # Polar Stereo - South -71S
            self.southern_hemisphere = True
            self.void_value = -32767
            self.dtype = np.float32
            self.reference_year = 2010  # YYYY, the year the DEM's elevations are referenced to
            self.zarr_type = True

        # --------------------------------------------------------------------------------
        elif self.name == "arcticdem_100m_greenland":
            # 100m DEM (subarea of Greenland) extracted from ArcticDem v3
            # The void areas will contain null values (-9999) in lieu of the terrain elevations.
            filename = "arcticdem_mosaic_100m_v3.0_subarea_greenland.tif"
            filled_filename = ""  # No filled version available
            default_dir = f'{os.environ["CPDATA_DIR"]}/SATS/RA/DEMS/arctic_dem_100m'
            self.src_url = (
                "https://data.pgc.umn.edu/elev/dem/setsm/ArcticDEM/mosaic/v3.0/100m/"
                "arcticdem_mosaic_100m_v3.0.tif"
            )
            self.reference_year = 2010  # YYYY, the year the DEM's elevations are referenced to
            self.src_url_filled = ""
            self.dem_version = "3.0"
            self.src_institute = "PGC"
            self.long_name = "ArcticDem 100m, Greenland"
            self.crs_bng = CRS("epsg:3413")  # Polar Stereo - North -latitude of origin 70N, 45
            self.southern_hemisphere = False
            self.void_value = -9999
            self.dtype = np.float32
        # --------------------------------------------------------------------------------
        elif self.name == "arcticdem_1km_greenland_v4.1":
            # 1km DEM (subarea of Greenland) extracted from ArcticDem v4.1
            # The void areas will contain null values (-9999) in lieu of the terrain elevations.
            filename = "arcticdem_mosaic_1km_v4.1_subarea_greenland.tif"
            filled_filename = ""  # No filled version available
            default_dir = f'{os.environ["CPDATA_DIR"]}/SATS/RA/DEMS/arctic_dem_1km_v4.1'
            self.src_url = (
                "https://data.pgc.umn.edu/elev/dem/setsm/ArcticDEM/mosaic"
                "/latest/1km/arcticdem_mosaic_1km_v4.1_dem.tif"
            )
            self.reference_year = 2010  # YYYY, the year the DEM's elevations are referenced to
            self.src_url_filled = ""
            self.dem_version = "4.1"
            self.src_institute = "PGC"
            self.long_name = "ArcticDem 1km, Greenland"
            self.crs_bng = CRS("epsg:3413")  # Polar Stereo - North -latitude of origin 70N, 45
            self.southern_hemisphere = False
            self.void_value = -9999
            self.dtype = np.float32
        # --------------------------------------------------------------------------------
        elif self.name == "arcticdem_1km_greenland_v4.1_zarr":
            # 1km DEM (subarea of Greenland) extracted from ArcticDem v4.1
            # The void areas will contain null values (-9999) in lieu of the terrain elevations.
            filename = "arcticdem_mosaic_1km_v4.1_subarea_greenland.zarr"
            filled_filename = (
                "arcticdem_mosaic_1km_v4.1_subarea_greenland.zarr"  # No filled version available
            )
            default_dir = f'{os.environ["CPDATA_DIR"]}/SATS/RA/DEMS/arctic_dem_1km_v4.1'
            self.src_url = (
                "https://data.pgc.umn.edu/elev/dem/setsm/ArcticDEM/mosaic"
                "/latest/1km/arcticdem_mosaic_1km_v4.1_dem.tif"
            )
            self.reference_year = 2010  # YYYY, the year the DEM's elevations are referenced to
            self.src_url_filled = ""
            self.dem_version = "4.1"
            self.src_institute = "PGC"
            self.long_name = "ArcticDem 1km, Greenland"
            self.crs_bng = CRS("epsg:3413")  # Polar Stereo - North -latitude of origin 70N, 45
            self.southern_hemisphere = False
            self.void_value = -9999
            self.dtype = np.float32
            self.zarr_type = True
        # --------------------------------------------------------------------------------
        elif self.name == "arcticdem_100m_greenland_v4.1":
            # 100m DEM (subarea of Greenland) extracted from ArcticDem v4.1
            # The void areas will contain null values (-9999) in lieu of the terrain elevations.
            filename = "arcticdem_mosaic_100m_v4.1_subarea_greenland.tif"
            filled_filename = ""  # No filled version available
            default_dir = f'{os.environ["CPDATA_DIR"]}/SATS/RA/DEMS/arctic_dem_100m_v4.1'
            self.src_url = (
                "https://data.pgc.umn.edu/elev/dem/setsm/ArcticDEM/mosaic"
                "/latest/100m/arcticdem_mosaic_100m_v4.1_dem.tif"
            )
            self.reference_year = 2010  # YYYY, the year the DEM's elevations are referenced to
            self.src_url_filled = ""
            self.dem_version = "4.1"
            self.src_institute = "PGC"
            self.long_name = "ArcticDem 100m, Greenland"
            self.crs_bng = CRS("epsg:3413")  # Polar Stereo - North -latitude of origin 70N, 45
            self.southern_hemisphere = False
            self.void_value = -9999
            self.dtype = np.float32

        # --------------------------------------------------------------------------------
        elif self.name == "arcticdem_100m_greenland_v4.1_zarr":
            # 100m DEM (subarea of Greenland) extracted from ArcticDem v4.1
            # The void areas will contain null values (-9999) in lieu of the terrain elevations.
            filename = "arcticdem_mosaic_100m_v4.1_subarea_greenland.zarr"
            filled_filename = (
                "arcticdem_mosaic_100m_v4.1_subarea_greenland.zarr"  # No filled version available
            )
            default_dir = f'{os.environ["CPDATA_DIR"]}/SATS/RA/DEMS/arctic_dem_100m_v4.1'
            self.src_url = (
                "https://data.pgc.umn.edu/elev/dem/setsm/ArcticDEM/mosaic"
                "/latest/100m/arcticdem_mosaic_100m_v4.1_dem.tif"
            )
            self.reference_year = 2010  # YYYY, the year the DEM's elevations are referenced to
            self.src_url_filled = ""
            self.dem_version = "4.1"
            self.src_institute = "PGC"
            self.long_name = "ArcticDem 100m, Greenland"
            self.crs_bng = CRS("epsg:3413")  # Polar Stereo - North -latitude of origin 70N, 45
            self.southern_hemisphere = False
            self.void_value = -9999
            self.dtype = np.float32
            self.zarr_type = True

        # --------------------------------------------------------------------------------

        else:
            raise ValueError(f"{self.name} does not have load support")

        # Form the DEM file name and load the DEM
        try:
            demfile = self.get_filename(default_dir, filename, filled_filename)
        except OSError as exc:
            self.log.error("Could not form dem path for %s : %s", self.name, exc)
            return False

        if self.npz_type:
            self.load_npz(demfile)
        elif self.zarr_type:
            self.load_zarr(demfile)
        else:
            self.load_geotiff(demfile)

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
        """return a cropped segment of the DEM, flattened or as a grid

        Args:
            segment_bounds (List): [(minx,maxx),(miny,maxy)]
            grid_xy (bool, optional): return segment as a grid. Defaults to True.
            flatten (bool, optional): return segment as flattened list. Defaults to False.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: (xdem,ydem,zdem)
        """

        # ----------------------------------------------------------------------
        # Get coord bounds as index bounds
        # ----------------------------------------------------------------------

        minx_ind = (np.absolute(segment_bounds[0][0] - self.xdem)).argmin()
        maxx_ind = (np.absolute(segment_bounds[0][1] - self.xdem)).argmin()
        miny_ind = (np.absolute(segment_bounds[1][0] - self.ydem)).argmin()
        maxy_ind = (np.absolute(segment_bounds[1][1] - self.ydem)).argmin()

        # ----------------------------------------------------------------------
        # Crop full dem coords to segment bounds
        # ----------------------------------------------------------------------

        zdem = self.zdem[maxy_ind:miny_ind, minx_ind:maxx_ind]
        xdem = self.xdem[minx_ind:maxx_ind]
        ydem = self.ydem[maxy_ind:miny_ind]

        if grid_xy is True:
            xdem, ydem = np.meshgrid(xdem, ydem)

            # Set x,y to nan where z is nan
            zdem_nan = np.isnan(zdem)
            xdem[zdem_nan] = np.nan
            ydem[zdem_nan] = np.nan

        # ----------------------------------------------------------------------
        # Return, flattened if requested
        # ----------------------------------------------------------------------

        if flatten is False:
            return (xdem, ydem, zdem)

        return (xdem.flatten(), ydem.flatten(), zdem.flatten())

    def chunked_interpolation(
        self, x: np.ndarray, y: np.ndarray, myydem: np.ndarray, xdem: np.ndarray, method: str
    ) -> np.ndarray:
        """Interpolate DEM in chunks to handle large datasets efficiently.

        This function performs interpolation on a DEM stored in a Zarr array by
        extracting relevant chunks and creating a sub-grid for interpolation.

        Args:
            x (np.ndarray): Array of x coordinates in the DEM's projection (in meters).
            y (np.ndarray): Array of y coordinates in the DEM's projection (in meters).
            myydem (np.ndarray): Flipped y coordinates corresponding to the DEM grid.
            xdem (np.ndarray): x coordinates corresponding to the DEM grid.
            method (str): Interpolation method to use ('linear', 'nearest', etc.).

        Returns:
            np.ndarray: Interpolated DEM elevation values at the specified coordinates.
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

            # Determine the indices of the bounding box in the DEM grid
            x_indices = np.searchsorted(xdem, [x_min, x_max])
            y_indices = np.searchsorted(myydem, [y_min, y_max])

            # Expand the indices to ensure we cover the region adequately
            x_indices[0] = max(x_indices[0] - 1, 0)
            x_indices[1] = min(x_indices[1] + 1, len(xdem) - 1)
            y_indices[0] = max(y_indices[0] - 1, 0)
            y_indices[1] = min(y_indices[1] + 1, len(myydem) - 1)

            # Extract the sub-array
            sub_zarr = self.zdem_flip[
                y_indices[0] : y_indices[1] + 1, x_indices[0] : x_indices[1] + 1
            ]
            sub_zarr = np.array(sub_zarr)

            sub_myydem = myydem[y_indices[0] : y_indices[1] + 1]
            sub_xdem = xdem[x_indices[0] : x_indices[1] + 1]

            # Create an interpolator for the sub-array
            interpolator = RegularGridInterpolator(
                (sub_myydem, sub_xdem),
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
    # Interpolate DEM, input x,y can be arrays or single, units m, in projection (epsg:3031")
    # returns the interpolated elevation(s) at x,y
    # x,y : x,y cartesian coordinates in the DEM's projection in m
    # OR, when xy_is_latlon is True:
    # x,y : latitude, longitude values in degs N and E (note the order, not longitude, latitude!)
    #
    # method: string containing the interpolation method. Default is 'linear'. Options are
    # “linear” and “nearest”, and “splinef2d” (see scipy.interpolate.interpn docs).
    #
    # Where your input points are outside the DEM area, then np.nan values will be returned
    # ----------------------------------------------------------------------------------------------

    def interp_dem(self, x, y, method="linear", xy_is_latlon=False) -> np.ndarray:
        """Interpolate DEM to return elevation values corresponding to
           cartesian x,y in DEM's projection or lat,lon values

        Args:
            x (np.ndarray): x cartesian coordinates in the DEM's projection in m, or lat values
            y (np.ndarray): x cartesian coordinates in the DEM's projection in m, or lon values
            method (str, optional): linear, nearest, splinef2d. Defaults to "linear".
            xy_is_latlon (bool, optional): if True, x,y are lat, lon values. Defaults to False.

        Returns:
            np.ndarray: interpolated dem elevation values
        """

        x = np.array(x)
        y = np.array(y)

        # Transform to x,y if inputs are lat,lon
        if xy_is_latlon:
            x, y = self.lonlat_to_xy_transformer.transform(  # pylint: disable=E0633
                y, x
            )  # transform lon,lat -> x,y
        myydem = np.flip(self.ydem.copy())
        # If zdem is actually a zarr array instead of a numpy array
        # then we use the pre-flipped zarr version, but need to convert it to
        # a numpy array first (which is slow)
        if self.zarr_type:
            return self.chunked_interpolation(x, y, myydem, self.xdem, method)

        myzdem = np.flip(self.zdem.copy(), 0)
        return interpn(
            (myydem, self.xdem),
            myzdem,
            (y, x),
            method=method,
            bounds_error=False,
            fill_value=np.nan,
        )

    def gaussian_smooth(self, sigma=1.0):
        """
        perform a gaussian smooth on the current loaded DEM
        sigma : degree of smoothing, def=1.0
        """
        # Gaussian smooth DEM
        this_zdem = self.zdem.copy()
        this_zdem[np.isnan(self.zdem)] = 0
        f_zdem = gaussian_filter(this_zdem, sigma=sigma)
        www = 0 * self.zdem.copy() + 1
        www[np.isnan(self.zdem)] = 0
        f_www = gaussian_filter(www, sigma=sigma)
        self.zdem = f_zdem / f_www

    def hillshade(self, azimuth=225, pitch=45):
        """
        Convert the DEM 'z_dem' values to a hillshade value between 0..255
        azimuth: angle in degrees (0..360)
        pitch : angle in degrees (0..90)
        """
        azimuth = 360.0 - azimuth

        x, y = np.gradient(self.zdem)
        slope = np.pi / 2.0 - np.arctan(np.sqrt(x * x + y * y))
        aspect = np.arctan2(-x, y)
        azimuthrad = azimuth * np.pi / 180.0
        altituderad = pitch * np.pi / 180.0

        shaded = np.sin(altituderad) * np.sin(slope) + np.cos(altituderad) * np.cos(slope) * np.cos(
            (azimuthrad - np.pi / 2.0) - aspect
        )

        self.zdem = 255 * (shaded + 1) / 2
