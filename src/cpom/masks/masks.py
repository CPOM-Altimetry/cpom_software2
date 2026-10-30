"""Class for area masking"""

import csv
import glob
import hashlib
import importlib
import logging
import os
from collections.abc import Sequence
from multiprocessing.shared_memory import SharedMemory
from os.path import isfile
from typing import Any, Optional

import numpy as np
from matplotlib.path import Path  # for polygon masking
from netCDF4 import Dataset  # pylint: disable=E0611
from pyproj import CRS  # CRS definitions
from pyproj import Transformer  # for transforming between projections

from cpom.utils.definition_loading_helpers import import_module_from_file

# pylint: disable=R0801
# pylint: disable=C0302

log = logging.getLogger(__name__)

# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-statements
# pylint: disable=too-many-branches
# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments


def list_all_mask_definition_names_only(logger=None) -> list[str]:
    """return a list of all mask definition names (only the names)

    Raises:

    Returns:
        list[str]
    """

    if logger is None:
        logger = logging.getLogger()

    original_level = logger.level
    logger.setLevel(logging.ERROR)

    try:
        mask_def_directory = f"{os.environ['CPOM_SOFTWARE_DIR']}/src/cpom/masks/definitions"
        if not os.path.isdir(mask_def_directory):
            raise FileNotFoundError(f"{mask_def_directory} not found")

        all_defs = glob.glob(f"{mask_def_directory}/*.py")
        all_defs = [
            os.path.basename(thisdef).replace(".py", "")
            for thisdef in all_defs
            if "__init__" not in thisdef
        ]

        final_defs = []
        for thisdef in all_defs:
            _ = Mask(thisdef)
            final_defs.append(thisdef)
        return sorted(final_defs)

    finally:
        logger.setLevel(original_level)


class Mask:
    """class to handle area masking"""

    def __init__(
        self,
        mask_name: str,
        basin_numbers: Optional[list[int | str]] = None,
        mask_filename: Optional[str] = None,
        mask_path: Optional[str] = None,
        store_in_shared_memory: bool = False,
        thislog: logging.Logger | None = None,
    ) -> None:
        """class initialization

        Args:
            mask_name (str): mask name, must be in global mask_list
            basin_numbers (list[int|str], optional): list of grid values to select from grid
            masks.
            Can use grid value names instead of numbers. If None, all grid values in mask are used.
            mask_path (str, optional): override default path of mask data file
            mask_definition_path (str, optional): uses mask definition found at this path instead of
                looking in stored definitions
            store_in_shared_memory (bool, optional): stores/access mask array in SharedMemory
            thislog (logging.Logger|None, optional): attach to a different log instance
        """
        self.nomask = False
        if not mask_name:
            self.nomask = True
            return
        self.mask_name = mask_name
        self.mask_long_name = ""
        self.mask_grid: np.ndarray = np.array([])
        self.mask_grid_possible_values: list[int] = []
        self.basin_numbers = basin_numbers
        self.store_in_shared_memory = store_in_shared_memory
        self.shared_mem: Any = None
        self.shared_mem_child = False  # set to True if a child process
        self.polygons = None
        self.polygons_lon: list[list[float]] = []
        self.polygons_lat: list[list[float]] = []
        self.polygon = None
        self.polygon_lon = np.array([])
        self.polygon_lat = np.array([])
        self.mask_type = None  # 'xylimits', 'polygon', 'grid','latlimits'

        # default values
        self.num_x = 0
        self.num_y = 0
        self.binsize = 0
        self.minym = 0
        self.minxm = 0
        self.xlimits: list[float] = []
        self.ylimits: list[float] = []
        self.shapefile_path = None
        self.shapefile_column_name = None

        self.crs_wgs = CRS("epsg:4326")  # assuming you're using WGS84 geographic

        if thislog is not None:
            self.log = thislog  # optionally attach to a different log instance
        else:
            self.log = log

        # ---------------------------------------------------------------------------
        # Load mask definition from file
        # ---------------------------------------------------------------------------

        self.log.info("Setting up %s..", mask_name)
        try:
            self._load_mask(
                mask_filename=mask_filename,
                mask_path=mask_path,
            )

        except ImportError as exc:
            raise ImportError(f"{mask_name} not in supported mask list") from exc

        # -----------------------------------------------------------------------------

        # Setup the Transforms
        self.xy_to_lonlat_transformer = Transformer.from_proj(
            self.crs_bng, self.crs_wgs, always_xy=True
        )
        self.lonlat_to_xy_transformer = Transformer.from_proj(
            self.crs_wgs, self.crs_bng, always_xy=True
        )

    def _load_mask(self, mask_filename: str | None = None, mask_path: str | None = None):
        """Load mask settings for current mask name"""

        log.info("loading mask -%s-", self.mask_name)

        if mask_filename is None:
            try:
                module = importlib.import_module(f"cpom.masks.definitions.{self.mask_name}")
            except ImportError as exc:
                raise ImportError(f"Could not load mask definition: {self.mask_name}") from exc
        else:
            module, self.mask_name = import_module_from_file(mask_filename)

        mask_definition = module.mask_definition.copy()

        # names of masks this mask inherits definitions from (via use_definitions_from)
        self.base_mask_names: list[str] = []

        secondary_mask_name = mask_definition.get("use_definitions_from", None)
        while secondary_mask_name is not None:
            if "use_definitions_from" in mask_definition:
                del mask_definition["use_definitions_from"]
            log.info("loading secondary mask %s", secondary_mask_name)
            self.base_mask_names.append(secondary_mask_name)
            try:
                module2 = importlib.import_module(f"cpom.mask.definitions.{secondary_mask_name}")
            except ImportError as exc:
                raise ImportError(f"Could not load mask definition: {secondary_mask_name}") from exc
            mask_definition2 = module2.mask_definition.copy()
            secondary_mask_name = mask_definition2.get("use_definitions_from", None)
            if "use_definitions_from" in mask_definition2:
                del mask_definition2["use_definitions_from"]
            mask_definition2.update(mask_definition)
            mask_definition = mask_definition2

        for setting_name, setting_value in mask_definition.items():
            match setting_name:
                case "nc_mask":
                    if setting_value:
                        mask_file_path = (
                            mask_definition["default_mask_file"] if mask_path is None else mask_path
                        )

                        if not isfile(mask_file_path):
                            self.log.error("mask file %s does not exist", mask_file_path)
                            raise FileNotFoundError("mask file does not exist")

                        mask_flip = mask_definition.get("nc_mask_flip", False)
                        mask_var = mask_definition.get("nc_mask_var", "mask")
                        self.dtype = mask_definition.get(
                            "dtype", np.uint8
                        )  # read this now incase its not read yet
                        self.load_netcdf_mask(mask_file_path, mask_flip, mask_var)
                case "default_mask_file" | "nc_mask_flip" | "nc_mask_var":
                    continue

                # parse epsg code to CRS
                case "crs_epsg_code":
                    self.crs_bng = CRS(setting_value)

                # apply longname suffixes
                case "basin_longname_suffixes":
                    if callable(setting_value):
                        self.mask_long_name = setting_value(self.mask_long_name, self.basin_numbers)

                case "glacier_csv":
                    glacier_names = ["Unclassified"]
                    with open(setting_value, mode="r", encoding="utf-8") as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            glacier_names.append(row["NAME"])

                    self.grid_value_names = glacier_names
                    self.mask_grid_possible_values = list(range(len(glacier_names)))

                # anything else, set as an attribute
                case _:
                    setattr(self, setting_name, setting_value)

    def load_netcdf_mask(self, mask_file: str, flip=True, nc_mask_var: str = "mask"):
        """load mask array from netcdf grid masks

        Args:
            mask_file (str) : path of netcdf mask file
            flip (bool, optional): _description_. Defaults to True.
            nc_mask_var (str): variable name in netcdf file containing mask data, def='mask'
        """
        if self.store_in_shared_memory:
            # Create a unique 8 char name hashed mask name
            # this is required because shared memory doesn't like long names
            hash_name = hashlib.md5(self.mask_name.encode()).hexdigest()[:8]
            try:  # Attaching to existing shared memory with this mask name
                self.shared_mem = SharedMemory(name=hash_name, create=False)
                self.mask_grid = np.ndarray(
                    shape=(self.num_y, self.num_x),
                    dtype=self.dtype,
                    buffer=self.shared_mem.buf,
                )
                self.shared_mem_child = True

                self.log.info(
                    "child: attached to existing shared memory for mask %s ",
                    self.mask_name,
                )

            except FileNotFoundError:  # Create shared memory with this mask name
                # first, load the mask array from the netcdf file
                with Dataset(mask_file) as nc:
                    mask_grid = np.array(nc.variables[nc_mask_var][:]).astype(self.dtype)
                    if flip:
                        mask_grid = np.flipud(mask_grid)  # flip each column in the up/down dirn

                # Create the shared memory with the appropriate size
                self.shared_mem = SharedMemory(name=hash_name, create=True, size=mask_grid.nbytes)

                # Create an ndarray of the correct size linked to the shared mem
                self.mask_grid = np.ndarray(
                    mask_grid.shape,
                    dtype=mask_grid.dtype,
                    buffer=self.shared_mem.buf,
                )

                # Copy the data from mask_grid to the shared_np_array
                self.mask_grid[:] = mask_grid[:]

                self.log.info("created shared memory for mask %s", self.mask_name)

        else:  # load normally without using shared memory
            # read netcdf file
            with Dataset(mask_file) as nc:
                self.mask_grid = np.array(nc.variables[nc_mask_var][:].data)
                if flip:
                    self.mask_grid = np.flipud(self.mask_grid)

    def load_npz_mask(self, mask_file: str):
        """load mask array from npz grid masks

        Args:
            mask_file (str) : path of npz mask file
        """
        if self.store_in_shared_memory:
            # Create a unique 8 char name hashed mask name
            # this is required because shared memory doesn't like long names
            hash_name = hashlib.md5(self.mask_name.encode()).hexdigest()[:8]

            try:  # Attaching to existing shared memory with this mask name
                self.shared_mem = SharedMemory(name=hash_name, create=False)
                self.mask_grid = np.ndarray(
                    shape=(self.num_y, self.num_x),
                    dtype=self.dtype,
                    buffer=self.shared_mem.buf,
                )
                self.shared_mem_child = True

                self.log.info(
                    "child: attached to existing shared memory for mask %s ",
                    self.mask_name,
                )

            except FileNotFoundError:  # Create shared memory with this mask name
                # first, load the mask array from the npz file
                mask_grid = (
                    np.load(mask_file, allow_pickle=True).get("mask_grid").astype(self.dtype)
                )

                # Create the shared memory with the appropriate size
                self.shared_mem = SharedMemory(name=hash_name, create=True, size=mask_grid.nbytes)

                # Create an ndarray of the correct size linked to the shared mem
                self.mask_grid = np.ndarray(
                    mask_grid.shape,
                    dtype=mask_grid.dtype,
                    buffer=self.shared_mem.buf,
                )

                # Copy the data from mask_grid to the shared_np_array
                self.mask_grid[:] = mask_grid[:]

                self.log.info("created shared memory for mask %s", self.mask_name)

        else:  # load normally without using shared memory
            # read npz file
            self.mask_grid = (
                np.load(mask_file, allow_pickle=True).get("mask_grid").astype(self.dtype)
            )

    def points_inside(
        self,
        lats: np.ndarray | list,
        lons: np.ndarray | list,
        basin_numbers: Optional[list[int | str]] = None,
        inputs_are_xy: bool = False,
    ) -> tuple[np.ndarray, int]:
        """Given a list of lat,lon or x,y points, find the points that are inside the current mask

        Args:
            lats (np.ndarray|list[float]): list of latitude points
            lons (np.ndarray|list[float]): list of longitude points
            basin_numbers (list[int,], optional): list of basin numbers or grid value names.
                                                    Defaults to None.
            inputs_are_xy (bool, optional): lats, lons are already transformed to x,y.
                                            Defaults to False.

        Returns:
            inmask(np.ndarray) : boolean array same size as input list, indicating whether
            inputs points are inside (True) or outside (False) mask
            n_inside (int) : number inside mask
        """

        if not self.mask_name:
            inmask = np.zeros(len(lats), np.bool_)
            return inmask, 0

        if not isinstance(lats, np.ndarray):
            if isinstance(lats, list):
                lats = np.array(lats)
            else:
                raise TypeError("lats is wrong type. Must be np.ndarray or list[float]")

        if not isinstance(lons, np.ndarray):
            if isinstance(lons, list):
                lons = np.array(lons)
            else:
                raise TypeError("lons is wrong type. Must be np.ndarray or list[float]")

        if basin_numbers:  # turn in to a list if a scalar
            if not isinstance(basin_numbers, (list, np.ndarray)):
                basin_numbers = [basin_numbers]

        if self.basin_numbers:
            if not basin_numbers:
                basin_numbers = self.basin_numbers

        # basin numbers are strings convert
        if basin_numbers and isinstance(basin_numbers[0], str):
            basin_numbers = self.get_grid_value_from_grid_value_names(basin_numbers)

        if inputs_are_xy:
            x, y = lats, lons
        else:
            x, y = self.latlon_to_xy(lats, lons)  # pylint: disable=E0633

        inmask = np.zeros(lats.size, np.bool_)

        n_inside = 0
        # ---------------------------------------------------------
        # Find points inside a x,y rectangular limits mask
        # ---------------------------------------------------------

        if self.mask_type == "xylimits":
            for i in range(x.size):
                if (x[i] >= self.xlimits[0] and x[i] <= self.xlimits[1]) and (
                    y[i] >= self.ylimits[0] and y[i] <= self.ylimits[1]
                ):
                    inmask[i] = True
                    n_inside += 1
            return inmask, n_inside

        if self.mask_type == "grid":
            for i in range(x.size):
                # calculate equivalent (ii,jj) in mask array

                ii = int(np.around((x[i] - self.minxm) / self.binsize))
                jj = int(np.around((y[i] - self.minym) / self.binsize))

                # Check bounds of Basin Mask array
                if ii < 0 or ii >= self.num_x:
                    continue
                if jj < 0 or jj >= self.num_y:
                    continue

                if basin_numbers:
                    for basin in basin_numbers:
                        if self.mask_grid[jj, ii] == basin:
                            inmask[i] = True
                            n_inside += 1
                else:
                    if self.mask_grid[jj, ii] > 0:
                        inmask[i] = True
                        n_inside += 1
        elif self.mask_type == "polygon":
            points = np.column_stack((x, y))

            if len(self.polygon_lon) > 0:
                poly_x, poly_y = self.latlon_to_xy(self.polygon_lat, self.polygon_lon)
                poly_corners = np.column_stack((poly_x, poly_y))
                path = Path(poly_corners)
                inmask = path.contains_points(points)
            elif self.polygons_lon:
                inmask = np.zeros(x.size, np.bool_)
                for pi, plon in enumerate(self.polygons_lon):
                    poly_x, poly_y = self.latlon_to_xy(self.polygons_lat[pi], plon)
                    poly_corners = np.column_stack((poly_x, poly_y))
                    path = Path(poly_corners)
                    inmask |= path.contains_points(points)
            else:
                inmask = np.zeros(x.size, np.bool_)

            n_inside = int(np.sum(inmask))
        else:
            return inmask, 0

        return inmask, n_inside

    def get_grid_value_names_from_grid_value(self, basin_numbers):
        """Get grid value names from grid values"""
        return [
            self.grid_value_names[self.mask_grid_possible_values.index(v)] for v in basin_numbers
        ]

    def get_grid_value_from_grid_value_names(self, grid_value_names):
        """Get grid values from grid value names"""
        return [
            self.mask_grid_possible_values[self.grid_value_names.index(n)] for n in grid_value_names
        ]

    def resolve_basin_numbers(
        self, basin_numbers: Optional[list[int | str]] = None
    ) -> Optional[list[int]]:
        """Resolve caller-supplied or instance-default basin selection to integer values."""
        selected = basin_numbers if basin_numbers is not None else self.basin_numbers
        if not selected:
            return None
        selected = list(selected)
        if isinstance(selected[0], str):
            return self.get_grid_value_from_grid_value_names(selected)
        return [int(v) for v in selected]

    def resolve_basin_names(
        self, basin_numbers: Optional[list[int | str]] = None
    ) -> Optional[list[str]]:
        """Resolve caller-supplied or instance-default basin selection to grid value names."""
        resolved = self.resolve_basin_numbers(basin_numbers)
        if resolved is None:
            return None
        return self.get_grid_value_names_from_grid_value(resolved)

    def grid_mask_values(
        self,
        lats: np.ndarray,
        lons: np.ndarray,
        inputs_are_xy=False,
        unknown_value: int = 0,
    ) -> np.ndarray:
        """Return the grid mask value at each input lats, lons interpolated grid location

        Args:
            lats (np.ndarray): array of latitude (N) values in degrees
            lons (np.ndarray): array of longitude (E) values in degrees
            inputs_are_xy (bool): inputs are x,y values (m) instead of latitude, longitude values
            unknown_value (int): value returned for locations outside mask, or where mask
                                 grid includes an unclassified value (unknown_value will be
                                 substituted)
        Returns:
            mask_values (np.ndarray): grid mask value at each input lats, lons interpolated
                                 grid location or np.nan if outside area

        """

        if self.mask_type != "grid":
            raise ValueError(
                (
                    "grid_mask_values can only be used on grid mask types."
                    " Use points_inside() for other masks"
                )
            )

        if np.isscalar(lats):
            lats = np.asarray([lats])
            lons = np.asarray([lons])
        else:
            lats = np.asarray(lats)
            lons = np.asarray(lons)

        # Convert to x,y (m) in mask coordinate system
        if inputs_are_xy:
            x, y = lats, lons
        else:
            x, y = self.latlon_to_xy(lats, lons)  # pylint: disable=E0633

        mask_values = np.full(lats.size, unknown_value, dtype=np.uint8)

        for i in range(0, lats.size):
            # check that x,y is not Nan
            if not np.isfinite(x[i]):
                continue

            # calculate equivalent (ii,jj) in mask array
            ii = int(np.around((x[i] - self.minxm) / self.binsize))
            jj = int(np.around((y[i] - self.minym) / self.binsize))

            # Check bounds of Basin Mask array
            if ii < 0 or ii >= self.num_x:
                continue
            if jj < 0 or jj >= self.num_y:
                continue

            mask_values[i] = self.mask_grid[jj, ii]
        return mask_values

    def latlon_to_xy(
        self, lats: Sequence[float] | np.ndarray, lons: Sequence[float] | np.ndarray
    ) -> tuple:
        """
        :param lats: latitude points in degs
        :param lons: longitude points in degrees E
        :return: x,y in polar stereo projection of mask
        """
        return self.lonlat_to_xy_transformer.transform(lons, lats)

    def clean_up(self):
        """Free up, close or release any shared memory or other resources associated
        with mask
        """
        if self.store_in_shared_memory:
            try:
                if self.shared_mem is not None:
                    if self.shared_mem_child:
                        self.shared_mem.close()
                        self.log.info(
                            "closed shared memory for %s in child process",
                            self.mask_name,
                        )
                        self.log.info("closing in child for mask %s", self.mask_name)
                    else:
                        self.shared_mem.close()
                        self.shared_mem.unlink()
                        self.log.info(
                            "unlinked shared memory for %s",
                            self.mask_name,
                        )

            except Exception as exc:  # pylint: disable=broad-exception-caught
                self.log.error("Shared memory for %s could not be closed %s", self.mask_name, exc)
                raise IOError(
                    f'Shared memory for {self.mask_name} could not be closed {exc}"'
                ) from exc
