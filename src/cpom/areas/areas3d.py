"""cpom.areas.areas3d.py: Area class to define 3D areas for polar plotting"""

import glob
import importlib
import logging
import os
from typing import Optional

import numpy as np
from pyproj import CRS  # Transformer transforms between projections
from pyproj import Transformer  # Transformer transforms between projections

from cpom.areas.areas import import_module_from_file
from cpom.masks.masks import Mask

# pylint: disable=too-many-statements
# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-locals

log = logging.getLogger(__name__)


def list_all_3d_area_definition_names(logger=None) -> list[str]:
    """return a list of all area definition names and some additional
    info on each.

    Raises:

    Returns:
        list[str]: _description_
    """

    if logger is None:
        logger = logging.getLogger()

    original_level = logger.level
    logger.setLevel(logging.ERROR)

    blue = "\033[0;34m"
    green = "\033[0;32m"
    endc = "\033[0m"  # Reset to default color

    try:
        area_def_directory = f"{os.environ['CPOM_SOFTWARE_DIR']}/src/cpom/areas/definitions_3d"
        if not os.path.isdir(area_def_directory):
            raise FileNotFoundError(f"{area_def_directory} not found")

        all_defs = glob.glob(f"{area_def_directory}/*.py")
        all_defs = [
            os.path.basename(thisdef).replace(".py", "")
            for thisdef in all_defs
            if "__init__" not in thisdef
        ]

        final_defs = []
        for thisdef in all_defs:
            thisarea = Area3d(thisdef)
            color = green
            if thisarea.hemisphere == "north":
                color = blue
            if thisarea.area_summary:
                final_defs.append(f"{color}{thisdef}{endc} : {thisarea.area_summary} :")
            else:
                final_defs.append(f"{color}{thisdef}{endc} : {thisarea.long_name} :")
        return sorted(final_defs)

    finally:
        logger.setLevel(original_level)


def list_all_3d_area_definition_names_only(logger=None) -> list[str]:
    """return a list of all area definition names (only the names)

    Raises:

    Returns:
        list[str]
    """

    if logger is None:
        logger = logging.getLogger()

    original_level = logger.level
    logger.setLevel(logging.ERROR)

    try:
        area_def_directory = f"{os.environ['CPOM_SOFTWARE_DIR']}/src/cpom/areas/definitions_3d"
        if not os.path.isdir(area_def_directory):
            raise FileNotFoundError(f"{area_def_directory} not found")

        all_defs = glob.glob(f"{area_def_directory}/*.py")
        all_defs = [
            os.path.basename(thisdef).replace(".py", "")
            for thisdef in all_defs
            if "__init__" not in thisdef
        ]

        final_defs = []
        for thisdef in all_defs:
            _ = Area3d(thisdef)
            final_defs.append(thisdef)
        return sorted(final_defs)

    finally:
        logger.setLevel(original_level)


class Area3d:
    """class to define 3d polar areas for plotting etc"""

    def __init__(self, name: str, overrides: dict | None = None, area_filename: str | None = None):
        """class initialization

        Args:
            name (str): area name. Must be in all_areas
            overrides (dict|None): dictionary to override any parameters in area definition dicts
        """

        self.name = name
        self.mask: Optional[Mask] = None

        try:
            self.load_3d_area(overrides, area_filename)
        except ImportError as exc:
            raise ImportError(f"{name} not in supported area list") from exc

        if self.apply_area_mask_to_data:
            self.mask = Mask(self.maskname, self.basin_numbers)

    def load_3d_area(self, overrides: dict | None = None, area_filename: str | None = None):
        """Load area settings for current area name"""

        log.info("loading area -%s-", self.name)

        if area_filename is None:
            try:
                module = importlib.import_module(f"cpom.areas.definitions_3d.{self.name}")
            except ImportError as exc:
                raise ImportError(f"Could not load area definition: {self.name}") from exc
        else:
            module, self.name = import_module_from_file(area_filename)

        area_definition = module.area_definition.copy()

        secondary_area_name = area_definition.get("use_definitions_from", None)
        while secondary_area_name is not None:
            if "use_definitions_from" in area_definition:
                del area_definition["use_definitions_from"]
            log.info("loading secondary area %s", secondary_area_name)
            try:
                module2 = importlib.import_module(
                    f"cpom.areas.definitions_3d.{secondary_area_name}"
                )
            except ImportError as exc:
                raise ImportError(f"Could not load area definition: {secondary_area_name}") from exc
            area_definition2 = module2.area_definition.copy()
            secondary_area_name = area_definition2.get("use_definitions_from", None)
            if "use_definitions_from" in area_definition2:
                del area_definition2["use_definitions_from"]
            # log.info(f"1- {area_definition2}")
            area_definition2.update(area_definition)
            # log.info(f"2- {area_definition2}")
            area_definition = area_definition2

        if overrides is not None and isinstance(overrides, dict):
            area_definition.update(overrides)

        # log.info(f"{area_definition}")
        # store parameters from the area definition dict in class variables
        self.dem_name = area_definition["dem_name"]
        self.smooth_dem = area_definition["smooth_dem"]
        self.long_name = area_definition["long_name"]
        self.area_summary = area_definition.get("area_summary", "")
        # 3D Area spec.
        self.hemisphere = area_definition["hemisphere"]
        self.centre_lon = area_definition.get("centre_lon")
        self.centre_lat = area_definition.get("centre_lat")
        self.lon_0 = area_definition.get("lon_0")
        self.width_km = area_definition.get("width_km")
        self.height_km = area_definition.get("height_km")
        self.crs_number = area_definition.get("crs_number")
        self.specify_by_centre = area_definition.get("specify_by_centre", False)
        self.specify_by_bounding_lat = area_definition.get("specify_by_bounding_lat", False)
        self.specify_plot_area_by_lowerleft_corner = area_definition.get(
            "specify_plot_area_by_lowerleft_corner", False
        )
        self.llcorner_lat = area_definition.get("llcorner_lat")
        self.llcorner_lon = area_definition.get("llcorner_lon")
        self.epsg_number = area_definition.get("epsg_number")
        self.round = area_definition.get("round", False)
        self.bounding_lat = area_definition.get("bounding_lat")
        self.min_elevation = area_definition.get("min_elevation", 0)
        self.max_elevation = area_definition.get("max_elevation", 5000)
        self.max_elevation_dem = area_definition.get("max_elevation_dem", 5000)

        # Data filtering
        self.apply_area_mask_to_data = area_definition.get("apply_area_mask_to_data", True)
        self.minlon = area_definition.get("minlon")
        self.maxlon = area_definition.get("maxlon")
        self.minlat = area_definition.get("minlat")
        self.maxlat = area_definition.get("maxlat")
        self.maskname = area_definition.get("maskname")
        self.masktype = area_definition.get("masktype")
        self.basin_numbers = area_definition.get("basin_numbers", None)
        self.show_polygon_mask = area_definition.get("show_polygon_mask", False)
        self.polygon_mask_color = area_definition.get("polygon_mask_color", "red")

        # Colormap
        self.cmap_name = area_definition.get("cmap_name", "RdYlBu_r")
        self.cmap_over_color = area_definition.get("cmap_over_color", "#A85754")
        self.cmap_under_color = area_definition.get("cmap_under_color", "#3E4371")
        self.cmap_extend = area_definition.get("cmap_extend", "both")

        self.page_width = area_definition.get("page_width")
        self.page_height = area_definition.get("page_height")
        self.dem_stride = area_definition.get("dem_stride")
        self.zaxis_multiplier = area_definition.get("zaxis_multiplier")
        self.add_mss_layer = area_definition.get("add_mss_layer")
        self.mss_gridarea = area_definition.get("mss_gridarea")
        self.mss_binsize_km = area_definition.get("mss_binsize_km")
        self.view_angle_elevation = area_definition.get("view_angle_elevation")
        self.view_angle_azimuth = area_definition.get("view_angle_azimuth")
        self.plot_zoom = area_definition.get("plot_zoom")
        self.zaxis_limits = area_definition.get("zaxis_limits")
        self.light_xdirection = area_definition.get("light_xdirection")
        self.light_ydirection = area_definition.get("light_ydirection")
        self.light_zdirection = area_definition.get("light_zdirection")
        self.place_annotations = area_definition.get("place_annotations")
        self.lat_annotations = area_definition.get("lat_annotations")
        self.lon_annotations = area_definition.get("lon_annotations")
        self.lat_lines = area_definition.get("lat_lines")
        self.lon_lines = area_definition.get("lon_lines")
        self.latlon_line_colour = area_definition.get("latlon_line_colour", "white")
        self.latlon_lines_increment = area_definition.get("latlon_lines_increment", 0.01)
        self.latlon_lines_size = area_definition.get("latlon_lines_size", 0.3)
        self.latlon_lines_opacity = area_definition.get("latlon_lines_opacity", 0.5)
        self.latlon_lines_elevation = area_definition.get("latlon_lines_elevation", 200)

        self.crs_wgs = CRS("epsg:4326")  # assuming you're using WGS84 geographic
        self.crs_bng = CRS(f"epsg:{self.epsg_number}")

        # Setup the Transforms
        self.xy_to_lonlat_transformer = Transformer.from_proj(
            self.crs_bng, self.crs_wgs, always_xy=True
        )
        self.lonlat_to_xy_transformer = Transformer.from_proj(
            self.crs_wgs, self.crs_bng, always_xy=True
        )

    def latlon_to_xy(self, lats: np.ndarray | float | list, lons: np.ndarray | float | list):
        """convert latitude and longitude to x,y in area's projection

        Args:
            lats (np.ndarray|float|list): latitude values
            lons (np.ndarray|float|list): longitude values

        Returns:
            (np.ndarray,np.ndarray): x,y
        """
        return self.lonlat_to_xy_transformer.transform(lons, lats)

    def xy_to_latlon(self, x: np.ndarray | float | list, y: np.ndarray | float | list):
        """convert from x,y to latitide, longitiude in area's projection

        Args:
            x (np.ndarray): x coordinates
            y (np.ndarray): y coordinates

        Returns:
            (np.ndarray,np.ndarray): latitude values, longitude values
        """
        return self.xy_to_lonlat_transformer.transform(x, y)

    def inside_xy_extent(
        self,
        lats: np.ndarray,
        lons: np.ndarray,
        inputs_are_xy=False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
        """filter points based on x,y extent of area

        Args:
            lats (np.ndarray): latitude values (degs N)
            lons (np.ndarray): longitude values (deg E)
            inputs_are_xy (bool): if True treat inputs as cartesian: x=lats, y=lons

        Returns:
            (lats_inside, lons_inside, x_inside, y_inside, indices_inside, n_inside):
            lats_inside (np.ndarray): lat values inside area
            lons_inside (np.ndarray): lon values inside area
            x_inside (np.ndarray): projected x coords inside area
            y_inside (np.ndarray): projected y coords inside area
            indices_inside (np.ndarray): indices of original lats,lons that are inside
            n_inside (int): number of original lats, lons that were inside
        """

        # Check the type of lats, lons. they needs to be np.array

        lats = np.atleast_1d(lats)
        lons = np.atleast_1d(lons)

        if inputs_are_xy:
            x, y = lats, lons
        else:
            x, y = self.latlon_to_xy(lats, lons)  # pylint: disable=E0633

        # Fine areas extent
        if self.specify_by_centre:
            centre_x, centre_y = self.latlon_to_xy(  # pylint: disable=E0633
                self.centre_lat, self.centre_lon
            )
            xmin = centre_x - self.width_km * 1000 / 2
            xmax = centre_x + self.width_km * 1000 / 2
            ymin = centre_y - self.height_km * 1000 / 2
            ymax = centre_y + self.height_km * 1000 / 2
        elif self.specify_plot_area_by_lowerleft_corner:
            ll_x, ll_y = self.latlon_to_xy(  # pylint: disable=E0633
                self.llcorner_lat, self.llcorner_lon
            )
            xmin = ll_x
            xmax = ll_x + self.width_km * 1000
            ymin = ll_y
            ymax = ll_y + self.height_km * 1000
        elif self.specify_by_bounding_lat:
            # no x,y extent filtering in this case
            return lats, lons, x, y, np.arange(lats.size), lats.size
        else:
            assert False, "area must specify by centre,lower left, or bounding lat"

        # Filter points within the specified extent
        inside_area = (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)

        # Get lats,lons,x,y inside the area using numpy's boolean indexing
        lats_inside = lats[inside_area]  # will be empty np.array([]) if no points
        lons_inside = lons[inside_area]
        x_inside = x[inside_area]
        y_inside = y[inside_area]

        # Find indices of points inside the area
        indices_inside = np.where(inside_area)[0]  # will be empty np.array([]) if no points

        # Count the number of points inside
        n_inside = len(indices_inside)

        return (
            lats_inside,
            lons_inside,
            x_inside,
            y_inside,
            indices_inside,
            n_inside,
        )

    def inside_latlon_bounds(
        self, lats: np.ndarray, lons: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """find if input latitude and longitude locations are inside area's lat/lon extent
           bounds

        Args:
            lats (np.ndarray): array of latitude values (degs N)
            lons (np.ndarray): array of longitude values (degs E)

        Returns:
            (bounded_lats|None, bounded_lons|None, bounded_indices|None, bounded_indices.size):
        """

        in_lat_area = np.logical_and(lats >= self.minlat, lats <= self.maxlat)
        in_lon_area = np.logical_and(lons >= self.minlon, lons <= self.maxlon)
        bounded_indices = np.flatnonzero(in_lat_area & in_lon_area)
        if bounded_indices.size > 0:
            bounded_lats = lats[bounded_indices]
            bounded_lons = lons[bounded_indices]
            return bounded_lats, bounded_lons, bounded_indices, bounded_indices.size

        return np.array([]), np.array([]), np.array([]), 0

    def inside_mask(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, int]:
        """Find indices of x,y coords inside the area's data mask (if there is one).

        Args:
            x (np.ndarray): x coordinates in areas's projection
            y (np.ndarray): y coordinates in areas's projection

        Returns:
            indices_in_maskarea (np.ndarray) : indices inside mask or empty np.ndarray
            n_inside (int) : number of points inside mask
        """

        # Check the type of lats, lons. they needs to be np.array

        x = np.atleast_1d(x)
        y = np.atleast_1d(y)

        if self.mask is None:
            # Check if there is no mask specified for this area
            # if so, return all the locations
            if self.masktype is None:
                # No mask so return all locations
                return np.arange(x.size), x.size
            # No mask class is currently loaded so we need to load it now
            self.mask = Mask(self.maskname, self.basin_numbers)

        if self.mask.nomask:
            return np.arange(x.size), x.size

        # Mask the lat,lon locations
        inmask, _ = self.mask.points_inside(
            x, y, self.basin_numbers, inputs_are_xy=True
        )  # returns (1s for inside, 0s outside), x,y locations of all lat/lon points
        indices_in_maskarea = np.flatnonzero(inmask)

        return (
            indices_in_maskarea,
            indices_in_maskarea.size,
        )
