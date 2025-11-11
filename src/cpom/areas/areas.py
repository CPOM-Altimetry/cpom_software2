"""cpom.areas.areas.py: Area class to define areas for polar plotting"""

import glob
import importlib
import logging
import os
import sys
from typing import Optional

import numpy as np
import polars as pl
from pyproj import CRS  # Transformer transforms between projections
from pyproj import Transformer  # Transformer transforms between projections

from cpom.masks.masks import Mask

# pylint: disable=too-many-statements
# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-locals

log = logging.getLogger(__name__)


def list_all_area_definition_names(logger=None) -> list[str]:
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
        area_def_directory = f"{os.environ['CPOM_SOFTWARE_DIR']}/src/cpom/areas/definitions"
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
            thisarea = Area(thisdef)
            color = green
            if thisarea.hemisphere == "north":
                color = blue
            if thisarea.area_summary:
                final_defs.append(
                    f"{color}{thisdef}{endc} : {thisarea.area_summary} :"
                    f" background:{thisarea.background_image}"
                )
            else:
                final_defs.append(
                    f"{color}{thisdef}{endc} : {thisarea.long_name} :"
                    f" background:{thisarea.background_image}"
                )
        return sorted(final_defs)

    finally:
        logger.setLevel(original_level)


def list_all_area_definition_names_only(logger=None) -> list[str]:
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
        area_def_directory = f"{os.environ['CPOM_SOFTWARE_DIR']}/src/cpom/areas/definitions"
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
            _ = Area(thisdef)
            final_defs.append(thisdef)
        return sorted(final_defs)

    finally:
        logger.setLevel(original_level)


def import_module_from_file(file_path):
    """Imports a Python module from a specified file path.

    The function dynamically loads a Python module from the given file path.
    The module name is derived from the file name, excluding its extension
    and directory path. The module is also added to `sys.modules` for
    standard import access.

    Args:
        file_path (str): The file path to the Python module to be imported.

    Raises:
        ImportError: If the module cannot be imported, either due to an
                     invalid file path or a failure during execution.

    Returns:
        tuple: A tuple containing:
            - module (module): The imported module object.
            - module_name (str): The name of the imported module.
    """
    # Calculate the module name from the file path
    module_name = os.path.splitext(os.path.basename(file_path))[0]

    # Load the module
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec and spec.loader:
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module  # Add to sys.modules
        spec.loader.exec_module(module)  # Execute the module
        return module, module_name

    raise ImportError(f"Cannot import module from file path: {file_path}")


class Area:
    """class to define polar areas for plotting etc"""

    def __init__(self, name: str, overrides: dict | None = None, area_filename: str | None = None):
        """class initialization

        Args:
            name (str): area name. Must be in all_areas
            overrides (dict|None): dictionary to override any parameters in area definition dicts
        """

        self.name = name
        self.mask: Optional[Mask] = None

        try:
            self.load_area(overrides, area_filename)
        except ImportError as exc:
            raise ImportError(f"{name} not in supported area list") from exc

        if self.apply_area_mask_to_data:
            self.mask = Mask(self.maskname, self.basin_numbers)

    def load_area(self, overrides: dict | None = None, area_filename: str | None = None):
        """Load area settings for current area name"""

        log.info("loading area -%s-", self.name)

        if area_filename is None:
            try:
                module = importlib.import_module(f"cpom.areas.definitions.{self.name}")
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
                module2 = importlib.import_module(f"cpom.areas.definitions.{secondary_area_name}")
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
        self.long_name = area_definition["long_name"]
        self.area_summary = area_definition.get("area_summary", "")
        # Area spec.
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

        # Plot parameters
        self.axes = area_definition.get("axes")
        self.simple_axes = area_definition.get("simple_axes")
        self.background_color = area_definition.get("background_color", None)
        self.background_image = area_definition.get("background_image", None)
        self.background_image_alpha = area_definition.get("background_image_alpha", 1.0)
        self.background_image_resolution = area_definition.get(
            "background_image_resolution", "medium"
        )
        self.hillshade_params = area_definition.get("hillshade_params", None)
        self.draw_axis_frame = area_definition.get("draw_axis_frame")
        self.add_lakes_feature = area_definition.get("add_lakes_feature", None)
        self.add_rivers_feature = area_definition.get("add_rivers_feature", None)
        self.add_country_boundaries = area_definition.get("add_country_boundaries", None)
        self.add_province_boundaries = area_definition.get("add_province_boundaries", None)
        self.show_polygon_overlay_in_main_map = area_definition.get(
            "show_polygon_overlay_in_main_map", True
        )
        self.grid_polygon_overlay_mask = area_definition.get("grid_polygon_overlay_mask", None)
        self.apply_hillshade_to_vals = area_definition.get("apply_hillshade_to_vals", False)
        self.draw_coastlines = area_definition.get("draw_coastlines", True)
        self.coastline_color = area_definition.get("coastline_color", "grey")
        self.use_antarctica_medium_coastline = area_definition.get(
            "use_antarctica_medium_coastline", False
        )
        self.use_cartopy_coastline = area_definition.get("use_cartopy_coastline", None)
        self.show_gridlines: bool = area_definition.get("show_gridlines", True)
        # Annotation
        self.varname_annotation_position_xy = area_definition.get(
            "varname_annotation_position_xy", (0.1, 0.95)
        )
        self.varname_annotation_position_xy_simple = area_definition.get(
            "varname_annotation_position_xy_simple", (0.1, 0.95)
        )
        self.stats_position_x_offset = area_definition.get("stats_position_x_offset", 0)
        self.stats_position_y_offset = area_definition.get("stats_position_y_offset", 0)
        self.stats_position_x_offset_simple = area_definition.get(
            "stats_position_x_offset_simple", 0
        )
        self.stats_position_y_offset_simple = area_definition.get(
            "stats_position_y_offset_simple", 0
        )
        self.position_stats_manually = area_definition.get("position_stats_manually", False)
        self.nvals_position = area_definition.get("nvals_position", (0.0, 0.0))
        self.stdev_position = area_definition.get("stdev_position", (0.0, 0.0))
        self.min_position = area_definition.get("min_position", (0.0, 0.0))
        self.max_position = area_definition.get("max_position", (0.0, 0.0))
        self.mad_position = area_definition.get("mad_position", (0.0, 0.0))
        self.mean_position = area_definition.get("mean_position", (0.0, 0.0))
        self.median_position = area_definition.get("median_position", (0.0, 0.0))

        # Flag Settings
        self.include_flag_legend = area_definition.get("include_flag_legend", False)
        self.flag_legend_xylocation = area_definition.get("flag_legend_xylocation", [None, None])
        self.flag_legend_location = area_definition.get("flag_legend_location", "upper right")
        self.include_flag_percents = area_definition.get("include_flag_percents", True)
        self.flag_perc_axis = area_definition.get(
            "flag_perc_axis",
            [
                0.84,
                0.25,
                0.09,
            ],
        )
        self.area_long_name_position = area_definition.get("area_long_name_position", None)
        self.area_long_name_position_simple = area_definition.get(
            "area_long_name_position_simple", None
        )
        self.area_long_name_fontsize = area_definition.get("area_long_name_fontsize", 12)
        self.mask_long_name_position = area_definition.get("mask_long_name_position", None)
        self.mask_long_name_position_simple = area_definition.get(
            "mask_long_name_position_simple", None
        )
        self.mask_long_name_fontsize = area_definition.get("mask_long_name_fontsize", 9)
        # Colormap
        self.cmap_name = area_definition.get("cmap_name", "RdYlBu_r")
        self.cmap_over_color = area_definition.get("cmap_over_color", "#A85754")
        self.cmap_under_color = area_definition.get("cmap_under_color", "#3E4371")
        self.cmap_extend = area_definition.get("cmap_extend", "both")
        # Colour bar
        self.draw_colorbar = area_definition.get("draw_colorbar", True)
        self.colorbar_orientation = area_definition.get("colorbar_orientation", "vertical")
        self.vertical_colorbar_axes = area_definition.get(
            "vertical_colorbar_axes",
            [
                0.04,
                0.05,
                0.02,
                0.55,
            ],
        )
        self.vertical_colorbar_axes_simple = area_definition.get(
            "vertical_colorbar_axes_simple",
            [
                0.04,
                0.05,
                0.02,
                0.55,
            ],
        )
        self.horizontal_colorbar_axes = area_definition.get(
            "horizontal_colorbar_axes",
            [
                0.08,
                0.05,
                0.55,
                0.02,
            ],
        )
        self.horizontal_colorbar_axes_simple = area_definition.get(
            "horizontal_colorbar_axes_simple",
            [
                0.08,
                0.05,
                0.55,
                0.02,
            ],
        )

        # Grid lines
        self.longitude_gridlines = np.asarray(area_definition.get("longitude_gridlines")).astype(
            "float"
        )
        self.longitude_gridlines[self.longitude_gridlines > 180.0] -= 360.0

        self.latitude_gridlines = area_definition.get("latitude_gridlines")
        self.gridline_color: str = area_definition.get("gridline_color", "lightgrey")
        self.gridlabel_color = area_definition.get("gridlabel_color", "darkgrey")
        self.gridlabel_size = area_definition.get("gridlabel_size", 9)
        self.draw_gridlabels = area_definition.get("draw_gridlabels", True)

        self.inner_gridlabel_color = area_definition.get("inner_gridlabel_color", "k")
        self.inner_gridlabel_size = area_definition.get("inner_gridlabel_size", 9)
        self.latitude_of_radial_labels = area_definition.get("latitude_of_radial_labels", None)
        self.labels_at_top = area_definition.get("labels_at_top", False)
        self.labels_at_bottom = area_definition.get("labels_at_bottom", True)
        self.labels_at_left = area_definition.get("labels_at_left", True)
        self.labels_at_right = area_definition.get("labels_at_right", True)

        # Mini-map
        self.show_minimap = area_definition.get("show_minimap")
        self.minimap_axes = area_definition.get("minimap_axes")
        self.minimap_bounding_lat = area_definition.get("minimap_bounding_lat")
        self.minimap_circle = area_definition.get("minimap_circle", None)
        self.minimap_draw_gridlines = area_definition.get("minimap_draw_gridlines", True)
        self.minimap_val_scalefactor = area_definition.get("minimap_val_scalefactor", 1.0)
        self.minimap_legend_pos = area_definition.get("minimap_legend_pos", (1.0, 1.0))

        # Bad data mini-map
        self.show_bad_data_map = area_definition.get("show_bad_data_map", True)
        self.bad_data_minimap_axes = area_definition.get("bad_data_minimap_axes")
        self.bad_data_minimap_draw_gridlines = area_definition.get(
            "bad_data_minimap_draw_gridlines", True
        )
        self.bad_data_minimap_gridlines_color = area_definition.get(
            "bad_data_minimap_gridlines_color", "grey"
        )
        self.bad_data_latitude_lines = area_definition.get("bad_data_latitude_lines", [])
        self.bad_data_longitude_lines = area_definition.get("bad_data_longitude_lines", [])

        self.bad_data_minimap_val_scalefactor = area_definition.get(
            "bad_data_minimap_val_scalefactor", 1.0
        )
        self.bad_data_minimap_legend_pos = area_definition.get(
            "bad_data_minimap_legend_pos", (1.0, 1.0)
        )
        self.bad_data_minimap_coastline_resolution = area_definition.get(
            "bad_data_minimap_coastline_resolution", "low"
        )

        # Scale bar
        self.show_scalebar = area_definition.get("show_scalebar")
        self.mapscale = area_definition.get("mapscale")

        self.crs_wgs = CRS("epsg:4326")  # assuming you're using WGS84 geographic
        self.crs_bng = CRS(f"epsg:{self.epsg_number}")
        # Histograms
        self.show_histograms = area_definition.get("show_histograms", True)
        self.histogram_plotrange_axes = area_definition.get(
            "histogram_plotrange_axes",
            [
                0.735,  # left
                0.3,  # bottom
                0.08,  # width (axes fraction)
                0.35,  # height (axes fraction)
            ],
        )
        self.histogram_fullrange_axes = area_definition.get(
            "histogram_fullrange_axes",
            [
                0.89,  # left
                0.3,  # bottom
                0.08,  # width (axes fraction)
                0.35,  # height (axes fraction)
            ],
        )

        self.show_latitude_scatter = area_definition.get("show_latitude_scatter", True)

        self.latvals_axes = area_definition.get(
            "latvals_axes",
            [
                0.77,  # left
                0.05,  # bottom
                0.17,  # width (axes fraction)
                0.2,  # height (axes fraction)
            ],
        )

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
        return self.xy_to_lonlat_transformer.transform(x, y)[::-1]

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

    def inside_area(self, lats: np.ndarray, lons: np.ndarray) -> tuple[np.ndarray, int]:
        """find if input latitude and longitude locations are inside area's
            extent and any mask

        Args:
            lats (np.ndarray): array of latitude values (degs N)
            lons (np.ndarray): array of longitude values (degs E)

        Returns:
            bool_mask (ndarray[bool]): boolean mask of points inside
            n_inside (int) : number of points inside mask
        """

        bool_mask = np.full_like(lats, False, dtype=bool)

        if self.specify_by_bounding_lat:
            (lats_inside, lons_inside, indices_inside_orig, n_inside) = self.inside_latlon_bounds(
                lats, lons
            )
            x_inside, y_inside = self.latlon_to_xy(lats_inside, lons_inside)
        else:
            (lats_inside, lons_inside, x_inside, y_inside, indices_inside_orig, n_inside) = (
                self.inside_xy_extent(lats, lons)
            )
        if n_inside > 0:
            (indices_inside_mask, n_inside) = self.inside_mask(x_inside, y_inside)
            if n_inside > 0:
                indice_inside_orig = indices_inside_orig[indices_inside_mask]
                bool_mask[indice_inside_orig] = True

        return (bool_mask, n_inside)

    ########################################################
    # Functions to work with Polars DataFrames or LazyFrames #
    #########################################################

    def inside_latlon_bounds_polars(
        self, df: pl.DataFrame | pl.LazyFrame, lat_col: str, lon_col: str, return_pl_dataframe=False
    ) -> pl.DataFrame | pl.LazyFrame:
        """Find if input latitude and longitude locations are inside area's lat/lon extent
        bounds.
        Filter polars DataFrame or LazyFrame to only include rows within the area's lat/lon bounds.
        Args:
            df (pl.DataFrame|pl.LazyFrame): Polars DataFrame or LazyFrame containing lat/lon data
            lat_col (str): name of latitude column in df
            lon_col (str): name of longitude column in df
            return_pl_dataframe (bool): if True and input is LazyFrame, return a DataFrame
        Returns:
            pl.DataFrame|pl.LazyFrame: filtered DataFrame or LazyFrame with only rows inside bounds
        """

        df = df.filter(
            (pl.col(lat_col) >= self.minlat)
            & (pl.col(lat_col) <= self.maxlat)
            & (pl.col(lon_col) >= self.minlon)
            & (pl.col(lon_col) <= self.maxlon)
        )

        return df.collect() if (isinstance(df, pl.LazyFrame) and return_pl_dataframe) else df

    # pylint: disable=R0913
    def latlon_to_xy_polars(
        self,
        df: pl.DataFrame | pl.LazyFrame,
        lat_col: str = "latitude",
        lon_col: str = "longitude",
        out_x_col: str = "x",
        out_y_col: str = "y",
        return_pl_dataframe=False,
    ) -> pl.DataFrame | pl.LazyFrame:
        """
        Convert latitude and longitude columns in a Polars DataFrame or LazyFrame to x,y
        coordinates in the area's projection.
        Return a new DataFrame or LazyFrame with added x,y columns.
        Args:
            df (pl.DataFrame|pl.LazyFrame): Polars DataFrame or LazyFrame containing lat/lon
            latitude_col (str): name of latitude column in df
            longitude_col (str): name of longitude column in df
            out_x_col (str): name of output x column to create
            out_y_col (str): name of output y column to create
        Returns:
            pl.DataFrame|pl.LazyFrame: DataFrame or LazyFrame with added x,y
        """
        if isinstance(df, pl.DataFrame):
            df = df.lazy()

        schema = df.collect_schema()
        schema[out_x_col], schema[out_y_col] = pl.Float64, pl.Float64

        def transform_batch(df):
            x, y = self.latlon_to_xy(df[lat_col].to_numpy(), (df[lon_col].to_numpy()))
            df = df.with_columns(
                [
                    pl.Series(out_x_col, x, dtype=pl.Float64),
                    pl.Series(out_y_col, y, dtype=pl.Float64),
                ]
            )
            return df

        df = df.map_batches(
            transform_batch,
            schema=schema,
        )

        return df.collect() if isinstance(df, pl.LazyFrame) & return_pl_dataframe else df

    def xy_to_latlon_polars(
        self,
        df: pl.DataFrame | pl.LazyFrame,
        x_col: str = "x",
        y_col: str = "y",
        out_lat_col: str = "latitude",
        out_lon_col: str = "longitude",
        return_pl_dataframe=False,
    ) -> pl.DataFrame | pl.LazyFrame:
        """
        Convert latitude and longitude columns in a Polars DataFrame or LazyFrame to x,y
        coordinates in the area's projection.
        Return a new DataFrame or LazyFrame with added x,y columns.
        Args:
            df (pl.DataFrame|pl.LazyFrame): Polars DataFrame or LazyFrame containing lat/lon
            x_col (str): name of x column in df
            y_col (str): name of y column in df
            out_lat_col (str): name of output latitude column to create
            out_lon_col (str): name of output longitude column to create
        Returns:
            pl.DataFrame|pl.LazyFrame: DataFrame or LazyFrame with added x,y
        """
        if isinstance(df, pl.DataFrame):
            df = df.lazy()

        schema = df.collect_schema()
        schema[out_lat_col], schema[out_lon_col] = pl.Float64, pl.Float64

        def transform_batch(df):
            x, y = self.xy_to_latlon(df[x_col].to_numpy(), (df[y_col].to_numpy()))
            df = df.with_columns(
                [
                    pl.Series(out_lat_col, x, dtype=pl.Float64),
                    pl.Series(out_lon_col, y, dtype=pl.Float64),
                ]
            )
            return df

        df = df.map_batches(
            transform_batch,
            schema=schema,
        )

        return df.collect() if isinstance(df, pl.LazyFrame) & return_pl_dataframe else df

    def inside_mask_polars(
        self,
        df: pl.DataFrame | pl.LazyFrame,
        x_col: str = "x",
        y_col: str = "y",
        return_pl_dataframe=False,
    ):
        """Find indices of x,y coords inside the area's data mask (if there is one).
        Args:
            df (pl.DataFrame|pl.LazyFrame): Polars DataFrame or LazyFrame containing x,y data
            x_col (str): name of x column in df
            y_col (str): name of y column in df
            return_pl_dataframe (bool): if True and input is LazyFrame, return a DataFrame
        Returns:
            indices_in_maskarea (np.ndarray) : indices inside mask or empty np.ndarray
            n_inside (int) : number of points inside mask
        """

        # Check there is a mask specified for the area
        if self.masktype is None:
            return df  # No mask to return entire dataframe
        self.mask = Mask(self.maskname, self.basin_numbers)

        if self.mask.nomask:
            return df

        df = self.mask.points_inside_polars(
            df,
            x_col=x_col,
            y_col=y_col,
            basin_numbers=self.basin_numbers,
            return_pl_dataframe=return_pl_dataframe,
        )

        return df.collect() if (isinstance(df, pl.LazyFrame) and return_pl_dataframe) else df

    def inside_area_polars(
        self,
        df: pl.DataFrame | pl.LazyFrame,
        lat_col: str = "latitude",
        lon_col: str = "longitude",
        return_pl_dataframe=False,
        return_xy: bool = False,
    ) -> pl.DataFrame | pl.LazyFrame:
        """
        Find if input latitude and longitude locations are inside area's
        extent and any mask.
        Return a polars DataFrame or LazyFrame with only rows inside the area.
        Args:
            df (pl.DataFrame|pl.LazyFrame): Polars DataFrame or LazyFrame containing lat/lon
            lat_col (str): name of latitude column in df
            lon_col (str): name of longitude column in df
            return_pl_dataframe (bool): if True and input is LazyFrame, return a DataFrame
            return_xy (bool): if True, add x,y columns to output
        Returns:
            pl.DataFrame|pl.LazyFrame: filtered DataFrame or LazyFrame with only rows inside area
        """
        if self.specify_by_bounding_lat:
            df = self.inside_latlon_bounds_polars(
                df, lat_col=lat_col, lon_col=lon_col, return_pl_dataframe=False
            )
            if return_xy:
                df = self.latlon_to_xy_polars(
                    df, lat_col=lat_col, lon_col=lon_col, out_x_col="x", out_y_col="y"
                )

        df = self.latlon_to_xy_polars(
            df, lat_col=lat_col, lon_col=lon_col, out_x_col="x", out_y_col="y"
        )
        df = self.inside_mask_polars(
            df, x_col="x", y_col="y", return_pl_dataframe=return_pl_dataframe
        )
        if not return_xy:
            df = df.drop_columns(["x", "y"])

        if isinstance(df, pl.LazyFrame) and return_pl_dataframe:
            df = df.collect()

        return df
