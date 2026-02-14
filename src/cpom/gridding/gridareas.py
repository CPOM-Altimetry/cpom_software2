#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Class to define rectangular polar stereo grid areas for processing output. Does not define the
grids internal spacing.

Each area is defined by projection, bottom left corner x,y coordinates in m, width and height in m

Author: Alan Muir (CPOM/UCL)
Date: 2019
Copyright: UCL/MSSL/CPOM. Not to be used outside CPOM/MSSL without permission of author

History:
Updated 24/09/21 by Lin Gilbert to use pyproj CRS rather than Proj
"""
from typing import Tuple, Union

import numpy as np
from pyproj import CRS, Transformer  # co-ord transforms and definitions

# pylint: disable=too-many-branches,too-many-locals,too-many-statements
# pylint: disable=too-many-instance-attributes

all_grid_areas = [
    "antarctica",
    "greenland",
    "antarctic_ocean",
    "greenland_wider_area",
    "greenland_centre",
    "amundsen_sea_area",
    "spirit_all",
    "ross_1600km",
    "filchner_1300km",
    "arctic_russia",
    "svalbard",
    "novaya_zemlya",
    "franz_josef_land",
    "severnaya_zemlya",
    "arctic",
    "chongtar",
]


class GridArea:
    """class to define and use named grid areas for polar regions"""

    def __init__(self, name: str, binsize: int):
        """initialization function for GridArea class

        Args:
            name (str): GridArea area name
            binsize (int): bin size of grid to use in m
        """
        self.name = name
        self.binsize = binsize  # bin width in meters
        self.halfbinsize = binsize / 2

        self.crs_wgs = CRS("epsg:4326")  # ellipsoid : WGS84

        if name == "antarctica":
            self.long_name = "Antarctica"
            self.minxm = -2820e3  # bottom left corner x coordinate of grid in m
            self.minym = -2420e3  # bottom left corner y coordinate of grid in m
            self.grid_x_size = 5640e3  # Width of grid in integer m
            self.grid_y_size = 4840e3  # Width of grid in integer m
            # For lat/lon -> x,y transforms for each area
            self.coordinate_reference_system = (
                "epsg:3031"  # WGS 84 / Antarctic Polar Stereographic, lon0=0E,
            )
            # X along 90E, Y along 0E
            self.hemisphere = "s"  # north='n', south='s'

        if name == "antarctic_ocean":
            self.long_name = "Antarctica Ocean"
            self.minxm = -3820e3  # bottom left corner x coordinate of grid in m
            self.minym = -3420e3  # bottom left corner y coordinate of grid in m
            self.grid_x_size = 7640e3  # Width of grid in integer m
            self.grid_y_size = 6840e3  # Width of grid in integer m
            # For lat/lon -> x,y transforms for each area
            self.coordinate_reference_system = (
                "epsg:3031"  # WGS 84 / Antarctic Polar Stereographic, lon0=0E,
            )
            # X along 90E, Y along 0E
            self.hemisphere = "s"  # north='n', south='s'

        if name == "greenland":
            self.long_name = "Greenland"
            self.minxm = -1000e3  # bottom left corner x coordinate of grid in m
            self.minym = -3500e3  # bottom left corner y coordinate of grid in m
            self.grid_x_size = 2100e3  # Width of grid in integer m
            self.grid_y_size = 3100e3  # Width of grid in integer m
            # For lat/lon -> x,y transforms for each area
            self.coordinate_reference_system = (
                "epsg:3413"  # WGS 84 / NSIDC Sea Ice Polar Stereographic North: lon0=45W,
            )
            # X along 45E, Y along 135E
            self.hemisphere = "n"  # north='n', south='s'

        if name == "greenland_wider_area":
            self.long_name = "Greenland"
            self.minxm = -972134.6133  # bottom left corner x coordinate of grid in m
            self.minym = -3412788.5034  # bottom left corner y coordinate of grid in m
            self.grid_x_size = 2150e3  # Width of grid in integer m
            self.grid_y_size = 2900e3  # Width of grid in integer m
            # For lat/lon -> x,y transforms for each area
            self.coordinate_reference_system = (
                "epsg:3413"  # WGS 84 / NSIDC Sea Ice Polar Stereographic North: lon0=45W,
            )
            # X along 45E, Y along 135E
            self.hemisphere = "n"  # north='n', south='s'

        if name == "greenland_centre":
            self.long_name = "small area in centre of Greenland"
            self.minxm = 102863.21  # bottom left corner x coordinate of grid in m
            self.minym = -1962747.09  # bottom left corner y coordinate of grid in m
            self.grid_x_size = 100e3  # Width of grid in integer m
            self.grid_y_size = 100e3  # Width of grid in integer m
            # For lat/lon -> x,y transforms for each area
            self.coordinate_reference_system = (
                "epsg:3413"  # WGS 84 / NSIDC Sea Ice Polar Stereographic North: lon0=45W,
            )
            # X along 45E, Y along 135E
            self.hemisphere = "n"  # north='n', south='s'

        if name == "amundsen_sea_area":
            self.long_name = "Amundsen Sea Area"
            self.minxm = -1767344  # bottom left corner x coordinate of grid in m
            self.minym = -739296  # bottom left corner y coordinate of grid in m
            self.grid_x_size = 600e3  # Width of grid in integer m
            self.grid_y_size = 760e3  # Width of grid in integer m
            # For lat/lon -> x,y transforms for each area
            self.coordinate_reference_system = (
                "epsg:3031"  # WGS 84 / Antarctic Polar Stereographic, lon0=0E,
            )
            # X along 90E, Y along 0E
            self.hemisphere = "s"  # north='n', south='s'

        if name == "spirit_all":
            self.long_name = "SPIRIT all"
            self.minxm = 1288489  # bottom left corner x coordinate of grid in m
            self.minym = -2231728  # bottom left corner y coordinate of grid in m
            self.grid_x_size = 697e3  # Width of grid in integer m
            self.grid_y_size = 500e3  # Width of grid in integer m
            # For lat/lon -> x,y transforms for each area
            self.coordinate_reference_system = (
                "epsg:3031"  # WGS 84 / Antarctic Polar Stereographic, lon0=0E,
            )
            # X along 90E, Y along 0E
            self.hemisphere = "s"  # north='n', south='s'

        if name == "ross_1600km":
            self.long_name = "Ross Ice Shelf Area"
            self.minxm = -799999.99  # bottom left corner x coordinate of grid in m
            self.minym = -1670577.355  # bottom left corner y coordinate of grid in m
            self.grid_x_size = 1600e3  # Width of grid in integer m
            self.grid_y_size = 1600e3  # Width of grid in integer m
            # For lat/lon -> x,y transforms for each area
            self.coordinate_reference_system = (
                "epsg:3031"  # WGS 84 / Antarctic Polar Stereographic, lon0=0E,
            )
            # X along 90E, Y along 0E
            self.hemisphere = "s"  # north='n', south='s'

        if name == "filchner_1300km":
            self.long_name = "Filchner Ice Shelf Area"
            self.minxm = -1688115.9815  # bottom left corner x coordinate of grid in m
            self.minym = -50643.4586  # bottom left corner y coordinate of grid in m
            self.grid_x_size = 1300e3  # Width of grid in integer m
            self.grid_y_size = 1300e3  # Width of grid in integer m
            # For lat/lon -> x,y transforms for each area
            self.coordinate_reference_system = (
                "epsg:3031"  # WGS 84 / Antarctic Polar Stereographic, lon0=0E,
            )
            # X along 90E, Y along 0E
            self.hemisphere = "s"  # north='n', south='s'

        if (
            name == "arctic_russia"
        ):  # Covers Svalbard, Franz Josef Land, Novaya Zemlya and Severnaya Zemlya
            self.long_name = "Arctic Russia Area"
            self.minxm = 500000.0  # bottom left corner x coordinate of grid in m
            self.minym = -800000.0  # bottom left corner y coordinate of grid in m
            self.grid_x_size = 1400000.0  # Width of grid in integer m
            self.grid_y_size = 1900000.0  # Width of grid in integer m
            # For lat/lon -> x,y transforms for each area
            self.coordinate_reference_system = (
                "epsg:3413"  # WGS 84 / NSIDC Sea Ice Polar Stereographic North: lon0=45W,
            )
            # X along 45E, Y along 135E
            self.hemisphere = "n"  # north='n', south='s'

        if name == "svalbard":  # Svalbard
            self.long_name = "Svalbard Area"
            self.minxm = 900000.0  # bottom left corner x coordinate of grid in m
            self.minym = -800000.0  # bottom left corner y coordinate of grid in m
            self.grid_x_size = 450000.0  # Width of grid in integer m
            self.grid_y_size = 600000.0  # Width of grid in integer m
            # For lat/lon -> x,y transforms for each area
            self.coordinate_reference_system = (
                "epsg:3413"  # WGS 84 / NSIDC Sea Ice Polar Stereographic North: lon0=45W,
            )
            # X along 45E, Y along 135E
            self.hemisphere = "n"  # north='n', south='s'

        if name == "novaya_zemlya":  # Novaya Zemlya
            self.long_name = "Novaya Zemlya Area"
            self.minxm = 1250000.0  # bottom left corner x coordinate of grid in m
            self.minym = 150000.0  # bottom left corner y coordinate of grid in m
            self.grid_x_size = 650000.0  # Width of grid in integer m
            self.grid_y_size = 500000.0  # Width of grid in integer m
            # For lat/lon -> x,y transforms for each area
            self.coordinate_reference_system = (
                "epsg:3413"  # WGS 84 / NSIDC Sea Ice Polar Stereographic North: lon0=45W,
            )
            # X along 45E, Y along 135E
            self.hemisphere = "n"  # north='n', south='s'

        if name == "franz_josef_land":  # Franz Josef Land
            self.long_name = "Franz Josef Land Area"
            self.minxm = 800000.0  # bottom left corner x coordinate of grid in m
            self.minym = -50000.0  # bottom left corner y coordinate of grid in m
            self.grid_x_size = 350000.0  # Width of grid in integer m
            self.grid_y_size = 450000.0  # Width of grid in integer m
            # For lat/lon -> x,y transforms for each area
            self.coordinate_reference_system = (
                "epsg:3413"  # WGS 84 / NSIDC Sea Ice Polar Stereographic North: lon0=45W,
            )
            # X along 45E, Y along 135E
            self.hemisphere = "n"  # north='n', south='s'

        if name == "severnaya_zemlya":  # Severnaya Zemlya
            self.long_name = "Severnaya Zemlya Area"
            self.minxm = 550000.0  # bottom left corner x coordinate of grid in m
            self.minym = 650000.0  # bottom left corner y coordinate of grid in m
            self.grid_x_size = 250000.0  # Width of grid in integer m
            self.grid_y_size = 450000.0  # Width of grid in integer m
            # For lat/lon -> x,y transforms for each area
            self.coordinate_reference_system = (
                "epsg:3413"  # WGS 84 / NSIDC Sea Ice Polar Stereographic North:
            )
            # lon0=45W, X along 45E, Y along 135E
            self.hemisphere = "n"  # north='n', south='s'

        if name == "arctic":  # Arctic
            self.long_name = "Arctic Area"
            self.minxm = -3850000.0  # bottom left corner x coordinate of grid in m
            self.minym = -5350000.0  # bottom left corner y coordinate of grid in m
            self.grid_x_size = 7600000.0  # Width of grid in integer m
            self.grid_y_size = 11200000.0  # Height of grid in integer m
            # For lat/lon -> x,y transforms for each area
            self.coordinate_reference_system = "epsg:3413"  # WGS 84 / NSIDC Sea Ice Polar
            # Stereographic North: lon0=45W, X along 45E, Y along 135E
            self.hemisphere = "n"  # north='n', south='s'

        if name == "chongtar":  # Chongtar surging glacier region of high moutain Asia
            self.long_name = "Chongtar Area"
            self.minxm = 6125000.0  # bottom left corner x coordinate of grid in m
            self.minym = -1541000.0  # bottom left corner y coordinate of grid in m
            self.grid_x_size = 67000.0  # Width of grid in integer m
            self.grid_y_size = 68000.0  # Height of grid in integer m
            # For lat/lon -> x,y transforms for each area
            self.coordinate_reference_system = (
                "epsg:3995"  # WGS 84 / NSIDC Sea Ice Polar Stereographic North: lon0=0E lat0=71N
            )
            self.hemisphere = "n"  # north='n', south='s'

        self.maxxm = self.minxm + self.grid_x_size
        self.maxym = self.minym + self.grid_y_size
        # create mesh grid

        # make x,y cell centre coordinate arrays. Note these are centre coordinates so,
        # start half a bin in from the lower left corner,
        # and stop half bin from the maximum extent
        self.cell_x_centres = np.arange(
            self.minxm + binsize / 2.0, self.maxxm, binsize
        )  # start, stop, step : stop value not included
        self.cell_y_centres = np.arange(self.minym + binsize / 2.0, self.maxym, binsize)

        # Create a grid mesh of ccordinates
        self.xmesh, self.ymesh = np.meshgrid(self.cell_x_centres, self.cell_y_centres)

        # setup coordinate reference system (crs) for this projection
        self.crs_bng = CRS(self.coordinate_reference_system)

        # Setup the Transforms
        self.xy_to_lonlat_transformer = Transformer.from_proj(
            self.crs_bng, self.crs_wgs, always_xy=True
        )
        self.lonlat_to_xy_transformer = Transformer.from_proj(
            self.crs_wgs, self.crs_bng, always_xy=True
        )

    def info(self):
        """print info on the class instance"""
        print("Grid Name : ", self.name)
        print("    Grid Long Name : ", self.long_name)
        print("    Grid CRS : ", self.coordinate_reference_system)
        print("    Width (in X) : ", self.grid_x_size, "(m)")
        print("    Height (in Y) : ", self.grid_y_size, "(m)")
        print("    Bottom left coordinates (min x,y): ", self.minxm, self.minym, "(m)")
        print("Binsize", self.binsize, " (m)")
        print("Ncols, Nrows=", self.get_ncols_nrows())

    def get_ncols_nrows(self):
        """
        returns the number of grid columns (x) and rows (y) for a given bin size in m
        """

        ncols = int(self.grid_x_size / self.binsize)
        nrows = int(self.grid_y_size / self.binsize)
        return ncols, nrows

    def get_col_row_from_x_y(self, x, y):
        """
        returns the grid column and row for a given x,y and bin size in m
        """
        ii = (np.floor((x - self.minxm) / self.binsize)).astype(int)
        jj = (np.floor((y - self.minym) / self.binsize)).astype(int)
        return ii, jj

    def get_cellcentre_x_y_from_col_row(self, col, row):
        """
        returns cell centre x,y in (m) from the grid column and row indices and bin size
        """
        x = self.minxm + col * self.binsize + (self.halfbinsize)
        y = self.minym + row * self.binsize + (self.halfbinsize)

        return x, y

    def get_cellcentre_lat_lon_from_col_row(self, col, row):
        """
        returns latitude and longitude E of cell centre from the grid column and row indices
        and bin size
        """
        x = self.minxm + col * self.binsize + (self.halfbinsize)
        y = self.minym + row * self.binsize + (self.halfbinsize)

        # transform to lat, lon

        lon, lat = self.xy_to_lonlat_transformer.transform(x, y)
        lon = np.mod(lon, 360)

        return lat, lon

    def get_xy_relative_to_cellcentre(self, x, y, col, row):
        """
        returns  the offset in x and y from grid cell (ncol,nrow) centre
        """

        return x - (self.minxm + col * self.binsize + (self.halfbinsize)), y - (
            self.minym + row * self.binsize + (self.halfbinsize)
        )

    def get_xy_relative_to_cellcentres(
        self, x: Union[float, np.ndarray], y: Union[float, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns the offsets in x and y from the centers of the grid cells
        that the input x,y points are located in.

        Args:
            x (Union[float, np.ndarray]): The x-coordinate or an array of x-coordinates.
            y (Union[float, np.ndarray]): The y-coordinate or an array of y-coordinates.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing the x and y offsets from the
            center of the grid cell that each input x,y point is located in.
            If inputs are scalars, the outputs will be scalar arrays.
        """
        # 1. Get integer grid column and row for each x,y
        col, row = self.get_col_row_from_x_y(x, y)

        # 2. Compute the offsets from the cell center
        offset_x = x - (self.minxm + col * self.binsize + self.halfbinsize)
        offset_y = y - (self.minym + row * self.binsize + self.halfbinsize)

        return offset_x, offset_y

    def transform_x_y_to_lat_lon(self, x, y):
        """
        returns latitude and longitude E of cell centre from the arrays of x,y
        """
        lon, lat = self.xy_to_lonlat_transformer.transform(x, y)
        lon = np.mod(lon, 360)

        return lat, lon

    def transform_lat_lon_to_x_y(self, lats, lons):
        """
        returns x,y in grid projection of latitude and longitude E
        """

        return self.lonlat_to_xy_transformer.transform(lons, lats)
