"""Class for area masking"""

# pylint: disable=C0302
import csv
import hashlib
import logging
from multiprocessing.shared_memory import SharedMemory
from os import environ
from os.path import isfile
from typing import Any, Optional

import numpy as np
import polars as pl
from netCDF4 import Dataset  # pylint: disable=E0611
from pyproj import CRS  # CRS definitions
from pyproj import Transformer  # for transforming between projections

# pylint: disable=R0801

log = logging.getLogger(__name__)

# list of all supported mask names
mask_list = [
    "ase_xylimits_mask",  # rectangular mask for Amundsen Sea Embayment (ASE)
    "ronne_filchner_xylimits_mask",  # rectangular mask for Ronne Filchner (Antarctica)
    "trilinear_singular",  # rectangular mask for Greenland
    "antarctica_bedmachine_v2_grid_mask",  # Antarctic Bedmachine v2 surface type mask
    "greenland_bedmachine_v3_grid_mask",  # Greenland Bedmachine v3 surface type mask
    "antarctica_iceandland_dilated_10km_grid_mask",  # Antarctic ice (grounded+floating) and
    # ice free land mask (source BedMachine v2) ,
    # dilated by 10km out into the ocean. NetCDF compressed version.
    "greenland_iceandland_dilated_10km_grid_mask",  # Greenland ice (grounded+floating) and ice free
    # land mask  (source BedMachine v3) , dilated by
    # 10km out into the ocean
    "antarctic_grounded_and_floating_2km_grid_mask",
    # Antarctic grounded and floating ice, 2km grid, source: Zwally 2012
    "greenland_icesheet_2km_grid_mask",
    # Greenland ice sheet grounded ice mask, from 2km grid, source: Zwally 2012. Can select basins
    "antarctic_icesheet_2km_grid_mask_rignot2016",
    # Antarctic ice sheet grounded ice mask + islands, from 2km grid, source: Rignot 2016
    "greenland_icesheet_2km_grid_mask_rignot2016",
    # Greenland ice sheet grounded ice mask, from 2km grid, source: Rignot 2016. Can select basins
    "greenland_icesheet_2km_grid_mask_mouginot2019",
    # Greenland ice sheet grounded ice mask, from 2km grid, source: Mouginot 2019. Can select basins
    "greenland_icesheet_2km_grid_mask_mouginot2019_glaciers",
    # Greenland ice sheet grounded ice
    # + glaciers mask, from 2km grid, source: Mouginot 2019. Can select glaciers
]

# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-statements
# pylint: disable=too-many-branches
# pylint: disable=too-many-arguments


class Mask:
    """class to handle area masking"""

    def __init__(
        self,
        mask_name: str,
        basin_numbers: Optional[list[int]] = None,
        mask_path: Optional[str] = None,
        store_in_shared_memory: bool = False,
        thislog: logging.Logger | None = None,
    ) -> None:
        """class initialization

        Args:
            mask_name (str): mask name, must be in global mask_list
            basin_numbers (list[int], optional): list of grid values to select from grid masks
                                                 def=None
            mask_path (str, optional): override default path of mask data file
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
        self.basin_numbers = basin_numbers
        self.store_in_shared_memory = store_in_shared_memory
        self.shared_mem: Any = None
        self.shared_mem_child = False  # set to True if a child process
        self.polygons = None
        self.polygons_lon = np.array([])
        self.polygons_lat = np.array([])
        self.polygon = None
        self.polygon_lon = np.array([])
        self.polygon_lat = np.array([])
        self.mask_type = None  # 'xylimits', 'polygon', 'grid','latlimits'

        self.crs_wgs = CRS("epsg:4326")  # assuming you're using WGS84 geographic

        if thislog is not None:
            self.log = thislog  # optionally attach to a different log instance
        else:
            self.log = log

        # ---------------------------------------------------------------------------
        # Define the limits,bounds or polygons of each mask
        # ---------------------------------------------------------------------------

        if mask_name not in mask_list:
            raise ValueError(f"{mask_name} not in supported mask_list")

        self.log.info("Setting up %s..", mask_name)

        # -----------------------------------------------------------------------------

        if mask_name == "greenland_area_xylimits_mask":
            # Greenland rectangular mask for rapid masking
            self.mask_type = "xylimits"  # 'xylimits', 'polygon', 'grid','latlimits'

            self.xlimits = [
                -630000,
                904658,
            ]  # [minx, maxx] in m, in current  coordinate system
            self.ylimits = [
                -3355844,
                -654853,
            ]  # [miny, maxy] in m, in current  coordinate system
            self.crs_bng = CRS("epsg:3413")  # Polar Stereo - North -latitude of origin 70N, 45W

        elif mask_name == "ase_xylimits_mask":
            # ASE rectangular mask for rapid masking
            self.mask_type = "xylimits"  # 'xylimits', 'polygon', 'grid','latlimits'

            self.xlimits = [
                -1996781 - 50000,
                -1196781 + 50000,
            ]  # [minx, maxx] in m, in current  coordinate system
            self.ylimits = [
                -768646 - 50000,
                31353 + 50000,
            ]  # [miny, maxy] in m, in current  coordinate system
            self.crs_bng = CRS("epsg:3031")  # Polar Stereo - South

        elif mask_name == "ronne_filchner_xylimits_mask":
            # ASE rectangular mask for rapid masking
            self.mask_type = "xylimits"  # 'xylimits', 'polygon', 'grid','latlimits'

            self.xlimits = [
                -1685615,
                -390616,
            ]  # [minx, maxx] in m, in current  coordinate system
            self.ylimits = [
                -48143,
                1246856,
            ]  # [miny, maxy] in m, in current  coordinate system
            self.crs_bng = CRS("epsg:3031")  # Polar Stereo - South

        # -----------------------------------------------------------------------------

        elif mask_name == "antarctica_bedmachine_v2_grid_mask":
            # Antarctica surface type grid mask from BedMachine v2, 500m resolution
            #    - 0='Ocean', 1='Ice-free land',2='Grounded ice',3='Floating ice',
            #      4='Lake Vostok'
            # src=https://nsidc.org/data/nsidc-0756/versions/2

            self.mask_type = "grid"  # 'xylimits', 'polygon', 'grid','latlimits'

            if mask_path is None:
                mask_file = (
                    f'{environ["CPDATA_DIR"]}/RESOURCES/surface_discrimination_masks'
                    "/antarctica/bedmachine_v2/BedMachineAntarctica_2020-07-15_v02.nc"
                )
            else:
                mask_file = mask_path

            if not isfile(mask_file):
                self.log.error("mask file %s does not exist", mask_file)
                raise FileNotFoundError("mask file does not exist")

            self.num_x = 13333  # mask dimension in x direction
            self.num_y = 13333  # mask dimension in y direction
            self.dtype = np.uint8  # data type used for mask array.
            # values are 0..4, so using uint8 to reduce memory
            self.bad_mask_value = 255  # value in unknown grid cells in mask

            self.load_netcdf_mask(mask_file, flip=True)

            self.minxm = -3333000
            self.minym = -3333000
            self.binsize = 500

            self.crs_bng = CRS("epsg:3031")  # Polar Stereo - South (71S, 0E)
            self.mask_grid_possible_values = [0, 1, 2, 3, 4]  # values in the mask_grid
            self.grid_value_names = [
                "Ocean",
                "Ice-free land",
                "Grounded ice",
                "Floating ice",
                "Lake Vostok",
            ]
            self.grid_colors = ["blue", "brown", "grey", "green", "red"]

            if basin_numbers is not None:
                if 0 in basin_numbers:
                    self.mask_long_name += "Ocean"
                if 1 in basin_numbers:
                    self.mask_long_name += " Ice-free land"
                if 2 in basin_numbers:
                    self.mask_long_name += " Grounded ice"
                if 3 in basin_numbers:
                    self.mask_long_name += " Floating ice"
                if 4 in basin_numbers and 2 not in basin_numbers:
                    self.mask_long_name += " Lake Vostok"
                self.mask_long_name += " (bedmachine v2)"

        # -----------------------------------------------------------------------------

        elif mask_name == "greenland_bedmachine_v3_grid_mask":
            # Greenland surface type grid mask from BedMachine v3, 150m resolution
            #   - 0='Ocean', 1='Ice-free land',2='Grounded ice',3='Floating ice',
            #     4='non-Greenland land'

            self.mask_type = "grid"  # 'xylimits', 'polygon', 'grid','latlimits'
            # read netcdf file

            if not mask_path:
                mask_file = (
                    f'{environ["CPDATA_DIR"]}/RESOURCES/surface_discrimination_masks'
                    "/greenland/bedmachine_v3/BedMachineGreenland-2017-09-20.nc"
                )
            else:
                mask_file = mask_path

            if not isfile(mask_file):
                self.log.error("mask file %s does not exist", mask_file)
                raise FileNotFoundError("mask file does not exist")

            self.num_x = 10218  # mask dimension in x direction
            self.num_y = 18346  # mask dimension in y direction
            self.dtype = np.uint8  # data type used for mask array.
            # values are 0..4, so using uint8 to reduce memory
            self.bad_mask_value = 255  # value in unknown grid cells in mask

            self.load_netcdf_mask(mask_file, flip=True)

            self.binsize = 150
            self.minxm = -652925
            self.minym = -632675 - (self.num_y * self.binsize)

            self.crs_bng = CRS("epsg:3413")  # Polar Stereo - South (70N, 45W)
            self.mask_grid_possible_values = [0, 1, 2, 3, 4]  # values in the mask_grid
            self.grid_value_names = [
                "Ocean",  # 0
                "Ice-free land",  # 1
                "Grounded ice",  # 2
                "Floating ice",  # 3
                "Non-Greenland land",  # 4
            ]
            self.grid_colors = ["blue", "brown", "grey", "green", "white"]

            if basin_numbers is not None:
                if 0 in basin_numbers:
                    self.mask_long_name += "Ocean"
                if 1 in basin_numbers:
                    self.mask_long_name += " Ice-free land"
                if 2 in basin_numbers:
                    self.mask_long_name += " Grounded ice"
                if 3 in basin_numbers:
                    self.mask_long_name += " Floating ice"
                if 4 in basin_numbers:
                    self.mask_long_name += "Non-Greenland land"
                self.mask_long_name += " (bedmachine v3)"

        # -----------------------------------------------------------------------------
        # Antarctica surface type grid mask derived from BedMachine v2, 500m resolution
        #    - 0='Other', 1='Ice (grounded+floating)+ice-free land, dilated by 10km in to  ocean'
        elif mask_name == "antarctica_iceandland_dilated_10km_grid_mask":
            self.mask_type = "grid"  # 'xylimits', 'polygon', 'grid','latlimits'

            if not mask_path:
                mask_file = (
                    f'{environ["CPDATA_DIR"]}/RESOURCES/surface_discrimination_masks'
                    "/antarctica/bedmachine_v2/ant_dilated_grid_mask.nc"
                )
            else:
                mask_file = mask_path

            if not isfile(mask_file):
                self.log.error("mask file %s does not exist", mask_file)
                raise FileNotFoundError(f"mask file {mask_file} does not exist")

            self.num_x = 13333
            self.num_y = 13333
            self.dtype = np.uint8  # data type used for mask array.
            self.bad_mask_value = 255  # value in unknown grid cells in mask
            self.minxm = -3333000
            self.minym = -3333000
            self.binsize = 500  # meters
            self.crs_bng = CRS("epsg:3031")  # Polar Stereo - South (71S, 0E)
            self.mask_grid_possible_values = [0, 1]  # values in the mask_grid
            self.grid_value_names = ["outside", "inside Antarctic dilated mask"]
            self.grid_colors = ["blue", "darkgrey"]
            self.load_netcdf_mask(mask_file, flip=False, nc_mask_var="mask")

            self.mask_long_name = "Dilated by 10km in to  Ocean"
        # -----------------------------------------------------------------------------

        # Greenland surface type grid mask derived from BedMachine v3, 150m resolution
        #    - 0='Other', 1='Ice (grounded+floating)+ice-free land, dilated by 10km in to  ocean'
        elif mask_name == "greenland_iceandland_dilated_10km_grid_mask":
            self.mask_type = "grid"  # 'xylimits', 'polygon', 'grid','latlimits'

            if not mask_path:
                mask_file = (
                    f'{environ["CPDATA_DIR"]}/RESOURCES/surface_discrimination_masks'
                    "/greenland/bedmachine_v3/grn_dilated_grid_mask.nc"
                )
            else:
                mask_file = mask_path

            if not isfile(mask_file):
                self.log.error("mask file %s does not exist", mask_file)
                raise FileNotFoundError(f"mask file {mask_file} does not exist")

            self.num_x = 10218
            self.num_y = 18346
            self.dtype = np.uint8  # data type used for mask array.
            self.bad_mask_value = 255  # value in unknown grid cells in mask
            self.binsize = 150  # meters
            self.minxm = -652925
            self.minym = -3384575
            self.crs_bng = CRS("epsg:3413")  # Polar Stereo - North (70N, 45W)
            self.mask_grid_possible_values = [0, 1]  # values in the mask_grid
            self.grid_value_names = ["outside", "inside Greenland dilated mask"]
            self.grid_colors = ["blue", "darkgrey"]
            self.mask_long_name = "Dilated by 10km in to  Ocean"

            with Dataset(mask_file) as nc:
                self.mask_grid = np.array(nc.variables["mask"][:]).astype("i1")
            self.mask_grid = np.transpose(self.mask_grid)

        # -----------------------------------------------------------------------------

        elif mask_name == "antarctic_grounded_and_floating_2km_grid_mask":
            self.mask_type = "grid"  # 'xylimits', 'polygon', 'grid','latlimits'

            if not mask_path:
                mask_file = (
                    f'{environ["CPOM_SOFTWARE_DIR"]}/resources/drainage_basins/antarctica'
                    "/zwally_2012_imbie1_ant_grounded_and_floating_icesheet_basins/"
                    "basins/zwally_2012_imbie1_ant_grounded_and_floating_icesheet_basins_2km.nc"
                )
            else:
                mask_file = mask_path

            if not isfile(mask_file):
                self.log.error("mask file %s does not exist", mask_file)
                raise FileNotFoundError("mask file does not exist")

            self.num_x = 2820
            self.num_y = 2420
            self.dtype = np.uint8
            self.bad_mask_value = 255  # value in unknown grid cells in mask

            self.load_netcdf_mask(
                mask_file,
                flip=False,
                nc_mask_var="ANT_ZWALLY_BASINMASK_INCFLOATING_ICE",
            )

            self.minxm = -2820000
            self.minym = -2420000
            self.binsize = 2000  # km

            self.mask_grid_possible_values = list(range(28))  # values in the mask_grid
            self.grid_value_names = [f"Basin-{i}" for i in range(28)]
            self.grid_value_names[0] = "Unknown"

            self.crs_bng = CRS("epsg:3031")  # Polar Stereo - South -71S
            self.mask_long_name = "Zwally grounded and floating ice 2km grid"

        elif mask_name == "greenland_icesheet_2km_grid_mask":
            self.mask_type = "grid"  # 'xylimits', 'polygon', 'grid','latlimits'

            if not mask_path:
                mask_file = (
                    f'{environ["CPOM_SOFTWARE_DIR"]}/resources/drainage_basins/greenland/'
                    "zwally_2012_grn_icesheet_basins/basins/Zwally_GIS_basins_2km.nc"
                )
            else:
                mask_file = mask_path

            if not isfile(mask_file):
                self.log.error("mask file %s does not exist", mask_file)
                raise FileNotFoundError("mask file does not exist")

            self.num_x = 1000
            self.num_y = 1550
            self.dtype = np.uint8
            self.bad_mask_value = 255  # value in unknown grid cells in mask

            self.minxm = -1000000
            self.minym = -3500000
            self.binsize = 2000

            self.load_netcdf_mask(mask_file, flip=False, nc_mask_var="gre_basin_mask")

            self.crs_bng = CRS("epsg:3413")  # Polar Stereo - North -latitude of origin 70N, 45
            self.grid_value_names = [
                "None",
                "1.1",
                "1.2",
                "1.3",
                "1.4",
                "2.1",
                "2.2",
                "3.1",
                "3.2",
                "3.3",
                "4.1",
                "4.2",
                "4.3",
                "5.0",
                "6.1",
                "6.2",
                "7.1",
                "7.2",
                "8.1",
                "8.2",
            ]
            self.mask_grid_possible_values = list(range(20))  # values in the mask_grid
            self.grid_colors = [
                "blue",
                "bisque",
                "darkorange",
                "moccasin",
                "gold",
                "greenyellow",
                "yellowgreen",
                "gray",
                "lightgray",
                "silver",
                "purple",
                "sandybrown",
                "peachpuff",
                "coral",
                "tomato",
                "navy",
                "lavender",
                "olivedrab",
                "lightyellow",
                "sienna",
            ]
            self.mask_long_name = "Zwally grounded and floating ice 2km grid"

        elif mask_name == "antarctic_icesheet_2km_grid_mask_rignot2016":
            # basin mask values are : 0..18, or 255 (unknown)
            #
            self.mask_type = "grid"  # 'xylimits', 'polygon', 'grid','latlimits'

            if not mask_path:
                mask_file = (
                    f'{environ["CPOM_SOFTWARE_DIR"]}/resources/drainage_basins/antarctica/'
                    "rignot_2016_imbie2_ant_grounded_icesheet_basins/basins/"
                    "rignot_2016_imbie2_ant_grounded_icesheet_basins_2km.nc"
                )
            else:
                mask_file = mask_path

            if not isfile(mask_file):
                self.log.error("mask file %s does not exist", mask_file)
                raise FileNotFoundError("mask file does not exist")

            self.num_x = 2820
            self.num_y = 2420
            self.dtype = np.uint8
            self.bad_mask_value = 255  # value in unknown grid cells in mask

            self.minxm = -2820000  # meters
            self.minym = -2420000  # meters
            self.binsize = 2000  # meters

            self.dtype = np.uint8
            self.load_netcdf_mask(mask_file, flip=False, nc_mask_var="basinmask")

            self.mask_grid_possible_values = list(range(19))  # values in the mask_grid
            self.grid_value_names = [
                "Islands",
                "West H-Hp",
                "West F-G",
                "East E-Ep",
                "East D-Dp",
                "East Cp-D",
                "East B-C",
                "East A-Ap",
                "East Jpp-K",
                "West G-H",
                "East Dp-E",
                "East Ap-B",
                "East C-Cp",
                "East K-A",
                "West J-Jpp",
                "Peninsula Ipp-J",
                "Peninsula I-Ipp",
                "Peninsula Hp-I",
                "West Ep-F",
            ]

            self.crs_bng = CRS("epsg:3031")  # Polar Stereo - South -71S
            self.shapefile_path = (
                f"{environ["CPOM_SOFTWARE_DIR"]}/resources/drainage_basins/"
                "antarctica/rignot_2016_imbie2_ant_grounded_icesheet_basins/data/"
                "/ANT_Basins_IMBIE2_v1.6.shp"
            )
            self.shapefile_column_name = "SUBREGION1"
            self.mask_long_name = "Rignot (2016) 2km grid"

        elif mask_name == "greenland_icesheet_2km_grid_mask_rignot2016":
            self.mask_type = "grid"  # 'xylimits', 'polygon', 'grid','latlimits'

            if not mask_path:
                mask_file = (
                    f'{environ["CPOM_SOFTWARE_DIR"]}/resources/drainage_basins/greenland/'
                    "GRE_Basins_IMBIE2_v1.3/basins/"
                    "rignot_2016_imbie2_grn_grounded_icesheet_basins_2km.nc"
                )
            else:
                mask_file = mask_path

            if not isfile(mask_file):
                self.log.error("mask file %s does not exist", mask_file)
                raise FileNotFoundError("mask file does not exist")

            self.num_x = 1000
            self.num_y = 1550
            self.minxm = -1000000  # meters
            self.minym = -3500000  # meters
            self.binsize = 2000  # meters
            self.dtype = np.uint8
            self.bad_mask_value = 0  # value in unknown grid cells in mask

            self.load_netcdf_mask(mask_file, flip=False, nc_mask_var="basinmask")

            self.mask_grid_possible_values = list(range(57))  # values in the mask_grid

            # 0 (unclassified), 1-50 (ice caps), 51 (NW), 52(CW), 53(SW), 54(SE), 55(NE), 56(NO)
            self.grid_value_names = ["Ice cap -" + str(i) for i in range(57)]
            self.grid_value_names[0] = "unclassified"
            self.grid_value_names[51] = "NW"
            self.grid_value_names[52] = "CW"
            self.grid_value_names[53] = "SW"
            self.grid_value_names[54] = "SE"
            self.grid_value_names[55] = "NE"
            self.grid_value_names[56] = "NO"

            self.crs_bng = CRS("epsg:3413")  # Polar Stereo - North -latitude of origin 70N, 45
            self.shapefile_path = (
                f"{environ["CPOM_SOFTWARE_DIR"]}/resources/drainage_basins/greenland/"
                "GRE_Basins_IMBIE2_v1.3/shpfiles/GRE_IceSheet_IMBIE2_v1.3.shp"
            )
            self.shapefile_column_name = "SUBREGION1"
            self.mask_long_name = "Rignot (2016) 2km grid"
        elif mask_name == "greenland_icesheet_2km_grid_mask_mouginot2019":
            self.mask_type = "grid"  # 'xylimits', 'polygon', 'grid','latlimits'

            if not mask_path:
                mask_file = (
                    f'{environ["CPOM_SOFTWARE_DIR"]}/resources/drainage_basins/greenland/'
                    "Mouginot/"
                    "mouginot_2019_grn_grounded_icesheet_basins_2km.nc"
                )
            else:
                mask_file = mask_path

            if not isfile(mask_file):
                self.log.error("mask file %s does not exist", mask_file)
                raise FileNotFoundError("mask file does not exist")

            self.num_x = 1000
            self.num_y = 1550
            self.minxm = -1000000  # meters
            self.minym = -3500000  # meters
            self.binsize = 2000  # meters
            self.dtype = np.uint8
            self.bad_mask_value = 0  # value in unknown grid cells in mask

            self.load_netcdf_mask(mask_file, flip=False, nc_mask_var="basinmask")

            self.mask_grid_possible_values = list(range(7))  # values in the mask_grid

            self.grid_value_names = [
                "Unclassified [0]",
                "CE [1]",
                "CW [2]",
                "NO [3]",
                "NE [4]",
                "NW [5]",
                "SE [6]",
                "SW [7]",
            ]
            self.mask_grid_possible_values = list(range(len(self.grid_value_names)))

            self.crs_bng = CRS("epsg:3413")  # Polar Stereo - North -latitude of origin 70N, 45
            self.shapefile_path = (
                f'{environ["CPOM_SOFTWARE_DIR"]}/resources/drainage_basins/greenland/'
                "Mouginot/"
                "Greenland_Basins_PS_v1.4.2.shp"
            )
            self.shapefile_column_name = "SUBREGION1"

            self.mask_long_name = "Mouginot (2019) 2km grid"

        elif mask_name == "greenland_icesheet_2km_grid_mask_mouginot2019_glaciers":
            self.mask_type = "grid"  # 'xylimits', 'polygon', 'grid','latlimits'

            if not mask_path:
                mask_file = (
                    f'{environ["CPOM_SOFTWARE_DIR"]}/resources/drainage_basins/greenland/'
                    "Mouginot/mouginot_subbasins/"
                    "mouginot_2019_grn_grounded_icesheet_glacier_basins_2km.nc"
                )
            else:
                mask_file = mask_path

            if not isfile(mask_file):
                self.log.error("mask file %s does not exist", mask_file)
                raise FileNotFoundError("mask file does not exist")

            self.num_x = 1000
            self.num_y = 1550
            self.minxm = -1000000  # meters
            self.minym = -3500000  # meters
            self.binsize = 2000  # meters
            self.dtype = np.uint16
            self.bad_mask_value = 0  # value in unknown grid cells in mask

            self.load_netcdf_mask(mask_file, flip=False, nc_mask_var="basinmask")
            # Load glacier names from CSV
            glacier_csv = (
                f'{environ["CPOM_SOFTWARE_DIR"]}/resources/drainage_basins/greenland/'
                "Mouginot/mouginot_subbasins/glacier_names.csv"
            )
            # Mask values are 1-260, CSV has 260 rows. Prepend entry for mask value 0.
            glacier_names = ["Unclassified"]
            with open(glacier_csv, mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    glacier_names.append(row["NAME"])

            self.grid_value_names = glacier_names

            self.mask_grid_possible_values = list(range(len(self.grid_value_names)))
            self.shapefile_path = (
                f'{environ["CPOM_SOFTWARE_DIR"]}/resources/drainage_basins/greenland/'
                "Mouginot/mouginot_subbasins/"
                "Greenland_Basins_PS_v1.4.2.shp"
            )
            self.shapefile_column_name = "NAME"

            self.crs_bng = CRS("epsg:3413")  # Polar Stereo - North -latitude of origin 70N, 45
            self.mask_long_name = "Mouginot (2019) 2km glacier grid"
        else:
            raise ValueError(f"mask name: {mask_name} not supported")

        # -----------------------------------------------------------------------------

        # Setup the Transforms
        self.xy_to_lonlat_transformer = Transformer.from_proj(
            self.crs_bng, self.crs_wgs, always_xy=True
        )
        self.lonlat_to_xy_transformer = Transformer.from_proj(
            self.crs_wgs, self.crs_bng, always_xy=True
        )

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
        basin_numbers: Optional[list[int]] = None,
        inputs_are_xy: bool = False,
    ) -> tuple[np.ndarray, int]:
        """Given a list of lat,lon or x,y points, find the points that are inside the current mask

        Args:
            lats (np.ndarray|list[float]): list of latitude points
            lons (np.ndarray|list[float]): list of longitude points
            basin_numbers (list[int,], optional): list of basin numbers. Defaults to None.
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

        if inputs_are_xy:
            x, y = lats, lons
        else:
            x, y = self.latlon_to_xy(lats, lons)  # pylint: disable=E0633

        # get count of xy that are infinity
        np.sum(np.isinf(x)) + np.sum(np.isinf(y))

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
        else:
            return inmask, 0

        return inmask, n_inside

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
                                 grid location or np.NaN if outside area

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
            (x, y) = self.latlon_to_xy(lats, lons)  # pylint: disable=E0633

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

    def latlon_to_xy(self, lats: np.ndarray, lons: np.ndarray) -> tuple:
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

    ######################################################
    # Function to Mask to a polars LazyFrame or DataFrame #
    ######################################################
    # pylint: disable=R0917
    def points_inside_polars(
        self,
        df: pl.LazyFrame | pl.DataFrame,
        x_col: str = "x",
        y_col: str = "y",
        basin_numbers: Optional[list[int]] = None,
        return_pl_dataframe=False,
    ) -> pl.LazyFrame | pl.DataFrame:
        """Given a list of lat,lon or x,y points, find the points that are inside the current mask
        Args:
            df (pl.LazyFrame|pl.DataFrame): polars LazyFrame or DataFrame with x,y columns
            x_col (str): name of column in df containing x values
            y_col (str): name of column in column in df containing y values
            basin_numbers (list[int,], optional): list of basin numbers. Defaults to None.
            return_pl_dataframe (bool, optional): return a polars DataFrame instead of LazyFrame
                                                  Defaults to False.
        Returns:
            pl.LazyFrame|pl.DataFrame: polars LazyFrame or DataFrame with only points inside mask
        """
        # Ensure LazyFrame
        if isinstance(df, pl.DataFrame):
            df = df.lazy()

        # Basin numbers assignment
        if basin_numbers and not isinstance(basin_numbers, (list, np.ndarray)):
            basin_numbers = [basin_numbers]
        if self.basin_numbers and not basin_numbers:
            basin_numbers = self.basin_numbers

        # Filter by mask type
        if self.mask_type == "xylimits":
            df = df.filter(
                (pl.col(x_col) >= self.xlimits[0])
                & (pl.col(x_col) <= self.xlimits[1])
                & (pl.col(y_col) >= self.ylimits[0])
                & (pl.col(y_col) <= self.ylimits[1])
            )
        elif self.mask_type == "grid":
            df = df.with_columns(
                [
                    ((pl.col(x_col) - self.minxm) / self.binsize)
                    .round()
                    .cast(pl.Int64)
                    .alias("ii"),
                    ((pl.col(y_col) - self.minym) / self.binsize)
                    .round()
                    .cast(pl.Int64)
                    .alias("jj"),
                ]
            )
            df = (
                df.with_columns(
                    [
                        (
                            (pl.col("ii") >= 0)
                            & (pl.col("ii") < self.num_x)
                            & (pl.col("jj") >= 0)
                            & (pl.col("jj") < self.num_y)
                        ).alias("in_bounds")
                    ]
                )
                .filter(pl.col("in_bounds"))
                .drop("in_bounds")
            )
            jj = df.select("jj").collect().to_numpy().flatten()
            ii = df.select("ii").collect().to_numpy().flatten()

            # Get mask values from grid
            mask_values = self.mask_grid[jj, ii]
            df = df.with_columns([pl.Series("mask_value", mask_values)])

            if self.basin_numbers:
                df = df.filter(pl.col("mask_value").is_in(self.basin_numbers))
            else:
                df = df.filter(pl.col("mask_value") > 0)
            df = df.drop(["ii", "jj", "mask_value"])

        return df.collect() if return_pl_dataframe else df
