"""
Class to manage background images for polar plotting
"""

import gc  # garbage collection
import logging
import os
import sys

import cartopy
import cartopy.feature as cfeature
import cartopy.io.img_tiles as cimgt
import matplotlib.pyplot as plt  # plotting functions
import numpy as np
import pyproj
from matplotlib import colormaps
from matplotlib.colors import LinearSegmentedColormap
from netCDF4 import Dataset  # pylint: disable=E0611
from PIL import Image
from pyproj import Transformer
from skimage import exposure  # from scikit-image

from cpom.areas.areas import Area
from cpom.dems.dems import Dem  # DEM reading

log = logging.getLogger(__name__)

# pylint: disable=unpacking-non-sequence
# pylint: disable=too-many-lines
# pylint: disable=too-many-arguments
# pylint: disable=too-many-locals
# pylint: disable=too-many-statements
# pylint: disable=too-many-branches
# pylint: disable=too-few-public-methods

# dict of available plot background names and test areas
all_backgrounds = {
    "basic_land": ["antarctica_is"],
    "cartopy_stock": ["antarctica_is", "arctic"],
    "cartopy_stock_ocean": ["antarctica_is", "arctic"],
    "arcgis_shaded_relief": ["antarctica_is", "arctic"],
    # "google_satellite": ["canadianlake_woods"],
    "bluemarble": ["antarctica_is", "arctic"],
    "natural_earth_cbh": ["antarctica_is", "arctic"],
    "natural_earth_cbh_oceanmasked": ["antarctica_is", "arctic"],
    "natural_earth_cbh_ocean": ["antarctica_is", "arctic"],
    "natural_earth_gray": ["antarctica_is", "arctic"],
    "natural_earth1": ["antarctica_is", "arctic"],
    "stamen": ["antarctica_is", "arctic"],
    "natural_earth_faded": ["antarctica_is", "arctic"],
    "moa": ["antarctica_is", "vostok"],
    "cpom_dem": ["antarctica_is"],
    "awi_gis_dem": ["greenland"],
    "arcticdem_1km": ["arctic"],
    "rema_dem_1km": ["antarctica_is"],
    "grn_s1_mosaic": ["greenland"],
    "hillshade": ["antarctica_is", "greenland"],
    "ant_iceshelves": ["antarctica_is"],
    "ibcso_bathymetry": ["antarctica_is"],
    "ibcao_bathymetry": ["arctic"],
}

all_background_resolutions = ["low", "medium", "high", "vhigh", "vvhigh"]

# all_backgrounds = {"google_satellite": 'canadianlake_woods',}


class Background:
    """class to handle background images"""

    def __init__(self, name, area):
        """class initialization

        Args:
            name (str): background name
            area (str): area name from cpom.areas.areas
        """
        if isinstance(area, str):
            self.thisarea = Area(area)
        elif isinstance(area, Area):
            self.thisarea = area

        if name is None:
            name = self.thisarea.background_image

        if name not in all_backgrounds:
            log.error("background name not valid %s", name)
            sys.exit(1)

        self.name = name

        self.moa_image = None
        self.moa_zimage = None

    def load(
        self,
        ax,  # axis to display background
        dataprj,  # cartopy coordinate reference system (crs)
        alpha=None,  # transparency of background (0..1). If None Area.background_image_alpha
        resolution=None,  # resolution str for backgrounds that have this option
        include_features=True,  # cartopy features to overlay for backgrounds that include this
        # option
        hillshade_params=None,  # dict of hillshade parameters for use with the
        # 'hillshade' background
        zorder=None,  # zorder number to use to display background
        cmap=None,  # colormap to use for backgrounds that support different colourmaps
    ):
        """
        param: ax : axis
        param: dataprj : cartopy coordinate reference system (crs)
        param: background: replace default background image (thisarea.background_image) for plot
        with one of the available backgrounds

        cartopy_stock : stock image for Cartopy, uses a downgraded natural earth image.
        Only one resolution
        cartopy_stock_ocean : stock image for Cartopy with land blanked out in a single colour.
        Only one resolution
        arcgis_shaded_relief : resolution (low, medium, default is high) : : ArcGIS World Shaded
        Relief tiles
        google_satellite : resolution (low, medium, high, vhigh, vvhigh (default)
        bluemarble : resolution (low, medium, high) : NASA Blue Marble world image
        natural_earth_cbh : resolution (low, medium, default is high): Cross Blended Hypso with
        Shaded Relief and Water
        https://www.naturalearthdata.com/downloads/50m-raster-data/50m-cross-blend-hypso/
        natural_earth_cbh_oceanmasked : resolution (low, medium, default is high) : as for
        natural_earth_cbh, but with oceans set to white
        natural_earth_cbh_ocean : resolution (low, medium, high)
        natural_earth_gray : resolution (low, medium, high):  Gray Earth with Shaded Relief,
        Hypsography, and Ocean Bottom :
        https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/50m/raster/GRAY_50M_SR_OB.zip
        natural_earth1 : resolution (low, medium, default is high): Natural Earth I with Shaded
        Relief and Water :
        https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/50m/raster/NE1_50M_SR_W.zip
        stamen : resolution (low, medium, default is high):terrain-background
        http://maps.stamen.com/terrain-background/#12/37.7706/-122.3782
        natural_earth_faded
        basic_land : resolution (low, medium, high) : basic ocean and land plot
        moa : MODIS Mosaic of Antarctica 2008-2009 (MOA2009) Image Map at 750m resolution,
        gray scale
        cpom_dem : CPOM Antarctic DEM at 1km resolution, gray scale
        hillshade : hillshade_params={"azimuth": f,”pitch": f,”dem": “str”,”alpha": f}
        ant_iceshelves
        ibcso_bathymetry
        ibcao_bathymetry
        awi_gis_dem : Greenland DEM from AWI 2014
        grn_s1_mosaic : resolution (low, medium, high, vhigh)
        arcticdem_1km :  ArcticDEM at 1km resolution
        rema_dem_1km : REMA Antarctic DEM at 1km resolution
                '
        param: resolution : 'low', 'medium','high','vhigh','vvhigh', if None,
        self.thisarea.background_image_resolution is used
        alpha:  set the background transparency (alpha), 0..1. If None,
        self.thisarea.background_image_alpha is used
        :return:
        """

        # Select background image to use. Default is from cpom.areas.Area.background_image
        if not self.name:
            if isinstance(self.thisarea.background_image, list):
                self.name = self.thisarea.background_image[0]
            else:
                self.name = self.thisarea.background_image
        if not resolution:
            if isinstance(self.thisarea.background_image_resolution, list):
                resolution = self.thisarea.background_image_resolution[0]
            else:
                resolution = self.thisarea.background_image_resolution

        if not alpha:
            if isinstance(self.thisarea.background_image_alpha, list):
                alpha = self.thisarea.background_image_alpha[0]
            else:
                alpha = self.thisarea.background_image_alpha

        print("-------------------------------------------------------------")
        print(f"Background: {self.name}")

        log.info("resolution=%s", resolution)
        print("alpha=", alpha)

        # Most background files are stored here (unless they are too big)
        os.environ["CARTOPY_USER_BACKGROUNDS"] = (
            os.environ["CPOM_SOFTWARE_DIR"] + "/resources/backgrounds/"
        )

        if self.name == "cartopy_stock":
            print("Loading cartopy stock background")
            ax.stock_img()
            return
        if self.name == "cartopy_stock_ocean":
            print("Loading cartopy stock background")
            ax.stock_img()
            # blank out land with single colour (default land feature colour)
            ax.add_feature(cfeature.LAND, zorder=5)
            return
        if self.name == "arcgis_shaded_relief":
            print("Loading arcgis background")
            url = (
                "https://server.arcgisonline.com/ArcGIS/rest/services"
                "/World_Shaded_Relief/MapServer/tile/{z}/{y}/{x}.jpg"
            )
            image = cimgt.GoogleTiles(url=url)

            if resolution == "low":
                ax.add_image(image, 2, zorder=0)
            elif resolution == "medium":
                ax.add_image(image, 4, zorder=0)
            else:
                ax.add_image(image, 8, zorder=0)
        elif self.name == "google_satellite":
            print("Loading google_satellite tiles background")
            image = cimgt.GoogleTiles(style="satellite")

            if resolution == "low":
                print("Adding Google Earth Satellite background at low res")
                ax.add_image(image, 2, alpha=alpha)
            elif resolution == "medium":
                print("Adding Google Earth Satellite background at medium res")
                ax.add_image(image, 4, alpha=alpha)
            elif resolution == "high":
                print("Adding Google Earth Satellite background at high res")
                ax.add_image(image, 8, alpha=alpha)
            elif resolution == "vhigh":
                print("Adding Google Earth Satellite background at high res")
                ax.add_image(image, 12, alpha=alpha)
            elif resolution == "vvhigh":
                print("Adding Google Earth Satellite background at high res")
                ax.add_image(image, 14, alpha=alpha)
            else:
                print("Adding Google Earth Satellite background at very high res")
                ax.add_image(image, 14, alpha=alpha)

        elif self.name == "bluemarble":
            print("Loading bluemarble background")
            if resolution == "low":
                ax.background_img(name="blue_marble", resolution="low")
            elif resolution == "medium":
                ax.background_img(name="blue_marble", resolution="medium")
            else:
                ax.background_img(name="blue_marble", resolution="high")

        elif self.name == "natural_earth_cbh":
            print("Loading natural_earth cbh background : ", resolution)
            if resolution == "low":
                ax.background_img(name="natural_earth_cbh", resolution="low")
            elif resolution == "medium":
                ax.background_img(name="natural_earth_cbh", resolution="medium")
            else:
                ax.background_img(name="natural_earth_cbh", resolution="high")

        elif self.name == "natural_earth_cbh_oceanmasked":
            print("Loading natural_earth cbh oceanmasked background : ", resolution)
            if resolution == "low":
                ax.background_img(name="natural_earth_cbh_oceanmasked", resolution="low")
            elif resolution == "medium":
                ax.background_img(name="natural_earth_cbh_oceanmasked", resolution="medium")
            else:
                ax.background_img(name="natural_earth_cbh_oceanmasked", resolution="high")

        elif self.name == "natural_earth_cbh_ocean":
            print("Loading natural_earth cbh background")
            if resolution == "low":
                ax.background_img(name="natural_earth_cbh", resolution="low")
            elif resolution == "medium":
                ax.background_img(name="natural_earth_cbh", resolution="medium")
            else:
                ax.background_img(name="natural_earth_cbh", resolution="high")
            ax.add_feature(cfeature.LAND, zorder=5)

        elif self.name == "natural_earth_gray":
            print("Loading natural_earth_gray background")
            if resolution == "low":
                ax.background_img(name="natural_earth_gray", resolution="low")
            elif resolution == "medium":
                ax.background_img(name="natural_earth_gray", resolution="medium")
            else:
                ax.background_img(name="natural_earth_gray", resolution="high")

        elif self.name == "natural_earth1":
            print("Loading natural_earth1 background")
            if resolution == "low":
                ax.background_img(name="natural_earth_1", resolution="low")
            elif resolution == "medium":
                ax.background_img(name="natural_earth_1", resolution="medium")
            else:
                ax.background_img(name="natural_earth_1", resolution="high")

        elif self.name == "stamen":
            print("Loading stamen:terrain-background background")
            stamen_terrain = cimgt.Stamen("terrain-background")
            if resolution == "low":
                ax.add_image(stamen_terrain, 2, zorder=0)  # 8 is high res, 4, 1 lower
            elif resolution == "medium":
                ax.add_image(stamen_terrain, 4, zorder=0)  # 8 is high res, 4, 1 lower
            else:
                ax.add_image(stamen_terrain, 8, zorder=0)  # 8 is high res, 4, 1 lower

        elif self.name == "natural_earth_faded":
            print("Loading natural_earth_faded background")

            # Add a semi-transparent white layer to fade the background
            ax.add_feature(
                cfeature.NaturalEarthFeature(
                    "physical", "land", "10m", edgecolor="none", facecolor="whitesmoke"
                ),
                zorder=1,
                alpha=0.4,
            )

            ax.background_img(name="natural_earth_cbh_oceanmasked", resolution="medium")

        elif self.name == "basic_land":
            print("Loading basic_land background")
            # ax.stock_img()
            ax.set_facecolor("#F2F7FF")
            land_color = cfeature.COLORS["land"]
            land_color = "#F5F5F5"
            if resolution == "low":
                land = cfeature.NaturalEarthFeature(
                    "physical",
                    "land",
                    "110m",
                    edgecolor="face",
                    facecolor=land_color,
                )
            elif resolution == "medium":
                land = cfeature.NaturalEarthFeature(
                    "physical",
                    "land",
                    "50m",
                    edgecolor="face",
                    facecolor=land_color,
                )
            elif resolution == "high":
                land = cfeature.NaturalEarthFeature(
                    "physical",
                    "land",
                    "10m",
                    edgecolor="black",
                    facecolor=land_color,
                )
            else:
                land = cfeature.NaturalEarthFeature(
                    "physical",
                    "land",
                    "110m",
                    edgecolor="face",
                    facecolor=land_color,
                )
            ax.add_feature(land, zorder=0)

        elif self.name == "moa":
            # -------------------------------------------------------------------------------------
            # 	Load CPOM Antarctic 1km DEM
            # -------------------------------------------------------------------------------------
            print("----------------------------------------------------------")

            # new_img = plt.imread(moa_image)

            moa_image_file = (
                os.environ["CPDATA_DIR"]
                + "/SATS/OPTICAL/MODIS/MOA2009/750m/moa750_2009_hp1_v1.1.tif"
            )
            print("reading MOA 750m Antarctic image..")
            print(moa_image_file)

            self.moa_image = Image.open(moa_image_file)
            print("processing MOA image...")
            ncols, nrows = self.moa_image.size
            zimage = np.array(self.moa_image.getdata()).reshape((nrows, ncols))
            self.moa_zimage = np.flip(zimage, 0)

            ximage = np.linspace(-3174450.0, 2867550.0, 8056, endpoint=True)
            yimage = np.linspace(-2816675.0, 2406325.0, 6964, endpoint=True)
            minimagex = ximage.min()
            minimagey = yimage.min()
            imagebinsize = 750.0
            imagebinsize_km = imagebinsize / 1000.0

            # Plot MOA

            if self.thisarea.hemisphere == "north":
                print("Can not plot Antarctic MOA background in Northern Hemisphere")
                sys.exit()

            # Get tie point

            prj = pyproj.CRS("epsg:3031")
            wgs_prj = pyproj.CRS("epsg:4326")  # WGS84

            if self.thisarea.specify_by_centre:
                c_x, c_y = self.thisarea.latlon_to_xy(
                    self.thisarea.centre_lat, self.thisarea.centre_lon
                )
                xll = c_x - ((self.thisarea.width_km * 1000) / 2)
                yll = c_y - ((self.thisarea.height_km * 1000) / 2)
            else:
                tie_point_lonlat_to_xy_transformer = Transformer.from_proj(
                    wgs_prj, prj, always_xy=True
                )
                xll, yll = tie_point_lonlat_to_xy_transformer.transform(
                    self.thisarea.llcorner_lon, self.thisarea.llcorner_lat
                )

            image_bin_offset_x = int((xll - minimagex) / imagebinsize)
            image_bin_offset_y = int((yll - minimagey) / imagebinsize)

            zimage = self.moa_zimage[
                image_bin_offset_y : image_bin_offset_y
                + int(self.thisarea.height_km / imagebinsize_km),
                image_bin_offset_x : image_bin_offset_x
                + int(self.thisarea.width_km / imagebinsize_km),
            ]

            ximage = ximage[
                image_bin_offset_x : image_bin_offset_x
                + int(self.thisarea.width_km / imagebinsize_km)
            ]
            yimage = yimage[
                image_bin_offset_y : image_bin_offset_y
                + int(self.thisarea.height_km / imagebinsize_km)
            ]

            x_grid, y_grid = np.meshgrid(ximage, yimage)

            zimage = exposure.equalize_adapthist(zimage)

            # thiscmap = plt.cm.get_cmap("Greys_r") - depreciated
            thiscmap = colormaps["Greys_r"]

            print("pcolormesh")
            plt.pcolormesh(x_grid, y_grid, zimage, cmap=thiscmap, shading="auto", transform=dataprj)

        elif self.name == "cpom_dem":
            print("----------------------------------------------------------")
            print("reading CS2 DEM..")
            demfile = (
                os.environ["CPDATA_DIR"] + "/SATS/RA/DEMS/ant_cpom_cs2_1km/nc/ant_cpom_cs2_1km.nc"
            )
            nc_dem = Dataset(demfile)
            xdem = nc_dem.variables["x"][:]
            ydem = nc_dem.variables["y"][:]
            zdem = nc_dem.variables["z"][:]
            print("DEM zdem shape ", zdem.data.shape)
            print("DEM xdem.min,xdem.max= ", xdem.min(), xdem.max())
            print("DEM ydem.min,ydem.max= ", ydem.min(), ydem.max())
            print("----------------------------------------------------------")
            mindemx = xdem.min()
            mindemy = ydem.min()

            # Get tie point

            prj = pyproj.CRS("epsg:3031")
            wgs_prj = pyproj.CRS("epsg:4326")  # WGS84
            tie_point_lonlat_to_xy_transformer = Transformer.from_proj(wgs_prj, prj, always_xy=True)

            if self.thisarea.specify_plot_area_by_lowerleft_corner:
                xll, yll = tie_point_lonlat_to_xy_transformer.transform(
                    self.thisarea.llcorner_lon, self.thisarea.llcorner_lat
                )
            else:
                xc, yc = tie_point_lonlat_to_xy_transformer.transform(
                    self.thisarea.centre_lon, self.thisarea.centre_lat
                )
                xll = xc - (self.thisarea.width_km * 1e3) / 2
                yll = yc - (self.thisarea.height_km * 1e3) / 2

            dem_bin_offset_x = int((xll - mindemx) / 1000.0)
            dem_bin_offset_y = int((yll - mindemy) / 1000.0)

            print("dem_bin_offset =", dem_bin_offset_x, dem_bin_offset_y)
            dem_bin_offset_x = max(dem_bin_offset_x, 0)
            dem_bin_offset_y = max(dem_bin_offset_y, 0)

            zdem = np.flip(zdem, 0)

            zdem = zdem[
                dem_bin_offset_y : dem_bin_offset_y + int(self.thisarea.height_km),
                dem_bin_offset_x : dem_bin_offset_x + int(self.thisarea.width_km),
            ]

            xdem = xdem[dem_bin_offset_x : dem_bin_offset_x + int(self.thisarea.width_km)]
            ydem = np.flip(ydem)
            ydem = ydem[dem_bin_offset_y : dem_bin_offset_y + int(self.thisarea.height_km)]

            x_grid, y_grid = np.meshgrid(xdem, ydem)

            # thiscmap = plt.cm.get_cmap("Greys", 48) - depreciated
            thiscmap = colormaps["Greys"].resampled(48)

            thiscmap.set_bad(color="aliceblue")
            ax.pcolormesh(
                x_grid,
                y_grid,
                zdem,
                cmap=thiscmap,
                shading="auto",
                vmin=self.thisarea.min_elevation - 50.0,
                vmax=self.thisarea.max_elevation + 50.0,
                transform=dataprj,
            )

        elif self.name == "hillshade":
            print(f"Loading background : {self.name}")

            # Define default hill shade params for southern hemisphere
            def_hillshade_params = {
                "azimuth": 235.0,
                "pitch": 45.0,
                "dem": "awi_ant_1km",
                "alpha": 0.3,
            }
            # Define default hill shade params for northern hemisphere
            if self.thisarea.hemisphere == "north":
                def_hillshade_params = {
                    "azimuth": 140.0,
                    "pitch": 45.0,
                    "dem": "awi_grn_1km",
                    "alpha": 0.3,
                }

            if self.thisarea.hillshade_params is not None:
                def_hillshade_params.update(self.thisarea.hillshade_params)

            if hillshade_params is not None:
                def_hillshade_params.update(hillshade_params)

            print("background hillshade params: ", def_hillshade_params)

            thisdem = Dem(def_hillshade_params["dem"])

            print("Applying hillshade to DEM..")
            thisdem.hillshade(
                azimuth=def_hillshade_params["azimuth"],
                pitch=def_hillshade_params["pitch"],
            )  # convert dem elevations to hill shaded values (0..256)
            print("done..")

            # Get lower left x,y coordinates of area in m
            if self.thisarea.specify_plot_area_by_lowerleft_corner:
                xll, yll = thisdem.lonlat_to_xy_transformer.transform(
                    self.thisarea.llcorner_lon, self.thisarea.llcorner_lat
                )
            else:
                xc, yc = thisdem.lonlat_to_xy_transformer.transform(
                    self.thisarea.centre_lon, self.thisarea.centre_lat
                )
                xll = xc - (self.thisarea.width_km * 1e3) / 2
                yll = yc - (self.thisarea.height_km * 1e3) / 2

            dem_bin_offset_x = int((xll - thisdem.mindemx) / thisdem.binsize)
            dem_bin_offset_y = int((yll - thisdem.mindemy) / thisdem.binsize)

            dem_bin_offset_x = max(dem_bin_offset_x, 0)
            dem_bin_offset_y = max(dem_bin_offset_y, 0)

            zdem = np.flip(thisdem.zdem, 0)

            zdem = zdem[
                dem_bin_offset_y : dem_bin_offset_y
                + int(self.thisarea.height_km * 1e3 / thisdem.binsize),
                dem_bin_offset_x : dem_bin_offset_x
                + int(self.thisarea.width_km * 1e3 / thisdem.binsize),
            ]

            xdem = thisdem.xdem[
                dem_bin_offset_x : dem_bin_offset_x
                + int(self.thisarea.width_km * 1e3 / thisdem.binsize)
            ]
            ydem = np.flip(thisdem.ydem)
            ydem = ydem[
                dem_bin_offset_y : dem_bin_offset_y
                + int(self.thisarea.height_km * 1e3 / thisdem.binsize)
            ]

            x_grid, y_grid = np.meshgrid(xdem, ydem)

            thiscmap = colormaps["Greys"]
            if cmap:
                thiscmap = cmap
            # thiscmap.set_bad(color="aliceblue")

            ax.pcolormesh(
                x_grid,
                y_grid,
                zdem,
                cmap=thiscmap,
                shading="auto",
                vmin=np.nanmin(zdem),
                vmax=np.nanmax(zdem),
                transform=dataprj,
                alpha=def_hillshade_params["alpha"],
                zorder=zorder if zorder else None,
            )

            del thisdem
            gc.collect()

        # this background applies a white ice shelf layer
        elif self.name == "ant_iceshelves":
            print(f"Loading background : {self.name}")

            # Define default hill shade params for southern hemisphere
            def_hillshade_params = {
                "azimuth": 235.0,
                "pitch": 45.0,
                "dem": "awi_ant_1km_floating",
                "alpha": 1.0,
            }

            print("background hillshade params: ", def_hillshade_params)

            thisdem = Dem(def_hillshade_params["dem"])
            thisdem.hillshade(
                azimuth=def_hillshade_params["azimuth"],
                pitch=def_hillshade_params["pitch"],
            )  # convert dem elevations to hill shaded values (0..256)

            # Get lower left x,y coordinates of area in m
            if self.thisarea.specify_plot_area_by_lowerleft_corner:
                xll, yll = thisdem.lonlat_to_xy_transformer.transform(
                    self.thisarea.llcorner_lon, self.thisarea.llcorner_lat
                )
            else:
                xc, yc = thisdem.lonlat_to_xy_transformer.transform(
                    self.thisarea.centre_lon, self.thisarea.centre_lat
                )
                xll = xc - (self.thisarea.width_km * 1e3) / 2
                yll = yc - (self.thisarea.height_km * 1e3) / 2

            dem_bin_offset_x = int((xll - thisdem.mindemx) / thisdem.binsize)
            dem_bin_offset_y = int((yll - thisdem.mindemy) / thisdem.binsize)

            zdem = np.flip(thisdem.zdem, 0)

            zdem = zdem[
                dem_bin_offset_y : dem_bin_offset_y
                + int(self.thisarea.height_km * 1e3 / thisdem.binsize),
                dem_bin_offset_x : dem_bin_offset_x
                + int(self.thisarea.width_km * 1e3 / thisdem.binsize),
            ]

            xdem = thisdem.xdem[
                dem_bin_offset_x : dem_bin_offset_x
                + int(self.thisarea.width_km * 1e3 / thisdem.binsize)
            ]
            ydem = np.flip(thisdem.ydem)
            ydem = ydem[
                dem_bin_offset_y : dem_bin_offset_y
                + int(self.thisarea.height_km * 1e3 / thisdem.binsize)
            ]

            x_grid, y_grid = np.meshgrid(xdem, ydem)

            thiscmap = LinearSegmentedColormap.from_list(
                "white_viridis",
                [
                    (0, "#808080"),
                    (1, "#ffffff"),
                ],
                N=256,
            )

            if cmap:
                thiscmap = cmap

            ax.pcolormesh(
                x_grid,
                y_grid,
                zdem,
                cmap=thiscmap,
                shading="auto",
                vmin=np.nanmin(zdem),
                vmax=np.nanmax(zdem),
                transform=dataprj,
                alpha=def_hillshade_params["alpha"],
                zorder=zorder if zorder else None,
            )

        # Antarctic bathymetry from IBCSO v2, available in 'low','medium','high' resolutions
        elif self.name == "ibcso_bathymetry":
            bgfile = (
                f'{os.environ["CPDATA_DIR"]}/RESOURCES/'
                f"backgrounds/IBCSO_v2_bed_RGB_{resolution}.npz"
            )

            print(f"Loading background : {self.name} : {bgfile}")

            try:
                print("Loading IBCSO_v2_bed_RGB topo map background..")
                data = np.load(bgfile, allow_pickle=True)
                zdem = data["zdem"]
                x_grid = data["X"]
                y_grid = data["Y"]
            except IOError:
                log.error("Could not read %s", bgfile)
                sys.exit(f"Could not read {bgfile}")

            # thiscmap = plt.cm.get_cmap("Blues_r", 8) - depreciated
            thiscmap = colormaps["Blues_r"].resampled(8)

            ax.pcolormesh(
                x_grid,
                y_grid,
                zdem,
                cmap=thiscmap,
                shading="auto",
                alpha=alpha,
                vmin=np.nanmin(zdem),
                vmax=np.nanmax(zdem),
                transform=dataprj,
            )

        # Arctic bathymetry from IBCAO_v4.2
        elif self.name == "ibcao_bathymetry":
            print(f"Loading background : {self.name}")

            bgfile = (
                f'{os.environ["CPDATA_DIR"]}/RESOURCES/backgrounds/'
                f"IBCAO_v4.2_bathymetry_{resolution}.npz"
            )

            try:
                print(f"Loading IBCAO_v4.2 topo map background {bgfile}..")
                data = np.load(bgfile, allow_pickle=True)
                zdem = data["zdem"]
                x_grid = data["X"]
                y_grid = data["Y"]
            except IOError:
                log.error("Could not read %s", bgfile)

            base_cmap = colormaps["Blues"].reversed()
            new_colors = base_cmap(np.linspace(0, 1, 8))
            thiscmap = LinearSegmentedColormap.from_list("custom_blues", new_colors)

            ax.pcolormesh(
                x_grid,
                y_grid,
                zdem,
                cmap=thiscmap,
                shading="auto",
                alpha=alpha,
                vmin=np.nanmin(zdem),
                vmax=np.nanmax(zdem),
                transform=dataprj,
            )

        elif self.name == "awi_gis_dem":
            print(f"Loading background : {self.name}")
            thisdem = Dem("awi_grn_1km")

            if self.thisarea.specify_plot_area_by_lowerleft_corner:
                xll, yll = thisdem.lonlat_to_xy_transformer.transform(
                    self.thisarea.llcorner_lon, self.thisarea.llcorner_lat
                )
            else:
                xc, yc = thisdem.lonlat_to_xy_transformer.transform(
                    self.thisarea.centre_lon, self.thisarea.centre_lat
                )
                xll = xc - (self.thisarea.width_km * 1e3) / 2
                yll = yc - (self.thisarea.height_km * 1e3) / 2

            dem_bin_offset_x = int((xll - thisdem.mindemx) / thisdem.binsize)
            dem_bin_offset_y = int((yll - thisdem.mindemy) / thisdem.binsize)

            zdem = np.flip(thisdem.zdem, 0)

            zdem = zdem[
                dem_bin_offset_y : dem_bin_offset_y + int(self.thisarea.height_km),
                dem_bin_offset_x : dem_bin_offset_x + int(self.thisarea.width_km),
            ]

            xdem = thisdem.xdem[dem_bin_offset_x : dem_bin_offset_x + int(self.thisarea.width_km)]
            ydem = np.flip(thisdem.ydem)
            ydem = ydem[dem_bin_offset_y : dem_bin_offset_y + int(self.thisarea.height_km)]

            x_grid, y_grid = np.meshgrid(xdem, ydem)

            # thiscmap = plt.cm.get_cmap("Greys", 48) - depreciated
            thiscmap = colormaps["Greys"].resampled(48)

            ax.pcolormesh(
                x_grid,
                y_grid,
                zdem,
                cmap=thiscmap,
                shading="auto",
                vmin=self.thisarea.min_elevation - 50.0,
                vmax=self.thisarea.max_elevation + 50.0,
                transform=dataprj,
            )

        elif self.name == "grn_s1_mosaic":
            if resolution == "low":
                demfile = (
                    os.environ["CPDATA_DIR"]
                    + "/RESOURCES/backgrounds/greenland/S1_mosaic_v2_1km.tiff"
                )
                binsize = 1000  # 1000m grid resolution
            elif resolution == "medium":
                demfile = (
                    os.environ["CPDATA_DIR"]
                    + "/RESOURCES/backgrounds/greenland/S1_mosaic_v2_500m.tiff"
                )
                binsize = 500  # 500m grid resolution
            elif resolution == "high":
                demfile = (
                    os.environ["CPDATA_DIR"]
                    + "/RESOURCES/backgrounds/greenland/S1_mosaic_v2_200m.tiff"
                )
                binsize = 200  # 200m grid resolution
            elif resolution == "vhigh":
                demfile = (
                    os.environ["CPDATA_DIR"]
                    + "/RESOURCES/backgrounds/greenland/S1_mosaic_v2_100m.tiff"
                )
                binsize = 100  # 100m grid resolution
            else:
                raise ValueError(f"resolution {resolution} not supported")

            print("Loading S1 Sigma0 Mosaic background..")
            print(demfile)
            im = Image.open(demfile)
            print("opened")

            ncols, nrows = im.size
            print("ncols, nrows: ", ncols, nrows)

            zdem = np.array(im.getdata(0)).reshape((nrows, ncols))

            # Set void data to Nan
            # void_data = np.where(zdem == -9999)
            # if np.any(void_data):
            # 	zdem[void_data] = np.nan

            # self.zdem = np.flip(self.zdem,0)
            xdem = np.linspace(-660050.000, 859950.000, ncols, endpoint=True)
            ydem = np.linspace(-3380050.000, -630050.000, nrows, endpoint=True)
            ydem = np.flip(ydem)
            mindemx = xdem.min()
            mindemy = ydem.min()

            # Get tie point

            prj = pyproj.CRS("epsg:3413")
            wgs_prj = pyproj.CRS("epsg:4326")  # WGS84
            tie_point_lonlat_to_xy_transformer = Transformer.from_proj(wgs_prj, prj, always_xy=True)

            if self.thisarea.specify_plot_area_by_lowerleft_corner:
                xll, yll = tie_point_lonlat_to_xy_transformer.transform(
                    self.thisarea.llcorner_lon, self.thisarea.llcorner_lat
                )
            else:
                xc, yc = tie_point_lonlat_to_xy_transformer.transform(
                    self.thisarea.centre_lon, self.thisarea.centre_lat
                )
                xll = xc - (self.thisarea.width_km * 1e3) / 2
                yll = yc - (self.thisarea.height_km * 1e3) / 2

            dem_bin_offset_x = int((xll - mindemx) / binsize)
            dem_bin_offset_y = int((yll - mindemy) / binsize)

            dem_bin_offset_x = max(dem_bin_offset_x, 0)
            dem_bin_offset_y = max(dem_bin_offset_y, 0)

            zdem = np.flip(zdem, 0)

            zdem = zdem[
                dem_bin_offset_y : dem_bin_offset_y + int(self.thisarea.height_km * 1e3 / binsize),
                dem_bin_offset_x : dem_bin_offset_x + int(self.thisarea.width_km * 1e3 / binsize),
            ]

            xdem = xdem[
                dem_bin_offset_x : dem_bin_offset_x + int(self.thisarea.width_km * 1e3 / binsize)
            ]
            ydem = np.flip(ydem)
            ydem = ydem[
                dem_bin_offset_y : dem_bin_offset_y + int(self.thisarea.height_km * 1e3 / binsize)
            ]

            x_grid, y_grid = np.meshgrid(xdem, ydem)

            # thiscmap = plt.cm.get_cmap("Greys", 48) - this method depreciated
            thiscmap = colormaps["Greys"].resampled(48)

            thiscmap.set_bad(color="aliceblue")
            ax.pcolormesh(
                x_grid,
                y_grid,
                zdem,
                cmap=thiscmap,
                shading="auto",
                # vmin=self.thisarea.min_elevation - 50.,
                # vmax=self.thisarea.max_elevation + 50.,
                transform=dataprj,
            )

        elif self.name == "arcticdem_1km":
            demfile = (
                os.environ["CPDATA_DIR"]
                + "/SATS/RA/DEMS/arctic_dem_1km/arcticdem_mosaic_1km_v3.0.tif"
            )
            print("Loading ArcticDEM 1km Greenland DEM..")
            print(demfile)
            im = Image.open(demfile)

            ncols, nrows = im.size
            zdem = np.array(im.getdata()).reshape((nrows, ncols))
            # Set void data to Nan
            void_data = np.where(zdem == -9999)
            if np.any(void_data):
                zdem[void_data] = np.nan
            # self.zdem = np.flip(self.zdem,0)
            xdem = np.linspace(-4000000.000, 3400000.000, ncols, endpoint=True)
            ydem = np.linspace(-3400000.000, 4100000.000, nrows, endpoint=True)
            ydem = np.flip(ydem)
            mindemx = xdem.min()
            mindemy = ydem.min()
            binsize = 1e3  # 1km grid resolution in m

            # Get tie point

            prj = pyproj.CRS("epsg:3413")
            wgs_prj = pyproj.CRS("epsg:4326")  # WGS84
            tie_point_lonlat_to_xy_transformer = Transformer.from_proj(wgs_prj, prj, always_xy=True)

            if self.thisarea.specify_plot_area_by_lowerleft_corner:
                xll, yll = tie_point_lonlat_to_xy_transformer.transform(
                    self.thisarea.llcorner_lon, self.thisarea.llcorner_lat
                )
            else:
                xc, yc = tie_point_lonlat_to_xy_transformer.transform(
                    self.thisarea.centre_lon, self.thisarea.centre_lat
                )
                xll = xc - (self.thisarea.width_km * 1e3) / 2
                yll = yc - (self.thisarea.height_km * 1e3) / 2

            dem_bin_offset_x = int((xll - mindemx) / 1000.0)
            dem_bin_offset_y = int((yll - mindemy) / 1000.0)

            dem_bin_offset_x = max(dem_bin_offset_x, 0)
            dem_bin_offset_y = max(dem_bin_offset_y, 0)

            zdem = np.flip(zdem, 0)

            zdem = zdem[
                dem_bin_offset_y : dem_bin_offset_y + int(self.thisarea.height_km),
                dem_bin_offset_x : dem_bin_offset_x + int(self.thisarea.width_km),
            ]

            xdem = xdem[dem_bin_offset_x : dem_bin_offset_x + int(self.thisarea.width_km)]
            ydem = np.flip(ydem)
            ydem = ydem[dem_bin_offset_y : dem_bin_offset_y + int(self.thisarea.height_km)]

            x_grid, y_grid = np.meshgrid(xdem, ydem)

            # thiscmap = plt.cm.get_cmap("Greys", 48) - depreciated
            thiscmap = colormaps["Greys"].resampled(48)

            thiscmap.set_bad(color="aliceblue")
            ax.pcolormesh(
                x_grid,
                y_grid,
                zdem,
                cmap=thiscmap,
                shading="auto",
                vmin=self.thisarea.min_elevation - 50.0,
                vmax=self.thisarea.max_elevation + 50.0,
                transform=dataprj,
            )

        elif self.name == "rema_dem_1km":
            demfile = (
                os.environ["CPDATA_DIR"]
                + "/SATS/RA/DEMS/rema_1km_dem/REMA_1km_dem_filled_uncompressed.tif"
            )
            print("Loading REMA 1km DEM ..")
            print(demfile)
            im = Image.open(demfile)

            ncols, nrows = im.size
            zdem = np.array(im.getdata()).reshape((nrows, ncols))
            # Set void data to Nan
            void_data = np.where(zdem == -9999)
            if np.any(void_data):
                zdem[void_data] = np.nan
            # self.zdem = np.flip(self.zdem,0)
            xdem = np.linspace(-2700000.0, 2800000.0, ncols, endpoint=True)
            ydem = np.linspace(-2200000.0, 2300000.0, nrows, endpoint=True)
            ydem = np.flip(ydem)
            mindemx = xdem.min()
            mindemy = ydem.min()
            binsize = 1e3  # 1km grid resolution in m

            # Get tie point

            prj = pyproj.CRS("epsg:3031")  # Polar Stereo - South -71S
            wgs_prj = pyproj.CRS("epsg:4326")  # WGS84
            tie_point_lonlat_to_xy_transformer = Transformer.from_proj(wgs_prj, prj, always_xy=True)

            if self.thisarea.specify_plot_area_by_lowerleft_corner:
                xll, yll = tie_point_lonlat_to_xy_transformer.transform(
                    self.thisarea.llcorner_lon, self.thisarea.llcorner_lat
                )
            else:
                xc, yc = tie_point_lonlat_to_xy_transformer.transform(
                    self.thisarea.centre_lon, self.thisarea.centre_lat
                )
                xll = xc - (self.thisarea.width_km * 1e3) / 2
                yll = yc - (self.thisarea.height_km * 1e3) / 2

            dem_bin_offset_x = int((xll - mindemx) / 1000.0)
            dem_bin_offset_y = int((yll - mindemy) / 1000.0)

            dem_bin_offset_x = max(dem_bin_offset_x, 0)
            dem_bin_offset_y = max(dem_bin_offset_y, 0)

            zdem = np.flip(zdem, 0)

            zdem = zdem[
                dem_bin_offset_y : dem_bin_offset_y + int(self.thisarea.height_km),
                dem_bin_offset_x : dem_bin_offset_x + int(self.thisarea.width_km),
            ]

            xdem = xdem[dem_bin_offset_x : dem_bin_offset_x + int(self.thisarea.width_km)]
            ydem = np.flip(ydem)
            ydem = ydem[dem_bin_offset_y : dem_bin_offset_y + int(self.thisarea.height_km)]

            x_grid, y_grid = np.meshgrid(xdem, ydem)

            # thiscmap = plt.cm.get_cmap("Greys", 64) - depreciated
            thiscmap = colormaps["Greys"].resampled(64)

            thiscmap.set_bad(color="aliceblue")
            max_elevation = self.thisarea.max_elevation
            if self.thisarea.max_elevation:
                max_elevation = self.thisarea.max_elevation
            ax.pcolormesh(
                x_grid,
                y_grid,
                zdem,
                cmap=thiscmap,
                shading="auto",
                vmin=self.thisarea.min_elevation - 50.0,
                vmax=max_elevation + 50.0,
                transform=dataprj,
            )

        if include_features:
            if self.thisarea.add_lakes_feature:
                lakes = cartopy.feature.NaturalEarthFeature(
                    "physical",
                    "lakes",
                    scale="10m",
                    edgecolor="lightslategray",
                    facecolor="powderblue",
                )

                ax.add_feature(lakes, zorder=1)

            if self.thisarea.add_rivers_feature:
                rivers = cartopy.feature.NaturalEarthFeature(
                    "physical",
                    "rivers_lake_centerlines",
                    scale="10m",
                    edgecolor="b",
                    facecolor="none",
                )

                ax.add_feature(rivers, linewidth=1.0, zorder=1)

            if self.thisarea.add_country_boundaries:
                # country boundaries
                country_bodr = cartopy.feature.NaturalEarthFeature(
                    category="cultural",
                    name="admin_0_boundary_lines_land",
                    scale="50m",
                    facecolor="none",
                    edgecolor="k",
                )
                ax.add_feature(
                    country_bodr, linestyle="--", linewidth=0.8, edgecolor="k", zorder=1
                )  # USA/Canada

            if self.thisarea.add_province_boundaries:
                # province boundaries
                provinc_bodr = cartopy.feature.NaturalEarthFeature(
                    category="cultural",
                    name="admin_1_states_provinces_lines",
                    scale="50m",
                    facecolor="none",
                    edgecolor="k",
                )
                ax.add_feature(provinc_bodr, linestyle="--", linewidth=0.6, edgecolor="k", zorder=1)
        print("-------------------------------------------------------------")
