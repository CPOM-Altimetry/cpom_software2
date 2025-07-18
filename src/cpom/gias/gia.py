"""cpom.gia.gia.py

Gia class to load glacial isostatic adjustment (GIA) data and 
interpolation latitude and longitude.
"""

import importlib
import os

import numpy as np
from pyproj import CRS, Transformer  # coord transforms, definitions
from scipy.interpolate import griddata, interpn  # interpolation functions

gia_definitions = {
    "ij05": {
        "file": "/MODELS/GIA/ij05_model.npz",
        "crs_wgs": 4326,  # WGS 84 geographic
        "crs_bng": 3031,  # PS 0E, 71S
    },
    "ice5g": {
        "file": "/MODELS/GIA/ice5g_model.npz",
        "crs_wgs": 4326,  # WGS 84 geographic
        "crs_bng": 3413,  # PS 45W 70N
    },
}


class GIA:
    """class to load and interpolate GIA data."""

    def __init__(self, gia_name: str):
        self.gia_name = gia_name
        try:
            self.load_gia()
        except ImportError as exc:
            raise ImportError(f"{gia_name} not in supported area list") from exc

    def load_gia(self):
        """Load gia settings for current gia name"""

        this_gia = np.load(
            os.environ["CPDATA_DIR"] + gia_definitions[self.gia_name]["file"], allow_pickle=True
        )

        self.lat = this_gia["gia_lat"]
        self.lon = this_gia["gia_lon"]
        self.gia = this_gia["gia"]
        self.crs_wgs = CRS(gia_definitions[self.gia_name]["crs_wgs"])
        self.crs_bng = CRS(gia_definitions[self.gia_name]["crs_bng"])

        # # Setup the Transforms
        self.xy_to_lonlat_transformer = Transformer.from_proj(
            self.crs_bng, self.crs_wgs, always_xy=True
        )
        self.lonlat_to_xy_transformer = Transformer.from_proj(
            self.crs_wgs, self.crs_bng, always_xy=True
        )

        self._prepare_gia_grid()

    def _prepare_gia_grid(self, ncols=1000, nrows=1000, method="cubic"):
        """Construct 2D mesh grid from 1D longitude and latitude arrays.
            Create a regular grid. (6.5km resolution by default)
            Interpolate GIA data onto regular grid using scipy griddata.

        Args:
            ncols (int, optional): _description_. Defaults to 1000.
            nrows (int, optional): _description_. Defaults to 1000.
            method (str, optional): _description_. Defaults to 'cubic'.
        """
        self.lonmesh, self.latmesh = np.meshgrid(self.lon, self.lat)
        self.input_x, self.input_y = self.lonlat_to_xy_transformer.transform(
            self.lonmesh, self.latmesh
        )
        self.minx, self.maxx = np.min(self.input_x), np.max(self.input_x)
        self.miny, self.maxy = np.min(self.input_y), np.max(self.input_y)

        self.x = np.linspace(self.minx, self.maxx, ncols, endpoint=True)
        self.y = np.linspace(self.miny, self.maxy, nrows, endpoint=True)
        self.xmesh, self.ymesh = np.meshgrid(self.x, self.y)

        self.gia_grid = griddata(
            (self.input_x.flatten(), self.input_y.flatten()),
            self.gia.flatten(),
            (self.xmesh, self.ymesh),
            method=method,
        )

    def interp_gia(self, crs_bng, crs_wgs, x, y, method="linear"):
        """Interpolate GIA data at given coordinates.

        Args:
            crs_bng (_type_): _description_
            crs_wgs (_type_): _description_
            x (_type_): _description_
            y (_type_): _description_
            method (str, optional): _description_. Defaults to 'linear'.

        Returns:
            _type_: _description_
        """
        # Transform input coordinates to GIA map projection if needed
        if crs_bng != self.crs_bng or crs_wgs != self.crs_wgs:
            this_xy_to_lonlat_transformer = Transformer.from_proj(
                crs_bng, self.crs_wgs, always_xy=True
            )
            this_lon, this_lat = this_xy_to_lonlat_transformer.transform(x, y)
            x, y = self.lonlat_to_xy_transformer.transform(this_lon, this_lat)

        x = np.array(x)
        y = np.array(y)
        badx = (x < self.minx) | (x > self.maxx)
        bady = (y < self.miny) | (y > self.maxy)
        x[badx], y[bady] = 0.0, 0.0  # Set out of bounds to 0.0
        gia_data = interpn(
            (self.y, self.x),
            self.gia_grid,
            (y, x),
            method=method,
            bounds_error=False,
            fill_value=np.nan,
        )
        # gia_data[badx | bady] = np.nan
        return gia_data

    def interp_gia_from_lat_lon(self, lat, lon, method="linear"):
        thisx, thisy = self.lonlat_to_xy_transformer.transform(lon, lat)
        return self.interp_gia(self.crs_bng, self.crs_wgs, thisx, thisy, method=method)
