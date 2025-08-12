"""3D Area Definition

# Antarctica 100m DEM : ATL14 A3 (SW) Sector

"""

area_definition = {
    "long_name": "Antarctica 100m DEM : ATL14 A3 (SW) Sector",
    # --------------------------------------------
    # Area definition
    # --------------------------------------------
    "hemisphere": "south",  # area is in  'south' or 'north' or 'both'
    "epsg_number": 3031,  # EPSG number for area's projection
    #   --------
    "round": False,  # False=rectangular, True = round map area
    "specify_by_bounding_lat": False,  # for round hemisphere views
    "bounding_lat": -63.15,  # limiting latitude for round areas or None
    #   --------
    "specify_by_centre": False,  # specify plot area by centre lat/lon, width, height (km)
    "centre_lon": 320.0,  # degrees E
    "centre_lat": -70.0,  # degrees N
    #   --------
    "specify_plot_area_by_lowerleft_corner": True,  # specify by lower left corner, w,h
    "llcorner_lat": -67.852,  # lower left corner latitude
    "llcorner_lon": 235.2,  # lower left corner longitude
    #   --------
    "lon_0": None,  # None or projection y-axis longitude (used for mercator)
    #   --------
    "width_km": 2000,  # width in km of plot area (x direction)
    "height_km": 1390,  # height in km of plot area (y direction)
    # --------------------------------------------
    # Area characteristics
    # --------------------------------------------
    "min_elevation": -50,  # minimum expected elevation in area (m)
    "max_elevation": 4200,  # maximum expected elevation in area (m)
    # --------------------------------------------
    # Data filtering using lat/lon extent (used as a quick data pre-filter before masking)
    # --------------------------------------------
    #   Area min/max lat/lon for initial data filtering
    "minlon": 0.0,  # minimum longitude to initially filter records for area (0..360E)
    "maxlon": 360.0,  # maximum longitude to initially filter records for area (0..360E)
    "minlat": -90.0,  # minimum latitude to initially filter records for area
    "maxlat": -62.0,  # maximum latitude to initially filter records for area
    # ------------------------------------------------------
    # Default colormap for primary dataset (can be overridden in dataset dicts)
    # ------------------------------------------------------
    "cmap_name": "RdYlBu_r",  # colormap name to use for this dataset
    "cmap_over_color": "#A85754",  # or None
    "cmap_under_color": "#3E4371",  # or None
    "cmap_extend": "both",  # 'neither','min', 'max','both'
    # 3d plot settings
    # ---------------------------------------------
    "dem_name": "alt14_ant_a3_100m_004_004_zarr",  # DEM used for 3d plots in this area
    "smooth_dem": False,
    "page_width": 1400,  # browser page width (doesn't affect aspect ratio)
    "dem_stride": 50,
    "zaxis_multiplier": 0.05,  # aspect ratio for Z-axis
    "add_mss_layer": True,  # True if MSS sea decoration required"
    "mss_gridarea": "antarctic_ocean",  # grid area to specify the MSS ocean layer
    "mss_binsize_km": 10,
    "view_angle_elevation": 90,
    "view_angle_azimuth": 90,
    "plot_zoom": 1.2,  # default=10, smaller zooms in, larger zooms out"
    "zaxis_limits": [-300.0, 4500],  # in m
    "light_xdirection": 1e4,
    "light_ydirection": 0,
    "light_zdirection": 1.0,
    "place_annotations": (
        (-80, 190, 3234.0, "Ross IS", "Black", "White", 10, 0.9),
        (-76, 302, 200.0, "Ronne IS", "Black", "White", 10, 0.9),
        (-70.5, 256, 100.0, "Amundsen Sea", "white", None, 9, 0.9),
        (-71.5, 315, 100.0, "Weddell Sea", "white", None, 10, 0.9),
        (-73.5, 190, 100.0, "Ross Sea", "white", None, 10, 0.9),
        (-67, 278, 100.0, "Bellingshausen Sea", "white", None, 10, 0.9),
        (-69, 71, 100.0, "Amery Ice Shelf", "Black", "White", 10, 0.9),
        (-69, 116, 100.0, "Totten Glacier", "Black", "White", 10, 0.9),
        (-75.9, 253.25, 100.0, "Thwaites Glacier", "Black", "White", 10, 0.9),
        (-77.3, 106.0, 3500, "Vostok", "Black", "White", 10, 0.9),
        (-90.0, 0.0, 2840, "Pole", "Black", "White", 10, 0.9),
    ),  # place annotations, [[latitude, longitude,elevation,string,color, bg_color,size],..]"
    # latitude, longitude annotations, placed at each lat, lon,pair.
    # [lat,lon,elevation, text_size,xshift,yshift]
    "lat_annotations": ((-70, 220, 150, 10, 2, 0), (-60, 220, 150, 10, 2, 0)),
    "lon_annotations": (
        [-66, 120, 150, 10, 2, 0],
        [-66, 160, 150, 10, 2, 0],
        [-66, 200, 150, 10, 2, 0],
        [-66, 240, 150, 10, 2, 0],
        [-66, 280, 150, 10, 2, 0],
        [-66, 320, 150, 10, 2, 0],
        [-66, 0, 150, 10, 2, 0],
        [-66, 40, 150, 10, 2, 0],
        [-66, 80, 150, 10, 2, 0],
    ),
    "lat_lines": range(-60, -80, -10),  # min_lat, max_lat, step
    "lon_lines": range(0, 360, 20),  # min_lonE, max_lonE, step
    "latlon_line_colour": "white",  # color or lat/lon lines
    "latlon_lines_increment": 0.01,  # spacing between latlon lines points
    "latlon_lines_size": 0.3,  # point size of lat/lon lines
    "latlon_lines_opacity": 0.5,  # opacity (0..1) of lat/lon lines
    "latlon_lines_elevation": 200,  # m, elevation to plot lat/lon lines at
}
