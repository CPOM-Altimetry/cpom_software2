"""3D Area Definition

# Greenland DEM: ArcticDEM 1km::4

"""

area_definition = {
    "long_name": "Greenland DEM: ArcticDEM 1km::4",
    # --------------------------------------------
    # Area definition
    # --------------------------------------------
    "hemisphere": "north",  # area is in  'south' or 'north' or 'both'
    "epsg_number": 3413,  # EPSG number for area's projection
    #   --------
    "round": False,  # False=rectangular, True = round map area
    "specify_by_bounding_lat": False,  # for round hemisphere views
    "bounding_lat": None,  # limiting latitude for round areas or None
    #   --------
    "specify_by_centre": True,  # specify plot area by centre lat/lon, width, height (km)
    "centre_lon": -41.75,  # degrees E
    "centre_lat": 71.5,  # degrees N
    #   --------
    "specify_plot_area_by_lowerleft_corner": False,  # specify by lower left corner, w,h
    "llcorner_lat": None,  # lower left corner latitude
    "llcorner_lon": None,  # lower left corner longitude
    #   --------
    "lon_0": None,  # None or projection y-axis longitude (used for mercator)
    #   --------
    "width_km": 1540,  # width in km of plot area (x direction)
    "height_km": 2740,  # height in km of plot area (y direction)
    # --------------------------------------------
    # Area characteristics
    # --------------------------------------------
    "min_elevation": -50,  # minimum expected elevation in area (m)
    "max_elevation": 4200,  # maximum expected elevation in area (m)
    # --------------------------------------------
    # Data filtering using lat/lon extent (used as a quick data pre-filter before masking)
    # --------------------------------------------
    #   Area min/max lat/lon for initial data filtering
    "minlon": 285.0,  # minimum longitude to initially filter records for area (0..360E)
    "maxlon": 350.0,  # maximum longitude to initially filter records for area (0..360E)
    "minlat": 59.0,  # minimum latitude to initially filter records for area
    "maxlat": 85.0,  # maximum latitude to initially filter records for area
    # ------------------------------------------------------
    # Default colormap for primary dataset (can be overridden in dataset dicts)
    # ------------------------------------------------------
    "cmap_name": "RdYlBu_r",  # colormap name to use for this dataset
    "cmap_over_color": "#A85754",  # or None
    "cmap_under_color": "#3E4371",  # or None
    "cmap_extend": "both",  # 'neither','min', 'max','both'
    # --------------------------------------------
    #    mask from clev2er.utils.masks.Mask
    # --------------------------------------------
    "apply_area_mask_to_data": False,  # filter data using areas mask
    "maskname": None,  # from  clev2er.utils.masks.Mask
    "masktype": None,  # mask is a polar stereo grid of Nkm resolution
    "basin_numbers": None,  # [n1,n2,..] if mask allows basin numbers
    # for bedmachine v3, 2=grounded ice, 3=floating
    "show_polygon_mask": False,  # show mask polygon
    "polygon_mask_color": "red",  # color to draw mask polygon
    # 3d plot settings
    # ---------------------------------------------
    "dem_name": "arcticdem_1km_greenland_v4.1_zarr",  # DEM used for 3d plots in this area
    "smooth_dem": True,
    "page_width": 1400,  # browser page width (doesn't affect aspect ratio)
    "dem_stride": 1,
    "zaxis_multiplier": 0.04,  # aspect ratio for Z-axis
    "add_mss_layer": True,  # True if MSS sea decoration required"
    "mss_gridarea": "greenland",  # grid area to specify the MSS ocean layer
    "mss_binsize_km": 10,
    "view_angle_elevation": 90,
    "view_angle_azimuth": 90,
    "plot_zoom": 1.2,  # default=10, smaller zooms in, larger zooms out"
    "zaxis_limits": [-200.0, 4000],  # in m
    "light_xdirection": 1e4,
    "light_ydirection": 0,
    "light_zdirection": 1.0,
}
