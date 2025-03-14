"""3D Area Definition

# Lake Vostok

"""

area_definition = {
    "long_name": "Lake Vostok, E. Antarctica, 400km sq. Area, DEM 1km::1",
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
    "specify_by_centre": True,  # specify plot area by centre lat/lon, width, height (km)
    "centre_lon": 105.0,  # degrees E
    "centre_lat": -77.2,  # degrees N
    #   --------
    "specify_plot_area_by_lowerleft_corner": False,  # specify by lower left corner, w,h
    "llcorner_lat": None,  # lower left corner latitude
    "llcorner_lon": None,  # lower left corner longitude
    #   --------
    "lon_0": None,  # None or projection y-axis longitude (used for mercator)
    #   --------
    "width_km": 400,  # width in km of plot area (x direction)
    "height_km": 400,  # height in km of plot area (y direction)
    # --------------------------------------------
    # Area characteristics
    # --------------------------------------------
    "min_elevation": 3478.0,  # minimum expected elevation in area (m)
    "max_elevation": 3529.0,  # maximum expected elevation in area (m)
    # --------------------------------------------
    # Data filtering using lat/lon extent (used as a quick data pre-filter before masking)
    # --------------------------------------------
    #   Area min/max lat/lon for initial data filtering
    "minlon": 101.0,  # minimum longitude to initially filter records for area (0..360E)
    "maxlon": 107.5,  # maximum longitude to initially filter records for area (0..360E)
    "minlat": -78.7,  # minimum latitude to initially filter records for area
    "maxlat": -76.1,  # maximum latitude to initially filter records for area
    # ------------------------------------------------------
    # Default colormap for primary dataset (can be overridden in dataset dicts)
    # ------------------------------------------------------
    "cmap_name": "RdYlBu_r",  # colormap name to use for this dataset
    "cmap_over_color": "#A85754",  # or None
    "cmap_under_color": "#3E4371",  # or None
    "cmap_extend": "both",  # 'neither','min', 'max','both'
    # --------------------------------------------
    #    mask from cpom.masks.Mask
    # --------------------------------------------
    "apply_area_mask_to_data": True,  # filter data using areas cpom.masks.Mask
    "maskname": "antarctica_bedmachine_v2_grid_mask",  # from  cpom.masks.Mask
    "basin_numbers": [4],  # [n1,n2,..] if mask allows basin numbers
    # for bedmachine v2, 2=grounded ice, 3=floating, 4=vostok
    "show_polygon_mask": False,  # show mask polygon
    "polygon_mask_color": "red",  # color to draw mask polygon
    # 3d plot settings
    "dem_name": "atl14_ant_100m_004_004_zarr",  # DEM used for 3d plots in this area
    "smooth_dem": False,  # Smooth DEM before displaying
    "page_width": 1200,  # browser page width (doesn't affect aspect ratio)
    "dem_stride": 10,  # sampling stride of the DEM
    "zaxis_multiplier": 0.7,  # aspect ratio for Z-axis
    "add_mss_layer": False,  # True if MSS sea decoration required"
    "mss_gridarea": "antarctic_ocean",  # grid area to specify the MSS ocean layer
    "mss_binsize_km": 10,
    "view_angle_elevation": 35,
    "view_angle_azimuth": 330,
    "plot_zoom": 1.15,  # default=10, smaller zooms in, larger zooms out"
    "zaxis_limits": [2400.0, 4000],  # in m
    "light_xdirection": 1e4,
    "light_ydirection": 10,
    "light_zdirection": 0.0,
}
