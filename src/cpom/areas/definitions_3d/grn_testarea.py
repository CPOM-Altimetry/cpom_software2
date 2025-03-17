"""3D Area Definition

# Greenland DEM: ArcticDEM 1km::4

"""

area_definition = {
    "long_name": "Greenland Test Area",
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
    "centre_lon": -41,  # degrees E
    "centre_lat": 70,  # degrees N
    #   --------
    "specify_plot_area_by_lowerleft_corner": False,  # specify by lower left corner, w,h
    "llcorner_lat": None,  # lower left corner latitude
    "llcorner_lon": None,  # lower left corner longitude
    #   --------
    "lon_0": None,  # None or projection y-axis longitude (used for mercator)
    #   --------
    "width_km": 500,  # width in km of plot area (x direction)
    "height_km": 500,  # height in km of plot area (y direction)
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
    "dem_name": "arcticdem_100m_greenland_v4.1_zarr",  # DEM used for 3d plots in this area
    "smooth_dem": False,
    "page_width": 1400,  # browser page width (doesn't affect aspect ratio)
    "dem_stride": 4,
    "zaxis_multiplier": 0.2,  # aspect ratio for Z-axis
    "add_mss_layer": False,  # True if MSS sea decoration required"
    "mss_gridarea": "greenland",  # grid area to specify the MSS ocean layer
    "mss_binsize_km": 10,
    "view_angle_elevation": 90,
    "view_angle_azimuth": 90,
    "plot_zoom": 1.2,  # default=10, smaller zooms in, larger zooms out"
    "zaxis_limits": [-200.0, 4000],  # in m
    "light_xdirection": 1e4,
    "light_ydirection": 0,
    "light_zdirection": 1.0,
    "lat_lines": range(68, 73, 1),  # min_lat, max_lat, step
    "lon_lines": (316, 320, 324),  # min_lonE, max_lonE, step
    "latlon_line_colour": "darkgrey",  # color or lat/lon lines
    "latlon_lines_increment": 0.01,  # spacing between latlon lines points
    "latlon_lines_size": 2.0,  # point size of lat/lon lines
    "latlon_lines_opacity": 0.5,  # opacity (0..1) of lat/lon lines
    "latlon_lines_elevation": None,  # m, elevation to plot lat/lon lines at
    "raise_latlon_lines_above_dem": 20,  # m, raise lat/lon lines above DEM
    # latitude, longitude annotations, placed at each lat, lon,pair.
    # [lat,lon,elevation, text_size,xshift,yshift]
    "lat_annotations": (
        (68, 320, 2600, 10, 0, 0),
        (69, 320, 2600, 10, 0, 0),
        (70, 320, 2600, 10, 0, 0),
        (71, 320, 2600, 10, 0, 0),
    ),
    "lon_annotations": (
        (68.5, 316, 350, 10, 0, 0),
        (68.5, 320, 350, 10, 0, 0),
    ),
    "place_annotations": (),
    # place annotations, [[latitude, longitude,elevation,string,color, bg_color,size],..]"
}
