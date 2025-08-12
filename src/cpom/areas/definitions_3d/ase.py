"""3D Area Definition

# Amundsen Sea Embayment, West Antarctica DEM: 1km REMA Gapless v1::1

"""

area_definition = {
    "long_name": "Amundsen Sea Embayment DEM: 1km REMA Gapless v1::1",
    # --------------------------------------------
    # Area definition
    # --------------------------------------------
    "hemisphere": "south",  # area is in  'south' or 'north' or 'both'
    "epsg_number": 3031,  # EPSG number for area's projection
    #   --------
    "round": False,  # False=rectangular, True = round map area
    "specify_by_bounding_lat": False,  # for round hemisphere views
    "bounding_lat": None,  # limiting latitude for round areas or None
    #   --------
    "specify_by_centre": True,  # specify plot area by centre lat/lon, width, height (km)
    "centre_lon": -103.0,  # degrees E
    "centre_lat": -75.0,  # degrees N
    #   --------
    "specify_plot_area_by_lowerleft_corner": False,  # specify by lower left corner, w,h
    "llcorner_lat": None,  # lower left corner latitude
    "llcorner_lon": None,  # lower left corner longitude
    #   --------
    "lon_0": None,  # None or projection y-axis longitude (used for mercator)
    #   --------
    "width_km": 900,  # width in km of plot area (x direction)
    "height_km": 900,  # height in km of plot area (y direction)
    # --------------------------------------------
    # Area characteristics
    # --------------------------------------------
    "min_elevation": -500,  # minimum expected elevation in area (m)
    "max_elevation": 2900,  # maximum expected elevation in area (m)
    # --------------------------------------------
    # Data filtering using lat/lon extent (used as a quick data pre-filter before masking)
    # --------------------------------------------
    #   Area min/max lat/lon for initial data filtering
    "minlon": 0,  # minimum longitude to initially filter records for area (0..360E)
    "maxlon": 360,  # maximum longitude to initially filter records for area (0..360E)
    "minlat": -80,  # minimum latitude to initially filter records for area
    "maxlat": -71,  # maximum latitude to initially filter records for area
    # --------------------------------------------
    #    mask from clev2er.utils.masks.Mask
    # --------------------------------------------
    "apply_area_mask_to_data": True,  # filter data using areas clev2er.utils.masks.Mask
    "maskname": "ase_xylimits_mask",  # from  clev2er.utils.masks.Mask
    "masktype": "xylimits",
    "basin_numbers": [],  # [n1,n2,..] if mask allows basin numbers
    # for bedmachine v2, 2=grounded ice, 3=floating, 4=vostok
    "show_polygon_mask": False,  # show mask polygon
    "polygon_mask_color": "red",  # color to draw mask polygon
    # ------------------------------------------------------
    # Default colormap for primary dataset (can be overridden in dataset dicts)
    # ------------------------------------------------------
    "cmap_name": "RdYlBu_r",  # colormap name to use for this dataset
    "cmap_over_color": "#A85754",  # or None
    "cmap_under_color": "#3E4371",  # or None
    "cmap_extend": "both",  # 'neither','min', 'max','both'
    # 3d plot settings
    # ---------------------------------------------
    "dem_name": "rema_gapless_1km_zarr",  # DEM used for 3d plots in this area
    "smooth_dem": False,
    "page_width": 1300,  # browser page width (doesn't affect aspect ratio)
    "dem_stride": 1,
    "zaxis_multiplier": 0.07,  # aspect ratio for Z-axis
    "add_mss_layer": True,  # True if MSS sea decoration required"
    "mss_gridarea": "antarctic_ocean",  # grid area to specify the MSS ocean layer
    "mss_binsize_km": 10,
    "view_angle_elevation": 45,
    "view_angle_azimuth": 215,
    "plot_zoom": 0.8,  # default=10, smaller zooms in, larger zooms out"
    "zaxis_limits": [-100.0, 2500],  # in m
    "light_xdirection": 1e4,
    "light_ydirection": 0,  # 0
    "light_zdirection": 1.0,  # 1.0
    "place_annotations": (
        [-74.1, 253, 100.0, "Pine Island Bay", "white", None, 11, 0.9],
        [-73.2, 253, 100.0, "AMUNDSEN SEA", "white", None, 13, 0.9],
        [-75.9, 253.25, 100.0, "Thwaites Glacier", "Black", "White", 10, 0.9],
        [-74.4, -112.667, 150.0, "Dotson Ice Shelf", "Black", "White", 11, 0.9],
        [-75.5, 265, 150.0, "Pine Island Glacier", "Black", "White", 10, 0.9],
        [-78.6, 255, 600.0, "WEST ANTARCTIC ICE SHEET", "Black", "White", 9, 0.9],
        [-73.08333, -105.1, 150, "Burke Island", "White", None, 10, 0.9],
    ),  # place annotations, [[latitude, longitude,elevation,string,color,
    # bg_color,size,opacity],..]
    # latitude, longitude annotations, placed at each lat, lon,pair.
    # [lat,lon,elevation, text_size,xshift,yshift]
    "lat_annotations": (
        [-72, 254, 150, 8, 2, 0],
        [-74, 254, 150, 8, 2, 0],
    ),  # [lat,lon,elevation, text_size,xshift,yshift]
    "lon_annotations": (
        [-73, 250, 150, 8, 2, 0],
        [-73, 245, 150, 8, 2, 0],
        [-71.8, 265, 150, 8, 2, 0],
        [-71.8, 255, 150, 8, 2, 0],
        [-71.5, 260, 150, 8, 2, 0],
        [-71.8, 270, 150, 8, 2, 0],
    ),  # [lat,lon,elevation, text_size,xshift,yshift]
    "lat_lines": (-72, -74),  # min_lat, max_lat, step
    "lon_lines": (245, 250, 255, 260, 265, 270),  # min_lonE, max_lonE, step
    "latlon_line_colour": "white",  # color or lat/lon lines
    "latlon_lines_increment": 0.01,  # spacing between latlon lines points
    "latlon_lines_size": 0.8,  # point size of lat/lon lines
    "latlon_lines_opacity": 0.5,  # opacity (0..1) of lat/lon lines
    "latlon_lines_elevation": 200,  # m, elevation to plot lat/lon lines at
}
