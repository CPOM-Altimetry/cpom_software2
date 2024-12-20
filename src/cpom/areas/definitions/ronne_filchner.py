"""Area definition"""

# pylint: disable=R0801 # warning for similar lines

area_definition = {
    "long_name": "Ronne Filchner",
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
    "centre_lon": 300.0,  # degrees E
    "centre_lat": -79.0,  # degrees N
    #   --------
    "specify_plot_area_by_lowerleft_corner": False,  # specify by lower left corner, w,h
    "llcorner_lat": None,  # lower left corner latitude
    "llcorner_lon": None,  # lower left corner longitude
    #   --------
    "lon_0": None,  # None or projection y-axis longitude (used for mercator)
    #   --------
    "width_km": 1295,  # width in km of plot area (x direction)
    "height_km": 1295,  # height in km of plot area (y direction)
    # --------------------------------------------
    # Area characteristics
    # --------------------------------------------
    "min_elevation": -50,  # minimum expected elevation in area (m)
    "max_elevation": 2000,  # maximum expected elevation in area (m)
    # --------------------------------------------
    # Data filtering using lat/lon extent (used as a quick data pre-filter before masking)
    # --------------------------------------------
    #   Area min/max lat/lon for initial data filtering
    "minlon": 265.0,  # minimum longitude to initially filter records for area (0..360E)
    "maxlon": 350.0,  # maximum longitude to initially filter records for area (0..360E)
    "minlat": -87.0,  # minimum latitude to initially filter records for area
    "maxlat": -70.0,  # maximum latitude to initially filter records for area
    # --------------------------------------------
    #    mask from clev2er.utils.masks.Mask
    # --------------------------------------------
    "apply_area_mask_to_data": False,  # filter data using areas clev2er.utils.masks.Mask
    "maskname": "",  # from  clev2er.utils.masks.Mask
    "basin_numbers": [],  # [n1,n2,..] if mask allows basin numbers
    # for bedmachine v2, 2=grounded ice, 3=floating, 4=vostok
    "show_polygon_mask": False,  # show mask polygon
    "polygon_mask_color": "red",  # color to draw mask polygon
    # --------------------------------------------
    # Plot parameters for this area
    # --------------------------------------------
    "axes": [  # define plot axis position
        -0.02,  # left
        0.11,  # bottom
        0.71,  # width (axes fraction)
        0.71,  # height (axes fraction)
    ],
    "simple_axes": [  # define plot axis position in the map_only plot
        0.09,  # left
        0.12,  # bottom
        0.79,  # width (axes fraction)
        0.79,  # height (axes fraction)
    ],
    "draw_axis_frame": True,
    "background_color": None,  # background color of map
    "background_image": "basic_land",  # background image. see clev2er.utils.backgrounds
    "background_image_alpha": 1.0,  # 0..1.0, default is 1.0, image transparency
    "background_image_resolution": "low",  # None, 'low','medium', 'high'
    "hillshade_params": None,  # hill shade parameter dict or None
    "show_polygon_overlay_in_main_map": True,  # Overlay the area polygon outline in the main map
    "grid_polygon_overlay_mask": None,
    "apply_hillshade_to_vals": False,  # Apply a hillshade to plotted vals (True or False)
    "draw_coastlines": True,  # Draw coastlines
    "coastline_color": "grey",  # Colour to draw coastlines
    "use_antarctica_medium_coastline": True,  # True,False: Antarctic coastline including iceshelves
    "use_cartopy_coastline": "no",  # 'no', 'low','medium', 'high' resolution
    "show_gridlines": True,  # True|False, display lat/lon grid lines
    "area_long_name_position": (0.20, 0.86),  # for default annotation position of area long name
    "area_long_name_fontsize": 16,  # font size of area.long_name
    "area_long_name_position_simple": (
        0.4,
        0.96,
    ),  # for default annot. pos of area name (map_only)
    "mask_long_name_position": (0.20, 0.84),  # for default annotation position of area long name
    "mask_long_name_position_simple": (
        0.4,
        0.94,
    ),  # for default annotation position of area long name
    "mask_long_name_fontsize": 9,  # font size of area.long_name
    # ------------------------------------------------------
    # Default Annotation
    # ------------------------------------------------------
    "varname_annotation_position_xy": (
        0.20,
        0.92,
    ),  # normalized position of default varname annotation in plot
    "varname_annotation_position_xy_simple": (
        0.04,
        0.94,
    ),  # normalized position of default varname annotation in map_only plot
    "stats_position_x_offset": 0,  # x offset to stats position
    "stats_position_y_offset": 0.77,  # y offset to stats position
    "stats_position_x_offset_simple": -0.2,  # x offset to stats when plotting map_only
    "stats_position_y_offset_simple": -0.08,  # y offset to stats when plotting map_only
    # ------------------------------------------------------
    # Flag plot settings
    # ------------------------------------------------------
    "include_flag_legend": False,  # include or not the flag legend
    "flag_legend_xylocation": [
        None,
        None,
    ],  # x, y of flag legend lower right bbox
    "flag_legend_location": "upper right",  # position of flag legend bbox
    "include_flag_percents": True,  # include or not the flag percentage sub-plot
    "flag_perc_axis": [
        0.74,
        0.25,
        0.10,
    ],  # [left,bottom, width] of axis. Note height is auto set
    # ------------------------------------------------------
    # Default colormap for primary dataset (can be overridden in dataset dicts)
    # ------------------------------------------------------
    "cmap_name": "RdYlBu_r",  # colormap name to use for this dataset
    "cmap_over_color": "#A85754",  # or None
    "cmap_under_color": "#3E4371",  # or None
    "cmap_extend": "both",  # 'neither','min', 'max','both'
    # ------------------------------------------------------
    # Colour bar
    # ------------------------------------------------------
    "draw_colorbar": True,
    "colorbar_orientation": "horizontal",  # vertical, horizontal
    "vertical_colorbar_axes": [
        0.04,
        0.05,
        0.02,
        0.55,
    ],  # [ left, bottom, width, height (fractions of axes)]
    "vertical_colorbar_axes_simple": [
        0.04,
        0.05,
        0.02,
        0.55,
    ],  # [ left, bottom, width, height (fractions of axes)]
    "horizontal_colorbar_axes": [
        0.08,
        0.05,
        0.55,
        0.02,
    ],  # [ left, bottom, width, height (fractions of axes)]
    "horizontal_colorbar_axes_simple": [
        0.28,
        0.05,
        0.55,
        0.02,
    ],  # [ left, bottom, width, height (fractions of axes)]
    # ------------------------------------------------------
    #       Lat/lon grid lines to show in main area
    #           - use empty lists to not include
    # ------------------------------------------------------
    "longitude_gridlines": [270, 280, 290, 300, 310, 320, 330],  # deg E
    "latitude_gridlines": [-71, -73, -75, -77, -79, -81, -83, -85, -87],  # deg N
    "gridline_color": "lightgrey",  # color to use for lat/lon grid lines
    "gridlabel_color": "darkgrey",  # color of grid labels
    "gridlabel_size": 8,  # size of grid labels
    "draw_gridlabels": True,  # whether to draw the grid labels
    "inner_gridlabel_color": "white",  # color of grid labels
    "inner_gridlabel_size": 8,  # size of grid labels
    "latitude_of_radial_labels": -58.3,  # latitude for radial grid line labels for circular plots
    "labels_at_top": False,  # allow lat or lon labels at top of plot
    "labels_at_bottom": True,  # allow lat or lon labels at bottom of plot
    "labels_at_left": True,  # allow lat or lon labels at left of plot
    "labels_at_right": True,  # allow lat or lon labels at right of plot
    # ------------------------------------------------------
    #       Show a scale bar in km
    # ------------------------------------------------------
    "show_scalebar": True,
    "mapscale": [
        -85.0,  # longitude to position scale bar
        -84.0,  # latitide to position scale bar
        -65.0,  # longitude of true scale (ie centre of area)
        -80.0,  # latitude of true scale (ie centre of area)
        200,  # width of scale bar (km)
        "black",  # color of scale bar
        20,  # size of scale bar
    ],
    # --------------------------------------------------------
    # Histograms
    # --------------------------------------------------------
    "show_histograms": True,  # Whether to show the histogram plots
    "histogram_plotrange_axes": [
        0.735,  # left
        0.3,  # bottom
        0.08,  # width (axes fraction)
        0.35,  # height (axes fraction)
    ],  # axis location of plot range histogram for Polarplot.plot_points()
    "histogram_fullrange_axes": [
        0.89,  # left
        0.3,  # bottom
        0.08,  # width (axes fraction)
        0.35,  # height (axes fraction)
    ],  # axis location of plot range histogram for Polarplot.plot_points()
    # --------------------------------------------------------
    # Latitude vs Values plot
    # --------------------------------------------------------
    "show_latitude_scatter": True,  # Whether to show the latitude scatter plot
    "latvals_axes": [
        0.77,  # left
        0.05,  # bottom
        0.17,  # width (axes fraction)
        0.2,  # height (axes fraction)
    ],  # axis location of latitude vs values scatter plot
    # --------------------------------------------------------
    # Bad Data Mini-map plot - to show locations of data flagged as bad
    # due to Nan,None, FV, Out or Range data
    # --------------------------------------------------------
    "show_bad_data_map": True,
    "bad_data_minimap_axes": [  # define minimap axis position
        0.65,  # left
        0.69,  # bottom
        0.26,  # width (axes fraction)
        0.26,  # height (axes fraction)
    ],
    "bad_data_minimap_bounding_lat": None,  # None or bounding latitude if used for mini-map
    # uses 40N for northern hemisphere or 50N for southern.
    # Override with this parameter
    "bad_data_minimap_circle": None,  # None or [lat,lon,circle_radius_m,color_str]
    "bad_data_minimap_draw_gridlines": True,
    "bad_data_minimap_gridlines_color": "lightgrey",  # color of gridlines drawn in bad data minimap
    "bad_data_latitude_lines": [-85, -81, -77, -73],  # latitude lines to draw in bad data minimap
    "bad_data_longitude_lines": [
        -30,
        -40,
        -50,
        -60,
        -70,
        -80,
        -90,
        -100,
    ],  # longitude lines to draw in bad data minimap
    "bad_data_minimap_val_scalefactor": 1.0,  # scale factor for plotting bad values on minimap
    "bad_data_minimap_legend_pos": (1.5, 1.0),  # position of minimap legend (upper right)
    # relative to bad_data_minimap axis
    "bad_data_minimap_coastline_resolution": "medium",  # low, medium, high resolution coastline
    # ------------------------------------------------------------------
    # Mini-map (with box showing actual area) - purpose to show where a
    # smaller area is on a larger map.
    # -------------------------------------------------------------------
    "show_minimap": True,  # show the overview minmap
    "minimap_axes": [  # define minimap axis position
        0.5,  # left
        0.84,  # bottom
        0.14,  # width (axes fraction)
        0.14,  # height (axes fraction)
    ],
    "minimap_bounding_lat": None,  # None or bounding latitude if used for mini-map
    # uses 40N for northern hemisphere or 50N for southern.
    # Override with this parameter
    "minimap_circle": None,  # None or [lat,lon,circle_radius_m,color_str]
    "minimap_draw_gridlines": False,
    "minimap_val_scalefactor": 1.0,  # scale factor for plotting bad values on minimap
    "minimap_legend_pos": (1.38, 1.1),  # position of minimap legend (upper right) in minimap axis
}
