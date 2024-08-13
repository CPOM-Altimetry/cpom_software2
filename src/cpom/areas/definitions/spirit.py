"""Area definition"""

# pylint: disable=R0801 # warning for similar lines

area_definition = {
    "long_name": "SPIRIT Zone, E. Antarctica",
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
    "centre_lon": 141.0,  # degrees E
    "centre_lat": -68,  # degrees N
    #   --------
    "specify_plot_area_by_lowerleft_corner": False,  # specify by lower left corner, w,h
    "llcorner_lat": -67.5,  # lower left corner latitude
    "llcorner_lon": 150.0,  # lower left corner longitude
    #   --------
    "lon_0": None,  # None or projection y-axis longitude (used for mercator)
    #   --------
    "width_km": 600,  # width in km of plot area (x direction)
    "height_km": 600,  # height in km of plot area (y direction)
    # --------------------------------------------
    # Area characteristics
    # --------------------------------------------
    "min_elevation": 0.0,  # minimum expected elevation in area (m)
    "max_elevation": 2400.0,  # maximum expected elevation in area (m)
    # --------------------------------------------
    # Data filtering using lat/lon extent (used as a quick data pre-filter before masking)
    # --------------------------------------------
    #   Area min/max lat/lon for initial data filtering
    "minlon": 130.0,  # minimum longitude to initially filter records for area (0..360E)
    "maxlon": 152.0,  # maximum longitude to initially filter records for area (0..360E)
    "minlat": -72.0,  # minimum latitude to initially filter records for area
    "maxlat": -64.0,  # maximum latitude to initially filter records for area
    # --------------------------------------------
    #    mask from cpom.masks.Mask
    # --------------------------------------------
    "apply_area_mask_to_data": False,  # filter data using areas cpom.masks.Mask
    "maskname": "",  # from  cpom.masks.Mask
    "basin_numbers": [],  # [n1,n2,..] if mask allows basin numbers
    # for bedmachine v2, 2=grounded ice, 3=floating, 4=vostok
    "show_polygon_mask": False,  # show mask polygon
    "polygon_mask_color": "red",  # color to draw mask polygon
    # --------------------------------------------
    # Plot parameters for this area
    # --------------------------------------------
    "axes": [  # define plot axis position
        0.01,  # left
        0.16,  # bottom
        0.66,  # width (axes fraction)
        0.66,  # height (axes fraction)
    ],
    "simple_axes": [  # define plot axis position in the map_only plot
        0.13,  # left
        0.16,  # bottom
        0.69,  # width (axes fraction)
        0.69,  # height (axes fraction)
    ],
    "draw_axis_frame": True,
    "background_color": "white",  # background color of map
    "background_image": [
        "ibcso_bathymetry",
        "hillshade",
    ],
    "background_image_alpha": [0.14, 0.28],
    "background_image_resolution": "high",  # None, 'low','medium', 'high'
    "hillshade_params": None,  # hill shade parameter dict or None
    "show_polygon_overlay_in_main_map": True,  # Overlay the area polygon outline in the main map
    "grid_polygon_overlay_mask": None,
    "apply_hillshade_to_vals": False,  # Apply a hillshade to plotted vals (True or False)
    "draw_coastlines": True,  # Draw coastlines
    "coastline_color": "grey",  # Colour to draw coastlines
    "use_antarctica_medium_coastline": True,  # True,False: Antarctic coastline inc iceshelves
    "use_cartopy_coastline": "no",  # 'no', 'low','medium', 'high' resolution
    "show_gridlines": True,  # True|False, display lat/lon grid lines
    "area_long_name_position": (0.29, 0.89),  # for default annotation position of area long name
    "area_long_name_fontsize": 16,  # font size of area.long_name
    "area_long_name_position_simple": (
        0.36,
        0.92,
    ),  # for default annot. pos of area name (map_only)
    "mask_long_name_position": (0.26, 0.86),  # for default annotation position of area long name
    "mask_long_name_position_simple": (
        0.36,
        0.9,
    ),  # for default annotation position of area long name
    "mask_long_name_fontsize": 10,  # font size of area.long_name
    # ------------------------------------------------------
    # Default Annotation
    # ------------------------------------------------------
    "varname_annotation_position_xy": (
        0.04,
        0.9,
    ),  # normalized position of default varname annotation in plot
    "varname_annotation_position_xy_simple": (
        0.04,
        0.91,
    ),  # normalized position of default varname annotation in map_only plot
    # ------------------------------------------------------
    "position_stats_manually": True,  # manually select each stats position
    "nvals_position": (0.0, 0.06),  # x,y position of nvals stat in relation to top left of cmap
    "min_position": (0.0, 0.04),  # x,y position of stdev stat in relation to top left of cmap
    "max_position": (0.0, 0.02),  # x,y position of stdev stat in relation to top left of cmap
    "mean_position": (0.2, 0.06),  # x,y position of stdev stat in relation to top left of cmap
    "median_position": (0.2, 0.04),  # x,y position of stdev stat in relation to top left of cmap
    "mad_position": (0.4, 0.06),  # x,y position of stdev stat in relation to top left of cmap
    "stdev_position": (0.4, 0.04),  # x,y position of stdev stat in relation to top left of cmap
    # ------------------------------------------------------
    "stats_position_x_offset": 0,  # x offset to stats position
    "stats_position_y_offset": 0,  # y offset to stats position
    "stats_position_x_offset_simple": -0.13,  # x offset to stats when plotting map_only
    "stats_position_y_offset_simple": 0,  # y offset to stats when plotting map_only
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
        0.22,
        0.05,
        0.55,
        0.02,
    ],  # [ left, bottom, width, height (fractions of axes)]
    # ------------------------------------------------------
    #       Lat/lon grid lines to show in main area
    #           - use empty lists to not include
    # ------------------------------------------------------
    "latitude_gridlines": [-64, -66, -68, -70],  # deg N
    "longitude_gridlines": [135, 140, 145, 150],  # deg E
    "gridline_color": "black",  # color to use for lat/lon grid lines
    "gridlabel_color": "black",  # color of grid labels
    "gridlabel_size": 8,  # size of grid labels
    "draw_gridlabels": True,  # whether to draw the grid labels
    "inner_gridlabel_color": "white",  # color of grid labels
    "inner_gridlabel_size": 8,  # size of grid labels
    "latitude_of_radial_labels": None,  # latitude for radial grid line labels for circular plots
    "labels_at_top": True,  # allow lat or lon labels at top of plot
    "labels_at_bottom": False,  # allow lat or lon labels at bottom of plot
    "labels_at_left": True,  # allow lat or lon labels at left of plot
    "labels_at_right": True,  # allow lat or lon labels at right of plot
    # ------------------------------------------------------
    #       Show a scale bar in km
    # ------------------------------------------------------
    "show_scalebar": True,
    "mapscale": [
        142,  # longitude to position scale bar
        -65.6,  # latitide to position scale bar
        100,  # longitude of true scale (ie centre of area)
        -68,  # latitude of true scale (ie centre of area)
        100,  # width of scale bar (km)
        "black",  # color of scale bar
        8,  # size of scale bar
    ],
    # --------------------------------------------------------
    # Histograms
    # --------------------------------------------------------
    "show_histograms": True,  # Whether to show the histogram plots
    "histogram_plotrange_axes": [
        0.705,  # left
        0.3,  # bottom
        0.08,  # width (axes fraction)
        0.35,  # height (axes fraction)
    ],  # axis location of plot range histogram for Polarplot.plot_points()
    "histogram_fullrange_axes": [
        0.88,  # left
        0.3,  # bottom
        0.08,  # width (axes fraction)
        0.35,  # height (axes fraction)
    ],  # axis location of plot range histogram for Polarplot.plot_points()
    # --------------------------------------------------------
    # Latitude vs Values plot
    # --------------------------------------------------------
    "show_latitude_scatter": True,  # Whether to show the latitude scatter plot
    "latvals_axes": [
        0.75,  # left
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
        0.8,  # left
        0.75,  # bottom
        0.19,  # width (axes fraction)
        0.19,  # height (axes fraction)
    ],
    "bad_data_minimap_bounding_lat": None,  # None or bounding latitude if used for mini-map
    # uses 40N for northern hemisphere or 50N for southern.
    # Override with this parameter
    "bad_data_minimap_circle": None,  # None or [lat,lon,circle_radius_m,color_str]
    "bad_data_minimap_draw_gridlines": True,
    "bad_data_minimap_gridlines_color": "lightgrey",  # color of gridlines drawn in bad data minimap
    "bad_data_latitude_lines": [-50, -70],  # latitude lines to draw in bad data minimap
    "bad_data_longitude_lines": [
        0,
        60,
        120,
        180,
        -120,
        -60,
    ],  # longitude lines to draw in bad data minimap
    "bad_data_minimap_val_scalefactor": 1.0,  # scale factor for plotting bad values on minimap
    "bad_data_minimap_legend_pos": (0.75, 0.0),  # position of minimap legend (upper right)
    # relative to bad_data_minimap axis
    "bad_data_minimap_coastline_resolution": "medium",  # low, medium, high resolution coastline
    # ------------------------------------------------------------------
    # Mini-map (with box showing actual area) - purpose to show where a
    # smaller area is on a larger map.
    # -------------------------------------------------------------------
    "show_minimap": True,  # show the overview minmap
    "minimap_axes": [  # define minimap axis position
        0.63,  # left
        0.81,  # bottom
        0.18,  # width (axes fraction)
        0.18,  # height (axes fraction)
    ],
    "minimap_bounding_lat": None,  # None or bounding latitude if used for mini-map
    # uses 40N for northern hemisphere or 50N for southern.
    # Override with this parameter
    "minimap_circle": [-68, 141, 200000, "red"],  # None or [lat,lon,circle_radius_m,color_str]
    "minimap_draw_gridlines": True,
    "minimap_val_scalefactor": 1.0,  # scale factor for plotting bad values on minimap
    "minimap_legend_pos": (1.38, 1.1),  # position of minimap legend (upper right) in minimap axis
}
