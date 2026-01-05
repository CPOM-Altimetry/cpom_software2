"""
# Area definition

## Summary:

Based on area: amundsen_sea_embayment

"""

area_definition = {
    "use_definitions_from": "amundsen_sea_embayment",
    "area_summary": "Amundsen Sea Embayment (Thwaites, PIG glaciers)",
    "width_km": 900,  # width in km of plot area (x direction)
    # --------------------------------------------
    #    mask from clev2er.utils.masks.Mask
    # --------------------------------------------
    "apply_area_mask_to_data": True,  # filter data using areas clev2er.utils.masks.Mask
    "maskname": "ase_xylimits_mask",  # from  clev2er.utils.masks.Mask
    "masktype": "xylimits",
    "axes": [  # define plot axis position
        -0.0,  # left
        0.09,  # bottom
        0.73,  # width (axes fraction)
        0.73,  # height (axes fraction)
    ],
    "labels_at_top": False,  # allow lat or lon labels at top of plot
    "labels_at_bottom": False,  # allow lat or lon labels at bottom of plot
    "labels_at_left": False,  # allow lat or lon labels at left of plot
    "labels_at_right": False,  # allow lat or lon labels at right of plot
    # ....
    # --------------------------------------------------------
    # Histograms
    # --------------------------------------------------------
    "show_histograms": True,  # Whether to show the histogram plots
    "histogram_plotrange_axes": [
        0.735,  # left
        0.47,  # bottom
        0.08,  # width (axes fraction)
        0.35,  # height (axes fraction)
    ],  # axis location of plot range histogram for Polarplot.plot_points()
    "histogram_fullrange_axes": [
        0.89,  # left
        0.47,  # bottom
        0.08,  # width (axes fraction)
        0.35,  # height (axes fraction)
    ],  # axis location of plot range histogram for Polarplot.plot_points()
    # --------------------------------------------------------
    # Latitude vs Values plot
    # --------------------------------------------------------
    "show_latitude_scatter": True,  # Whether to show the latitude scatter plot
    "latvals_axes": [
        0.77,  # left
        0.22,  # bottom
        0.17,  # width (axes fraction)
        0.2,  # height (axes fraction)
    ],  # axis location of latitude vs values scatter plot
    # ------------------------------------------------------------------
    # Mini-map (with box showing actual area) - purpose to show where a
    # smaller area is on a larger map.
    # -------------------------------------------------------------------
    "show_minimap": True,  # show the overview minmap
    "minimap_axes": [  # define minimap axis position
        0.86,  # left
        0.85,  # bottom
        0.13,  # width (axes fraction)
        0.13,  # height (axes fraction)
    ],
    "minimap_bounding_lat": None,  # None or bounding latitude if used for mini-map
    # uses 40N for northern hemisphere or 50N for southern.
    # Override with this parameter
    "minimap_circle": None,  # None or [lat,lon,circle_radius_m,color_str]
    "minimap_draw_gridlines": False,
    "minimap_val_scalefactor": 1.0,  # scale factor for plotting bad values on minimap
    "minimap_legend_pos": (1.38, 1.1),  # position of minimap legend (upper right) in minimap axis
    "stats_position_x_offset": 0.77,  # x offset to stats position
    "stats_position_y_offset": -0.06,  # y offset to stats position
}
