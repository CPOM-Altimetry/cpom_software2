"""
# Area definition

## Summary:

Based on area: amundsen_sea_embayment

"""

area_definition = {
    "use_definitions_from": "amundsen_sea_embayment",
    "area_summary": "Amundsen Sea Embayment (Thwaites, PIG glaciers)",
    "centre_lon": -104.3,  # degrees E
    "centre_lat": -76.6,  # degrees N
    "width_km": 1020,  # width in km of plot area (x direction)
    "height_km": 1020,  # height in km of plot area (y direction)
    # --------------------------------------------
    # Plot parameters for this area
    # --------------------------------------------
    "axes": [  # define plot axis position
        -0.025,  # left
        0.125,  # bottom
        0.75,  # width (axes fraction)
        0.75,  # height (axes fraction)
    ],
    "minimap_axes": [  # define minimap axis position
        0.7,  # left
        0.7,  # bottom
        0.25,  # width (axes fraction)
        0.25,  # height (axes fraction)
    ],
    "show_stats": False,  # whether to show stats info on the plot
    "longitude_gridlines": [
        230,
        235,
        240,
        245,
        250,
        255,
        260,
        265,
        270,
        275,
        280,
        285,
        290,
        300,
    ],  # deg E
    "latitude_gridlines": [-70, -72, -74, -76, -78, -80],  # deg N
    "apply_area_mask_to_data": True,  # filter data using areas clev2er.utils.masks.Mask
    "maskname": "rignot_2016_basin_9_polygon_mask",  # from  clev2er.utils.masks.Mask
    "masktype": "polygon",
    "basin_numbers": [],  # [n1,n2,..] if mask allows basin numbers
    # for bedmachine v2, 2=grounded ice, 3=floating, 4=vostok
    "show_polygon_mask": False,  # show mask polygon
    "polygon_mask_color": "black",  # color to draw mask polygon
}
