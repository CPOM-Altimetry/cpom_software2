"""
# Area definition

## Summary:

Based on area: amundsen_sea_embayment

"""

area_definition = {
    "use_definitions_from": "amundsen_sea_embayment",
    "area_summary": "Amundsen Sea Embayment (Thwaites, PIG glaciers)",
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
}
