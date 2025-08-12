"Area definition" ""

# pylint: disable=R0801 # warning for similar lines

area_definition = {
    "use_definitions_from": "ross_iceshelf",
    "area_summary": "Ross Ice Shelf [hillshade]",
    "background_image": ["ibcso_bathymetry", "hillshade", "ant_iceshelves"],
    "background_image_resolution": ["low", "_"],
    "background_image_alpha": [0.1, 0.3, 0.2],
    "background_color": "aliceblue",
    "gridline_color": "darkgrey",
    "hillshade_params": {"dem": "awi_ant_1km_grounded", "azimuth": 120.0},
    "draw_coastlines": True,  # Draw coastlines
}
