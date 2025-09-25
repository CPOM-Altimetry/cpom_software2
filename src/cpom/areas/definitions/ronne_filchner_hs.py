"""
# Area definition

## Summary:
Based on area: ronne_filchner
**background_image: hillshade**

"""

area_definition = {
    "use_definitions_from": "ronne_filchner",
    "area_summary": "Ronne Filchner ice shelves [hillshade]",
    "background_image": ["ibcso_bathymetry", "hillshade", "ant_iceshelves"],
    "background_image_resolution": ["low", "_"],
    "background_image_alpha": [0.1, 0.3, 0.2],
    "background_color": "aliceblue",
    "gridline_color": "darkgrey",
    "hillshade_params": {"dem": "awi_ant_1km_grounded", "azimuth": 120.0},
    "draw_coastlines": False,  # Draw coastlines
}
