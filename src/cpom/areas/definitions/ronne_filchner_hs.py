"""
# Area definition

## Summary: 
Based on area: ronne_filchner
**background_image: hillshade**

"""

area_definition = {
    "use_definitions_from": "ronne_filchner",
    "area_summary": "Ronne Filchner ice shelves [hillshade]",
    # --------------------------------------------
    #    mask from clev2er.utils.masks.Mask
    # --------------------------------------------
    "background_image": ["ibcso_bathymetry", "hillshade", "ant_iceshelves"],
    "background_image_resolution": ["low", "_"],
    "background_image_alpha": [0.5, 0.7, 0.3],
    "background_color": "aliceblue",
    "gridline_color": "darkgrey",
    "hillshade_params": {"dem": "awi_ant_1km_grounded", "azimuth": 120.0},
    "draw_coastlines": False,  # Draw coastlines
}
