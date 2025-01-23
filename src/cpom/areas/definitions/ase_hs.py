"""
# Area definition

## Summary:

Based on area: amundsen_sea_embayment

"""

area_definition = {
    "use_definitions_from": "amundsen_sea_embayment",
    # --------------------------------------------
    #    mask from cpom.masks.Mask
    # --------------------------------------------
    "background_image": [
        "ibcso_bathymetry",
        "hillshade",
    ],
    "background_image_alpha": [0.14, 0.18],
    "background_color": "white",
}
