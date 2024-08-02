"""
# Area definition

## Summary: 
Based on area: greenland
**background_image: hillshade**

"""
area_definition = {
    "use_definitions_from": "greenland",
    # --------------------------------------------
    #    mask from cpom.masks.Mask
    # --------------------------------------------
    "background_image": [
        "ibcao_bathymetry",
        "hillshade",
    ],
    "background_image_alpha": [0.14, 0.18],
    "background_color": "white",
}
