"""
# Area definition

## Summary:
Based on area: arctic
North polar area (> 60N) used by the CryoSat-2 CSQA performance monitoring service
"""

area_definition = {
    "use_definitions_from": "arctic",
    "long_name": "North Polar (>60N)",
    "background_image": "natural_earth_faded",
    "bounding_lat": 60.0,  # limiting latitude for round areas or None
    # Data filtering using lat/lon extent (used as a quick data pre-filter before masking)
    "minlat": 60.0,  # minimum latitude to initially filter records for area
}
