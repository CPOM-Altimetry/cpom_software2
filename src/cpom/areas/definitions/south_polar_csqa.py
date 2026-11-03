"""
# Area definition

## Summary:
Based on area: antarctica
South polar area (< 60S) used by the CryoSat-2 CSQA performance monitoring service
"""

area_definition = {
    "use_definitions_from": "antarctica",
    "long_name": "South Polar (<60S)",
    "bounding_lat": -60,  # limiting latitude for round areas or None
    # Data filtering using lat/lon extent (used as a quick data pre-filter before masking)
    "maxlat": -60.0,  # maximum latitude to initially filter records for area
}
