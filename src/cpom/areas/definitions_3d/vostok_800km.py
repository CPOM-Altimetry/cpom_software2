"""3D Area Definition

# Lake Vostok at 800km using 100m REMA Gapless DEM (sampled by 4)

"""

area_definition = {
    "long_name": "Lake Vostok, E. Antarctica, 800km sq. Area, DEM 100m::4",
    "use_definitions_from": "vostok",
    "width_km": 800,  # width in km of plot area (x direction)
    "height_km": 800,  # height in km of plot area (y direction)
    # 3d plot settings
    "dem_name": "rema_gapless_100m_zarr",  # DEM name used for 3d plots from CPOM Dem class
    "page_width": 1500,  # browser page width (doesn't affect aspect ratio)
    "dem_stride": 4,
    "zaxis_multiplier": 0.3,  # aspect ratio for Z-axis
}
