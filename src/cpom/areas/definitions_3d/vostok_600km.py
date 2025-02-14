"""3D Area Definition

# Lake Vostok at 600km using 100m REMA Gapless DEM (sampled by 5)

"""

area_definition = {
    "long_name": "Lake Vostok, E. Antarctica, 600km sq. Area, DEM 100m::5",
    "use_definitions_from": "vostok",
    "width_km": 600,  # width in km of plot area (x direction)
    "height_km": 600,  # height in km of plot area (y direction)
    # 3d plot settings
    "dem_name": "rema_ant_1km_zarr",  # DEM name used for 3d plots from CPOM Dem class
    "page_width": 1500,  # browser page width (doesn't affect aspect ratio)
    "dem_stride": 1,
    "zaxis_multiplier": 0.25,  # aspect ratio for Z-axis
}
