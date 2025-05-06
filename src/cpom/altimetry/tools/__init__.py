"""
# Tools

The following command line tools are available:

## plot_map.py

Plot variables from NetCDF file(s) on a selectable cryosphere map.

Further details at `cpom.altimetry.tools.plot_map`

## find_files_in_area.py

Tool to to find files within a specified directory (and default is to search recursive sub-dirs) 
that contain lat/lon locations within a CPOM Area's
mask or within a radius (km) of a specified lat,lon point. Optionally plot tracks 
in area map. 


## nc_vals.py

Tool which prints netcdf parameters. A bit like ncdump but with more options
and allows single parameters to be printed more easily.

## validate_l2_altimetry_elevations.py

This tool compares a selected month of radar altimetry mission elevation data 
against laser altimetry reference elevations. 
It also allows comparing a reference mission to itself.

Further details at `cpom.altimetry.tools.validate_l2_altimetry_elevations`


"""
