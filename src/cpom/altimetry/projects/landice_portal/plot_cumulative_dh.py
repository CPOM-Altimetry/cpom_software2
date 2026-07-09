"""
cpom.altimetry.projects.landice_portal.plot_cumulative_dh.py

# Purpose

Plot parameters from CPOM Antarctic cumulative dH products

Product naming example:
CPOM-AIS-L3C-DH-MULTIMISSION-5KM-19950321-20050308-fv1.nc

Product CDL:
netcdf CPOM-AIS-L3C-DH-MULTIMISSION-5KM-19950321-20051213-fv1 {
dimensions:
        ny = 968 ;
        nx = 1128 ;
        time_period = 1 ;
variables:
        float dH(time_period, ny, nx) ;
                dH:long_name = "surface elevation change" ;
                dH:units = "m" ;
                dH:source = "multi-mission radar altimetry" ;
                dH:grid_mapping = "grid_projection" ;
                string dH:comment = "We fill data gaps outside the pole hole (81.5 S)
                    wherever possible by triangulation, and we fill the small number of
                    remaining gaps elsewhere in this region using the average elevation
                    rate of the remainder of each drainage basin. To estimate elevation
                    changes beyond the satellite orbital limits, we divided the
                    unobserved region into the inland and downstream catchments of the
                    Kamb, Whillans, and Mercer ice streams and the interior remainder.
                    We extrapolated the average elevation trends from the observed
                    portions of each area within the 80–81°S annulus to their
                    respective unobserved portions." ;
        float dH_uncertainty(time_period, ny, nx) ;
                dH_uncertainty:long_name = "uncertainty in surface elevation change" ;
                dH_uncertainty:units = "m" ;
                dH_uncertainty:grid_mapping = "grid_projection" ;
                string dH_uncertainty:comment = "uncertainty in surface elevation change
                    is calculated as ... (note uncompleted)" ;
        float x(nx) ;
                x:long_name = "Cartesian x-coordinate - easting, of centre of each grid cell" ;
                x:units = "meters" ;
                x:standard_name = "projection_x_coordinate" ;
                x:min_val = -2817500. ;
                x:binsize = 5000. ;
        float y(ny) ;
                y:long_name = "Cartesian y-coordinate - northing, of centre of each grid cell" ;
                y:units = "meters" ;
                y:standard_name = "projection_y_coordinate" ;
                y:min_val = -2417500. ;
                y:binsize = 5000. ;
        char grid_projection ;
                grid_projection:ellipsoid = "WGS84" ;
                grid_projection:crs = "epsg:3031" ;
                grid_projection:latitude_of_origin = -71. ;
                grid_projection:grid_mapping_name = "polar_stereographic_south" ;
                grid_projection:false_easting = 0. ;
                grid_projection:false_northing = 0. ;
                grid_projection:central_meridian = 0. ;
        double lat(ny, nx) ;
                lat:units = "degrees_north" ;
                lat:standard_name = "latitude" ;
                lat:long_name = "latitude coordinate" ;
                lat:min_val = -89.9674601532943 ;
                lat:max_val = -56.7587107166777 ;
        double lon(ny, nx) ;
                lon:units = "degrees_east" ;
                lon:standard_name = "longitude" ;
                lon:long_name = "longitude coordinate" ;
                lon:min_val = 0.0592510435250638 ;
                lon:max_val = 359.940748956475 ;
        byte surface_type(ny, nx) ;
                surface_type:_FillValue = -128b ;
                surface_type:coordinates = "lon lat" ;
                surface_type:long_name = "surface type from mask" ;
                surface_type:flag_values = 0b, 1b, 2b, 3b, 4b ;
                surface_type:flag_meanings =
                    "ocean ice_free_land grounded_ice floating_ice lake_vostok" ;
                surface_type:source = "https://nsidc.org/data/nsidc-0756/versions/2" ;
                surface_type:valid_min = 0b ;
                surface_type:valid_max = 4b ;
                surface_type:comment = "Surface type identifier, for use in
                    discriminating different surfaces types within the SEC grid; derived
                    from the BedMachine Antarctica version 2 (Morlighem, 2020) datasets." ;
        byte basin_id(ny, nx) ;
                basin_id:_FillValue = -128b ;
                basin_id:coordinates = "lon lat" ;
                basin_id:long_name = "Glacialogical basin identification number" ;
                basin_id:comment = "IMBIE glacialogical basin id number (Rignot et al.,
                    2016) associated with each measurement. Values are : 0 (outside
                    mask), 1-18 (basin values for Antarctica)" ;
                basin_id:source = "IMBIE imbie.org/imbie-3/drainage-basins/" ;
                basin_id:valid_min = 0b ;
                basin_id:valid_max = 18b ;
                basin_id:flag_values = 0LL, 1LL, 2LL, 3LL, 4LL, 5LL, 6LL, 7LL, 8LL, 9LL,
                    10LL, 11LL, 12LL, 13LL, 14LL, 15LL, 16LL, 17LL, 18LL ;
                string basin_id:flag_meanings = "Islands", "West H-Hp", "West F-G",
                    "East E-Ep", "East D-Dp", "East Cp-D", "East B-C", "East A-Ap",
                    "East Jpp-K", "West G-H", "East Dp-E", "East Ap-B", "East C-Cp",
                    "East K-A", "West J-Jpp", "Peninsula Ipp-J", "Peninsula I-Ipp",
                    "Peninsula Hp-I", "West Ep-F" ;
        float start_time(time_period) ;
                start_time:comment = "the start time of each period used to calculate
                    surface elevation change, in decimal years" ;
                start_time:long_name = "start time" ;
                start_time:units = "years" ;
        float end_time(time_period) ;
                end_time:comment = "the end time of each period used to calculate
                    surface elevation change, in decimal years" ;
                end_time:long_name = "end time" ;
                end_time:units = "years" ;

// global attributes:
                :title = "Ice Sheet Surface Elevation Change at 5.0km resolution from
                    1995-03-21 to 2005-12-13 from Multi-Mission Radar Altimetry" ;
                :institution = "Centre for Polar Observation and Modelling (CPOM)" ;
                :creator_name = "Alan Muir, Tom Slater, Jennifer Maddalena, Ben Palmer(CPOM)" ;
                :creator_url = "http://www.cpom.org.uk" ;
                :references = "Trends in Antarctic Ice Sheet Elevation and Mass,
                    Shepherd et al, GRL,2019, doi: 10.1029/2019GL082182" ;
                :source = "ERS-2 FDR4ALT v1, ENVISAT FDR4ALT v1, CryoSat-2 CryoTEMPO LI
                    Baseline-Dd" ;
                :history = "2026-06-10T14:37:46Z - Product generated by CPOM Software
                    Processor, commit 37160c710" ;
                :tracking_id = "90f3e26a-64d1-11f1-bd26-aa52cba6b339" ;
                :Conventions = "CF-1.10" ;
                :product_version = 1. ;
                :summary = "This dataset contains the surface elevation change of the
                    grounded ice sheet between 21/03/1995 and 13/12/2005, at 5.0 km
                    resolution, on a polar stereo grid, derived from radar altimetry
                    missions since 1995: ERS-2, ENVISAT, CryoSat-2." ;
                :keywords = "satellite, ice, ice sheets, ice growth/melt, cryospheric
                    indicators, climate indicators" ;
                :id = "CPOM-AIS-L3C-DH-MULTIMISSION-5KM-19950321-20051213-fv1.nc" ;
                :naming_authority = "cpom.org.uk" ;
                :keywords_vocabulary = "NASA Global Change Master Directory (GCMD)
                    Science Keywords" ;
                :date_created = "20260610T143746Z" ;
                :creator_email = "cpom@northumbria.ac.uk" ;
                :region = "Antarctic grounded ice" ;
                :gia_correction_model = "ij05" ;
                :grid_resolution = "5.0 km" ;
                :geospatial_lat_min = -89.9674601532943 ;
                :geospatial_lat_max = -56.7587107166777 ;
                :geospatial_lon_min = 0.0592510435250638 ;
                :geospatial_lon_max = 359.940748956475 ;
                :data_start_time = "1995-03-21T00:00:00Z" ;
                :data_end_time = "2005-12-13T00:00:00Z" ;
                :time_coverage_start = "19950321T000000Z" ;
                :time_coverage_end = "20051213T000000Z" ;
                :key_variables = "dH, dH_uncertainty" ;
                :spatial_resolution = "5.0 km" ;
                :dh_sw_version = "d3b859955" ;
                :product_sw_version = "37160c710" ;
                :product_created = "2026-06-10T14:37:46Z" ;
                :netCDF_version = "NETCDF4" ;
}
"""

import argparse
import glob
import os
import re
import sys

import numpy as np
from netCDF4 import Dataset  # pylint: disable=no-name-in-module

from cpom.areas.area_plot import Annotation, Polarplot


def main():
    """main function for tool"""

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prod_dir",
        "-d",
        help="path of directory containing input cumulative dH products",
    )

    parser.add_argument(
        "--prod_filename",
        "-f",
        help="path of a single input cumulative dH product file",
    )

    parser.add_argument(
        "--outdir",
        "-od",
        help=("output directory for plot files. Default is the input product directory."),
    )
    parser.add_argument(
        "--area",
        "-a",
        help="Area of interest. Default is antarctica_cpom_portal.",
        default="antarctica_cpom_portal",
    )

    parser.add_argument(
        "--parameter",
        "-p",
        help=("parameter to plot: dH, dH_uncertainty, basin_id, surface_type"),
        default="dH",
    )

    if len(sys.argv) == 1:
        print("no args provided")
        parser.print_help()
        sys.exit()

    args = parser.parse_args()

    # Find CPOM cumulative dH netcdf files: example:
    # CPOM-AIS-L3C-DH-MULTIMISSION-5KM-<YYYYMMDD>-<YYYYMMDD>-fv1.nc

    if args.prod_dir:
        input_files = sorted(
            glob.glob(f"{args.prod_dir}/CPOM-AIS-L3C-DH-MULTIMISSION-5KM-*-fv*.nc")
        )
        if not input_files:
            sys.exit(f"No CPOM-AIS-L3C-DH-MULTIMISSION-5KM-*-fv*.nc files found in {args.prod_dir}")
    elif args.prod_filename:
        input_files = [args.prod_filename]
    else:
        sys.exit("Must have either --prod_filename or --prod_dir")

    output_dir = args.outdir
    if not output_dir:
        output_dir = os.path.dirname(input_files[0])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for input_file in input_files:
        print(f"Processing {input_file}")
        prod_name = os.path.basename(input_file)
        print(f"Base name {prod_name}")

        out_file = f"{output_dir}/{prod_name.replace('.nc', f'-{args.parameter.lower()}')}"
        print(f"Output file base {out_file}")

        if args.parameter == "dH":
            param_long_name = "Cumulative Height Change (dH)"
        elif args.parameter == "dH_uncertainty":
            param_long_name = "Uncertainty of Cumulative dH"
        elif args.parameter == "basin_id":
            param_long_name = "Glaciological Basin ID (Rignot 2016)"
        elif args.parameter == "surface_type":
            param_long_name = "Ice Surface Type"
        else:
            param_long_name = args.parameter

        # Extract dates YYYYMMDD-YYYYMMDD from the product name
        date_match = re.search(r"(\d{8})-(\d{8})", prod_name)
        if not date_match:
            sys.exit(f"Could not find YYYYMMDD-YYYYMMDD dates in product name {prod_name}")
        dh_start_year = date_match.group(1)[0:4]
        dh_start_month = date_match.group(1)[4:6]
        dh_start_day = date_match.group(1)[6:8]
        dh_end_year = date_match.group(2)[0:4]
        dh_end_month = date_match.group(2)[4:6]
        dh_end_day = date_match.group(2)[6:8]
        print(f"dh period start: {dh_start_year}-{dh_start_month}-{dh_start_day}")
        print(f"dh period end: {dh_end_year}-{dh_end_month}-{dh_end_day}")

        with Dataset(input_file) as nc:
            lats = np.ma.filled(nc.variables["lat"][:], np.nan)
            lons = np.ma.filled(nc.variables["lon"][:], np.nan)

            # dH and dH_uncertainty have dims (time_period, ny, nx) with time_period=1,
            # whereas basin_id and surface_type are (ny, nx)
            var_data = nc.variables[args.parameter]
            plot_data = var_data[0][:] if var_data.ndim == 3 else var_data[:]
            plot_var = np.ma.filled(np.ma.asarray(plot_data).astype(float), np.nan)

            long_name = nc[args.parameter].long_name
            try:
                units = nc[args.parameter].units
            except (KeyError, AttributeError):
                units = ""

            # Print numpy array dimensions: shape and number of dimensions
            print(f"lats.shape = {lats.shape}, lats.ndim = {lats.ndim}")
            print(f"lons.shape = {lons.shape}, lons.ndim = {lons.ndim}")
            print(f"plot_var.shape = {plot_var.shape}, plot_var.ndim = {plot_var.ndim}")

            # Plot parameter

            # Creating the dataset
            dataset = {
                "name": long_name,
                "units": units,
                "lats": lats,
                "lons": lons,
                "vals": plot_var,
                "plot_size_scale_factor": 0.01,
                "min_plot_range": -20.0,
                "max_plot_range": 20.0,
                "cmap_name": "RdBu",  # Colormap name, could use RdYlBu
                "cmap_over_color": "#150685",  # Optional: Over color for colormap
                "cmap_under_color": "#9E0005",  # Optional: Under color for colormap
                "cmap_extend": "both",  # Optional: Extend colormap
            }

            if "uncertainty" in args.parameter:
                dataset["cmap_name"] = "RdBu_r"
                dataset["cmap_over_color"] = "#9E0005"
                dataset["cmap_under_color"] = "#150685"
                dataset["min_plot_range"] = 0.0
                dataset["max_plot_range"] = 2.0

            if args.parameter == "surface_type":
                dataset = {
                    "name": long_name,
                    "units": units,
                    "lats": lats,
                    "lons": lons,
                    "vals": plot_var,
                    "plot_size_scale_factor": 0.01,
                    "flag_values": [0, 1, 2, 3, 4],  # Optional: List of flag values
                    "flag_names": [
                        "Ocean",
                        "Ice-free Land",
                        "Grounded Ice",
                        "Floating Ice",
                        "Lake Vostok",
                    ],  # Optional: List of flag names
                    "flag_colors": [
                        "#F2F7FF",
                        "green",
                        "orange",
                        "yellow",
                        "red",
                    ],  # Optional: Colors for flags or colormap
                }

            if args.parameter == "basin_id":
                flag_colors = [
                    "#F2F7FF",
                    "blue",
                    "green",
                    "orange",
                    "purple",
                    "brown",
                    "pink",
                    "gray",
                    "olive",
                    "cyan",
                    "magenta",
                    "gold",
                    "navy",
                    "teal",
                    "coral",
                    "lime",
                    "indigo",
                    "maroon",
                    "orchid",
                ]

                dataset = {
                    "name": long_name,
                    "units": units,
                    "lats": lats,
                    "lons": lons,
                    "vals": plot_var,
                    "plot_size_scale_factor": 0.01,
                    "flag_values": list(range(0, 19)),
                    "flag_names": [
                        "Outside:00",
                        "West H-Hp:01",
                        "West F-G:02",
                        "East E-Ep:03",
                        "East D-Dp:04",
                        "East Cp-D:05",
                        "East B-C:06",
                        "East A-Ap:07",
                        "East Jpp-K:08",
                        "West G-H:09",
                        "East Dp-E:10",
                        "East Ap-B:11",
                        "East C-Cp:12",
                        "East K-A:13",
                        "West J-Jpp:14",
                        "Peninsula Ipp-J:15",
                        "Peninsula I-Ipp:16",
                        "Peninsula Hp-I:17",
                        "West Ep-F:18",
                    ],
                    "flag_colors": flag_colors,
                }

                print(f"min {np.nanmin(plot_var)} max {np.nanmax(plot_var)}")

            xpos = 0.02
            ypos = 0.86
            ysep = 0.04
            xsep = 0.05

            annot = Annotation(
                0.269,
                0.958,
                f"{param_long_name}",
                {
                    "boxstyle": "round",  # Style of the box (e.g.,'round','square')
                    "facecolor": "aliceblue",  # Background color of the box
                    "alpha": 1.0,  # Transparency of the box (0-1)
                    "edgecolor": "lightgrey",  # Color of the box edge
                },
                18,
                fontweight="bold",
            )
            annotation_list = [annot]

            annotation_list.append(
                Annotation(
                    xpos,
                    ypos + 0.01,
                    "dH period:",
                    None,
                    10,
                )
            )

            annotation_list.append(
                Annotation(
                    xpos,
                    ypos - ysep + 0.007,
                    f"{dh_start_month}",
                    None,
                    16,
                    fontweight="normal",
                    color="grey",
                )
            )
            annotation_list.append(
                Annotation(
                    xpos + xsep - 0.02,
                    ypos - ysep,
                    f"{dh_start_year}",
                    None,
                    24,
                    fontweight="bold",
                )
            )

            annotation_list.append(
                Annotation(
                    xpos,
                    ypos - ysep * 2 + 0.007,
                    f"{dh_end_month}",
                    None,
                    16,
                    fontweight="normal",
                    color="grey",
                )
            )
            annotation_list.append(
                Annotation(
                    xpos + xsep - 0.02,
                    ypos - ysep * 2,
                    f"{dh_end_year}",
                    None,
                    24,
                    fontweight="bold",
                )
            )

            annotation_list.append(
                Annotation(
                    0.261,
                    0.89,
                    f"{prod_name}",
                    None,
                    12,
                    fontweight="normal",
                )
            )

            annotation_list.append(
                Annotation(
                    0.237,
                    0.235,
                    "66S",
                    None,
                    8,
                    fontweight="normal",
                    color="grey",
                )
            )
            annotation_list.append(
                Annotation(
                    0.275,
                    0.283,
                    "70S",
                    None,
                    8,
                    fontweight="normal",
                    color="grey",
                )
            )
            annotation_list.append(
                Annotation(
                    0.315,
                    0.33,
                    "74S",
                    None,
                    8,
                    fontweight="normal",
                    color="grey",
                )
            )

            area_overrides = {
                "show_bad_data_map": False,
            }

            if args.parameter == "basin_id":
                area_overrides["flag_perc_axis"] = (
                    0.8,
                    0.1,
                    0.05,
                )  # [left,bottom, width] of axis. Note height is auto set

            if args.parameter == "surface_type":
                area_overrides["flag_perc_axis"] = (
                    0.82,
                    0.4,
                    0.1,
                )  # [left,bottom, width] of axis. Note height is auto set

            Polarplot(args.area, area_overrides).plot_points(
                dataset,
                # map_only=True,
                output_file=out_file,
                use_default_annotation=False,
                annotation_list=annotation_list,
                image_format="webp",
                use_cmap_in_hist=(args.parameter != "dH"),
                dpi=75,
                webp_settings=(95, 6),
            )

            # -----------------------------------------------------------------------
            # Redo plots with hillshade
            # -----------------------------------------------------------------------

            area_overrides["apply_hillshade_to_vals"] = True

            Polarplot(args.area, area_overrides).plot_points(
                dataset,
                # map_only=True,
                output_file=f"{out_file}-hs",
                use_default_annotation=False,
                annotation_list=annotation_list,
                image_format="webp",
                use_cmap_in_hist=(args.parameter != "dH"),
                dpi=75,
                webp_settings=(95, 6),
            )

            print(f"Output: {out_file}")


if __name__ == "__main__":
    main()
