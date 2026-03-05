#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate CPOM sea ice thickness NetCDF products from input ascii .map files."""

import argparse
import os
import re
import sys
from datetime import datetime, timezone

import pandas as pd
from netCDF4 import Dataset  # pylint: disable=no-name-in-module


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate NetCDF products from input ascii .map files."
    )
    parser.add_argument(
        "--nrt", action="store_true", help="Generate near real time (NRT) products."
    )
    parser.add_argument(
        "--ntc", action="store_true", help="Generate non time critical (NTC) monthly products."
    )
    parser.add_argument("-m", "--month", type=int, help="Month to process for NTC (1-12).")
    parser.add_argument("-y", "--year", type=int, help="Year to process for NTC (YYYY).")
    parser.add_argument(
        "--arco", action="store_true", help="Process only Arctic (arco) region for NTC."
    )
    parser.add_argument(
        "--anto", action="store_true", help="Process only Antarctic (anto) region for NTC."
    )
    parser.add_argument(
        "--latest", action="store_true", help="Process the previous 3 months for NTC."
    )
    parser.add_argument("-o", "--output", required=True, help="Path to output directory.")
    parser.add_argument(
        "--max-cog-dist",
        type=float,
        default=15.0,
        help="Maximum COG distance in km (default: 15.0). Data points exceeding this are excluded.",
    )
    parser.add_argument(
        "--version",
        "-v",
        type=int,
        default=1,
        help="Product version number (default: 1).",
    )
    # For testing and flexibility
    parser.add_argument(
        "--input-dir",
        default="/cpnet/altimetry/seaice/nrt_system/latest",
        help="Override input directory (default: /cpnet/altimetry/seaice/nrt_system/latest).",
    )
    return parser.parse_args()


# Map parameter filenames to standard NetCDF CF-like variable names
NTC_PARAMETERS = {
    "FloeChordLength": "floe_chord_length",
    "IceConcentration": "ice_concentration",
    "IceType": "ice_type",
    "LeadFraction": "lead_fraction",
    "FloeFraction": "floe_fraction",
    "RadarFreeboard": "radar_freeboard",
    "SeaLevelAnomaly": "sea_level_anomaly",
    "WarrenSnowDepth": "warren_snow_depth",
    "UnkFraction": "unk_fraction",
}

# Metadata map for NTC parameters
NTC_PARAMETER_METADATA = {
    "radar_freeboard": {
        "units": "meters",
        "comment": (
            "Radar freeboard is the difference between the height of the ice surface observed "
            "by the radar and the sea surface height. It will equal true freeboard when the "
            "sea ice has no snow cover but in the presence of snow the freeboard will appear "
            "low due to the reduced propagation speed of light through the snow cover. A "
            "model for the snow depth on sea ice is required to convert radar freeboard to "
            "true freeboard. As an approximation, true freeboard is greater than radar "
            "freeboard by one quarter of the snow depth."
        ),
    },
    "floe_chord_length": {
        "units": "meters",
        "comment": (
            "An indication of floe chord length is computed by measuring the distance over "
            "which continuous sequences of radar echoes classified as floes are found. A gap "
            "of one record not classified as a floe is permitted but a gap of two or more "
            "marks the end of the floe. This should be regarded as an indication of floe "
            "length rather than a precise measurement as the results are highly dependent on "
            "details of the method used."
        ),
    },
    "ice_concentration": {
        "units": "percentage ice",
        "comment": (
            "Echoes returning from ice floes are very similar to those returning from open "
            "water, so we use sea ice concentration is used to distinguish between the two. "
            "For an echo to have come from an ice floe the sea ice concentration must exceed "
            "75%. We use daily sea ice concentration data that are generated at the NASA "
            "Goddard Space Flight Centre (GSFC) and are available through the National Snow "
            "and Ice Data Centre (NSIDC) (Cavalieri et al., 1996, updated yearly). The data "
            "are generated from brightness temperature data derived from satellite passive "
            "microwave sensors using the NASA Team algorithm."
        ),
    },
    "ice_type": {
        "units": "tbd",
        "comment": (
            "Sea ice type (Arctic only) is required to modify the snow depth climatology "
            "used to compute ice thickness from freeboard.  The monthly mean snow depth is "
            "halved over first year ice to account for the reduced accumulation compared "
            "with multi-year ice. Sea ice type data are provided by the Norwegian "
            "Meteorological Service (NMS) Ocean and Sea Ice Satellite Application Facility "
            "(OSI SAF) (Andersen et al., 2012)."
        ),
    },
    "sea_level_anomaly": {
        "units": "meters",
        "comment": (
            "Sea level anomaly is the difference between the observed sea surface height "
            "and a mean sea surface. The sea surfaces heights over sea ice are constructed "
            "using the elevations of echoes returning from leads. A mean sea surface "
            "constructed from CryoSat-2 data is then subtracted from these heights to give "
            "sea level anomaly."
        ),
    },
    "warren_snow_depth": {
        "units": "meters",
        "comment": (
            "(Arctic Only): This is the snow depth model used to convert Arctic radar "
            "freeboard to true freeboard, and then true freeboard to ice thickness using a "
            "buoyancy calculation. It is based on a climatology built from in situ "
            "measurements of snow depth and snow density collected over multi-year ice in "
            "the central Arctic between 1954 and 1991 (Warren et al., 1999). The "
            "climatology comes in the form of monthly two-dimensional quadric functions "
            "which we average over the central Arctic to give a single mean monthly snow "
            "depth which is applied everywhere including outside the central Arctic. The "
            "monthly mean is halved over first year ice to account for the reduced "
            "accumulation compared with multi-year ice."
        ),
    },
    "unk_fraction": {
        "units": "percentage",
        "comment": (
            "Each radar echo from the earth's surface is classified as coming from open "
            "water, ice floes or the leads between floes based on properties of the echo. "
            "In addition, some echoes are unclassified. This is the fraction of each block "
            "or 20 echoes which are unclassified."
        ),
    },
    "floe_fraction": {
        "units": "percentage",
        "comment": (
            "Each radar echo from the earth's surface is classified as coming from open "
            "water, ice floes or the leads between floes based on properties of the echo. "
            "In addition, some echoes are unclassified. This is the fraction of each block "
            "or 20 echoes classified as floes."
        ),
    },
    "lead_fraction": {
        "units": "percentage",
        "comment": (
            "Each radar echo from the earth's surface is classified as coming from open "
            "water, ice floes or the leads between floes based on properties of the echo. "
            "In addition, some echoes are unclassified. This is the fraction of each block "
            "or 20 echoes classified as leads."
        ),
    },
}


def to_snake_case(name):
    """Convert PascalCase/camelCase parameter names to snake_case."""
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def process_map_file(latency, input_dir, output_dir, max_cog_dist, version):
    """Process a single map file to generate a NetCDF product.

    Args:
        latency (str): The latency of the sea ice thickness data (e.g., "02", "14", "28").
        input_dir (str): The directory containing the input map and info files.
        output_dir (str): The directory where the output NetCDF file will be saved.
        max_cog_dist (float): The maximum centre of gravity distance in km to
        filter out data points.
        version (int): The product version number.
    """
    map_filename = f"thk_{latency}.map"
    info_filename = f"thk_{latency}.info"

    map_path = os.path.join(input_dir, map_filename)
    info_path = os.path.join(input_dir, info_filename)

    if not os.path.exists(map_path):
        print(f"Warning: Map file not found: {map_path}")
        return
    if not os.path.exists(info_path):
        print(f"Warning: Info file not found: {info_path}")
        return

    # Read the info file to get dates
    with open(info_path, "r", encoding="utf-8") as f:
        line = f.readline().strip()
        parts = line.split()
        if len(parts) >= 6:
            start_d, start_m, start_y = int(parts[0]), int(parts[1]), int(parts[2])
            end_d, end_m, end_y = int(parts[3]), int(parts[4]), int(parts[5])
        else:
            print(f"Error parsing dates in {info_path}")
            return

    start_date = datetime(start_y, start_m, start_d)
    end_date = datetime(end_y, end_m, end_d)
    ndays = (end_date - start_date).days + 1

    # Format DDMMYYYY
    start_str = start_date.strftime("%d%m%Y")
    end_str = end_date.strftime("%d%m%Y")

    # Read the data file
    # Format: <latitude>  <longitude east>  <thickness in m>  <stdev> <numvals> <cog>
    df = pd.read_csv(
        map_path, sep=r"\s+", names=["lat", "lon", "thickness", "stdev", "numvals", "cog"]
    )

    # Filter out data where cog_dist is greater than the threshold
    initial_len = len(df)
    df = df[df["cog"] <= max_cog_dist]
    length = len(df)
    if initial_len != length:
        print(f"Filtered out {initial_len - length} points with cog_dist > {max_cog_dist}")

    # Output file path
    out_filename = (
        f"cpom_nrt_sea_ice_thickness_{latency}days_"
        f"{start_str}_{end_str}_5km_sparse_grid_v{version:03d}.nc"
    )
    out_path = os.path.join(output_dir, out_filename)

    os.makedirs(output_dir, exist_ok=True)

    print(f"Generating {out_path} for latency {latency} ({length} points)...")

    # Create NetCDF
    with Dataset(out_path, "w", format="NETCDF4") as nc:
        nc.Title = (
            "Near Real Time (NRT) Arctic Sea Ice Thickness Product"
            " from CryoSat-2 at 5km sparse grid resolution"
        )
        nc.Conventions = "CF-1.11"
        nc.featureType = "point"

        nc.history = (
            f"{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}"
            " - Created by gen_seaice_products.py"
        )
        nc.institution = "Centre for Polar Observation and Modelling (CPOM)"
        nc.source = "CryoSat-2 Altimetry"

        nc.createDimension("length", length)
        nc.createDimension("time", 1)

        # Helper to create scalar variables
        def create_scalar(name, datatype, units, value):
            var = nc.createVariable(name, datatype)
            var.units = units
            var[0] = value

        # Create scalar variables
        create_scalar("start_day", "i2", "day of month(1-31)", start_d)
        create_scalar("start_month", "i2", "month of year (1-12)", start_m)
        create_scalar("start_year", "i2", "years YYYY", start_y)

        create_scalar("end_day", "i2", "day of month (1-31)", end_d)
        create_scalar("end_month", "i2", "month of year (1-12)", end_m)
        create_scalar("end_year", "i2", "years YYYY", end_y)

        create_scalar("ndays", "i2", "days", ndays)
        create_scalar("grid_spacing", "i2", "km", 5)  # Assuming 5km, update if necessary

        # Create CF-compliant time variable
        mid_date = start_date + (end_date - start_date) / 2
        time_var = nc.createVariable("time", "f8", ("time",))
        time_var.units = "days since 1970-01-01 00:00:00"
        time_var.standard_name = "time"
        time_var.long_name = "time"
        time_var.calendar = "gregorian"
        time_var[0] = (mid_date - datetime(1970, 1, 1)).total_seconds() / 86400.0

        # Create array variables with compression level 5
        def create_array(name, datatype, units, standard_name, long_name, data, nc_ref=nc):
            var = nc_ref.createVariable(name, datatype, ("length",), zlib=True, complevel=5)
            var.units = units
            if standard_name:
                var.standard_name = standard_name
            if long_name:
                var.long_name = long_name
            var.coordinates = "time latitude longitude"
            var[:] = data
            return var

        lat_var = create_array(
            "latitude", "f4", "degrees_north", "latitude", "latitude", df["lat"].values
        )
        lat_var.coordinates = ""  # Dim variables should not have coordinates pointing to themselves
        lon_var = create_array(
            "longitude", "f4", "degrees_east", "longitude", "longitude", df["lon"].values
        )
        lon_var.coordinates = ""

        create_array(
            "thickness", "f4", "m", "sea_ice_thickness", "sea ice thickness", df["thickness"].values
        )
        create_array(
            "thk_stdev",
            "f4",
            "m",
            None,
            "standard deviation of thickness at location",
            df["stdev"].values,
        )
        create_array(
            "n_thk", "i4", "1", None, "number of thickness measurements used", df["numvals"].values
        )
        create_array(
            "cog_dist",
            "f4",
            "km",
            None,
            "distance of centre of gravity from operator centre",
            df["cog"].values,
        )

    print(f"Successfully created {out_path}")


def process_ntc_monthly(year, month, output_dir, max_cog_dist, process_arco, process_anto, version):
    """Process NTC monthly products combining base thickness and parameter maps."""
    date_str = f"{year:04d}{month:02d}"

    # Base input directory structure based on requirement
    base_dirs = {
        "arco": "/cpnet/altimetry/seaice/CS2/arco/archive",
        "anto": "/cpnet/altimetry/seaice/CS2/anto/archive",
    }

    # Determine which regions to process
    regions_to_process = []
    if not process_arco and not process_anto:
        # Default to both if neither is explicitly passed
        regions_to_process = ["arco", "anto"]
    else:
        if process_arco:
            regions_to_process.append("arco")
        if process_anto:
            regions_to_process.append("anto")

    for hemisphere_name in regions_to_process:
        base_dir = base_dirs[hemisphere_name]
        base_map = os.path.join(base_dir, f"{date_str}.map")
        if not os.path.exists(base_map):
            print(
                f"Skipping NTC monthly {date_str} for {hemisphere_name} - "
                f"base map missing: {base_map}"
            )
            continue

        print(f"Processing NTC monthly {date_str} for {hemisphere_name}...")

        # Read the base thickness data file
        # Format: <latitude>  <longitude east>  <thickness in m>  <stdev> <numvals> <cog>
        df = pd.read_csv(
            base_map, sep=r"\s+", names=["lat", "lon", "thickness", "stdev", "numvals", "cog"]
        )

        # Filter out data where cog_dist is greater than the threshold
        initial_len = len(df)
        df = df[df["cog"] <= max_cog_dist]
        length = len(df)
        if initial_len != length:
            print(f"Filtered out {initial_len - length} points with cog_dist > {max_cog_dist}")

        # Search for and merge parameter files
        merged_params = {}
        processed_params = []
        for param, nc_var_name in NTC_PARAMETERS.items():
            param_file = os.path.join(base_dir, f"{date_str}.{param}.map")
            if os.path.exists(param_file):
                print(f"  Found parameter: {param}")
                # Parameter files have the same strict format: lat, lon, value, stdev, numvals, cog
                # We do an inner join on lat/lon to ensure exact mapping of sparse grid points
                param_df = pd.read_csv(
                    param_file, sep=r"\s+", names=["lat", "lon", "value", "stdev", "numvals", "cog"]
                )

                # To merge efficiently, we round lat/lon slightly to account for floating point
                # Since the grid is identical, exact merge is preferred.
                param_df = param_df.set_index(["lat", "lon"])
                merged_params[nc_var_name] = param_df["value"]
                processed_params.append(nc_var_name)

        # Merge all into the main dataframe based on lat/lon
        df = df.set_index(["lat", "lon"])
        for var_name, series in merged_params.items():
            df[var_name] = series
        df = df.reset_index()

        # Missing values (e.g. from outer join mismatch, though should be exact)
        # will be filled with NaN
        df = df.dropna(subset=["thickness"])
        length = len(df)

        # Output file path
        out_filename = (
            f"cpom_ntc_{hemisphere_name}_sea_ice_thickness_"
            f"{date_str}_5km_sparse_grid_v{version:03d}.nc"
        )

        year_out_dir = os.path.join(output_dir, str(year))
        out_path = os.path.join(year_out_dir, out_filename)
        os.makedirs(year_out_dir, exist_ok=True)

        print(f"Generating NTC {out_path} ({length} points) with parameters: {processed_params}")

        # Create NetCDF
        with Dataset(out_path, "w", format="NETCDF4") as nc:
            nc.Title = (
                f"Non-Time Critical (NTC) Arctic Sea Ice Thickness Product"
                f" from CryoSat-2 at 5km sparse grid resolution for {date_str}"
            )
            nc.Conventions = "CF-1.11"
            nc.featureType = "point"

            nc.history = (
                f"{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}"
                " - Created by gen_seaice_products.py"
            )
            nc.institution = "Centre for Polar Observation and Modelling (CPOM)"
            nc.source = "CryoSat-2 Altimetry"

            nc.createDimension("length", length)
            nc.createDimension("time", 1)

            # Helper to create scalar variables
            def create_scalar(name, datatype, units, value, nc_ref=nc):
                var = nc_ref.createVariable(name, datatype)
                if units:
                    var.units = units
                var[0] = value

            # Create scalar variables
            create_scalar("year", "i2", "years YYYY", year)
            create_scalar("month", "i2", "month of year (1-12)", month)
            create_scalar("grid_spacing", "i2", "km", 5)

            # Since monthly, center time on the 15th of the month
            mid_date = datetime(year, month, 15)
            time_var = nc.createVariable("time", "f8", ("time",))
            time_var.units = "days since 1970-01-01 00:00:00"
            time_var.standard_name = "time"
            time_var.long_name = "time"
            time_var.calendar = "gregorian"
            time_var[0] = (mid_date - datetime(1970, 1, 1)).total_seconds() / 86400.0

            # Create array variables with compression level 5
            def create_array(
                name, datatype, units, standard_name, long_name, data, comment=None, nc_ref=nc
            ):
                var = nc_ref.createVariable(name, datatype, ("length",), zlib=True, complevel=5)
                if units:
                    var.units = units
                if standard_name:
                    var.standard_name = standard_name
                if long_name:
                    var.long_name = long_name
                if comment:
                    var.comment = comment
                var.coordinates = "time latitude longitude"

                # Replace NaNs with a valid fill value if needed, NetCDF handles NaNs well in floats
                var[:] = data
                return var

            lat_var = create_array(
                "latitude", "f4", "degrees_north", "latitude", "latitude", df["lat"].values
            )
            lat_var.coordinates = ""
            lon_var = create_array(
                "longitude", "f4", "degrees_east", "longitude", "longitude", df["lon"].values
            )
            lon_var.coordinates = ""

            create_array(
                "thickness",
                "f4",
                "m",
                "sea_ice_thickness",
                "sea ice thickness",
                df["thickness"].values,
            )
            create_array(
                "thk_stdev",
                "f4",
                "m",
                None,
                "standard deviation of thickness at location",
                df["stdev"].values,
            )
            create_array(
                "n_thk",
                "i4",
                "1",
                None,
                "number of thickness measurements used",
                df["numvals"].values,
            )
            create_array(
                "cog_dist",
                "f4",
                "km",
                None,
                "distance of centre of gravity from operator centre",
                df["cog"].values,
            )

            # Create variables for the merged secondary parameters
            for var_name in processed_params:
                raw_data = df[var_name].values
                metadata = NTC_PARAMETER_METADATA.get(var_name, {})
                units = metadata.get("units")
                comment = metadata.get("comment")
                create_array(
                    var_name,
                    "f4",
                    units,
                    None,
                    var_name.replace("_", " "),
                    raw_data,
                    comment=comment,
                )


def main():
    """Main function to process command line arguments and generate plots for sea ice parameters"""
    args = parse_args()

    if args.nrt:
        latencies = ["02", "14", "28"]
        for lat in latencies:
            process_map_file(lat, args.input_dir, args.output, args.max_cog_dist, args.version)
    elif args.ntc:
        tasks = []

        if args.latest:
            now = datetime.now(timezone.utc)
            for i in [
                3,
                2,
            ]:  # Process now - 3 months and 2 months
                m = now.month - i
                y = now.year
                if m < 1:
                    m += 12
                    y -= 1
                tasks.append((y, m, args.arco, args.anto))
        elif args.year is not None and args.month is None:
            do_arco = args.arco or (not args.arco and not args.anto)
            do_anto = args.anto or (not args.arco and not args.anto)

            if do_arco:
                for m in [1, 2, 3, 4, 5, 10, 11, 12]:
                    tasks.append((args.year, m, True, False))
            if do_anto:
                for m in range(1, 13):
                    tasks.append((args.year, m, False, True))
        elif args.year is not None and args.month is not None:
            tasks.append((args.year, args.month, args.arco, args.anto))
        else:
            print("Error: For NTC, provide --latest OR --year (and optionally --month).")
            sys.exit(1)

        for y, m, arco_flag, anto_flag in tasks:
            process_ntc_monthly(
                y, m, args.output, args.max_cog_dist, arco_flag, anto_flag, args.version
            )
    else:
        print("Please specify a processing mode, e.g. --nrt or --ntc")
        sys.exit(1)


if __name__ == "__main__":
    main()
