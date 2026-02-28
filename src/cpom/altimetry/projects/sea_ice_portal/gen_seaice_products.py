"""Generate CPOM sea ice thickness NetCDF products from input ascii .map files."""

import argparse
import os
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
    parser.add_argument("-o", "--output", required=True, help="Path to output directory.")
    parser.add_argument(
        "--max-cog-dist",
        type=float,
        default=15.0,
        help="Maximum COG distance in km (default: 15.0). Data points exceeding this are excluded.",
    )
    # For testing and flexibility
    parser.add_argument(
        "--input-dir",
        default="/cpnet/altimetry/seaice/nrt_system/latest",
        help="Override input directory (default: /cpnet/altimetry/seaice/nrt_system/latest).",
    )
    return parser.parse_args()


def process_map_file(latency, input_dir, output_dir, max_cog_dist):
    """Process a single map file to generate a NetCDF product.

    Args:
        latency (str): The latency of the sea ice thickness data (e.g., "02", "14", "28").
        input_dir (str): The directory containing the input map and info files.
        output_dir (str): The directory where the output NetCDF file will be saved.
        max_cog_dist (float): The maximum centre of gravity distance in km to
        filter out data points.
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
        f"cpom_nrt_sea_ice_thickness_{latency}days_{start_str}_{end_str}_5km_sparse_grid.nc"
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
        def create_array(name, datatype, units, standard_name, long_name, data):
            var = nc.createVariable(name, datatype, ("length",), zlib=True, complevel=5)
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


def main():
    """Main function to process command line arguments and generate plots for sea ice parameters"""
    args = parse_args()

    if args.nrt:
        latencies = ["02", "14", "28"]
        for lat in latencies:
            process_map_file(lat, args.input_dir, args.output, args.max_cog_dist)
    else:
        print("Please specify a processing mode, e.g. --nrt")
        sys.exit(1)


if __name__ == "__main__":
    main()
