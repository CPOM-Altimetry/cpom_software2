#!/usr/bin/env python3
"""
Plot distances between latitude/longitude points in two NetCDF files,
with summary statistics in the plot footer.

Each file must contain 1D latitude and longitude variables of the same length.
Variables may live in nested groups, referenced by slash-separated paths.

Example:
    python distance_between_netcdf_points.py \
        --file1 data1.nc --file2 data2.nc \
        --lat1 group1/latitude --lon1 group1/longitude \
        --lat2 lat_var --lon2 lon_var
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset  # pylint: disable=no-name-in-module

# Earth radius in meters
EARTH_RADIUS = 6371000.0


def get_variable(dataset: Dataset, path: str) -> np.ndarray:
    """
    Retrieve a variable array from a NetCDF dataset, given a slash-delimited path.
    """
    parts = path.split("/")
    var_name = parts[-1]
    group = dataset
    # Traverse group hierarchy
    for p in parts[:-1]:
        if p in group.groups:
            group = group.groups[p]
        else:
            raise KeyError(f"Group '{p}' not found in dataset")
    if var_name not in group.variables:
        raise KeyError(f"Variable '{var_name}' not found in group '{'/'.join(parts[:-1]) or '/'}'")
    return group.variables[var_name][:]


def haversine(lat1, lon1, lat2, lon2):
    """
    Compute haversine distance (in meters) between two points or arrays of points.
    Inputs are in degrees; output is in meters.
    """
    # Convert to radians
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)

    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return EARTH_RADIUS * c


def main():
    """main tool function"""
    parser = argparse.ArgumentParser(
        description="Plot point-to-point distances between two NetCDF datasets, with summary stats."
    )
    parser.add_argument("--file1", required=True, help="First NetCDF file path")
    parser.add_argument("--file2", required=True, help="Second NetCDF file path")
    parser.add_argument("--lat1", required=True, help="Latitude variable path in first file")
    parser.add_argument("--lon1", required=True, help="Longitude variable path in first file")
    parser.add_argument("--lat2", required=True, help="Latitude variable path in second file")
    parser.add_argument("--lon2", required=True, help="Longitude variable path in second file")
    args = parser.parse_args()

    # Open datasets
    ds1 = Dataset(args.file1, mode="r")
    ds2 = Dataset(args.file2, mode="r")

    # Load variables
    lat1 = get_variable(ds1, args.lat1)
    lon1 = get_variable(ds1, args.lon1)
    lat2 = get_variable(ds2, args.lat2)
    lon2 = get_variable(ds2, args.lon2)

    ds1.close()
    ds2.close()

    # Check shapes
    if lat1.shape != lat2.shape or lon1.shape != lon2.shape:
        raise ValueError("Latitude/longitude arrays must have the same shape in both files.")

    # Compute distances
    distances = haversine(lat1, lon1, lat2, lon2)

    # Plot distances with larger default width
    plt.figure(figsize=(12, 6))
    plt.plot(distances)
    plt.xlabel("Index")
    plt.ylabel("Distance (m)")
    plt.title("Point-to-point distances between NetCDF files")
    plt.grid(True)

    # Compute summary statistics
    mean_dist = distances.mean()
    std_dist = distances.std()
    median_dist = np.median(distances)
    # Median Absolute Deviation
    mad_dist = np.median(np.abs(distances - median_dist))
    # Upper 5% trimmed mean & std (remove top 5% largest values)
    n = distances.size
    k = int(n * 0.05)
    if k > 0:
        sorted_dist = np.sort(distances)
        upper_trimmed = sorted_dist[:-k]
    else:
        upper_trimmed = distances
    trim_mean = upper_trimmed.mean()
    trim_std = upper_trimmed.std()

    # Create footer text
    footer = (
        f"Mean = {mean_dist:.2f} m, Std = {std_dist:.2f} m; "
        f"Median = {median_dist:.2f} m, MAD = {mad_dist:.2f} m; "
        f"Upper 5% Trimmed Mean = {trim_mean:.2f} m, "
        f"Upper 5% Trimmed Std = {trim_std:.2f} m"
    )

    # Adjust layout and add footer
    fig = plt.gcf()
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.25)
    fig.text(0.5, 0.01, footer, ha="center", va="bottom", fontsize="small")

    plt.show()


if __name__ == "__main__":
    main()
