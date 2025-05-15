#!/usr/bin/env python3
"""
Plot distances between latitude/longitude points in two NetCDF files,
with summary statistics in the plot footer, optional plot range, and histograms below the main plot.

Each file must contain 1D latitude and longitude variables of the same length.
Variables may live in nested groups, referenced by slash-separated paths.

Example:
    python distance_between_netcdf_points.py \
        --file1 data1.nc --file2 data2.nc \
        --lat1 group1/latitude --lon1 group1/longitude \
        --lat2 lat_var --lon2 lon_var \
        --plot-range 0 1000 --hist-bins 30
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
    Compute haversine distance (m) between points or arrays of points.
    Inputs in degrees; output in meters.
    """
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return EARTH_RADIUS * c


def main():
    """main tool function"""
    parser = argparse.ArgumentParser(
        description=(
            "Plot point-to-point distances between two NetCDF datasets "
            "with stats and histograms."
        )
    )
    parser.add_argument("--file1", required=True, help="First NetCDF file path")
    parser.add_argument("--file2", required=True, help="Second NetCDF file path")
    parser.add_argument("--lat1", required=True, help="Latitude variable path in first file")
    parser.add_argument("--lon1", required=True, help="Longitude variable path in first file")
    parser.add_argument("--lat2", required=True, help="Latitude variable path in second file")
    parser.add_argument("--lon2", required=True, help="Longitude variable path in second file")
    parser.add_argument(
        "--plot-range",
        nargs=2,
        type=float,
        metavar=("MIN", "MAX"),
        help="Y-axis limits for the distance plot",
    )
    parser.add_argument("--hist-bins", type=int, default=50, help="Number of bins for histograms")
    args = parser.parse_args()

    # Load data
    ds1, ds2 = Dataset(args.file1), Dataset(args.file2)
    lat1 = get_variable(ds1, args.lat1)
    lon1 = get_variable(ds1, args.lon1)
    lat2 = get_variable(ds2, args.lat2)
    lon2 = get_variable(ds2, args.lon2)
    ds1.close()
    ds2.close()

    if lat1.shape != lat2.shape or lon1.shape != lon2.shape:
        raise ValueError("Latitude/longitude arrays must have the same shape in both files.")

    distances = haversine(lat1, lon1, lat2, lon2)

    # Compute basic stats
    min_dist = distances.min()
    max_dist = distances.max()
    mean_dist = distances.mean()
    std_dist = distances.std()
    median_dist = np.median(distances)
    mad_dist = np.median(np.abs(distances - median_dist))

    # Prepare trimmed stats
    sorted_dist = np.sort(distances)
    k = int(distances.size * 0.05)
    upper_trimmed = sorted_dist[:-k] if k > 0 else distances
    trim_mean = upper_trimmed.mean()
    trim_std = upper_trimmed.std()

    # Determine subplot layout: 1 main + full-hist + optional range-hist
    n_hist = 1 + (1 if args.plot_range else 0)
    nrows = 1 + n_hist
    fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(12, 4 * nrows))

    # Ensure axes is iterable
    if nrows == 1:
        axes = [axes]

    # Main line plot
    ax_main = axes[0]
    ax_main.plot(distances)
    ax_main.set_xlabel("Index")
    ax_main.set_ylabel("Distance (m)")
    ax_main.set_title("Point-to-point distances between NetCDF files")
    ax_main.grid(True)
    if args.plot_range:
        ax_main.set_ylim(args.plot_range)

    # Histograms
    ax_hist_full = axes[1]
    ax_hist_full.hist(distances, bins=args.hist_bins)
    ax_hist_full.set_title("Histogram of distances (full range)")
    ax_hist_full.set_xlabel("Distance (m)")
    ax_hist_full.set_ylabel("Frequency")

    if args.plot_range:
        ax_hist_range = axes[2]
        lo, hi = args.plot_range
        mask = (distances >= lo) & (distances <= hi)
        ax_hist_range.hist(distances[mask], bins=args.hist_bins)
        ax_hist_range.set_title(f"Histogram of distances within [{lo}, {hi}] m")
        ax_hist_range.set_xlabel("Distance (m)")
        ax_hist_range.set_ylabel("Frequency")

    # Footer and stats lines: reserve bottom area for text
    plt.tight_layout(rect=[0, 0.15, 1, 0.95])

    # First line: trimmed stats footer
    footer = (
        f"Mean={mean_dist:.2f}m, Std={std_dist:.2f}m; "
        f"Median={median_dist:.2f}m, MAD={mad_dist:.2f}m; "
        f"Upper5%TrimMean={trim_mean:.2f}m, Upper5%TrimStd={trim_std:.2f}m"
    )
    fig.text(0.5, 0.12, footer, ha="center", va="bottom", fontsize="small")

    # Second line: basic stats
    stats_line = (
        f"Min={min_dist:.2f}m, Max={max_dist:.2f}m; "
        f"Mean={mean_dist:.2f}m, Median={median_dist:.2f}m, Std={std_dist:.2f}m"
    )
    fig.text(0.5, 0.06, stats_line, ha="center", va="bottom", fontsize="small")

    plt.show()


if __name__ == "__main__":
    main()
