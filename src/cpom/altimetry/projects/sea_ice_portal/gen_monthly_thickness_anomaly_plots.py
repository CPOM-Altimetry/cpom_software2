#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cpom.altimetry.projects.sea_ice_portal.gen_monthly_thickness_anomaly_plots.py

# Purpose:
    Generate monthly thickness anomaly map plots for sea ice portal of the Arctic and
    Antarcic oceans

Author: Alan Muir (MSSL)
Date: 2026
Copyright: UCL/MSSL/CPOM

# Inputs:

1. YYYY MM : year and month for which anomalies are to be calculated

2. 5km resolution monthly sparsely gridded sea ice thickness files

/cpnet/altimetry/seaice/<CS2,S3A,S3B>/<arco,anto>/archive/YYYYMM.map  [Thickness]

# Outputs:

<output_dir>/<CS2,S3A,S3B>/<arco,anto>/archive/YYYYMM.anomalies.webp  [Thickness]

# Task

1. We need to read in all the monthly thickness files for a given mission and region
   We need a method to grid the data onto a regular grid as the input files are sparsely gridded
   We can use a grid specification for the Arctic of:
    EPSG: 3413
    minxm = -3850000.0
    minym = -5350000.0
    grid_x_size = 7600000.0
    grid_y_size = 11200000.0
    binsize = 5e3

   We could use a parquet file to store the gridded data for each mission and region
   for efficiency

2. We need to calculate the mean thickness for each grid cell over the period 2010-11 to YYYY-MM
3. We need to calculate the anomaly for each grid cell for each month
4. We need to plot the anomaly for each month using the Polarplot.plot_points() method
   and using plot area definitions 'arctic0_seaiceportal' for the Arctic and
   'antarctic0_seaiceportal' for the Antarctic
5. We need to save the anomaly for each month as a .webp file as per the file naming above.

For the plots we can use a similar method to that used in plot_seaice_param.py

Command line arguments:
    --mission: mission: one of cs2, s3a, s3b, env
    --south: [optional] process southern hemisphere instead of north
    --year: YYYY
    --month: [optional, default is to process all months 1-12] month number
    --outdir: Output base directory for plots
    --latest: [optional] process latest 2 months of data (only for monthly processing)

"""

import argparse
import calendar
import os
import sys
from datetime import datetime

import pandas as pd

from cpom.areas.area_plot import Annotation, Polarplot
from cpom.areas.areas import Area
from cpom.gridding.gridareas import GridArea

PLOT_SCALE_FACTOR = 0.02
DPI = 85
CMAP_NAME = (
    "RdBu_r"  # Standard diverging map for anomalies (red is thick/positive, blue is thin/negative)
)
CMAP_OVER_COLOR = "#800000"  # Dark red
CMAP_UNDER_COLOR = "#000080"  # Dark blue


def get_mission_longname(mission):
    """Return the human readable mission name."""
    if mission == "cs2":
        return "CryoSat-2"
    if mission == "s3a":
        return "Sentinel-3A"
    if mission == "s3b":
        return "Sentinel-3B"
    if mission == "env":
        return "Envisat"
    return mission.upper()


def save_anomaly_plot(
    args,
    year,
    month,
    archive_area,
    mission,
    mission_longname,
    lats,
    lons,
    vals,
    plot_type,
    baseline_desc,
    baseline_years="",
):
    """
    Common function to render and save an anomaly/difference plot.
    """
    # Plotting setup
    area_name = "antarctica_ocean_seaiceportal" if args.south else "arctic0_seaiceportal"

    min_plot_range = -1.0
    max_plot_range = 1.0
    if args.south:
        min_plot_range = -2.0
        max_plot_range = 2.0

    param_name = "Thickness Anomaly" if plot_type == "anomaly" else "Thickness Difference"

    dataset = {
        "lats": lats,
        "lons": lons,
        "vals": vals,
        "name": param_name,
        "units": "m",
        "plot_size_scale_factor": PLOT_SCALE_FACTOR,
        "apply_area_mask_to_data": not args.south,
        "min_plot_range": min_plot_range,
        "max_plot_range": max_plot_range,
        "cmap_name": CMAP_NAME,
        "cmap_over_color": CMAP_OVER_COLOR,
        "cmap_under_color": CMAP_UNDER_COLOR,
    }

    annotation_list = []
    annotation_list.append(
        Annotation(0.027, 0.915, "Parameter:", fontsize=10, fontweight="normal", color="grey")
    )
    annotation_list.append(
        Annotation(
            0.03,
            0.895,
            param_name,
            bbox={
                "boxstyle": "round",
                "facecolor": "aliceblue",
                "alpha": 1.0,
                "edgecolor": "lightgrey",
            },
            fontsize=18,
            fontweight="bold",
        )
    )

    annotation_list.append(
        Annotation(
            0.83, 0.01, "processed by CPOM, U.K.", fontsize=10, fontweight="normal", color="#000000"
        )
    )

    area_obj = Area(area_name)
    if args.south:
        annotation_list.append(
            Annotation(
                0.40 - 0.005 * (len(area_obj.long_name) - 6),
                0.89,
                area_obj.long_name,
                fontsize=15,
                fontweight="normal",
            )
        )
        annotation_list.append(
            Annotation(
                0.49 - 0.005 * (len(area_obj.long_name) - 6) + 0.06,
                0.89,
                "(5km grid)",
                fontsize=12,
                fontweight="normal",
            )
        )
    else:
        annotation_list.append(
            Annotation(
                0.40 - 0.005 * (len(area_obj.long_name) - 6),
                0.87,
                area_obj.long_name,
                fontsize=15,
                fontweight="normal",
            )
        )
        annotation_list.append(
            Annotation(
                0.40 - 0.005 * (len(area_obj.long_name) - 6) + 0.06,
                0.87,
                "(5km grid)",
                fontsize=12,
                fontweight="normal",
            )
        )

    annotation_list.append(
        Annotation(0.685, 0.96, f"Mission: {mission_longname}", fontsize=18, fontweight="bold")
    )
    annotation_list.append(
        Annotation(0.685, 0.92, "Latency: Final, Precise Orbit", fontsize=14, fontweight="normal")
    )
    annotation_list.append(Annotation(0.685, 0.86, "Month:", fontsize=12, fontweight="normal"))
    annotation_list.append(
        Annotation(
            0.765,
            0.85,
            f"{month:02d}/{year}",
            fontsize=34,
            fontweight="bold",
            bbox={
                "boxstyle": "round",
                "facecolor": "white",
                "alpha": 1.0,
                "edgecolor": "lightgrey",
            },
        )
    )

    if plot_type == "anomaly":
        header_text = "Monthly Mean Anomaly"
    elif plot_type == "prev_year":
        header_text = "Annual Month Difference"
    elif plot_type == "prev_month":
        header_text = "Diff. to Previous Month"
    else:
        header_text = "Monthly Difference"
    full_baseline_text = (
        f"({baseline_desc} {baseline_years})" if baseline_years else f"({baseline_desc})"
    )
    if args.south:
        annotation_list.append(
            Annotation(0.023, 0.86, header_text, fontsize=18, fontweight="normal")
        )
        if plot_type == "anomaly":
            annotation_list.append(
                Annotation(0.023, 0.83, f"({baseline_desc}", fontsize=12, fontweight="normal")
            )
            annotation_list.append(
                Annotation(0.023, 0.81, f"{baseline_years})", fontsize=12, fontweight="normal")
            )
        else:
            annotation_list.append(
                Annotation(0.023, 0.83, full_baseline_text, fontsize=12, fontweight="normal")
            )
    else:
        annotation_list.append(
            Annotation(0.023, 0.84, header_text, fontsize=18, fontweight="normal")
        )
        if plot_type == "anomaly":
            annotation_list.append(
                Annotation(0.023, 0.81, f"({baseline_desc}", fontsize=12, fontweight="normal")
            )
            annotation_list.append(
                Annotation(0.023, 0.78, f"{baseline_years})", fontsize=12, fontweight="normal")
            )
        else:
            annotation_list.append(
                Annotation(0.023, 0.81, full_baseline_text, fontsize=12, fontweight="normal")
            )

    # Add reference annotations common to plot_seaice_param
    if not args.south:
        annotation_list.append(
            Annotation(0.388, 0.310, "76°N", fontsize=9, fontweight="normal", color="#626262")
        )
        annotation_list.append(
            Annotation(0.37, 0.212, "68°N", fontsize=9, fontweight="normal", color="#626262")
        )
        annotation_list.append(
            Annotation(0.354, 0.115, "60°N", fontsize=9, fontweight="normal", color="#626262")
        )
    else:
        annotation_list.append(
            Annotation(0.41, 0.41, "80°S", fontsize=9, fontweight="normal", color="#626262")
        )
        annotation_list.append(
            Annotation(0.388, 0.295, "70°S", fontsize=9, fontweight="normal", color="#626262")
        )
        annotation_list.append(
            Annotation(0.37, 0.195, "60°S", fontsize=9, fontweight="normal", color="#626262")
        )

    # Output directory architecture
    archive_outdir = os.path.join(args.outdir, mission.lower(), "ntc", archive_area, str(year))
    os.makedirs(archive_outdir, exist_ok=True)

    if plot_type == "anomaly":
        output_file = f"{mission}_{archive_area}_{year}{month:02d}_thickness_anomaly_grid"
    elif plot_type == "prev_year":
        output_file = f"{mission}_{archive_area}_{year}{month:02d}_thickness_diff_prev_year_grid"
    elif plot_type == "prev_month":
        output_file = f"{mission}_{archive_area}_{year}{month:02d}_thickness_diff_prev_month_grid"
    else:
        output_file = f"{mission}_{archive_area}_{year}{month:02d}_thickness_diff_grid"

    Polarplot(area_name).plot_points(
        dataset,
        output_dir=archive_outdir,
        output_file=output_file,
        annotation_list=annotation_list,
        use_default_annotation=False,
        figure_height=12,
        figure_width=12,
        image_format="webp",
        dpi=DPI,
    )
    print(f"Saved {plot_type} plot to {os.path.join(archive_outdir, output_file)}.webp")


def process_month(args, year, month, cache_df, grid_area, archive_area, mission, mission_longname):
    """Calculate the anomaly and save the plot types for a specific month."""
    if cache_df is None or cache_df.empty:
        print("Empty cache, nothing to plot.")
        return

    # 1. Prepare Target Month Data
    target_month_df = cache_df[(cache_df["year"] == year) & (cache_df["month"] == month)]
    if target_month_df.empty:
        print(f"No data available for the target month {year}-{month:02d}.")
        return

    # Helper to calculate differences
    def get_lat_lon_diff(current_df, baseline_df_full):
        # Calculate mean thickness for the baseline period
        mean_baseline = (
            baseline_df_full.groupby(["x_bin", "y_bin"])["thickness"].mean().reset_index()
        )
        mean_baseline.rename(columns={"thickness": "mean_thickness"}, inplace=True)

        merged = pd.merge(current_df, mean_baseline, on=["x_bin", "y_bin"], how="inner")
        merged["diff"] = merged["thickness"] - merged["mean_thickness"]

        lats, lons = grid_area.get_cellcentre_lat_lon_from_col_row(
            col=merged["x_bin"].values, row=merged["y_bin"].values
        )
        return lats, lons, merged["diff"].values

    # --- PLOT A: Long-term Anomaly (2011 to latest) ---
    month_cache = cache_df[cache_df["month"] == month]
    baseline_lt = month_cache[month_cache["year"] >= 2011]
    if not baseline_lt.empty:
        end_year_baseline = baseline_lt["year"].max()
        lats, lons, vals = get_lat_lon_diff(target_month_df, baseline_lt)
        save_anomaly_plot(
            args,
            year,
            month,
            archive_area,
            mission,
            mission_longname,
            lats,
            lons,
            vals,
            "anomaly",
            baseline_desc=f"Diff. to {calendar.month_name[month]} mean",
            baseline_years=f"2011-{end_year_baseline}",
        )

    # --- PLOT B: Difference to Previous Year ---
    baseline_py = month_cache[month_cache["year"] == (year - 1)]
    if not baseline_py.empty:
        lats, lons, vals = get_lat_lon_diff(target_month_df, baseline_py)
        save_anomaly_plot(
            args,
            year,
            month,
            archive_area,
            mission,
            mission_longname,
            lats,
            lons,
            vals,
            "prev_year",
            baseline_desc=f"Diff. to {calendar.month_name[month]}",
            baseline_years=f"{year - 1}",
        )
    else:
        print(f"No data for previous year ({year - 1}) to calculate difference.")

    # --- PLOT C: Difference to Previous Month ---
    if month == 1:
        prev_month, prev_year = 12, year - 1
    else:
        prev_month, prev_year = month - 1, year

    baseline_pm = cache_df[(cache_df["year"] == prev_year) & (cache_df["month"] == prev_month)]
    if not baseline_pm.empty:
        lats, lons, vals = get_lat_lon_diff(target_month_df, baseline_pm)
        save_anomaly_plot(
            args,
            year,
            month,
            archive_area,
            mission,
            mission_longname,
            lats,
            lons,
            vals,
            "prev_month",
            baseline_desc=f"Diff. to {calendar.month_name[prev_month]}",
            baseline_years=f"{prev_year}",
        )
    else:
        print(f"No data for previous month ({prev_year}-{prev_month:02d}) to calculate difference.")


def update_cache(args, grid_area, archive_area, mission):
    """Build or update a parquet cache of gridded monthly thicknesses."""
    cache_dir = os.path.join(args.outdir, mission, archive_area)
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "thickness_grid_cache.parquet")

    if os.path.exists(cache_path):
        cache_df = pd.read_parquet(cache_path)
        existing_months = set(zip(cache_df["year"], cache_df["month"]))
    else:
        cache_df = pd.DataFrame(columns=["year", "month", "x_bin", "y_bin", "thickness", "count"])
        existing_months = set()

    # Scan the archive directory to find the actual latest year and month
    archive_dir = f"/cpnet/altimetry/seaice/{mission.upper()}/{archive_area}/archive/"
    latest_year = 2010
    latest_month = 12

    if os.path.exists(archive_dir):
        map_files = [f for f in os.listdir(archive_dir) if f.endswith(".map")]
        if map_files:
            years_months = []
            for f in map_files:
                try:
                    ym = int(f.replace(".map", ""))
                    years_months.append((ym // 100, ym % 100))
                except ValueError:
                    continue
            if years_months:
                latest_year, latest_month = max(years_months)

    # Generate list of (year, month) from Nov 2010 to latest available data
    start_year = 2010
    start_month = 11
    end_year = latest_year
    end_month = latest_month

    new_data_frames = []

    for y in range(start_year, end_year + 1):
        m_start = start_month if y == start_year else 1
        m_end = end_month if y == end_year else 12
        for m in range(m_start, m_end + 1):
            if (y, m) in existing_months:
                continue

            map_file = (
                f"/cpnet/altimetry/seaice/{mission.upper()}/{archive_area}/archive/{y}{m:02d}.map"
            )
            if not os.path.exists(map_file):
                continue

            print(f"Reading and gridding {map_file} for cache...")
            try:
                pd_data = pd.read_csv(map_file, sep=r"\s+")
            except pd.errors.EmptyDataError:
                continue
            except FileNotFoundError:
                continue

            if pd_data.empty:
                continue

            pd_data.columns = ["lat", "lon", "thickness", "stdev", "numvals", "dist"]
            # Apply required distortion / dist filter
            pd_data = pd_data[pd_data["dist"] < 15]

            if pd_data.empty:
                continue

            # Convert lat/lon to grid x,y coordinates
            x_coords, y_coords = grid_area.transform_lat_lon_to_x_y(
                pd_data["lat"].values, pd_data["lon"].values
            )
            x_bin, y_bin = grid_area.get_col_row_from_x_y(x_coords, y_coords)

            month_df = pd.DataFrame(
                {"x_bin": x_bin, "y_bin": y_bin, "thickness": pd_data["thickness"].values}
            )

            # Average multiple data points falling into the same grid cell
            gridded = (
                month_df.groupby(["x_bin", "y_bin"])
                .agg(thickness=("thickness", "mean"), count=("thickness", "count"))
                .reset_index()
            )

            gridded["year"] = y
            gridded["month"] = m

            new_data_frames.append(gridded)

    if new_data_frames:
        cache_df = pd.concat([cache_df] + new_data_frames, ignore_index=True)
        # Ensure optimal types for parquet
        cache_df["year"] = cache_df["year"].astype(int)
        cache_df["month"] = cache_df["month"].astype(int)
        cache_df["x_bin"] = cache_df["x_bin"].astype(int)
        cache_df["y_bin"] = cache_df["y_bin"].astype(int)

        cache_df.to_parquet(cache_path)
        print(f"Updated cache with {len(new_data_frames)} new months.")

    return cache_df


def main():
    """Main function to process command line arguments and
    generate plots for sea ice thickness anomaly"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--mission", "-m", help="mission: one of cs2, s3a, s3b, env")
    parser.add_argument(
        "--south",
        "-s",
        help="[optional] process southern hemisphere instead of north",
        action="store_true",
    )
    parser.add_argument("--year", "-y", help="YYYY", type=int)
    parser.add_argument(
        "--month",
        "-mo",
        help="[optional, default is to process all months 1-12] month number",
        type=int,
    )
    parser.add_argument("--outdir", "-o", help="Output base directory for plots", required=True)
    parser.add_argument(
        "--latest", "-la", help="[optional] process latest 2 months of data", action="store_true"
    )

    args = parser.parse_args()

    if not args.mission:
        sys.exit("--mission missing")

    mission = args.mission.lower()
    mission_longname = get_mission_longname(mission)

    if args.latest:
        if args.year or args.month:
            sys.exit("Error: cannot use --latest with --year or --month")
        now = datetime.now()
        args.year = now.year
        if now.month <= 2:
            args.month = 11
            args.year -= 1
        else:
            args.month = now.month - 2

    if not args.year:
        sys.exit("Must include --year or --latest")

    archive_area = "anto" if args.south else "arco"
    grid_area_name = "antarctic_ocean" if args.south else "arctic"

    # Initialize grid area with 5km binsize matching instructions
    grid_area = GridArea(grid_area_name, binsize=5000)

    # 1. Update cache by reading all available historical map files
    cache_df = update_cache(args, grid_area, archive_area, mission)

    # 2. Process requested month(s)
    months_to_process = [args.month] if args.month else list(range(1, 13))

    for m in months_to_process:
        process_month(
            args, args.year, m, cache_df, grid_area, archive_area, mission, mission_longname
        )


if __name__ == "__main__":
    main()
