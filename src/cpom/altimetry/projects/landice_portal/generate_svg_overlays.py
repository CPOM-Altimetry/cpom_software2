#!/usr/bin/env python3
"""
generate_svg_overlays.py

Tool to generate SVG polygons for Antarctic glaciological basins
aligned perfectly with the 1500x1500 map plots produced by plot_multi_mission_dhdt.py.
"""

import argparse
import glob
import os
import re

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np

from cpom.areas.area_plot import Polarplot


# noqa
# pylint: disable=too-many-locals, too-many-statements, too-many-branches,
# pylint: disable=broad-exception-caught,unspecified-encoding,missing-function-docstring
def get_basin_number(filename: str) -> int:
    """Extract basin number from the filename."""
    match = re.search(r"basin(\d+)_latlon\.csv", filename)
    if match:
        return int(match.group(1))
    return 0


def generate_svg(points_str: str, basin_id: int, svg_width: int, svg_height: int) -> str:
    """Generate the SVG file content."""
    svg = (
        f'<svg viewBox="0 0 {svg_width} {svg_height}" '
        f'xmlns="http://www.w3.org/2000/svg" id="basin{basin_id}">\n'
        f'  <a xlink:href="" class="map-highlight">\n'
        f'    <polygon points="{points_str}" fill="none" stroke="red" stroke-width="3" />\n'
        f"    <title>Basin {basin_id}</title>\n"
        f"  </a>\n"
        f"</svg>\n"
    )
    return svg


def main():
    parser = argparse.ArgumentParser(description="Generate SVG polygon overlays for map plots.")
    parser.add_argument(
        "--input_dir",
        "-i",
        default="polygons/antarctica",
        help="Directory containing the rignot2016_basin<N>_latlon.csv files",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        default="polygons/antarctica",
        help="Directory to write the output SVG files",
    )
    parser.add_argument(
        "--area",
        "-a",
        default="antarctica_cpom_portal",
        help="Area projection to use",
    )

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    csv_files = glob.glob(os.path.join(args.input_dir, "rignot2016_basin*_latlon.csv"))
    if not csv_files:
        print(f"No basin csv files found in {args.input_dir}")
        return

    # Setup plot exactly as plot_multi_mission_dhdt.py
    # dpi=150, figsize=(10,10) means the output image is 1500x1500 pixels
    fig_width = 10
    fig_height = 10
    dpi = 150
    pixel_width = int(fig_width * dpi)
    pixel_height = int(fig_height * dpi)

    area_overrides = {"show_bad_data_map": False, "flag_perc_axis": (0.8, 0.1, 0.05)}

    print(f"Setting up projection for area {args.area}...")
    pp = Polarplot(args.area, area_overrides)
    fig = plt.figure(figsize=(fig_width, fig_height))
    fig.set_dpi(dpi)

    ax, dataprj, _ = pp.setup_projection_and_extent(pp.thisarea.simple_axes, draw_axis_frame=False)
    # Important: Draw the canvas so the transforms are properly initialized
    fig.canvas.draw()

    for csv_file in csv_files:
        basin_num = get_basin_number(csv_file)
        if basin_num == 0:
            continue

        print(f"Processing basin {basin_num} from {csv_file}...")

        # Read the csv file. Skipping the header row.
        # Format: latitude,longitude
        try:
            data = np.loadtxt(csv_file, delimiter=",", skiprows=1)
            lats = data[:, 0]
            lons = data[:, 1]
        except Exception as e:
            print(f"Failed to read {csv_file}: {e}")
            continue

        svg_points = []
        for lat, lon in zip(lats, lons):
            # 1. Transform lat/lon to map projection coordinates
            # Note: cartopy expects (lon, lat)
            x_proj, y_proj = dataprj.transform_point(lon, lat, src_crs=ccrs.PlateCarree())

            # 2. Transform map projection coordinates to display pixels
            px, py = ax.transData.transform((x_proj, y_proj))

            # 3. Invert Y coordinate for SVG space (which grows downwards)
            svg_y = pixel_height - py

            # Formatting to 1 decimal place to save space
            svg_points.append(f"{px:.1f},{svg_y:.1f}")

        points_str = " ".join(svg_points)

        # Write individual SVG
        svg_content = generate_svg(points_str, basin_num, pixel_width, pixel_height)
        svg_filename = os.path.join(args.output_dir, f"rignot2016_basin{basin_num}_overlay.svg")

        with open(svg_filename, "w") as f:
            f.write(svg_content)

        print(f"Wrote {svg_filename}")


if __name__ == "__main__":
    main()
