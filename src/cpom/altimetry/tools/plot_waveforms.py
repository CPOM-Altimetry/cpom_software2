#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""cpom.altimetry.tools.plot_waveforms.py

# Tool to plot altimetry waveforms. Directly supports:

CRISTAL L1b
CS2 L1b

+ any NetCDF file that has selectable latitude, longitude and waveform parameters

"""

import argparse
import os
import sys

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
from cartopy.mpl.geoaxes import GeoAxes
from matplotlib.widgets import Button
from netCDF4 import Dataset, Variable  # pylint: disable=no-name-in-module

# Define functions outside main()

# pylint: disable=too-many-statements
# pylint: disable=too-many-arguments
# pylint: disable=too-many-locals
# pylint: disable=too-many-branches


def get_scaled_font_size(file_name: str, max_len: int = 50, base_size: int = 16) -> int:
    """Calculate a dynamically scaled font size based on the length of the file name.

    Args:
        file_name (str): The file name to scale the font size for.
        max_len (int, optional): The maximum length for scaling. Defaults to 50.
        base_size (int, optional): The base font size. Defaults to 16.

    Returns:
        int: The calculated font size.
    """
    if len(file_name) > max_len:
        scale_factor = max_len / len(file_name)
        return max(int(base_size * scale_factor), 10)  # Set a minimum font size of 10

    return base_size


def get_projection_and_extent(lat: float) -> tuple[ccrs.Projection, list[float] | None]:
    """Determine the appropriate Cartopy projection and extent based on latitude.

    Args:
        lat (float): The latitude to determine the projection for.

    Returns:
        tuple[ccrs.Projection, list[float] | None]: The Cartopy projection and the extent.
    """
    if -50 <= lat <= 50:
        return ccrs.Mercator(), [-179.9, 179.9, -90, 90]
    if lat > 50:
        return ccrs.Stereographic(central_latitude=90, central_longitude=0), [-180, 180, 50, 90]

    return ccrs.Stereographic(central_latitude=-90, central_longitude=0), [-180, 180, -90, -50]


def update_map(
    ax: GeoAxes,
    current_lat: float,
    current_lon: float,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    extent: list[float] | None = None,
) -> None:
    """Update the map with the track and the current location.

    Args:
        ax (plt.Axes): The map axis to update.
        current_lat (float): The current latitude of the waveform.
        current_lon (float): The current longitude of the waveform.
        latitudes (np.ndarray): Array of latitude values for the track.
        longitudes (np.ndarray): Array of longitude values for the track.
        extent (list[float] | None, optional): The extent for the map projection. Defaults to None.
    """
    ax.clear()
    _ = ax.set_extent(extent, crs=ccrs.PlateCarree()) if extent else None

    ax.set_title("Position of Track (blue) & Waveform (red)")

    # Add land with light grey color
    land_feature = cfeature.NaturalEarthFeature(
        category="physical", name="land", scale="50m", facecolor="lightgrey"
    )
    ax.add_feature(land_feature)
    ax.coastlines()
    ax.gridlines()

    if extent:
        lat_filter = (latitudes >= extent[2]) & (latitudes <= extent[3])
        lon_filtered = longitudes[lat_filter]
        lat_filtered = latitudes[lat_filter]
    else:
        lon_filtered = longitudes
        lat_filtered = latitudes

    ax.scatter(
        lon_filtered, lat_filtered, color="blue", s=10, zorder=998, transform=ccrs.PlateCarree()
    )
    ax.scatter(
        current_lon, current_lat, color="red", s=50, zorder=999, transform=ccrs.PlateCarree()
    )


def update_plot(
    index: int,
    ax_waveform: plt.Axes,
    ax_map: GeoAxes,
    waveforms: np.ndarray,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    file_name: str,
    wf_name: str,
    fig: plt.Figure,
) -> None:
    """Update the waveform and map plots.

    Args:
        index (int): The current index of the waveform.
        ax_waveform (plt.Axes): The waveform plot axis.
        ax_map (plt.Axes): The map plot axis.
        waveforms (np.ndarray): Array of waveform data.
        latitudes (np.ndarray): Array of latitude values.
        longitudes (np.ndarray): Array of longitude values.
        file_name (str): The name of the file being displayed.
        wf_name (str): name of the waveform parameter being plotted
        fig (plt.Figure): The figure object for the plot.
    """
    ax_waveform.cla()

    file_name_font_size = get_scaled_font_size(file_name)
    ax_waveform.set_title(f"{file_name}", loc="center", fontsize=file_name_font_size, pad=30)
    ax_waveform.text(
        0.5,
        1.02,
        f"Waveform {index + 1}/{len(waveforms)}",
        transform=ax_waveform.transAxes,
        fontsize=12,
        ha="center",
    )

    ax_waveform.plot(waveforms[index], color="blue")
    ax_waveform.fill_between(
        np.arange(len(waveforms[index])), waveforms[index], color="lightblue", alpha=0.3
    )
    ax_waveform.set_xlabel("Sample Index")
    ax_waveform.set_ylabel(wf_name)
    ax_waveform.grid(True)

    current_lat = latitudes[index]
    current_lon = longitudes[index]
    projection, extent = get_projection_and_extent(current_lat)

    ax_map.projection = projection
    update_map(ax_map, current_lat, current_lon, latitudes, longitudes, extent)

    fig.canvas.draw_idle()


def get_default_parameter_names(
    file_name: str, waveform_parameter_name: str | None, latname: str | None, lonname: str | None
) -> tuple[str, str, str]:
    """Get default parameter names for specific file types if not provided.

    Args:
        file_name (str): The name of the file being processed.
        waveform_parameter_name (str | None): The waveform parameter name.
        latname (str | None): The latitude parameter name.
        lonname (str | None): The longitude parameter name.

    Returns:
        tuple[str, str, str,str]: The waveform parameter name,
        latitude name,longitude name.
    """

    wf_parameter = waveform_parameter_name
    latitude_name = latname
    longitude_name = lonname

    # Check if it is a CRISTAL L1b file in HR mode
    if "CRA_IR_1B_HR__" in file_name[: len("CRA_IR_1B_HR__")]:
        print("CRISTAL l1b (CRA_IR_1B_HR) identified")
        if waveform_parameter_name is None:
            wf_parameter = "data/ku/power_waveform_comb"
        if latname is None:
            latitude_name = "data/ku/latitude"
        if lonname is None:
            longitude_name = "data/ku/longitude"
    # Check if it is a CRISTAL L1b file in LR mode
    elif "CRA_IR_1B_LR__" in file_name[: len("CRA_IR_1B_LR__")]:
        print("CRISTAL l1b (CRA_IR_1B_LR__) identified")
        if waveform_parameter_name is None:
            wf_parameter = "data_20/ku/power_waveform_comb"
        if latname is None:
            latitude_name = "data_20/ku/latitude"
        if lonname is None:
            longitude_name = "data_20/ku/longitude"
    elif "SIR_LRM_1B" in file_name[8 : 8 + len("SIR_LRM_1B")]:
        print("CryoSat-2 LRM L1b file identified")
        if waveform_parameter_name is None:
            wf_parameter = "pwr_waveform_20_ku"
        if latname is None:
            latitude_name = "lat_20_ku"
        if lonname is None:
            longitude_name = "lon_20_ku"
    elif "SIR_SIN_1B" in file_name[8 : 8 + len("SIR_SIN_1B")]:
        print("CryoSat-2 SIN L1b file identified")
        if waveform_parameter_name is None:
            wf_parameter = "pwr_waveform_20_ku"
        if latname is None:
            latitude_name = "lat_20_ku"
        if lonname is None:
            longitude_name = "lon_20_ku"
    elif "SIR_SAR_1B" in file_name[8 : 8 + len("SIR_SAR_1B")]:
        print("CryoSat-2 SAR L1b file identified")
        if waveform_parameter_name is None:
            wf_parameter = "pwr_waveform_20_ku"
        if latname is None:
            latitude_name = "lat_20_ku"
        if lonname is None:
            longitude_name = "lon_20_ku"

    if wf_parameter is None or latitude_name is None or longitude_name is None:
        sys.exit(
            f"File {file_name} type not identified. "
            "In this case you must provide the --parameter <WAVEFORM_PARAM_NAME> "
            "and --latname <LATITUDE_NAME>, --lonname <LONGITUDE_NAME> to plot"
        )

    return wf_parameter, latitude_name, longitude_name


def get_scale_factor(nc: Dataset, file_name: str, wf_name: str, waveforms: np.ndarray):
    """retrieve an array of scale factors to convert waveform counts to Watts
    This is mission dependant

    Args:
        nc (Dataset):netcdf dataset
        file_name (str): file name
        wf_name (str): name of waveform parameter
    """

    if "CRA_IR_1B_HR__" in file_name[: len("CRA_IR_1B_HR__")]:
        if wf_name == "data/ku/power_waveform_comb":
            scale_factor = get_variable(nc, "data/ku/waveform_scale_factor_comb")[:].data
        elif wf_name == "data/ku/power_waveform_rx1":
            scale_factor = get_variable(nc, "data/ku/waveform_scale_factor_rx1")[:].data
        elif wf_name == "data/ku/power_waveform_rx2":
            scale_factor = get_variable(nc, "data/ku/waveform_scale_factor_rx2")[:].data
        elif wf_name == "data/ka/power_waveform_comb":
            scale_factor = get_variable(nc, "data/ka/waveform_scale_factor_comb")[:].data
        elif wf_name == "data/ka/power_waveform_rx1":
            scale_factor = get_variable(nc, "data/ka/waveform_scale_factor_rx1")[:].data
        elif wf_name == "data/ka/power_waveform_rx2":
            scale_factor = get_variable(nc, "data/ka/waveform_scale_factor_rx2")[:].data
        else:
            raise ValueError("wf parameter not supported")

    elif "CRA_IR_1B_LR__" in file_name[: len("CRA_IR_1B_LR__")]:
        if wf_name == "data_20/ku/power_waveform_comb":
            scale_factor = get_variable(nc, "data_20/ku/waveform_scale_factor_comb")[:].data
        elif wf_name == "data_20/ku/power_waveform_rx1":
            scale_factor = get_variable(nc, "data_20/ku/waveform_scale_factor_rx1")[:].data
        elif wf_name == "data_20/ku/power_waveform_rx2":
            scale_factor = get_variable(nc, "data_20/ku/waveform_scale_factor_rx2")[:].data
        elif wf_name == "data_20/ka/power_waveform_comb":
            scale_factor = get_variable(nc, "data_20/ka/waveform_scale_factor_comb")[:].data
        elif wf_name == "data_20/ka/power_waveform_rx1":
            scale_factor = get_variable(nc, "data_20/ka/waveform_scale_factor_rx1")[:].data
        elif wf_name == "data_20/ka/power_waveform_rx2":
            scale_factor = get_variable(nc, "data_20/ka/waveform_scale_factor_rx2")[:].data
        else:
            raise ValueError("wf parameter not supported")
    else:
        scale_factor = np.ones(np.shape(waveforms)[0], dtype=float)

    return scale_factor


def get_variable(nc: Dataset, nc_var_path: str) -> Variable:
    """Retrieve variable from NetCDF file, handling groups if necessary.

    This function navigates through groups in a NetCDF file to retrieve the specified variable.

    Args:
        nc (Dataset): The NetCDF dataset object.
        nc_var_path (str): The path to the variable within the NetCDF file,
                        with groups separated by '/'.

    Returns:
        Variable: The retrieved NetCDF variable.

    Raises:
        SystemExit: If the variable or group is not found in the NetCDF file.
    """
    parts = nc_var_path.split("/")
    var = nc
    for part in parts:
        try:
            var = var[part]
        except (KeyError, IndexError):
            sys.exit(f"NetCDF parameter '{nc_var_path}' not found in file")
    return var


def main(cmd_args: list[str]) -> None:
    """Main function to parse arguments and plot waveforms.

    Args:
        args (list[str]): List of command-line arguments passed to the script.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Tool to plot altimetry waveforms and show"
            " the location of waveform and track on a map. "
            "For CS2 and CRISTAL L1b files, default parameters"
            " will be used for waveforms and latitude "
            "and longitude unless specified with command line"
            " options."
        )
    )
    parser.add_argument(
        "--file", "-f", help=("file name of NetCDF file to plot waveforms from"), required=False
    )
    parser.add_argument(
        "--parameter",
        "-p",
        help=("waveform parameter to use. Example: data/ku/power_waveform_comb"),
        required=False,
    )
    parser.add_argument(
        "--latname",
        "-lat",
        help=("name of netcdf latitude parameter to use. Example: data/ku/latitude"),
        required=False,
    )
    parser.add_argument(
        "--lonname",
        "-lon",
        help=("name of netcdf longitude parameter to use." " Example: data/ku/longitude"),
        required=False,
    )
    parser.add_argument(
        "--index",
        "-i",
        help=("[optional, int, def=1] : waveform number to start displaying"),
        required=False,
        type=int,
        default=1,
    )
    parser.add_argument(
        "--delay",
        "-d",
        help=("[optional, float, def=0.2] : delay between waveforms when animating"),
        required=False,
        type=float,
        default=0.2,
    )
    parser.add_argument(
        "--scale_to_watts",
        "-w",
        help=("[optional] : if set, scale to watts"),
        required=False,
        action="store_true",
    )
    args = parser.parse_args(cmd_args)

    if not args.file:
        sys.exit("--file <file_name> missing command line arg")

    file_name = os.path.basename(args.file)

    wf_name, latitude_name, longitude_name = get_default_parameter_names(
        file_name, args.parameter, args.latname, args.lonname
    )

    with Dataset(args.file) as nc:
        waveforms = get_variable(nc, wf_name)[:].data
        scale_factor = get_scale_factor(nc, file_name, wf_name, waveforms)

        if args.scale_to_watts:
            waveforms_scaled = waveforms * scale_factor[:, np.newaxis]
            units = "watts"
        else:
            waveforms_scaled = waveforms
            units = "counts"
        latitudes = get_variable(nc, latitude_name)[:].data
        longitudes = get_variable(nc, longitude_name)[:].data

    fig = plt.figure(figsize=(15, 8))
    gs = fig.add_gridspec(1, 2, width_ratios=[2, 1], wspace=0.05)
    ax_waveform = fig.add_subplot(gs[0, 0])
    ax_map = fig.add_subplot(gs[0, 1], projection=ccrs.Mercator())

    plt.subplots_adjust(left=0.1, right=0.95, bottom=0.25, top=0.9)

    if args.index > len(waveforms_scaled) - 1:
        sys.exit("--index <WAVEFORM_INDEX> number is > (number of waveforms in file-1)")
    current_index = args.index - 1

    animating = False  # Flag to track animation state
    animation_delay = args.delay

    def next_waveform(_: object) -> None:
        nonlocal current_index
        if current_index < len(waveforms_scaled) - 1:
            current_index += 1
            update_plot(
                current_index,
                ax_waveform,
                ax_map,
                waveforms_scaled,
                latitudes,
                longitudes,
                file_name,
                wf_name + f"({units})",
                fig,
            )

    def prev_waveform(_: object) -> None:
        nonlocal current_index
        if current_index > 0:
            current_index -= 1
            update_plot(
                current_index,
                ax_waveform,
                ax_map,
                waveforms_scaled,
                latitudes,
                longitudes,
                file_name,
                wf_name + f"({units})",
                fig,
            )

    def rewind_waveform(_: object) -> None:
        nonlocal current_index
        current_index = 0
        update_plot(
            current_index,
            ax_waveform,
            ax_map,
            waveforms_scaled,
            latitudes,
            longitudes,
            file_name,
            wf_name + f"({units})",
            fig,
        )

    def forward_to_end(_: object) -> None:
        nonlocal current_index
        current_index = len(waveforms_scaled) - 1
        update_plot(
            current_index,
            ax_waveform,
            ax_map,
            waveforms_scaled,
            latitudes,
            longitudes,
            file_name,
            wf_name + f"({units})",
            fig,
        )

    def quit_program(_: object) -> None:
        sys.exit(0)

    def animate_waveforms(_: object) -> None:
        nonlocal current_index, animating, animation_delay
        if animating:
            animating = False  # Stop animation
            banimate.label.set_text("Animate >|")
            fig.canvas.draw_idle()  # Force update of the button label
        else:
            animating = True  # Start animation
            banimate.label.set_text("Stop")
            fig.canvas.draw_idle()  # Force update of the button label
            while animating and current_index < len(waveforms_scaled) - 1:
                current_index += 1
                update_plot(
                    current_index,
                    ax_waveform,
                    ax_map,
                    waveforms_scaled,
                    latitudes,
                    longitudes,
                    file_name,
                    wf_name + f"({units})",
                    fig,
                )
                plt.pause(animation_delay)  # Short delay between frames

    button_width = 0.1
    button_height = 0.075
    button_padding = 0.02
    start_x = (1 - 6 * (button_width + button_padding) + button_padding) / 2

    axrewind = plt.axes((start_x, 0.05, button_width, button_height))
    axprev = plt.axes(
        (start_x + (button_width + button_padding), 0.05, button_width, button_height)
    )
    axnext = plt.axes(
        (start_x + 2 * (button_width + button_padding), 0.05, button_width, button_height)
    )
    axforward = plt.axes(
        (start_x + 3 * (button_width + button_padding), 0.05, button_width, button_height)
    )
    axanimate = plt.axes(
        (start_x + 4 * (button_width + button_padding), 0.05, button_width, button_height)
    )
    axquit = plt.axes(
        (start_x + 5 * (button_width + button_padding), 0.05, button_width, button_height)
    )

    brewind = Button(axrewind, "|<")
    bprev = Button(axprev, "Previous")
    bnext = Button(axnext, "Next")
    bforward = Button(axforward, ">|")
    banimate = Button(axanimate, "Animate >|")
    bquit = Button(axquit, "Quit")

    brewind.on_clicked(rewind_waveform)
    bprev.on_clicked(prev_waveform)
    bnext.on_clicked(next_waveform)
    bforward.on_clicked(forward_to_end)
    banimate.on_clicked(animate_waveforms)
    bquit.on_clicked(quit_program)

    update_plot(
        current_index,
        ax_waveform,
        ax_map,
        waveforms_scaled,
        latitudes,
        longitudes,
        file_name,
        wf_name + f"({units})",
        fig,
    )

    plt.show()


if __name__ == "__main__":
    main(sys.argv[1:])
