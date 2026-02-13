"""cpom.areas.area_plot3d.py

# Purpose

Functions to create 3d plots (using plotly library)
"""

import logging
import os

import matplotlib.colors as mcolors
import numpy as np
import plotly.graph_objects as go
from matplotlib import cm
from numpy import ma  # masked arrays
from scipy.ndimage import gaussian_filter

from cpom.areas.area_plot import get_unique_colors
from cpom.areas.areas3d import Area3d
from cpom.dems.dems import Dem
from cpom.gridding.gridareas import GridArea

log = logging.getLogger(__name__)


def plot_3d_area(area_name: str, *data_sets, area_overrides: dict):
    """Functions to create 3d plots for an area

    Args:
        area_name (str) : 3d area definition name as in cpom/areas/definitions_3d/<area_name>.py
        `*data_sets` (dict, optional):
          One or more dataset dictionaries. Each dictionary must include the (lats, lons, vals)
          for a dataset and optional tunable plot parameters. Structure:
          ```python
          {
              "lats": np.array([]),             # Required: Latitude values
              "lons": np.array([]),             # Required: Longitude values
              "vals": np.array([]),             # Required: Data values
              "units": '',                      # Optional: Units of `vals`
              "name": "unnamed",                # Optional: Name of dataset
              "apply_area_mask_to_data": True,  # Optional: Apply area mask to data
              "fill_value": 9999,               # Optional: Fill value to ignore in `vals`
              "valid_range": [min, max],        # Optional: Valid range for `vals`
              "minimap_val_scalefactor": 1.0,   # Optional: Scale factor for bad data marker
              "flag_values": [],                # Optional: List of flag values
              "flag_names": [],                 # Optional: List of flag names
              "flag_colors": [],                # Optional: Colors for flags or colormap
              "color": str,                     # Optional: color to use to plot vals.
                                                #       can be "cmap:coolwarm_r" or color "red"
                                                #       cmap must be a supported python cmap name,
              "cmap_over_color": str,           # Optional: can specify an over color for cmap
              "cmap_under_color": str,          # Optional: can specify an under color for cmap
              "min_plot_range": None,           # Optional: Min range for colorbar
              "max_plot_range": None,           # Optional: Max range for colorbar
              "plot_size_scale_factor": 1.0,    # Optional: Marker size scale factor
              "plot_alpha": 1.0,                # Optional: Marker transparency (0 to 1)
              "var_stride": 1,                  # Optional: plot every N of input vals
              "plot_nan_and_fv": bool,          # if True, plot just the location of Nan/FV in data

          }
          ```
          area_overrides (dict): overide any area definitions in
                                 cpom/areas/definitions_3d/<area_name>.py
    """

    # Load 3d area definition from area_name

    thisarea = Area3d(area_name, area_overrides)

    # --------------------------------------------------------------------------------------------
    # find area of DEM required for AOI and create a sub-area of the DEM (uses transformer
    # created while calculating lat/long lines above)
    # --------------------------------------------------------------------------------------------

    if thisarea.specify_by_centre:
        # Get the bottom left corner of the area of interest in x,y coordinates in meters

        xll, yll = thisarea.latlon_to_xy(thisarea.centre_lat, thisarea.centre_lon)

        xll = xll - (thisarea.width_km * 1000.0 / 2.0)
        yll = yll - (thisarea.height_km * 1000.0 / 2.0)
    else:
        # Get the bottom left corner of the area of interest in x,y coordinates in meters
        xll, yll = thisarea.latlon_to_xy(thisarea.llcorner_lat, thisarea.llcorner_lon)

    # Load the DEM for this area

    thisdem = Dem(thisarea.dem_name)

    # find the offset in to the DEM grid of the areas bottom left corner

    dem_bin_offset_x = int((xll - thisdem.mindemx) / thisdem.binsize)

    dem_bin_offset_y = int((yll - thisdem.mindemy) / thisdem.binsize)
    # if the offset is < 0 in x or y, set to zero
    dem_bin_offset_y = max(dem_bin_offset_y, 0)
    dem_bin_offset_x = max(dem_bin_offset_x, 0)

    # zdem needs transposing if originated from geotiff
    if thisdem.zarr_type:
        zdem = thisdem.zdem_flip
        ydem = np.flip(thisdem.ydem)
    elif thisdem.zflip:
        zdem = np.flip(thisdem.zdem, 0)
        ydem = np.flip(thisdem.ydem)
    else:
        zdem = thisdem.zdem
        ydem = thisdem.ydem

    # Create a new DEM grid that corresponds to the AOI rectangle
    new_dem_max_x = dem_bin_offset_x + int(thisarea.width_km * 1000 / thisdem.binsize)
    new_dem_max_y = dem_bin_offset_y + int(thisarea.height_km * 1000 / thisdem.binsize)

    if new_dem_max_x >= zdem.shape[1]:
        new_dem_max_x = zdem.shape[1] - 1
    if new_dem_max_y >= zdem.shape[0]:
        new_dem_max_y = zdem.shape[0] - 1

    print(zdem.shape)

    # Override calculated DEM sampling for the plot from the command line inputs
    if not thisarea.dem_stride:
        # was previously: plot_surface_stride = 1 + int(zdem.shape[0] / thisarea.page_width)
        raise ValueError("no dem_stride provided in Area3d object")
    plot_surface_stride = thisarea.dem_stride

    print(f"plot_surface_stride={plot_surface_stride}")

    print("Subsetting DEM...")
    zdem = zdem[
        dem_bin_offset_y:new_dem_max_y:plot_surface_stride,
        dem_bin_offset_x:new_dem_max_x:plot_surface_stride,
    ]
    print("done...")

    # Do the same with x and y

    print("Subsetting xdem...")

    xdem = thisdem.xdem[dem_bin_offset_x:new_dem_max_x:plot_surface_stride]

    print("Subsetting ydem...")

    ydem = ydem[dem_bin_offset_y:new_dem_max_y:plot_surface_stride]

    # make the x,y coordinates start from 0,0
    shifted_xdem = xdem - xdem.min()
    shifted_ydem = ydem - ydem.min()

    print("creating x,y mesh...")

    # create a mesh of x,y coordinates
    x_grid, y_grid = np.meshgrid(shifted_xdem, shifted_ydem)

    x_grid = x_grid * 0.001  # convert to Km
    y_grid = y_grid * 0.001  # convert to Km

    # --------------------------------------------------------------------------------------------
    # Optionally smooth the DEM
    # --------------------------------------------------------------------------------------------

    if thisarea.smooth_dem:
        # Gaussian smooth DEM
        print("Smoothing")
        sigma = 1.0
        smooth_v = zdem.copy()
        smooth_v[np.isnan(zdem)] = 0
        smooth_vv = gaussian_filter(smooth_v, sigma=sigma)
        smooth_w = 0 * zdem.copy() + 1
        smooth_w[np.isnan(zdem)] = 0
        smooth_ww = gaussian_filter(smooth_w, sigma=sigma)
        zdem = smooth_vv / smooth_ww
        print("Done")

    # --------------------------------------------------------------------------------------------
    #  Create optional latitude and longitude, and Place annotations (uses transformer
    # created while calculating lat/long lines above)
    # --------------------------------------------------------------------------------------------

    annotations = []

    # Default light effects
    ambient_light = 0.4
    diffuse_light = 0.5
    roughness_light = 0.9
    specular_light = 1.0
    fresnel_light = 0.2

    msscolorscale = "Ice"

    if thisarea.place_annotations:
        for annotation in thisarea.place_annotations:
            print("Adding place annotation at ", annotation[0], annotation[1])
            xpos, ypos = thisarea.latlon_to_xy(annotation[0], annotation[1])
            xpos = (xpos - xdem.min()) * 0.001
            ypos = (ypos - ydem.min()) * 0.001
            annotations.append(
                {
                    "showarrow": False,
                    "x": xpos,
                    "font": {"color": annotation[4], "size": annotation[6]},
                    "y": ypos,
                    "z": annotation[2],
                    "text": str(annotation[3]),
                    "bgcolor": annotation[5],
                    "xanchor": "left",
                    "xshift": 2,
                    "opacity": annotation[7],
                }
            )

    # Add latitude number annotations
    if thisarea.lat_annotations:
        for annotation in thisarea.lat_annotations:
            print("Adding latitude annotation at ", annotation[0], annotation[1])
            if annotation[0] < 0:
                latstr = f"{annotation[0] * -1}S"
            else:
                latstr = f"{annotation[0]}N"
            xpos, ypos = thisarea.latlon_to_xy(annotation[0], annotation[1])

            if thisarea.raise_latlon_lines_above_dem:
                zpos = thisdem.interp_dem(xpos, ypos) + thisarea.raise_latlon_lines_above_dem
            else:
                zpos = annotation[2]

            xpos = (xpos - xdem.min()) * 0.001
            ypos = (ypos - ydem.min()) * 0.001
            annotations.append(
                {
                    "showarrow": False,
                    "x": xpos,
                    "font": {"color": "white", "size": annotation[3]},
                    "y": ypos,
                    "z": zpos,
                    "text": latstr,
                    "xanchor": "left",
                    "xshift": annotation[4],
                    "yshift": annotation[5],
                    "opacity": 0.7,
                }
            )

    # Add longitude number annotations
    if thisarea.lon_annotations:
        for annotation in thisarea.lon_annotations:
            print("Adding longitude annotation at ", annotation[0], annotation[1])
            xpos, ypos = thisarea.latlon_to_xy(annotation[0], annotation[1])
            if thisarea.raise_latlon_lines_above_dem:
                zpos = thisdem.interp_dem(xpos, ypos) + thisarea.raise_latlon_lines_above_dem
            else:
                zpos = annotation[2]

            xpos = (xpos - xdem.min()) * 0.001
            ypos = (ypos - ydem.min()) * 0.001
            if annotation[1] < 0:
                textstr = str(-1 * annotation[1]) + "W"
            else:
                textstr = str(annotation[1]) + "E"
            annotations.append(
                {
                    "showarrow": False,
                    "x": xpos,
                    "font": {"color": "white", "size": annotation[3]},
                    "y": ypos,
                    "z": zpos,
                    "text": textstr,
                    "xanchor": "left",
                    "xshift": annotation[4],
                    "yshift": annotation[5],
                    "opacity": 0.7,
                }
            )

    # ---------------------------------------------------------------------------------------------
    #  Specify the plot layout
    # 		z,y,z axises set to invisible
    #       set the aspect ratio of each axis.
    #       z-axis is squashed by a multiplier which is selected by the user or scene settings
    # 		x-axis's aspect is set by the area width/height ratio
    #       page_width
    # ---------------------------------------------------------------------------------------------

    layout = go.Layout(
        title_text=thisarea.long_name,
        scene={
            "xaxis_visible": False,
            "yaxis_visible": False,
            "zaxis_visible": False,
            "annotations": annotations,
            "aspectmode": "manual",
            "aspectratio": {
                "x": 1.0 * (thisarea.width_km) / (thisarea.height_km),
                "y": 1.0,
                "z": thisarea.zaxis_multiplier,
            },
        },
    )

    # Create a grid of points to calculate slope at
    # --------------------------------------------------------------------------------------------
    # Add a Mean Sea Surface layer to decorate the sea
    # --------------------------------------------------------------------------------------------
    if thisarea.add_mss_layer:
        mss_file = (
            os.environ["CPOM_SOFTWARE_DIR"]
            + f"/resources/mss/{thisarea.mss_binsize_km}km_{thisarea.mss_gridarea}_mean_mss.npz"
        )
        print("Load MSS..", mss_file)
        npfiles = np.load(mss_file, allow_pickle=True)
        mss_mean_grid = npfiles["mss_mean_grid"] + 5.0

        print("done")

        thisgridarea = GridArea(thisarea.mss_gridarea, thisarea.mss_binsize_km * 1e3)

        xmss = np.linspace(
            thisgridarea.minxm,
            thisgridarea.minxm + thisgridarea.grid_x_size,
            int(thisgridarea.grid_x_size / (thisarea.mss_binsize_km * 1e3)),
            endpoint=True,
        )
        ymss = np.linspace(
            thisgridarea.minym,
            thisgridarea.minym + thisgridarea.grid_y_size,
            int(thisgridarea.grid_y_size / (thisarea.mss_binsize_km * 1e3)),
            endpoint=True,
        )

        shifted_xmss = xmss - xdem.min()
        shifted_ymss = ymss - ydem.min()

        # create a mesh of x,y coordinates
        x_mss_grid, y_mss_grid = np.meshgrid(shifted_xmss, shifted_ymss)

        x_mss_grid = x_mss_grid * 0.001  # convert to Km
        y_mss_grid = y_mss_grid * 0.001  # convert to Km

    # --------------------------------------------------------------------------------------------
    # Add DEM surface layer
    # --------------------------------------------------------------------------------------------

    # Plot the DEM surface

    # Create a white colormap for 3d hillshade plots
    pl_white = [
        [0.0, "rgb(255,255,255)"],
        [1.0, "rgb(255,255,255)"],
    ]

    # Gather the light effects for this scene
    lighting_effects = {
        "ambient": ambient_light,
        "diffuse": diffuse_light,
        "roughness": roughness_light,
        "specular": specular_light,
        "fresnel": fresnel_light,
    }

    fig = go.Figure(
        data=[
            go.Surface(
                z=zdem,
                x=x_grid,
                y=y_grid,
                colorscale=pl_white,
                opacity=1.0,
                showscale=False,
                lighting=lighting_effects,
                lightposition={
                    "x": thisarea.light_xdirection,
                    "y": thisarea.light_ydirection,
                    "z": thisarea.light_zdirection,
                },
                contours=go.surface.Contours(
                    x=go.surface.contours.X(highlight=False),
                    y=go.surface.contours.Y(highlight=False),
                    z=go.surface.contours.Z(highlight=False),
                ),
                hoverinfo="none",
            ),
        ],
        layout=layout,
    )

    # --------------------------------------------------------------------------------------------
    # Add a Mean Sea Surface, surface layer if required
    # --------------------------------------------------------------------------------------------

    if thisarea.add_mss_layer:
        print("Adding MSS layer..")

        fig.add_trace(
            go.Surface(
                z=mss_mean_grid,
                x=x_mss_grid,
                y=y_mss_grid,
                colorscale=msscolorscale,
                showscale=False,  # 'Ice'
                lighting=lighting_effects,
                lightposition={
                    "x": thisarea.light_xdirection,
                    "y": thisarea.light_ydirection,
                    "z": thisarea.light_zdirection,
                },
                contours=go.surface.Contours(
                    x=go.surface.contours.X(highlight=False),
                    y=go.surface.contours.Y(highlight=False),
                    z=go.surface.contours.Z(highlight=False),
                ),
            )
        )
    # --------------------------------------------------------------------------------------------
    # Set initial 3d view angle, elevation
    # --------------------------------------------------------------------------------------------

    def sph2cart(azimuth, elevation, radius):
        az = azimuth * 0.0174533
        el = elevation * 0.0174533
        r = radius
        rcos_theta = r * np.cos(el)
        x = rcos_theta * np.cos(az)
        y = rcos_theta * np.sin(az)
        z = r * np.sin(el)
        return x, y, z

    xe, ye, ze = sph2cart(
        thisarea.view_angle_azimuth, thisarea.view_angle_elevation, thisarea.plot_zoom
    )
    print("xe,ye,ze", xe, ye, ze)

    fig.update_layout(
        showlegend=False,
        scene={
            "xaxis": {
                "nticks": 4,
                "range": [0, x_grid.max()],
            },
            "yaxis": {
                "nticks": 4,
                "range": [0, y_grid.max()],
            },
            "zaxis": {
                "nticks": 4,
                "range": [thisarea.zaxis_limits[0], thisarea.zaxis_limits[1]],
            },
        },
        scene_camera={
            "up": {"x": 0, "y": 0, "z": 1},
            "center": {"x": 0, "y": 0, "z": 0},
            "eye": {"x": xe, "y": ye, "z": ze},
        },
        title={
            "text": thisarea.long_name,
            "y": 0.95,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
        },
    )

    # ------------------------------------------------------------------------------------------
    #  Overplot the parameter values in params
    # ------------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------------
    # Load data sets
    # ------------------------------------------------------------------------------------------
    num_data_sets = len(data_sets)
    if num_data_sets > 0:
        log.info("Loading %d data sets", num_data_sets)

        for ds_num, data_set in enumerate(data_sets):
            print(f"loading data set {ds_num}: {data_set.get('name', 'unnamed')}")

            is_flag_data = len(data_set.get("flag_values", [])) > 0

            lats = data_set.get("lats", np.array([]))
            lons = data_set.get("lons", np.array([]))
            lons = np.array(lons) % 360  # ensure 0..360 degs E
            vals = data_set.get("vals", np.array([]))
            var_opacity = float(data_set.get("plot_alpha", 1.0))

            n_vals = len(vals)

            if n_vals != len(lats):
                raise ValueError(f"length of vals array must equal lats array in data set {ds_num}")
            if n_vals != len(lons):
                raise ValueError(f"length of vals array must equal lons array in data set {ds_num}")

            # convert to ndarray if a list
            if not isinstance(lats, np.ndarray):
                lats = np.asarray(lats)
            if not isinstance(lons, np.ndarray):
                lons = np.asarray(lons)
            if not isinstance(vals, np.ndarray):
                vals = np.asarray(vals)

            # Check if data is not 1-d. If n-d > 1, flatten to 1-d
            if lats.ndim > 1:
                lats = lats.flatten()
            if lons.ndim > 1:
                lons = lons.flatten()
            if vals.ndim > 1:
                vals = vals.flatten()

            # ------------------------------------------------------------------------------
            # check lats,lons for valid values before plotting
            # ------------------------------------------------------------------------------

            # Convert None to np.nan and ensure the array is of float type
            lats = np.array(lats, dtype=float)
            lons = np.array(lons, dtype=float)

            # Test for masked arrays
            if ma.is_masked(lats):
                lats[np.ma.getmaskarray(lats)] = np.nan

            if ma.is_masked(lons):
                lons[np.ma.getmaskarray(lons)] = np.nan

            # Step 1: Filter for valid values
            # Assuming latitude values must be between -90 and 90, and longitude
            # between -180 and 180 or 0 to 360
            valid_lat = (lats >= -90) & (lats <= 90)
            valid_long = (lons >= 0) & (lons <= 360)

            # Handling NaNs or None for both lats and lons
            valid_lat = valid_lat & ~np.isnan(lats)
            valid_long = valid_long & ~np.isnan(lons)

            # Step 3: Identify common indices
            valid_indices = np.where(valid_lat & valid_long)[0]

            if valid_indices.size == 0:
                log.error(
                    "No valid latitude and longitude values in dataset %s",
                    data_set.get("name", f"unnamed_{ds_num}"),
                )
                continue

            lats = lats[valid_indices]
            lons = lons[valid_indices]
            vals = vals[valid_indices]

            log.info(
                "%d valid lat/lon values found for dataset %s",
                lats.size,
                data_set.get("name", f"unnamed_{ds_num}"),
            )

            # ------------------------------------------------------------------------------
            # Area Mask data sets
            # ------------------------------------------------------------------------------

            # ------------------------------------------------------------------------------
            # All areas are filtered for the area's lat/lon bounds
            # ------------------------------------------------------------------------------

            # Lat/Lon bounds filter
            lats, lons, inside_area, n_inside = thisarea.inside_latlon_bounds(lats, lons)
            if n_inside > 0:
                vals = vals[inside_area]
            else:
                log.error("No data inside lat/lon bounds for data set %d", ds_num)
                continue

            log.info("Number of values inside lat/lon bounds %d of %d", n_inside, n_vals)

            # ------------------------------------------------------------------------------
            # All areas are filtered for the area's extent bounds
            #  (automatically ignored by circular areas)
            # ------------------------------------------------------------------------------

            (
                lats,
                lons,
                x_inside,
                y_inside,
                inside_area,
                n_inside,
            ) = thisarea.inside_xy_extent(lats, lons)
            if n_inside > 0:
                vals = vals[inside_area]
            else:
                log.error("No data inside extent bounds for data set %d", ds_num)
                continue
            log.info("Number of values inside extent bounds %d of %d", n_inside, n_vals)

            # ------------------------------------------------------------------------------
            # Optional Mask filtering : grid masks, polygon masks, etc
            # ------------------------------------------------------------------------------

            apply_area_mask = data_set.get(
                "apply_area_mask_to_data", thisarea.apply_area_mask_to_data
            )

            if apply_area_mask:
                log.info("Masking xy data with area's data mask..")

                inside_area, n_inside = thisarea.inside_mask(x_inside, y_inside)
                if n_inside > 0:
                    vals = vals[inside_area]
                    lats = lats[inside_area]
                    lons = lons[inside_area]
                else:
                    log.error("No data inside mask for data set %d", ds_num)
                    continue
                log.info("Number of values inside mask %d of %d", n_inside, n_vals)

            # ------------------------------------------------------------------------------
            # Check vals for Nan and FillValue before plotting
            # ------------------------------------------------------------------------------

            # convert None to Nan
            try:
                vals = np.array(vals, dtype=float)
            except ValueError:
                log.error("invalid value type in dataset found. Must be int or float")
                continue

            # find Nan values in data ------------------------------------------------------
            nan_vals_bool = np.isnan(vals)
            percent_nan = np.mean(nan_vals_bool) * 100.0
            # nan_indices = np.where(nan_vals_bool)[0]
            # if nan_indices.size > 0:
            #     nan_lats = lats[nan_indices]
            #     nan_lons = lons[nan_indices]
            # else:
            #     nan_lats = np.array([])
            #     nan_lons = np.array([])

            log.info("percent Nan %.2f", percent_nan)

            # find out of range values in data -------------------------------------------------

            if is_flag_data:
                outside_vals_bool = (vals < np.min(data_set.get("flag_values"))) | (
                    vals > np.max(data_set.get("flag_values"))
                )
                percent_outside = np.mean(outside_vals_bool) * 100.0
            else:
                if (
                    data_set.get("valid_range") is not None
                    and len(data_set.get("valid_range")) != 2
                ):
                    log.error("valid_range plot parameter must be of type [min,max]")
                if (
                    data_set.get("valid_range") is not None
                    and len(data_set.get("valid_range")) == 2
                ):
                    outside_vals_bool = np.zeros_like(vals, dtype=bool)
                    if data_set.get("valid_range")[0] is not None:
                        outside_vals_bool |= vals < data_set.get("valid_range")[0]
                    if data_set.get("valid_range")[1] is not None:
                        outside_vals_bool |= vals > data_set.get("valid_range")[1]
                    percent_outside = np.mean(outside_vals_bool) * 100.0
                else:
                    percent_outside = 0.0
                    outside_vals_bool = np.full_like(vals, False, bool)

            # outside_indices = np.where(outside_vals_bool)[0]
            # if outside_indices.size > 0:
            #     outside_lats = lats[outside_indices]
            #     outside_lons = lons[outside_indices]
            # else:
            #     outside_lats = np.array([])
            #     outside_lons = np.array([])

            log.info("percent outside valid range %.2f", percent_outside)

            # find fill values -------------------------------------------------------------
            if data_set.get("fill_value") is not None:
                log.info("finding fill_value %s", str(data_set.get("fill_value")))
                fv_vals_bool = vals == data_set["fill_value"]
                percent_fv = np.mean(fv_vals_bool) * 100.0
            else:
                percent_fv = 0.0
                fv_vals_bool = np.full_like(vals, False, bool)

            log.info("percent FV %.2f", percent_fv)

            # fv_indices = np.where(fv_vals_bool)[0]
            # if fv_indices.size > 0:
            #     fv_lats = lats[fv_indices]
            #     fv_lons = lons[fv_indices]
            # else:
            #     fv_lats = np.array([])
            #     fv_lons = np.array([])

            if data_set["plot_nan_and_fv"]:
                valid_vals_bool = nan_vals_bool | fv_vals_bool
            else:
                valid_vals_bool = (~nan_vals_bool & ~fv_vals_bool) & ~outside_vals_bool

            valid_indices = np.where(valid_vals_bool)[0]

            if valid_indices.size > 0:
                vals = vals[valid_indices]
                lats = lats[valid_indices]
                lons = lons[valid_indices]
                if data_set["plot_nan_and_fv"]:
                    vals = np.ones_like(lats, dtype=int)
            else:
                vals = np.array([])
                lats = np.array([])
                lons = np.array([])
                log.info("no valid values in dataset")
                continue

            percent_valid = np.mean(valid_vals_bool) * 100.0
            log.info("percent_valid=%.2f", percent_valid)

            # ------------------------------------------------------------------------------
            # Plot data
            # ------------------------------------------------------------------------------

            # sub-sample values to display if required
            var_stride = data_set.get("var_stride", 1)
            lats = lats[::var_stride]
            lons = lons[::var_stride]
            vals = vals[::var_stride]

            # Convert input lat/lon to x,y
            xlocs, ylocs = thisarea.latlon_to_xy(lats, lons)

            # Find DEM values at x,y
            dem_vals = thisdem.interp_dem(xlocs, ylocs)
            xlocs -= xdem.min()
            ylocs -= ydem.min()
            xlocs *= 0.001  # convert to km
            ylocs *= 0.001  # convert to km

            if data_set["use_colourmap"]:
                # Define min/max range
                cmin = data_set["plot_range"][0] if data_set["plot_range"] else np.nanmin(vals)
                cmax = data_set["plot_range"][1] if data_set["plot_range"] else np.nanmax(vals)

                # Define a Plotly colorscale
                colorscale = data_set["colourmap"]
                print(colorscale)

                # Define colors for over and under values
                cmap_under_color = data_set.get(
                    "cmap_under_color", None
                )  # Color for values below cmin
                cmap_over_color = data_set.get(
                    "cmap_under_color", None
                )  # Color for values above cmax

                # Convert Matplotlib colormap to a Plotly colorscale
                mpl_cmap = cm.get_cmap(data_set["colourmap"])
                colorscale = [[i / 255, mcolors.to_hex(mpl_cmap(i / 255))] for i in range(256)]

                if cmap_over_color:
                    # Append the over and under colors to the colorscale
                    colorscale.append([1.0, cmap_over_color])  # Over color at the end of the scale
                if cmap_under_color:
                    colorscale.insert(
                        0, [0.0, cmap_under_color]
                    )  # Under color at the beginning of the scale

                thiscolor = vals
            elif data_set.get("flag_values", None):
                print(f"flag_values = {data_set['flag_values']}")

                if "flag_colors" not in data_set:
                    flag_colors = get_unique_colors(len(data_set["flag_values"]), as_hex=True)
                else:
                    flag_colors = data_set["flag_colors"]

                # get_unique_colors(n: int, cmap_name_override: str | None = None):

                thiscolor = np.array([flag_colors[0] for r in vals])
                for index, flag_value in enumerate(data_set["flag_values"]):
                    ok = np.where(vals == flag_value)[0]
                    if ok.size > 0:
                        thiscolor[ok] = flag_colors[index]

            else:
                thiscolor = data_set["color"]

            # Create a marker dictionary which sets the characteristics (color,size, etc)
            # of the plotted point
            marker_dict = {
                "color": thiscolor,
                "size": data_set["point_size"],
                "showscale": True,
            }

            if data_set["use_colourmap"]:
                marker_dict["colorscale"] = colorscale
                marker_dict["cmin"] = cmin
                marker_dict["cmax"] = cmax

            print("Plotting vals...")

            fig.add_scatter3d(
                x=xlocs,
                y=ylocs,
                z=np.array(dem_vals) + float(data_set["raise_elevation"]),
                mode="markers",
                opacity=var_opacity,
                marker=marker_dict,
            )

    # --------------------------------------------------------------------------------------------
    #  lat lines
    # --------------------------------------------------------------------------------------------

    if thisarea.lat_lines:
        log.info("Adding latitude lines...")
        latx_tmp = []
        laty_tmp = []
        for latpoint in thisarea.lat_lines:
            for lonpoint in np.arange(0.0, 360.0, thisarea.latlon_lines_increment):
                xtmp, ytmp = thisarea.lonlat_to_xy_transformer.transform(lonpoint, latpoint)
                latx_tmp.append(xtmp)
                laty_tmp.append(ytmp)
        latx = np.array(latx_tmp)
        laty = np.array(laty_tmp)

        # lat/lon lines will be optionally plotted at an elevation raised above the DEM
        if thisarea.raise_latlon_lines_above_dem:
            latz = thisdem.interp_dem(latx, laty) + thisarea.raise_latlon_lines_above_dem
        else:
            # lat/lon lines will be optionally plotted at a fixed elevation
            latz = np.ones(laty.size) * thisarea.latlon_lines_elevation

        latx -= xdem.min()
        laty -= ydem.min()
        latx *= 0.001  # convert to km
        laty *= 0.001  # convert to km

        fig.add_scatter3d(
            x=latx,
            y=laty,
            z=latz,
            mode="markers",
            opacity=thisarea.latlon_lines_opacity,
            marker={
                "size": thisarea.latlon_lines_size,
                "color": thisarea.latlon_line_colour,
                "showscale": False,
            },
        )

    # --------------------------------------------------------------------------------------------
    #  lon lines
    # --------------------------------------------------------------------------------------------

    if thisarea.lon_lines:
        log.info("Adding longitude lines...")
        lonx_tmp = []
        lony_tmp = []
        for lonpoint in thisarea.lon_lines:
            if thisarea.hemisphere == "south":
                for latpoint in np.arange(-90, -50.0, thisarea.latlon_lines_increment):
                    xtmp, ytmp = thisarea.lonlat_to_xy_transformer.transform(lonpoint, latpoint)
                    lonx_tmp.append(xtmp)
                    lony_tmp.append(ytmp)
            else:
                for latpoint in np.arange(50, 90.0, thisarea.latlon_lines_increment):
                    xtmp, ytmp = thisarea.lonlat_to_xy_transformer.transform(lonpoint, latpoint)
                    lonx_tmp.append(xtmp)
                    lony_tmp.append(ytmp)
        lonx = np.array(lonx_tmp)
        lony = np.array(lony_tmp)

        # lat/lon lines will be optionally plotted at an elevation raised above the DEM
        if thisarea.raise_latlon_lines_above_dem:
            lonz = thisdem.interp_dem(lonx, lony) + thisarea.raise_latlon_lines_above_dem
        else:
            lonz = np.ones(lony.size) * thisarea.latlon_lines_elevation

        lonx -= xdem.min()
        lony -= ydem.min()
        lonx *= 0.001
        lony *= 0.001

        fig.add_scatter3d(
            x=lonx,
            y=lony,
            z=lonz,
            mode="markers",
            marker={
                "size": thisarea.latlon_lines_size,
                "color": thisarea.latlon_line_colour,
                "showscale": False,
            },
            opacity=thisarea.latlon_lines_opacity,
        )

    fig.show()
