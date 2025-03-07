"""
cpom.altimetry.geolocation.geolocate_roemer_sar.py

# Purpose

Slope correction/geolocation function for SAR using an adapted Roemer method
from :
Roemer, S., Legrésy, B., Horwath, M., and Dietrich, R.: Refined
analysis of radar altimetry data applied to the region of the
subglacial Lake Vostok/Antarctica, Remote Sens. Environ., 106,
269–284, https://doi.org/10.1016/j.rse.2006.02.026, 2007.

Adaption by A.Muir, M.McMillan, Q.Huang (CPOM), includes a two stage POCA location,
performed by minimizing the range to the satellite from DEM points located within firstly
an across track rectangular area (multi-looked Doppler footprint) centered on the nadir
location of width equal to the beam width,
and then within the pulse doppler limited footprint around the initial POCA
at a finer resampled DEM resolution.

Next the calculated POCA's range to satellite is checked to see if it is
within the range window (configurably trimmed). If not the slope
correction is failed.

Finally the slope correction and height is calculated from

height[i] = altitudes[i]
                    - (geo_corrected_tracker_range[i] + retracker_correction[i])
                    + slope_correction[i]

# Main function

geolocate_roemeri_sar()

"""

import logging
from datetime import datetime  # date and time functions
from typing import Tuple

import numpy as np
from matplotlib import path, transforms
from scipy.interpolate import interpn

from cpom.altimetry.geolocation.get_heading import get_heading
from cpom.dems.dems import Dem

log = logging.getLogger(__name__)

EARTH_RADIUS = 6378137.0


def find_poca(
    zdem: np.ndarray,
    xdem: np.ndarray,
    ydem: np.ndarray,
    nadir_x: float,
    nadir_y: float,
    alt_pt: float,
):
    """Function that finds the POCA using method similar to Roemer et al. (2007)
       Finds the point with the shortest range to the satellite in the DEM segment and
       computes the slope correction to height
       Adapted from original : CLS (python) of McMillan (Matlab) code

    Args:
        zdem (np.ndarray): DEM height values
        xdem (np.ndarray): x locations of DEM in polar stereo coordinates (m)
        ydem (np.ndarray): y locations of DEM in polar stereo coordinates (m)
        nadir_x (float): x location of nadir in polar stereo coordinates (m)
        nadir_y (float): y location of nadir in polar stereo coordinates (m)
        alt_pt (float): altitude at nadir (m)

    Returns:
        (float,float,float,float,float,bool): poca_x, poca_y, poca_z, slope_correction_to_height,
        range_to_satellite_of_poca,
        flg_success
    """

    # ----------------------------------------------------------------
    # compute horizontal plane distance to each cell in beam footprint
    # ----------------------------------------------------------------

    # compute x and y distance of all points in beam footprint from nadir coordinate of
    # current record
    dem_dx_vec = xdem - nadir_x
    dem_dy_vec = ydem - nadir_y

    # Compute the squared magnitude of distance from nadir
    dem_dmag_squared = dem_dx_vec**2 + dem_dy_vec**2

    # Account for earth curvature using the squared distance
    dem_dz_vec = zdem - alt_pt - dem_dmag_squared / (2.0 * EARTH_RADIUS)

    dem_range_vec = np.sqrt(dem_dmag_squared + (dem_dz_vec) ** 2)

    # find range to, and indices of, closest dem pixel
    [dem_rpoca, dempoca_ind] = np.nanmin(dem_range_vec), np.nanargmin(dem_range_vec)

    if np.isnan(zdem[dempoca_ind]) | (zdem[dempoca_ind] == -9999):
        return -999, -999, -999, -999, -999, 0

    # compute relocation correction to apply to assumed nadir altimeter elevation to move to poca
    slope_correction_to_height = dem_rpoca + zdem[dempoca_ind] - alt_pt

    flg_success = 1

    return (
        xdem[dempoca_ind],
        ydem[dempoca_ind],
        zdem[dempoca_ind],
        slope_correction_to_height,
        dem_rpoca,
        flg_success,
    )


def datetime2year(date_dt):
    """calculate decimal year from datetime

    Args:
        date_dt (datetime): datetime obj to process

    Returns:
        float: decimal year
    """
    year_part = date_dt - datetime(year=date_dt.year, month=1, day=1)
    year_length = datetime(year=date_dt.year + 1, month=1, day=1) - datetime(
        year=date_dt.year, month=1, day=1
    )
    return date_dt.year + year_part / year_length


def geolocate_roemer_sar(
    lats: np.ndarray,
    lons: np.ndarray,
    altitudes: np.ndarray,
    thisdem: Dem | None,
    thisdem_fine: Dem | None,
    config: dict,
    geo_corrected_tracker_range: np.ndarray,
    points_to_include: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Geolocate SAR measurements using an adapted Roemer (Roemer et al, 2007) method

    Args:
        l1b (Dataset): NetCDF Dataset of L1b file
        thisdem (Dem): Dem object used for Roemer correction
        thisdem_fine (Dem): Dem object used for fine Roemer correction (maybe same obj as thisdem)
        config (dict): config dictionary containing
            "roemer_geolocation": {
                "fine_grid_sampling": 10, # DEM sampling of second stage
                "max_poca_reloc_distance": 6600, # meters. Max distance from nadir allowed
                "reject_outside_range_window": True, # reject POCA is outside trimmed range window
                "range_window_lower_trim": 0, # meters
                "range_window_upper_trim": 0, # meters
            }
            "instrument": {
                "across_track_beam_width": 15000, # meters
                "pulse_limited_footprint_size": 1600, # meters
                "across_track_doppler_footprint_width": 1600, # meters
            }
        geo_corrected_tracker_range (np.ndarray) : geo-corrected tracker range (NOT retracked)
        points_to_include (np.ndarray) : boolean array of points to include (False == reject)
    Returns:
        (np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray):
        (slope_correction, lat_poca, lon_poca, slope_ok, relocation_distance)

    """

    if thisdem is None:
        raise ValueError("thisdem None value passed")
    if thisdem_fine is None:
        raise ValueError("thisdem_fine None value passed")

    # ------------------------------------------------------------------------------------
    # Retrieve configuration parameters (to avoid repeatedly getting from config dict
    # during loops further down the function)
    # ------------------------------------------------------------------------------------

    across_track_beam_width = config["instrument"]["across_track_beam_width"]  # meters
    along_track_beam_width = config["instrument"]["along_track_beam_width"]  # meters

    across_track_doppler_footprint_width = config["instrument"][
        "across_track_doppler_footprint_width"
    ]  # m
    reference_bin_index = config["instrument"]["ref_bin_index"]
    range_bin_size = config["instrument"]["range_bin_size"]  # meters
    num_bins = config["instrument"]["num_range_bins"]

    # Additional options

    max_poca_reloc_distance = config["roemer_geolocation"]["max_poca_reloc_distance"]
    fine_grid_sampling = config["roemer_geolocation"]["fine_grid_sampling"]
    reject_outside_range_window = config["roemer_geolocation"]["reject_outside_range_window"]
    range_window_lower_trim = config["roemer_geolocation"]["range_window_lower_trim"]
    range_window_upper_trim = config["roemer_geolocation"]["range_window_upper_trim"]

    # ------------------------------------------------------------------------------------

    # Check that longitude is in range 0..360
    lons = lons % 360.0

    # Transform to X,Y locs in DEM projection
    nadir_x, nadir_y = thisdem.lonlat_to_xy_transformer.transform(
        lons, lats
    )  # pylint: disable=unpacking-non-sequence

    # Create working parameter arrays
    poca_x = np.full_like(nadir_x, dtype=float, fill_value=np.nan)
    poca_y = np.full_like(nadir_x, dtype=float, fill_value=np.nan)
    poca_z = np.full_like(nadir_x, dtype=float, fill_value=np.nan)
    slope_correction = np.full_like(nadir_x, dtype=float, fill_value=np.nan)
    slope_ok = np.full_like(nadir_x, dtype=bool, fill_value=False)
    relocation_distance = np.full_like(nadir_x, dtype=float, fill_value=np.nan)

    # ------------------------------------------------------------------------------------
    #  Loop through each track record
    # ------------------------------------------------------------------------------------

    headings = get_heading(nadir_x, nadir_y)

    for i, _ in enumerate(nadir_x):
        # By default, set POCA x,y to nadir, and height to Nan
        poca_x[i] = nadir_x[i]
        poca_y[i] = nadir_y[i]
        poca_z[i] = np.nan

        # if record is excluded due to previous checks, then skip
        if not points_to_include[i]:
            continue

        # Intialise beam doppler footprint rectangule in xy with centre at 0,0 as mpl path object
        beam_footprint_xy = path.Path(
            [
                (-along_track_beam_width / 2, -across_track_beam_width / 2),
                (-along_track_beam_width / 2, across_track_beam_width / 2),
                (along_track_beam_width / 2, across_track_beam_width / 2),
                (along_track_beam_width / 2, -across_track_beam_width / 2),
            ]
        )

        # rotate to heading
        beam_footprint_xy = beam_footprint_xy.transformed(
            transforms.Affine2D().rotate_deg(90 - headings[i])
        )

        # translate to nadir
        beam_footprint_xy = beam_footprint_xy.transformed(
            transforms.Affine2D().translate(nadir_x[i], nadir_y[i])
        )

        beam_footprint_xy_verts = np.array(
            beam_footprint_xy.vertices
        ).copy()  # get footprint vertices

        # get bfp vertex bounds, adjusted for dem posting
        x_max = np.max(beam_footprint_xy_verts[:, 0]) + thisdem.binsize
        x_min = np.min(beam_footprint_xy_verts[:, 0]) - thisdem.binsize
        y_max = np.max(beam_footprint_xy_verts[:, 1]) + thisdem.binsize
        y_min = np.min(beam_footprint_xy_verts[:, 1]) - thisdem.binsize

        segment = [(x_min, x_max), (y_min, y_max)]

        # Extract the rectangular segment from the DEM
        try:
            xdem, ydem, zdem = thisdem.get_segment(segment, grid_xy=True, flatten=False)
        except (IndexError, ValueError, TypeError, AttributeError, MemoryError):
            continue
        except Exception:  # pylint: disable=W0718
            continue

        # Step 1: find the DEM points within a circular area centred on the nadir
        # point corresponding to a radius of half the beam width
        xdem = xdem.flatten()
        ydem = ydem.flatten()
        zdem = zdem.flatten()

        # ----------------------------------------------------------------------
        # Identify indices of dem segment coords that are within xy beam footprint (top down)
        # ----------------------------------------------------------------------

        # Get boolean array checking whether dem coords are within the footprint
        beam_footprint_mask_xy = beam_footprint_xy.contains_points(np.column_stack((xdem, ydem)))

        if not np.any(beam_footprint_mask_xy):
            continue

        xdem = xdem[beam_footprint_mask_xy]
        ydem = ydem[beam_footprint_mask_xy]
        zdem = zdem[beam_footprint_mask_xy]

        # Check remaining DEM points for bad height values and remove
        nan_mask = np.isnan(zdem)
        include_only_good_zdem_indices = np.where(~nan_mask)[0]
        if len(include_only_good_zdem_indices) < 1:
            continue

        xdem = xdem[include_only_good_zdem_indices]
        ydem = ydem[include_only_good_zdem_indices]
        zdem = zdem[include_only_good_zdem_indices]

        # Only keep DEM heights which are in a sensible range
        # this step removes DEM values set to most fill_values
        valid_dem_heights = np.where(np.abs(zdem) < 5000.0)[0]
        if len(valid_dem_heights) < 1:
            continue

        xdem = xdem[valid_dem_heights]
        ydem = ydem[valid_dem_heights]
        zdem = zdem[valid_dem_heights]

        # Find the POCA location and slope correction to height
        (
            this_poca_x,
            this_poca_y,
            this_poca_z,
            slope_correction_to_height,
            range_to_sat_of_poca,
            flg_success,
        ) = find_poca(zdem, xdem, ydem, nadir_x[i], nadir_y[i], altitudes[i])

        if not flg_success:
            continue
        poca_x[i] = this_poca_x
        poca_y[i] = this_poca_y
        poca_z[i] = this_poca_z

        if fine_grid_sampling > 0:
            # Create finer grid resolution around poca
            # -------------------------------------

            # get the rectangular bounds around the approx POCA,
            # adjusted for pulse limited width and the dem posting

            # Initialise beam doppler footprint rectangule in xy with centre at 0,0
            # as mpl path object
            beam_footprint_xy = path.Path(
                [
                    (-along_track_beam_width / 2, -across_track_doppler_footprint_width / 2),
                    (-along_track_beam_width / 2, across_track_doppler_footprint_width / 2),
                    (along_track_beam_width / 2, across_track_doppler_footprint_width / 2),
                    (along_track_beam_width / 2, -across_track_doppler_footprint_width / 2),
                ]
            )

            # rotate to heading
            beam_footprint_xy = beam_footprint_xy.transformed(
                transforms.Affine2D().rotate_deg(90 - headings[i])
            )

            # translate to approximate POCA
            beam_footprint_xy = beam_footprint_xy.transformed(
                transforms.Affine2D().translate(poca_x[i], poca_y[i])
            )

            beam_footprint_xy_verts = np.array(
                beam_footprint_xy.vertices
            ).copy()  # get footprint vertices

            # get bfp vertex bounds, adjusted for dem posting
            x_max = np.max(beam_footprint_xy_verts[:, 0]) + thisdem_fine.binsize
            x_min = np.min(beam_footprint_xy_verts[:, 0]) - thisdem_fine.binsize
            y_max = np.max(beam_footprint_xy_verts[:, 1]) + thisdem_fine.binsize
            y_min = np.min(beam_footprint_xy_verts[:, 1]) - thisdem_fine.binsize

            segment = [(x_min, x_max), (y_min, y_max)]

            # Extract the rectangular segment from the fine DEM
            try:
                xdem, ydem, zdem = thisdem_fine.get_segment(segment, grid_xy=False, flatten=False)
            except (IndexError, ValueError, TypeError, AttributeError, MemoryError):
                continue
            except Exception:  # pylint: disable=W0718
                continue

            # Define new grid for finer resolution
            grid_x, grid_y = np.mgrid[
                x_min:x_max:fine_grid_sampling, y_min:y_max:fine_grid_sampling
            ]
            grid_x = grid_x.flatten()
            grid_y = grid_y.flatten()

            new_z = interpn(
                (np.flip(ydem.copy()), xdem),
                np.flip(zdem.copy(), 0),
                (grid_y, grid_x),
                method="linear",
                bounds_error=False,
                fill_value=np.nan,
            )

            # ----------------------------------------------------------------------
            # Identify indices of dem segment coords that are within xy beam footprint
            # (top down)
            # ----------------------------------------------------------------------

            # Get boolean array checking whether dem coords are within the footprint
            beam_footprint_mask_xy = beam_footprint_xy.contains_points(
                np.column_stack((grid_x, grid_y))
            )

            if not np.any(beam_footprint_mask_xy):
                continue

            xdem = grid_x[beam_footprint_mask_xy]
            ydem = grid_y[beam_footprint_mask_xy]
            zdem = new_z[beam_footprint_mask_xy]

            # Check remaining DEM points for bad height values and remove
            nan_mask = np.isnan(zdem)
            include_only_good_zdem_indices = np.where(~nan_mask)[0]
            if len(include_only_good_zdem_indices) < 1:
                continue

            xdem = xdem[include_only_good_zdem_indices]
            ydem = ydem[include_only_good_zdem_indices]
            zdem = zdem[include_only_good_zdem_indices]

            # Only keep DEM heights which are in a sensible range
            # this step removes DEM values set to most fill_values
            valid_dem_heights = np.where(np.abs(zdem) < 5000.0)[0]
            if len(valid_dem_heights) < 1:
                continue

            xdem = xdem[valid_dem_heights]
            ydem = ydem[valid_dem_heights]
            zdem = zdem[valid_dem_heights]

            # Find the POCA location and slope correction to height
            (
                this_poca_x,
                this_poca_y,
                this_poca_z,
                slope_correction_to_height,
                range_to_sat_of_poca,
                flg_success,
            ) = find_poca(zdem, xdem, ydem, nadir_x[i], nadir_y[i], altitudes[i])

            if not flg_success:
                continue

            poca_x[i] = this_poca_x
            poca_y[i] = this_poca_y
            poca_z[i] = this_poca_z

            if reject_outside_range_window:
                range_to_window_start = (
                    range_window_lower_trim
                    + geo_corrected_tracker_range[i]
                    - (reference_bin_index) * range_bin_size
                )
                range_to_window_end = (
                    geo_corrected_tracker_range[i]
                    + ((num_bins - reference_bin_index) * range_bin_size)
                    - range_window_upper_trim
                )
                if (range_to_sat_of_poca < range_to_window_start) or (
                    range_to_sat_of_poca > range_to_window_end
                ):
                    continue

        slope_correction[i] = slope_correction_to_height
        relocation_distance[i] = np.sqrt(
            (this_poca_x - nadir_x[i]) ** 2 + (this_poca_y - nadir_y[i]) ** 2
        )
        if relocation_distance[i] < max_poca_reloc_distance:
            slope_ok[i] = True

    # Transform all POCA x,y to lon,lat
    lon_poca, lat_poca = thisdem.xy_to_lonlat_transformer.transform(poca_x, poca_y)

    return (slope_correction, lat_poca, lon_poca, slope_ok, relocation_distance)
