"""
Slope correction/geolocation function using an adapted Roemer method 
from :
Roemer, S., Legrésy, B., Horwath, M., and Dietrich, R.: Refined
analysis of radar altimetry data applied to the region of the
subglacial Lake Vostok/Antarctica, Remote Sens. Environ., 106,
269–284, https://doi.org/10.1016/j.rse.2006.02.026, 2007.

Adaption by A.Muir, M.McMillan, Q.Huang (CPOM), includes a two stage POCA location, 
performed by minimizing the range to the satellite from DEM points located within firstly
a circular area around the nadir location of radius equal to half the beam width, 
and then within the pulse limited footprint around the initial POCA
at a finer resampled DEM resolution.

Next the calculated POCA's range to satellite is checked to see if it is
within the range window (configurably trimmed). If not the slope
correction is failed.

Finally the slope correction and height is calculated from

height[i] = altitudes[i]
                    - (geo_corrected_tracker_range[i] + retracker_correction[i])
                    + slope_correction[i]

"""

import logging
from datetime import datetime  # date and time functions
from typing import Tuple

import numpy as np
from scipy.interpolate import interpn
from scipy.ndimage import median_filter

from cpom.dems.dems import Dem

log = logging.getLogger(__name__)

EARTH_RADIUS = 6378137.0


def calculate_distances3d(
    x1_coord: float,
    y1_coord: float,
    z1_coord: float,
    x2_array: np.ndarray | list,
    y2_array: np.ndarray | list,
    z2_array: np.ndarray | list,
    squared_only=False,
) -> list[float]:
    """calculates the distances between a  refernce cartesian point (x1,y1,z1) in 3d space
    and a list of other points : x2[],y2[],z2[]

    Args:
        x1_coord (float): x coordinate of ref point
        y1_coord (float): y coordinate of ref point
        z1_coord (float): z coordinate of ref point
        x2_array (np.ndarray): list of x coordinates
        y2_array (np.ndarray): list of y coordinates
        z2_array (np.ndarray): list of z coordinates
        squared_only (bool) : if True, only calculate the squares of diffs and not sqrt
                              this will be faster, but doesn't give actual distances

    Returns:
        list[float]: list of distances between points x1,y1,z1 and x2[],y2[],z2[]
    """

    x2_array = np.array(x2_array)
    y2_array = np.array(y2_array)
    z2_array = np.array(z2_array)

    distances = (x2_array - x1_coord) ** 2 + (y2_array - y1_coord) ** 2 + (z2_array - z1_coord) ** 2

    if not squared_only:
        distances = np.sqrt(distances)

    return distances  # Convert back to a regular Python list


def calculate_distances2d(
    x1_coord: float,
    y1_coord: float,
    x2_array: np.ndarray,
    y2_array: np.ndarray,
    squared_only=False,
) -> np.ndarray:
    """calculates the distances between a  refernce cartesian point (x1,y1) in 2d space
    and a list of other points : x2[],y2[]

    Args:
        x1_coord (float): x coordinate of ref point
        y1_coord (float): y coordinate of ref point
        x2_array (np.ndarray): list of x coordinates
        y2_array (np.ndarray): list of y coordinates
        squared_only (bool) : if True, only calculate the squares of diffs and not sqrt
                              this will be faster, but doesn't give actual distances

    Returns:
        np.ndarray: list of distances between points x1,y1,z1 and x2[],y2[]
    """

    x2_array = np.array(x2_array)
    y2_array = np.array(y2_array)

    distances = (x2_array - x1_coord) ** 2 + (y2_array - y1_coord) ** 2

    if squared_only:
        distances = (x2_array - x1_coord) ** 2 + (y2_array - y1_coord) ** 2
    else:
        distances = np.sqrt((x2_array - x1_coord) ** 2 + (y2_array - y1_coord) ** 2)

    return distances


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


def geolocate_roemer(
    lats: np.ndarray,
    lons: np.ndarray,
    altitudes: np.ndarray,
    thisdem: Dem | None,
    thisdem_fine: Dem | None,
    config: dict,
    surface_type_20_ku: np.ndarray,
    geo_corrected_tracker_range: np.ndarray,
    retracker_correction: np.ndarray,
    points_to_include: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Geolocate CS2 LRM measurements using an adapted Roemer (Roemer et al, 2007) method

    Args:
        l1b (Dataset): NetCDF Dataset of L1b file
        thisdem (Dem): Dem object used for Roemer correction
        thisdem_fine (Dem): Dem object used for fine Roemer correction (maybe same obj as thisdem)
        config (dict): config dictionary containing
            "roemer_geolocation": {
                "fine_grid_sampling": 10,
                "max_poca_reloc_distance": 6600,
                "range_window_lower_trim": 0,
                "range_window_upper_trim": 0,
                "median_filter": False,
                "median_filter_width": 7,
                "reject_outside_range_window": True,
                "use_sliding_window": False,
            }
            "instrument": {
                "across_track_beam_width": 15000, # meters
                "pulse_limited_footprint_size": 1600, # meters
            }
        surface_type_20_ku (np.ndarray): surface type for track, where 1 == grounded_ice
        geo_corrected_tracker_range (np.ndarray) : geo-corrected tracker range (NOT retracked)
        retracker_correction (np.ndarray) : retracker correction to range (m)
        points_to_include (np.ndarray) : boolean array of points to include (False == reject)
    Returns:
        (np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray):
        (height_20_ku, lat_poca_20_ku, lon_poca_20_ku, slope_ok, relocation_distance)
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
    pulse_limited_footprint_size = config["instrument"]["pulse_limited_footprint_size"]  # m
    reference_bin_index = config["instrument"]["ref_bin_index"]
    range_bin_size = config["instrument"]["range_bin_size"]  # meters
    num_bins = config["instrument"]["num_range_bins"]

    # Additional options

    max_poca_reloc_distance = config["roemer_geolocation"]["max_poca_reloc_distance"]

    fine_grid_sampling = config["roemer_geolocation"]["fine_grid_sampling"]

    median_filter_dem_segment = config["roemer_geolocation"]["median_filter"]
    # Adjusted to be close to CS2 PLF width of 1600m.
    median_filter_width = config["roemer_geolocation"]["median_filter_width"]
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
    slope_ok = np.full_like(nadir_x, dtype=bool, fill_value=True)
    height_20_ku = np.full_like(nadir_x, dtype=float, fill_value=np.nan)
    relocation_distance = np.full_like(nadir_x, dtype=float, fill_value=np.nan)

    # ------------------------------------------------------------------------------------
    #  Loop through each track record
    # ------------------------------------------------------------------------------------

    for i, _ in enumerate(nadir_x):
        # By default, set POCA x,y to nadir, and height to Nan
        poca_x[i] = nadir_x[i]
        poca_y[i] = nadir_y[i]
        poca_z[i] = np.nan

        # if record is excluded due to previous checks, then skip
        if not points_to_include[i]:
            continue

        # get the rectangular bounds about the track, adjusted for across track beam width and
        # the dem posting
        x_min = nadir_x[i] - (across_track_beam_width / 2 + thisdem.binsize)
        x_max = nadir_x[i] + (across_track_beam_width / 2 + thisdem.binsize)
        y_min = nadir_y[i] - (across_track_beam_width / 2 + thisdem.binsize)
        y_max = nadir_y[i] + (across_track_beam_width / 2 + thisdem.binsize)

        segment = [(x_min, x_max), (y_min, y_max)]

        # Extract the rectangular segment from the DEM
        try:
            xdem, ydem, zdem = thisdem.get_segment(segment, grid_xy=True, flatten=False)
        except (IndexError, ValueError, TypeError, AttributeError, MemoryError):
            slope_ok[i] = False
            continue
        except Exception:  # pylint: disable=W0718
            slope_ok[i] = False
            continue

        if median_filter_dem_segment:
            smoothed_zdem = median_filter(zdem, size=median_filter_width)
            zdem = smoothed_zdem

        # Step 1: find the DEM points within a circular area centred on the nadir
        # point corresponding to a radius of half the beam width
        xdem = xdem.flatten()
        ydem = ydem.flatten()
        zdem = zdem.flatten()

        # Compute 2d distance between each dem location and nadir in (x,y)
        dem_to_nadir_dists = calculate_distances2d(nadir_x[i], nadir_y[i], xdem, ydem)

        # find where dem_to_nadir_dists is within beam. ie extract circular area
        include_dem_indices = np.where(
            np.array(dem_to_nadir_dists) < (across_track_beam_width / 2.0)
        )[0]
        if len(include_dem_indices) == 0:
            slope_ok[i] = False
            continue

        xdem = xdem[include_dem_indices]
        ydem = ydem[include_dem_indices]
        zdem = zdem[include_dem_indices]

        # Check remaining DEM points for bad height values and remove
        nan_mask = np.isnan(zdem)
        include_only_good_zdem_indices = np.where(~nan_mask)[0]
        if len(include_only_good_zdem_indices) < 1:
            slope_ok[i] = False
            continue

        xdem = xdem[include_only_good_zdem_indices]
        ydem = ydem[include_only_good_zdem_indices]
        zdem = zdem[include_only_good_zdem_indices]

        # Only keep DEM heights which are in a sensible range
        # this step removes DEM values set to most fill_values
        valid_dem_heights = np.where(np.abs(zdem) < 5000.0)[0]
        if len(valid_dem_heights) < 1:
            slope_ok[i] = False
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
            slope_ok[i] = False
            continue
        poca_x[i] = this_poca_x
        poca_y[i] = this_poca_y
        poca_z[i] = this_poca_z

        if fine_grid_sampling > 0:
            # Create finer grid resolution around poca
            # -------------------------------------

            # get the rectangular bounds around the approx POCA,
            # adjusted for pulse limited width and the dem posting

            x_min = poca_x[i] - (pulse_limited_footprint_size / 2 + thisdem.binsize)
            x_max = poca_x[i] + (pulse_limited_footprint_size / 2 + thisdem.binsize)
            y_min = poca_y[i] - (pulse_limited_footprint_size / 2 + thisdem.binsize)
            y_max = poca_y[i] + (pulse_limited_footprint_size / 2 + thisdem.binsize)

            segment = [(x_min, x_max), (y_min, y_max)]

            # Extract the rectangular segment from the DEM
            try:
                xdem, ydem, zdem = thisdem_fine.get_segment(segment, grid_xy=False, flatten=False)
            except (IndexError, ValueError, TypeError, AttributeError, MemoryError):
                slope_ok[i] = False
                continue
            except Exception:  # pylint: disable=W0718
                slope_ok[i] = False
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

            # Compute distance between dem points and the approximate POCA in (x,y,z)
            dem_to_poca_dists = calculate_distances2d(poca_x[i], poca_y[i], grid_x, grid_y)

            # find where dem_to_poca_dists is within PLF. ie extract circular area
            include_dem_indices = np.where(
                np.array(dem_to_poca_dists) < (pulse_limited_footprint_size / 2.0)
            )[0]
            if len(include_dem_indices) == 0:
                slope_ok[i] = False
                continue

            xdem = grid_x[include_dem_indices]
            ydem = grid_y[include_dem_indices]
            zdem = new_z[include_dem_indices]

            # Check remaining DEM points for bad height values and remove
            nan_mask = np.isnan(zdem)
            include_only_good_zdem_indices = np.where(~nan_mask)[0]
            if len(include_only_good_zdem_indices) < 1:
                slope_ok[i] = False
                continue

            xdem = xdem[include_only_good_zdem_indices]
            ydem = ydem[include_only_good_zdem_indices]
            zdem = zdem[include_only_good_zdem_indices]

            # Only keep DEM heights which are in a sensible range
            # this step removes DEM values set to most fill_values
            valid_dem_heights = np.where(np.abs(zdem) < 5000.0)[0]
            if len(valid_dem_heights) < 1:
                slope_ok[i] = False
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
                slope_ok[i] = False
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
                    slope_ok[i] = False
                    continue

            slope_correction[i] = slope_correction_to_height
            relocation_distance[i] = np.sqrt(
                (this_poca_x - nadir_x[i]) ** 2 + (this_poca_y - nadir_y[i]) ** 2
            )
            if relocation_distance[i] > max_poca_reloc_distance:
                slope_ok[i] = False

        if config["roemer_geolocation"]["use_sliding_window"]:
            # Step 1: Calculate all distances once
            all_distances_flat = calculate_distances3d(
                x1_coord=nadir_x[i],
                y1_coord=nadir_y[i],
                z1_coord=altitudes[i],
                x2_array=xdem.flatten(),
                y2_array=ydem.flatten(),
                z2_array=zdem.flatten(),
                squared_only=False,
            )

            all_distances = np.array(all_distances_flat).reshape(xdem.shape)

            # Step 2: Slide the window over the pre-calculated distance grid
            min_distance = np.inf
            min_position = (0, 0)
            window_size = (int)(pulse_limited_footprint_size / thisdem.binsize)

            for ii in range(all_distances.shape[0] - window_size + 1):
                for jj in range(all_distances.shape[1] - window_size + 1):
                    # Extract the current window of distances
                    window = all_distances[ii : ii + window_size, jj : jj + window_size]

                    # Calculate the mean distance within the window
                    mean_distance = np.mean(window)

                    # Update the minimum mean and position if a new minimum is found
                    if mean_distance < min_distance:
                        min_distance = mean_distance
                        min_position = (ii, jj)

            # min_position is the position of the window with the smallest mean distance

            # --------------------------------------------------------------------------------------
            #  Find Location of POCA x,y
            # --------------------------------------------------------------------------------------

            poca_x[i] = xdem[min_position]
            poca_y[i] = ydem[min_position]

            # --------------------------------------------------------------------------------------
            #  Find Location of POCA z
            # --------------------------------------------------------------------------------------

            poca_z[i] = zdem[min_position]

            # --------------------------------------------------------------------------------------
            #  Calculate Slope Correction
            # --------------------------------------------------------------------------------------

            dem_to_sat_dists = calculate_distances3d(
                nadir_x[i],
                nadir_y[i],
                altitudes[i],
                [poca_x[i]],
                [poca_y[i]],
                [poca_z[i]],
            )

            # Calculate the slope correction to height
            slope_correction[i] = dem_to_sat_dists[0] + poca_z[i] - altitudes[i]

    # Transform all POCA x,y to lon,lat
    lon_poca_20_ku, lat_poca_20_ku = thisdem.xy_to_lonlat_transformer.transform(poca_x, poca_y)

    # Calculate height as altitude-(corrected range)+slope_correction
    height_20_ku = np.full_like(lats, np.nan)

    for i in range(len(lats)):  # pylint: disable=consider-using-enumerate
        if np.isfinite(geo_corrected_tracker_range[i]):
            if slope_ok[i] and surface_type_20_ku[i] == 1:  # grounded ice type only
                height_20_ku[i] = (
                    altitudes[i]
                    - (geo_corrected_tracker_range[i] + retracker_correction[i])
                    + slope_correction[i]
                )
            else:
                height_20_ku[i] = np.nan
        else:
            height_20_ku[i] = np.nan

        # Set POCA lat,lon to nadir if no slope correction
        if (
            (not np.isfinite(lat_poca_20_ku[i]))
            or (not np.isfinite(lon_poca_20_ku[i]))
            or (not slope_ok[i])
        ):
            lat_poca_20_ku[i] = lats[i]
            lon_poca_20_ku[i] = lons[i]
            height_20_ku[i] = np.nan

    return (height_20_ku, lat_poca_20_ku, lon_poca_20_ku, slope_ok, relocation_distance)
