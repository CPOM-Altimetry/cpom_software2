"""
Script to generate Cryo-TEMPO uncertainty lookup table, from specified inputs.

Uncertainty lookup table consists of binned difference values. Differences can be binned
by 1-5 variables, and values for each bin can be calculated using a number of metrics 
(median, MAD, RMSE, standard deviation upper estimate)

Only available for Antarctic currently.

Compatible for the following variables:
- Slope: interpolated from Slopes("rema_100m_900ws_slopes_zarr")
- Roughness: interpolated from Roughness ("rema_100m_900ws_roughness_zarr)
- Power: sigma 0 from L2i product
- Coherence: coherence from L2i product

Input is dh values, lat, lon from CS2-IS2 differences npz files, for example:
cs2_minus_is2_gt2lgt2r_p2p_diffs_antarctica_icesheets.npz

example usage: 
python clev2er_multilinear_uncertainty.py -area antarctica_icesheets -m median \
    -dh_file ~/downloads/cs2_minus_is2_gt2lgt2r_p2p_diffs_antarctica_icesheets.npz

Author: Karla Boxall

"""
# ---------------------------------------------------------------------------------------------------------------------
# Package Imports
# ---------------------------------------------------------------------------------------------------------------------

import argparse
import sys

import numpy as np
import pandas as pd
import math
from scipy.stats.distributions import chi2
from sklearn.linear_model import LinearRegression

from cpom.roughness.roughness import Roughness
from cpom.slopes.slopes import Slopes

# ---------------------------------------------------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------------------------------------------------

def calculate_binned_metric(
    delta_elevation: np.ndarray,
    slope: np.ndarray,
    roughness: np.ndarray,
    power: np.ndarray,
    coherence: np.ndarray,
    distance: np.ndarray,
    slope_bins: np.ndarray,
    roughness_bins: np.ndarray,
    power_bins: np.ndarray,
    coherence_bins: np.ndarray,
    distance_bins: np.ndarray,
    method: str,
) -> pd.DataFrame:

    """
    Calculate the binned elevation difference values.

    Args:
        delta_elevation (np.ndarray): Array of elevation differences between
                                      two measurement techniques.
        slope (np.ndarray): Array of surface slopes in meters.
        roughness (np.ndarray): Array of surface roughness values in meters.
        power (np.ndarray): Array of power (sigma 0) values.
        coherence (np.ndarray): Array of coherence (coherence_20_ku) values.
        distance (np.ndarray): Array of distance from POCA (nadir coord vs poca coord) values.
        slope_bins (np.ndarray): Bins to categorize slope values.
        roughness_bins (np.ndarray): Bins to categorize roughness values.
        power_bins (np.ndarray): Bins to categorise power values.
        coherence_bins (np.ndarray): Bins to categorise coherence values.
        distance_bins (np.ndarray): Bins to categorise distance values.
        method (str): Metric to calculate uncertainty within bins.

    Returns:
        pd.DataFrame: A pivot table where rows correspond to slope bins, columns to roughness 
                    and power bins, and values to the calculated elevation difference metric
                    within each bin.
    """

    # Create a DataFrame to hold the data
    data = pd.DataFrame(
        {
            "delta_elevation": np.abs(delta_elevation),  # Absolute elevation difference
            "slope": slope,
            "roughness": roughness,
            "power": power,
            "coherence": coherence,
            "distance": distance,
        }
    )

    # Bin the values
    data["slope_bin"] = pd.cut(
        data["slope"], bins=slope_bins, include_lowest=True, labels=slope_bins[:-1]
    )
    data["roughness_bin"] = pd.cut(
        data["roughness"], bins=roughness_bins, include_lowest=True, labels=roughness_bins[:-1]
    )
    data["power_bin"] = pd.cut(
        data["power"], bins=power_bins, include_lowest=True, labels=power_bins[:-1]
    )
    data["coherence_bin"] = pd.cut(
        data["coherence"], bins=coherence_bins, include_lowest=True, labels=coherence_bins[:-1]
    )
    data["distance_bin"] = pd.cut(
        data["distance"], bins=distance_bins, include_lowest=True, labels=distance_bins[:-1]
    )

    # calculate the metric within each bin
    # use median as metric
    if method == 'median':
        binned_metric = (
            data.assign(abs_delta_elevation=data["delta_elevation"].abs())
            .groupby(["slope_bin", "roughness_bin", "power_bin", "coherence_bin", "distance_bin"])["abs_delta_elevation"]
            .median()
            .reset_index()
        )
    
    # use MAD as metric
    elif method == 'mad':
        def mad(x):
            return np.median(np.abs(x - np.median(x)))

        binned_metric = (
            data.assign(abs_delta_elevation=data["delta_elevation"].abs())
            .groupby(["slope_bin", "roughness_bin", "power_bin", "coherence_bin", "distance_bin"])["abs_delta_elevation"].apply(mad).reset_index()
        )

    # use RMSE as metric
    elif method == 'rmse':
        def rmse(x):
            mse = np.square(x).mean()
            return math.sqrt(mse)

        binned_metric = (
            data.assign(abs_delta_elevation=data["delta_elevation"].abs())
            .groupby(["slope_bin", "roughness_bin", "power_bin", "coherence_bin", "distance_bin"])["abs_delta_elevation"].apply(rmse).reset_index()
        )

    # use upper estimate of standard deviation as metric
    elif method == 'std_ue':
        def std_ue(x):
            std_ue = np.std(x) * np.sqrt((len(x)-1)/chi2.ppf(0.975, len(x)-1)) 
            return std_ue

        binned_metric = (
            data.assign(abs_delta_elevation=data["delta_elevation"].abs())
            .groupby(["slope_bin", "roughness_bin", "power_bin", "coherence_bin", "distance_bin"])["abs_delta_elevation"].apply(std_ue).reset_index()
        )

    # Pivot the table to create a 3D matrix where rows are slope_bins and columns are roughness_bins and power_bins
    binned_metric_pivot = binned_metric.pivot(
        index="slope_bin", columns=["roughness_bin", "power_bin", "coherence_bin", "distance_bin"], values="abs_delta_elevation"
    )

    # print(binned_metric_pivot.stack(future_stack=True))

    return binned_metric_pivot


def calc_uncertainty_table(
    delta_elevation: np.ndarray,
    slope: np.ndarray,
    roughness: np.ndarray,
    power: np.ndarray,
    coherence: np.ndarray,
    distance: np.ndarray,
    slope_bins: np.ndarray,
    roughness_bins: np.ndarray,
    power_bins: np.ndarray,
    coherence_bins: np.ndarray,
    distance_bins: np.ndarray,
    method: str = "median",
) -> pd.DataFrame:

    """
    Main function to calculate the multi-dimensional uncertainty table.

    Args:
        delta_elevation (np.ndarray): Array of elevation differences between two
                                      measurement techniques.
        slope (np.ndarray): Array of surface slopes in meters.
        roughness (np.ndarray): Array of surface roughness values in meters.
        power (np.ndarray): Array of power values.
        coherence (np.ndarray): Array of coherence values.
        distance (np.ndarray): Array of distance values.
        slope_bins (np.ndarray): Bins to categorize slope values.
        roughness_bins (np.ndarray): Bins to categorize roughness values.
        power_bins (np.ndarray): Bins to categorise power values.
        coherence_bins (np.ndarray): Bins to categorise coherence values.
        distance_bins (np.ndarray): Bins to categorise distance values.
        method (str): Method to calculate the uncertainty ('median' 'mad' 'rmse' or 'std_ue'). Default is 'median'.

    Returns:
        pd.DataFrame: A pivot table where rows correspond to slope bins, columns to additional variables,
                      and values to the uncertainty metric within each bin.
    """
    binned_table = calculate_binned_metric(
        delta_elevation, slope, roughness, power, coherence, distance,
        slope_bins, roughness_bins, power_bins, coherence_bins, distance_bins, method,
    )

    return binned_table


def get_binned_values(
    slope_values: np.ndarray,
    roughness_values: np.ndarray,
    power_values: np.ndarray,
    coherence_values: np.ndarray,
    distance_values: np.ndarray,
    binned_table: pd.DataFrame,
    slope_bins: np.ndarray,
    roughness_bins: np.ndarray,
    power_bins: np.ndarray,
    coherence_bins: np.ndarray,
    distance_bins: np.ndarray,
) -> np.ndarray:

    """
    Retrieve the calculated metric of elevation difference for given variable values (ie for given
    slope, roughness and power values).

    Args:
        slope_values (np.ndarray): Array of slope values for which to retrieve calculated differences.
        roughness_values (np.ndarray): Array of roughness values for which to retrieve calculated differences.
        power_values (np.ndarray): Array of power values for which to retrieve calculated differences.
        coherence_values (np.ndarray): Array of coherence values for which to retrieve calculated differences.
        distance_values(np.ndarray): Array of distance values for which to retreive calculated difference.
        binned_table (pd.DataFrame): A pivot table of binned median absolute elevation differences.
        slope_bins (np.ndarray): Bins to categorize slope values.
        roughness_bins (np.ndarray): Bins to categorize roughness values.
        power_bins (np.ndarray): Bins to categorise power values.
        coherence_bins (np.ndarray): Bins to categorise coherence values.
        distance_bins (np.ndarray): Bins to categorise distance values.

    Returns:
        np.ndarray: An array of median absolute elevation differences corresponding to the
                    input values.
    """
    # Convert values to numpy arrays
    slope_values = np.asarray(slope_values)
    roughness_values = np.asarray(roughness_values)
    power_values = np.asarray(power_values)
    coherence_values = np.asarray(coherence_values)
    distance_values = np.asarray(distance_values)

    # Find the slope bin indices for the array of slope_values
    slope_bin_indices = np.digitize(slope_values, slope_bins) - 1
    slope_bin_indices = np.clip(
        slope_bin_indices, 0, len(slope_bins) - 2
    )  # Ensure indices are within range

    # Find the roughness bin indices for the array of roughness_values
    roughness_bin_indices = np.digitize(roughness_values, roughness_bins) - 1
    roughness_bin_indices = np.clip(
        roughness_bin_indices, 0, len(roughness_bins) - 2
    )  # Ensure indices are within range

    # Find the power bin indices for the array of power
    power_bin_indices = np.digitize(power_values, power_bins) - 1
    power_bin_indices = np.clip(
        power_bin_indices, 0, len(power_bins) - 2
    )  # Ensure indices are within range

    # Find the coherence bin indices for the array of coherence
    coherence_bin_indices = np.digitize(coherence_values, coherence_bins) - 1
    coherence_bin_indices = np.clip(
        coherence_bin_indices, 0, len(coherence_bins) - 2
    )  # Ensure indices are within range

    # Find the coherence bin indices for the array of coherence
    distance_bin_indices = np.digitize(distance_values, distance_bins) - 1
    distance_bin_indices = np.clip(
        distance_bin_indices, 0, len(distance_bins) - 2
    )  # Ensure indices are within range

    # Convert bin labels to row and column labels in the DataFrame
    row_indices = [slope_bins[idx] for idx in slope_bin_indices]
    col1_indices = [roughness_bins[idx] for idx in roughness_bin_indices]
    col2_indices = [power_bins[idx] for idx in power_bin_indices]
    col3_indices = [coherence_bins[idx] for idx in coherence_bin_indices]
    col4_indices = [distance_bins[idx] for idx in distance_bin_indices]

    binned_table_df = binned_table.stack(future_stack=True)
    # print(binned_table_df)

    # Retrieve the values using numpy indexing on the DataFrame values
    values = [binned_table_df.loc[row_indices[i]][col1_indices[i]][col2_indices[i]][col3_indices[i]][col4_indices[i]] for i in range(len(row_indices))]

    return values


def fit_linear_model(
    binned_table: np.ndarray,
    slope_bins: np.ndarray,
    roughness_bins: np.ndarray,
    power_bins: np.ndarray,
    coherence_bins: np.ndarray,
    distance_bins: np.ndarray,
) -> np.ndarray:
    """
    Fit a linear model to the lookup table to fill 'NaNs' using interpolation and extrapolation.

    Args:
        binned_table (np.ndarray): Uncertainty lookup table as a pivot table.
        slope_bins (np.ndarray): Bins to categorize slope values.
        roughness_bins (np.ndarray): Bins to categorize roughness values.
        power_bins (np.ndarray): Bins to categorise power values.
        coherence_bins (np.ndarray): Bins to categorise coherence values.
        distance_bins (np.ndarray): Bins to categorise distance values.

    Returns:
        np.ndarray: The pivot table with missing values filled using a linear model fit.
    
    NOTE: this is currently replacing existing values to fit model. Should this just be 
    filling empty spaces, rather than replacing existing lookup table values?

    """
    slope_bins = slope_bins[:-1]
    roughness_bins = roughness_bins[:-1]
    power_bins = power_bins[:-1]
    coherence_bins = coherence_bins[:-1]
    distance_bins = distance_bins[:-1]

    # Reshaping data into long format for fitting
    slope, roughness, power, coherence, distance = np.meshgrid(
        slope_bins, roughness_bins, power_bins, coherence_bins, distance_bins, indexing="ij")
    slope = slope.flatten()
    roughness = roughness.flatten()
    power = power.flatten()
    coherence = coherence.flatten()
    distance = distance.flatten()
    z_values = binned_table.values.flatten()  # Uncertainty values

    # Remove NaN values before fitting
    mask = ~np.isnan(z_values)
    slope_clean = slope[mask]
    roughness_clean = roughness[mask]
    power_clean = power[mask]
    coherence_clean = coherence[mask]
    distance_clean = distance[mask]
    z_clean = z_values[mask]

    # Create interaction term (slope * roughness)
    interaction_term = slope_clean * roughness_clean * power_clean * coherence_clean * distance_clean

    # Prepare the features matrix
    x_matrix = np.vstack([slope_clean, roughness_clean, power_clean, coherence_clean, distance_clean, interaction_term]).T

    # Perform linear regression
    reg = LinearRegression()
    reg.fit(x_matrix, z_clean)

    # Coefficients of the model
    a, b, c, d, e, f = reg.coef_
    g = reg.intercept_

    # Fit function
    def linear_fit(slope, roughness, power, coherence, distance):
        return a * slope + b * roughness + c * power + d * coherence + e * distance + f * slope * roughness * power * coherence * distance + g

    # Apply the fit to the original grid
    linear_table = pd.DataFrame(
        linear_fit(slope, roughness, power, coherence, distance).reshape(binned_table.shape), index=binned_table.index, columns=binned_table.columns
    )

    # Replace negative values with values from the original table
    linear_table = linear_table.where(linear_table >= 0, binned_table)

    return linear_table

# -------------------------------------------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------------------------------------------

def main():

    """main function for command line tool"""

    # initiate the command line parser
    parser = argparse.ArgumentParser()

    # add each argument
    parser.add_argument(
        "--area",
        "-a",
        choices=['antarctica_icesheets'],
        help=("choose the ice sheet: 'median' (default)"),
    )

    parser.add_argument(
        "--method",
        "-m",
        choices=["median", "mad", "rmse", "std_ue"],
        default="median",
        help=("choose the calculation method: 'median' (default) or 'mad' or 'rmse' or 'std_ci'"),
    )

    parser.add_argument(
        "-dh_file",
        "-dh",
        help=(
            "path of elevation difference npz file (for example "
            "/path/to/cs2_minus_is2_gt1lgt1rgt2lgt2rgt3lgt3r_p2p_diffs_antarctica_icesheets.npz)"
        ),
        type=str,
    )

    # read arguments from the command line
    args = parser.parse_args()

    if args.area != 'antarctica_icesheets':
        sys.exit("Must specify 'antarctica_icesheets'")

    # Define bins 
    the_slope_bins = np.arange(0, 2.1, 0.1)  # Define slope bins in degrees (0.1 degree steps from 0 to 2 degrees)
    the_roughness_bins = np.arange(0, 2.1, 0.1)  # Define roughness bins in meters (0.1 m steps from 0 to 2 meters)
    the_power_bins = np.arange(0, 10, 1)  # Define power bins (1 steps from 0 to 9)
    the_coherence_bins = np.arange(0, 1.1, 0.1)  # Define coherence bins (0.1 steps from 0 to 1)
    the_distance_bins = np.arange(0, 100, 10)  # Define distance bins (10 steps from 0 to 100)

    if args.area == 'antarctica_icesheets':
        this_slope = Slopes("rema_100m_900ws_slopes_zarr")
        this_roughness = Roughness("rema_100m_900ws_roughness_zarr")

    # Read npz file to get dh,lat,lon values
    print(f"reading npz file {args.dh_file}...")
    dh_data = np.load(args.dh_file, allow_pickle=True)

    # Extract lat, lon & elevation difference
    lats = dh_data.get("lats")
    lons = dh_data.get("lons")
    dh = dh_data.get("dh")
    
    # Extract variables
    print('Fetching slope data')
    slope = this_slope.interp_slopes(lats, lons, method="linear", xy_is_latlon=True)
    print('Fetching roughness data')
    roughness = this_roughness.interp_roughness(lats, lons, method="linear", xy_is_latlon=True)
    print('Fetching power data')
    power = dh_data.get("pow")
    print('Fetching coherence data')
    coherence = dh_data.get("coh")
    print('Fetching distance data')
    distance = dh_data.get("dis_poca")
    
    # calculate uncertainty lookup table
    print("calculating uncertainty table... ")
    binned_table = calc_uncertainty_table(
        dh,
        slope,
        roughness,
        power,
        coherence,
        distance,
        the_slope_bins,
        the_roughness_bins,
        the_power_bins,
        the_coherence_bins,
        the_distance_bins,
        method=args.method
    )
    print(binned_table)

    # fill the nans using linear interpolation
    binned_table_filled = fit_linear_model(
        binned_table,
        the_slope_bins,
        the_roughness_bins,
        the_power_bins,
        the_coherence_bins,
        the_distance_bins
    )
    print(binned_table_filled)

    # Test extraction from lookup table
    slope_values = [0.15, 0.4]  
    roughness_values = [0.25, 0.3]  
    power_values = [-1, 6]  
    coherence_values = [0.2, 0.8]
    distance_values = [37, 82]

    # Extract the corresponding value from the binned_table_filled for test values
    values = get_binned_values(
        slope_values, roughness_values, power_values, coherence_values, distance_values, binned_table_filled, 
        the_slope_bins, the_roughness_bins, the_power_bins, the_coherence_bins, the_distance_bins
    )
    print(
        f"Uncertainty for slope {slope_values} and roughness {roughness_values} and power {power_values} and coherence {coherence_values} and distance {distance_values}: {values} m")

    # Extract the corresponding value from the binned_table_filled for entire dataset
    print(f"Getting slope values from {this_slope.name}......")
    slope_values = this_slope.interp_slopes(lats, lons, method="linear", xy_is_latlon=True)
    print(f"Getting roughness values from {this_roughness.name}...")
    roughness_values = this_roughness.interp_roughness(
        lats, lons, method="linear", xy_is_latlon=True
    )
    print(f"Getting power values from [X] ...")
    # NOTE: need to decide upon input data for this. Needs to be the power value for the lat/lon of any given elevation value.
    print(f"Getting coherence values from [X] ...")
    # NOTE: need to decide upon input data for this. Needs to be the coherence value for the lat/lon of any given elevation value.
    print(f"Getting distance values from [X] ...")
    # NOTE: need to decide upon input data for this. Needs to be the distance value for the lat/lon of any given elevation value.
    # NOTE: ONLY NEED THE UNCERTAINTY VALUES FOR JOINED ELEVATION VALUES FOR THE UNCERTAINTY ASSESSMENT

    print("Getting uncertainty values from table...")
    # Extract the corresponding value from the binned_table
    # values = get_binned_values(
    #     slope_values, roughness_values, power_values, coherence_values, distance_values, binned_table_filled, the_slope_bins, the_roughness_bins, the_power_bins, the_coherence_bins, the_distance_bins
    #     )
    
    # save the final uncertainties to a pickle table?
    # insert code to do that here


if __name__ == "__main__":
    main()
