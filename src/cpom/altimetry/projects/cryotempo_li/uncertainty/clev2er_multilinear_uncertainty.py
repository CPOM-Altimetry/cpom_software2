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
    slope_bins: np.ndarray,
    roughness_bins: np.ndarray,
    power_bins: np.ndarray,
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
        slope_bins (np.ndarray): Bins to categorize slope values.
        roughness_bins (np.ndarray): Bins to categorize roughness values.
        power_bins (np.ndarray): Bins to categorise power values.
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

    # calculate the metric within each bin
    # use median as metric
    if method == 'median':
        binned_metric = (
            data.assign(abs_delta_elevation=data["delta_elevation"].abs())
            .groupby(["slope_bin", "roughness_bin", "power_bin"])["abs_delta_elevation"]
            .median()
            .reset_index()
        )
    
    # use MAD as metric
    elif method == 'mad':
        def mad(x):
            return np.median(np.abs(x - np.median(x)))

        binned_metric = (
            data.assign(abs_delta_elevation=data["delta_elevation"].abs())
            .groupby(["slope_bin", "roughness_bin", "power_bin"])["abs_delta_elevation"].apply(mad).reset_index()
        )

    # use RMSE as metric
    elif method == 'rmse':
        def rmse(x):
            mse = np.square(x).mean()
            return math.sqrt(mse)

        binned_metric = (
            data.assign(abs_delta_elevation=data["delta_elevation"].abs())
            .groupby(["slope_bin", "roughness_bin", "power_bin"])["abs_delta_elevation"].apply(rmse).reset_index()
        )

    # use upper estimate of standard deviation as metric
    elif method == 'std_ue':
        def std_ue(x):
            std_ue = np.std(x) * np.sqrt((len(x)-1)/chi2.ppf(0.975, len(x)-1)) 
            return std_ue

        binned_metric = (
            data.assign(abs_delta_elevation=data["delta_elevation"].abs())
            .groupby(["slope_bin", "roughness_bin", "power_bin"])["abs_delta_elevation"].apply(std_ue).reset_index()
        )

    # Pivot the table to create a 3D matrix where rows are slope_bins and columns are roughness_bins and power_bins
    binned_metric_pivot = binned_metric.pivot(
        index="slope_bin", columns=["roughness_bin", "power_bin"], values="abs_delta_elevation"
    )

    # print(binned_metric_pivot.stack(future_stack=True))

    return binned_metric_pivot


def calc_uncertainty_table(
    delta_elevation: np.ndarray,
    slope: np.ndarray,
    roughness: np.ndarray,
    power: np.ndarray,
    slope_bins: np.ndarray,
    roughness_bins: np.ndarray,
    power_bins: np.ndarray,
    method: str = "median",
) -> pd.DataFrame:

    """
    Main function to calculate the multi-dimensional uncertainty table.

    Args:
        delta_elevation (np.ndarray): Array of elevation differences between two
                                      measurement techniques.
        slope (np.ndarray): Array of surface slopes in meters.
        roughness (np.ndarray): Array of surface roughness values in meters.
        power (np.ndarray): Array if power values.
        slope_bins (np.ndarray): Bins to categorize slope values.
        roughness_bins (np.ndarray): Bins to categorize roughness values.
        power_bins (np.ndarray): Bins to categorise power values.
        method (str): Method to calculate the uncertainty ('median' 'mad' 'rmse' or 'std_ue'). Default is 'median'.

    Returns:
        pd.DataFrame: A pivot table where rows correspond to slope bins, columns to additional variables,
                      and values to the uncertainty metric within each bin.
    """
    binned_table = calculate_binned_metric(
        delta_elevation, slope, roughness, power, slope_bins, roughness_bins, power_bins, method,
    )

    return binned_table


def get_binned_values(
    slope_values: np.ndarray,
    roughness_values: np.ndarray,
    power_values: np.ndarray,
    binned_table: pd.DataFrame,
    slope_bins: np.ndarray,
    roughness_bins: np.ndarray,
    power_bins: np.ndarray,
) -> np.ndarray:

    """
    Retrieve the calculated metric of elevation difference for given variable values (ie for given
    slope, roughness and power values).

    Args:
        slope_values (np.ndarray): Array of slope values for which to retrieve median differences.
        roughness_values (np.ndarray): Array of roughness values for which to retrieve median differences.
        power_values (np.ndarray): Array of power values for which to retrieve median differences.
        binned_table (pd.DataFrame): A pivot table of binned median absolute elevation differences.
        slope_bins (np.ndarray): Bins to categorize slope values.
        roughness_bins (np.ndarray): Bins to categorize roughness values.
        power_bins (np.ndarray): Bins to categorise power values.

    Returns:
        np.ndarray: An array of median absolute elevation differences corresponding to the
                    input values.
    """
    # Convert values to numpy arrays
    slope_values = np.asarray(slope_values)
    roughness_values = np.asarray(roughness_values)
    power_values = np.asarray(power_values)

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

    # Convert bin labels to row and column labels in the DataFrame
    row_indices = [slope_bins[idx] for idx in slope_bin_indices]
    col1_indices = [roughness_bins[idx] for idx in roughness_bin_indices]
    col2_indices = [power_bins[idx] for idx in power_bin_indices]

    binned_table_df = binned_table.stack(future_stack=True)
    # print(binned_table_df)

    # Retrieve the values using numpy indexing on the DataFrame values
    values = [binned_table_df.loc[row_indices[i]][col1_indices[i]][col2_indices[i]] for i in range(len(row_indices))]

    return values


def fit_linear_model(
    binned_table: np.ndarray,
    slope_bins: np.ndarray,
    roughness_bins: np.ndarray,
    power_bins: np.ndarray,
) -> np.ndarray:
    """
    Fit a linear model to the lookup table to fill 'NaNs' using interpolation and extrapolation.

    Args:
        binned_table (np.ndarray): Uncertainty lookup table as a pivot table.
        slope_bins (np.ndarray): Bins to categorize slope values.
        roughness_bins (np.ndarray): Bins to categorize roughness values.
        power_bins (np.ndarray): Bins to categorise power values.

    Returns:
        np.ndarray: The pivot table with missing values filled using a linear model fit.
    
    NOTE: this is currently replacing existing values to fit model. Should this just be 
    filling empty spaces, rather than replacing existing lookup table values?

    """
    slope_bins = slope_bins[:-1]
    roughness_bins = roughness_bins[:-1]
    power_bins = power_bins[:-1]

    # Reshaping data into long format for fitting
    slope, roughness, power = np.meshgrid(slope_bins, roughness_bins, power_bins, indexing="ij")
    slope = slope.flatten()
    roughness = roughness.flatten()
    power = power.flatten()
    z_values = binned_table.values.flatten()  # Uncertainty values

    # Remove NaN values before fitting
    mask = ~np.isnan(z_values)
    slope_clean = slope[mask]
    roughness_clean = roughness[mask]
    power_clean = power[mask]
    z_clean = z_values[mask]

    # Create interaction term (slope * roughness)
    interaction_term = slope_clean * roughness_clean * power_clean

    # Prepare the features matrix
    x_matrix = np.vstack([slope_clean, roughness_clean, power_clean, interaction_term]).T

    # Perform linear regression
    reg = LinearRegression()
    reg.fit(x_matrix, z_clean)

    # Coefficients of the model
    a, b, c, d = reg.coef_
    e = reg.intercept_

    # Fit function
    def linear_fit(slope, roughness, power):
        return a * slope + b * roughness + c * power + d * slope * roughness * power + e

    # Apply the fit to the original grid
    linear_table = pd.DataFrame(
        linear_fit(slope, roughness, power).reshape(binned_table.shape), index=binned_table.index, columns=binned_table.columns
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
    the_power_bins = np.arange(0, 10, 1)  # Define power bins in meters (0.1 m steps from 0 to 2 meters)

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
    
    # calculate uncertainty lookup table
    print("calculating uncertainty table... ")
    binned_table = calc_uncertainty_table(
        dh,
        slope,
        roughness,
        power,
        the_slope_bins,
        the_roughness_bins,
        the_power_bins,
        method=args.method
    )
    print(binned_table)

    # fill the nans using linear interpolation
    binned_table_filled = fit_linear_model(
        binned_table,
        the_slope_bins,
        the_roughness_bins,
        the_power_bins
    )
    print(binned_table_filled)

    # Test extraction from lookup table
    slope_values = [0.15, 0.4]  
    roughness_values = [0.25, 0.3]  
    power_values = [-1, 6]  

    # Extract the corresponding value from the binned_table_filled for test values
    values = get_binned_values(
        slope_values, roughness_values, power_values, binned_table_filled, the_slope_bins, the_roughness_bins, the_power_bins,
    )
    print(f"Uncertainty for slope {slope_values} and roughness {roughness_values} and power {power_values}: {values} m")

    # Extract the corresponding value from the binned_table_filled for entire dataset
    print(f"Getting slope values from {this_slope.name}......")
    slope_values = this_slope.interp_slopes(lats, lons, method="linear", xy_is_latlon=True)
    print(f"Getting roughness values from {this_roughness.name}...")
    roughness_values = this_roughness.interp_roughness(
        lats, lons, method="linear", xy_is_latlon=True
    )
    print(f"Getting power values from [X] ...")
    # NOTE: need to decide upon input data for this. Needs to be the power value for the lat/lon of any given elevation value.

    print("Getting uncertainty values from table...")
    # Extract the corresponding value from the binned_table
    # values = get_binned_values(
    #     slope_values, roughness_values, power_values, binned_table_filled, the_slope_bins, the_roughness_bins, the_power_bins
    #     )
    
    # save the final uncertainties to a pickle table?
    # insert code to do that here


if __name__ == "__main__":
    main()
