"""
Script to generate Cryo-TEMPO uncertainty lookup table, from specified inputs.

Uncertainty lookup table consists of binned difference values. Differences can be binned
by 1-5 variables, and values for each bin can be calculated using a number of metrics 
(median, MAD, RMSE, standard deviation upper estimate).

Mean 2D slices of lookup table are plotted. 
Mean 2D slices of bin count and standard deviation are also plotted. 

Only available for Antarctic currently.

Compatible for the following variables:
- Slope: interpolated from Slopes("rema_100m_900ws_slopes_zarr")
- Roughness: interpolated from Roughness ("rema_100m_900ws_roughness_zarr)
- Power: sigma 0 from L2i product
- Coherence: coherence from L2i product

Input is dh values, lat, lon from CS2-IS2 differences npz files, for example:
cs2_minus_is2_gt2lgt2r_p2p_diffs_antarctica_icesheets.npz

example usage: 
python clev2er_multilinear_uncertainty.py -a antarctica_icesheets -m mad \
    -dh_dir /media/luna/boxallk/clev2er/uncertainty_assessment/uncertainty_tables/antarctica_icesheets/SIN/2020 \
        -o /media/luna/boxallk/clev2er/uncertainty_assessment/uncertainty_tables/test \
            -v slope,roughness,power,coherence,poca_distance

Author: Karla Boxall

"""
# ---------------------------------------------------------------------------------------------------------------------
# Package Imports
# ---------------------------------------------------------------------------------------------------------------------

import argparse
import sys
import glob
import pickle
import itertools

import numpy as np
import pandas as pd
import math
from scipy.stats.distributions import chi2
from sklearn.linear_model import LinearRegression
import seaborn as sns
import matplotlib.pyplot as plt

from cpom.roughness.roughness import Roughness
from cpom.slopes.slopes import Slopes

# ---------------------------------------------------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------------------------------------------------

def calculate_binned_metric(
    delta_elevation: np.ndarray,
    slope: np.ndarray,
    roughness: np.ndarray ,
    power: np.ndarray,
    coherence: np.ndarray,
    poca_distance: np.ndarray,
    slope_bins: np.ndarray,
    roughness_bins: np.ndarray,
    power_bins: np.ndarray,
    coherence_bins: np.ndarray,
    poca_distance_bins: np.ndarray,
    method: str,
) -> pd.DataFrame:

    """
    Calculate the binned elevation difference values.

    Args:
        delta_elevation (np.ndarray): Array of elevation differences between
                                      two measurement techniques.
        method (str): Metric to calculate uncertainty within bins.
        slope (np.ndarray) [default: None]: Array of surface slopes in meters.
        roughness (np.ndarray) [default: None]: Array of surface roughness values in meters.
        power (np.ndarray) [default: None]: Array of power (sigma 0) values.
        coherence (np.ndarray) [default: None]: Array of coherence (coherence_20_ku) values.
        poca_distance (np.ndarray) [default: None]: Array of distance from POCA (nadir coord vs poca coord) values.
        slope_bins (np.ndarray) [default: None]: Bins to categorize slope values.
        roughness_bins (np.ndarray) [default: None]: Bins to categorize roughness values.
        power_bins (np.ndarray) [default: None]: Bins to categorise power values.
        coherence_bins (np.ndarray) [default: None]: Bins to categorise coherence values.
        poca_distance_bins (np.ndarray) [default: None]: Bins to categorise poca distance values.

    Returns:
        binned_metric_pivot (pd.DataFrame): A pivot table where rows correspond to slope bins, columns to roughness 
                    and power bins, and values to the calculated elevation difference metric
                    within each bin.
        binned_count (pd.DataFrame): A dataframe containing the count within each bin
        binned_stdev(pd.DataFrame): A dataframe containing the standard deviation within each bin
        
    """

    print(f'{sum(x is not None for x in [slope, roughness, power, coherence, poca_distance])} variables ingested')

    # Create a DataFrame to hold the data
    data = pd.DataFrame({"delta_elevation": np.abs(delta_elevation)})  # Absolute elevation differences 

    grouping_bins = []

    # add data to dataframe
    # store bins to group by
    # bin the values
    if slope is not None: 
        data['slope'] = slope
        grouping_bins.append('slope_bin')
        data["slope_bin"] = pd.cut(data["slope"], bins=slope_bins, include_lowest=True, labels=slope_bins[:-1])
    if roughness is not None:
        data['roughness'] = roughness
        grouping_bins.append('roughness_bin')
        data["roughness_bin"] = pd.cut(
        data["roughness"], bins=roughness_bins, include_lowest=True, labels=roughness_bins[:-1])
    if power is not None: 
        data['power'] = power
        grouping_bins.append('power_bin')
        data["power_bin"] = pd.cut(
        data["power"], bins=power_bins, include_lowest=True, labels=power_bins[:-1])
    if coherence is not None:
        data['coherence'] = coherence
        grouping_bins.append('coherence_bin')
        data["coherence_bin"] = pd.cut(
        data["coherence"], bins=coherence_bins, include_lowest=True, labels=coherence_bins[:-1])
    if poca_distance is not None:
        data['poca_distance'] = poca_distance
        grouping_bins.append('poca_distance_bin')
        data["poca_distance_bin"] = pd.cut(
        data["poca_distance"], bins=poca_distance_bins, include_lowest=True, labels=poca_distance_bins[:-1])

    # calculate the metric within each bin
    # use median as metric
    if method == 'median':
        binned_metric = (
            data.assign(abs_delta_elevation=data["delta_elevation"].abs())
            .groupby(grouping_bins)["abs_delta_elevation"]
            .median()
            .reset_index()
        )
    
    # use MAD as metric
    elif method == 'mad':
        def mad(x):
            return np.median(np.abs(x - np.median(x)))

        binned_metric = (
            data.assign(abs_delta_elevation=data["delta_elevation"].abs())
            .groupby(grouping_bins)["abs_delta_elevation"].apply(mad).reset_index()
        )

    # use RMSE as metric
    elif method == 'rmse':
        def rmse(x):
            mse = np.square(x).mean()
            return math.sqrt(mse)

        binned_metric = (
            data.assign(abs_delta_elevation=data["delta_elevation"].abs())
            .groupby(grouping_bins)["abs_delta_elevation"].apply(rmse).reset_index()
        )

    # use upper estimate of standard deviation as metric
    elif method == 'std_ue':
        def std_ue(x):
            std_ue = np.std(x) * np.sqrt((len(x)-1)/chi2.ppf(0.975, len(x)-1)) 
            return std_ue

        binned_metric = (
            data.assign(abs_delta_elevation=data["delta_elevation"].abs())
            .groupby(grouping_bins)["abs_delta_elevation"].apply(std_ue).reset_index()
        )

    # Pivot the table to create a 3D matrix where rows are slope_bins and columns are roughness_bins and power_bins
    binned_metric_pivot = binned_metric.pivot(
        index=grouping_bins[0], columns=grouping_bins[1:], values="abs_delta_elevation"
    )

    # count the number of datapoints in each bin
    binned_count = (
        data.assign(abs_delta_elevation=data["delta_elevation"].abs())
        .groupby(grouping_bins)["abs_delta_elevation"].count().reset_index()
    )

    # calculate the standard deviation of the datapoints within each bin
    binned_stdev = (
        data.assign(abs_delta_elevation=data["delta_elevation"].abs())
        .groupby(grouping_bins)["abs_delta_elevation"].std().reset_index()
    )

    return binned_metric_pivot, binned_count, binned_stdev


def calc_uncertainty_table(
    delta_elevation: np.ndarray,
    slope: np.ndarray,
    roughness: np.ndarray,
    power: np.ndarray,
    coherence: np.ndarray,
    poca_distance: np.ndarray,
    slope_bins: np.ndarray,
    roughness_bins: np.ndarray,
    power_bins: np.ndarray,
    coherence_bins: np.ndarray,
    poca_distance_bins: np.ndarray,
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
        poca_distance (np.ndarray): Array of poca distance values.
        slope_bins (np.ndarray): Bins to categorize slope values.
        roughness_bins (np.ndarray): Bins to categorize roughness values.
        power_bins (np.ndarray): Bins to categorise power values.
        coherence_bins (np.ndarray): Bins to categorise coherence values.
        poca_distance_bins (np.ndarray): Bins to categorise poca distance values.
        method (str): Method to calculate the uncertainty ('median' 'mad' 'rmse' or 'std_ue'). Default is 'median'.

    Returns:
        binned_table (pd.DataFrame): A pivot table where rows correspond to slope bins, columns to additional variables,
                      and values to the uncertainty metric within each bin.
        binned_count (pd.DataFrame): A dataframe containing the count within each bin
        binned_stdev(pd.DataFrame): A dataframe containing the standard deviation within each bin
    """
    binned_table, binned_count, binned_stdev = calculate_binned_metric(
        delta_elevation, slope, roughness, power, coherence, poca_distance,
        slope_bins, roughness_bins, power_bins, coherence_bins, poca_distance_bins, method,
    )

    return binned_table, binned_count, binned_stdev

def get_binned_values(
    slope_values: np.ndarray,
    roughness_values: np.ndarray,
    power_values: np.ndarray,
    coherence_values: np.ndarray,
    poca_distance_values: np.ndarray,
    binned_table: pd.DataFrame,
    slope_bins: np.ndarray,
    roughness_bins: np.ndarray,
    power_bins: np.ndarray,
    coherence_bins: np.ndarray,
    poca_distance_bins: np.ndarray,
) -> list:

    """
    Retrieve the calculated metric of elevation difference for given variable values (ie for given
    slope, roughness and power values).

    Args:
        slope_values (np.ndarray): Array of slope values for which to retrieve calculated differences.
        roughness_values (np.ndarray): Array of roughness values for which to retrieve calculated differences.
        power_values (np.ndarray): Array of power values for which to retrieve calculated differences.
        coherence_values (np.ndarray): Array of coherence values for which to retrieve calculated differences.
        poca_distance_values(np.ndarray): Array of poca distance values for which to retreive calculated difference.
        binned_table (pd.DataFrame): A pivot table of binned median absolute elevation differences.
        slope_bins (np.ndarray): Bins to categorize slope values.
        roughness_bins (np.ndarray): Bins to categorize roughness values.
        power_bins (np.ndarray): Bins to categorise power values.
        coherence_bins (np.ndarray): Bins to categorise coherence values.
        poca_distance_bins (np.ndarray): Bins to categorise poca distance values.

    Returns:
        list: An array of median absolute elevation differences corresponding to the
                    input values.
    """
    ingested_variables = []

    if slope_values is not None:
        slope_values = np.asarray(slope_values)
        # Find the slope bin indices for the array of slope_values
        slope_bin_indices = np.digitize(slope_values, slope_bins) - 1
        slope_bin_indices = np.clip(
            slope_bin_indices, 0, len(slope_bins) - 2
        )  # Ensure indices are within range
        # Convert bin labels to row and column labels in the DataFrame
        slope_indices = [slope_bins[idx] for idx in slope_bin_indices]
        ingested_variables.append(slope_indices)

    if roughness_values is not None:
        roughness_values = np.asarray(roughness_values)
        # Find the roughness bin indices for the array of roughness_values
        roughness_bin_indices = np.digitize(roughness_values, roughness_bins) - 1
        roughness_bin_indices = np.clip(
            roughness_bin_indices, 0, len(roughness_bins) - 2
        )  # Ensure indices are within range
        # Convert bin labels to row and column labels in the DataFrame
        roughness_indices = [roughness_bins[idx] for idx in roughness_bin_indices]
        ingested_variables.append(roughness_indices)

    if power_values is not None:
        power_values = np.asarray(power_values)
        # Find the power bin indices for the array of power
        power_bin_indices = np.digitize(power_values, power_bins) - 1
        power_bin_indices = np.clip(
            power_bin_indices, 0, len(power_bins) - 2
        )  # Ensure indices are within range
        # Convert bin labels to row and column labels in the DataFrame
        power_indices = [power_bins[idx] for idx in power_bin_indices]
        ingested_variables.append(power_indices)

    if coherence_values is not None:
        coherence_values = np.asarray(coherence_values)
        # Find the coherence bin indices for the array of coherence
        coherence_bin_indices = np.digitize(coherence_values, coherence_bins) - 1
        coherence_bin_indices = np.clip(
            coherence_bin_indices, 0, len(coherence_bins) - 2
        )  # Ensure indices are within range
        # Convert bin labels to row and column labels in the DataFrame
        coherence_indices = [coherence_bins[idx] for idx in coherence_bin_indices]
        ingested_variables.append(coherence_indices)

    if poca_distance_values is not None:
        poca_distance_values = np.asarray(poca_distance_values)
        # Find the coherence bin indices for the array of coherence
        poca_distance_bin_indices = np.digitize(poca_distance_values, poca_distance_bins) - 1
        poca_distance_bin_indices = np.clip(
            poca_distance_bin_indices, 0, len(poca_distance_bins) - 2
        )  # Ensure indices are within range
        # Convert bin labels to row and column labels in the DataFrame
        poca_distance_indices = [poca_distance_bins[idx] for idx in poca_distance_bin_indices]
        ingested_variables.append(poca_distance_indices)

    binned_table_df = binned_table.stack(future_stack=True)
    # print(binned_table_df)

    if len(ingested_variables) == 1:
        values = [binned_table_df.loc[
            ingested_variables[0][i]][0] for i in range(len(ingested_variables[0]))]
    if len(ingested_variables) == 2:
        values = [binned_table_df.loc[
            ingested_variables[0][i]][ingested_variables[1][i]] for i in range(len(ingested_variables[0]))]
    if len(ingested_variables) == 3:
        values = [binned_table_df.loc[
            ingested_variables[0][i]][ingested_variables[1][i]][ingested_variables[2][i]] for i in range(len(ingested_variables[0]))]
    if len(ingested_variables) == 4:
        values = [binned_table_df.loc[
            ingested_variables[0][i]][ingested_variables[1][i]][ingested_variables[2][i]][ingested_variables[3][i]] for i in range(len(ingested_variables[0]))]
    if len(ingested_variables) == 5:
        values = [binned_table_df.loc[
            ingested_variables[0][i]][ingested_variables[1][i]][ingested_variables[2][i]][ingested_variables[3][i]][ingested_variables[4][i]] for i in range(len(ingested_variables[0]))]

    # Retrieve the values using numpy indexing on the DataFrame values
    # values = [binned_table_df.loc[row_indices[i]][col1_indices[i]][col2_indices[i]][col3_indices[i]][col4_indices[i]] for i in range(len(row_indices))]

    return values


def fit_linear_model(
    binned_table: np.ndarray,
    slope_bins: np.ndarray,
    roughness_bins: np.ndarray,
    power_bins: np.ndarray,
    coherence_bins: np.ndarray,
    poca_distance_bins: np.ndarray,
    variables: list
) -> np.ndarray:
    """
    Fit a linear model to the lookup table to fill 'NaNs' using interpolation and extrapolation.

    Args:
        binned_table (np.ndarray): Uncertainty lookup table as a pivot table.
        slope_bins (np.ndarray): Bins to categorize slope values.
        roughness_bins (np.ndarray): Bins to categorize roughness values.
        power_bins (np.ndarray): Bins to categorise power values.
        coherence_bins (np.ndarray): Bins to categorise coherence values.
        poca_distance_bins (np.ndarray): Bins to categorise poca distance values.
        variables (list): list of variable names ingested.

    Returns:
        np.ndarray: The pivot table with missing values filled using a linear model fit.

    """
    ingested_variables = []

    if 'slope' in variables:
        slope_bins = slope_bins[:-1]
        ingested_variables.append(slope_bins)
    if 'roughness' in variables:
        roughness_bins = roughness_bins[:-1]
        ingested_variables.append(roughness_bins)
    if 'power' in variables:
        power_bins = power_bins[:-1]
        ingested_variables.append(power_bins)
    if 'coherence' in variables:
        coherence_bins = coherence_bins[:-1]
        ingested_variables.append(coherence_bins)
    if 'poca_distance' in variables:
        poca_distance_bins = poca_distance_bins[:-1]
        ingested_variables.append(poca_distance_bins)
    
    z_values = binned_table.values.flatten()  # Uncertainty values

    var1 = None
    var2 = None
    var3 = None
    var4 = None
    var5 = None

    # Reshaping data into long format for fitting
    if len(ingested_variables)==1:
        var1 = np.meshgrid(ingested_variables[0], indexing="ij")
        var1 = np.asarray(var1)
        var1 = var1.flatten()

    if len(ingested_variables)==2:
        var1, var2 = np.meshgrid(ingested_variables[0], ingested_variables[1], indexing="ij")
        var1 = var1.flatten()
        var2 = var2.flatten()
    
    if len(ingested_variables)==3:
        var1, var2, var3 = np.meshgrid(
            ingested_variables[0], ingested_variables[1], ingested_variables[2], 
            indexing="ij")
        var1 = var1.flatten()
        var2 = var2.flatten()
        var3 = var3.flatten()

    if len(ingested_variables)==4:
        var1, var2, var3, var4 = np.meshgrid(
            ingested_variables[0], ingested_variables[1], ingested_variables[2], 
            ingested_variables[3], indexing="ij")
        var1 = var1.flatten()
        var2 = var2.flatten()
        var3 = var3.flatten()
        var4 = var4.flatten()
    
    if len(ingested_variables)==5:
        var1, var2, var3, var4, var5 = np.meshgrid(
            ingested_variables[0], ingested_variables[1], ingested_variables[2], 
            ingested_variables[3], ingested_variables[4], indexing="ij")
        var1 = var1.flatten()
        var2 = var2.flatten()
        var3 = var3.flatten()
        var4 = var4.flatten()
        var5 = var5.flatten()

    # Remove NaN values before fitting
    mask = ~np.isnan(z_values)
    z_clean = z_values[mask]

    if var1 is not None:
        var1_clean = var1[mask]
    if var2 is not None:
        var2_clean = var2[mask]
    if var3 is not None:
        var3_clean = var3[mask]
    if var4 is not None:
        var4_clean = var4[mask]
    if var5 is not None:
        var5_clean = var5[mask]

    if len(ingested_variables)==1:
        # Prepare the features matrix
        x_matrix = np.vstack([var1_clean]).T
        # Perform linear regression
        reg = LinearRegression()
        reg.fit(x_matrix, z_clean)
        # Coefficients of the model
        a = reg.coef_
        b = reg.intercept_
        # Fit function
        def linear_fit(var1):
            return a * var1 + b
        # Apply the fit to the original grid
        linear_table = pd.DataFrame(
        linear_fit(var1).reshape(binned_table.shape), index=binned_table.index)
    
    if len(ingested_variables)==2:
        # Create interaction term (slope * roughness)
        interaction_term = var1_clean * var2_clean
        # Prepare the features matrix
        x_matrix = np.vstack([var1_clean, var2_clean, interaction_term]).T
        # Perform linear regression
        reg = LinearRegression()
        reg.fit(x_matrix, z_clean)
        # Coefficients of the model
        a, b, c = reg.coef_
        d = reg.intercept_
        # Fit function
        def linear_fit(var1, var2):
            return a * var1 + b * var2 + c * var1 * var2 + d
        # Apply the fit to the original grid
        linear_table = pd.DataFrame(
        linear_fit(var1, var2).reshape(binned_table.shape), index=binned_table.index, columns=binned_table.columns)
    
    # repeat for different numbers of variables
    if len(ingested_variables)==3:
        interaction_term = var1_clean * var2_clean * var3_clean
        x_matrix = np.vstack([var1_clean, var2_clean, var3_clean, interaction_term]).T
        reg = LinearRegression()
        reg.fit(x_matrix, z_clean)
        a, b, c, d = reg.coef_
        e = reg.intercept_
        def linear_fit(var1, var2, var3):
            return a * var1 + b * var2 + c * var3 + d * var1 * var2 * var3 + e
        linear_table = pd.DataFrame(
        linear_fit(var1, var2, var3).reshape(binned_table.shape), index=binned_table.index, columns=binned_table.columns)

    if len(ingested_variables)==4:
        interaction_term = var1_clean * var2_clean * var3_clean * var4_clean
        x_matrix = np.vstack([var1_clean, var2_clean, var3_clean, var4_clean, interaction_term]).T
        reg = LinearRegression()
        reg.fit(x_matrix, z_clean)
        a, b, c, d, e = reg.coef_
        f = reg.intercept_
        def linear_fit(var1, var2, var3, var4):
            return a * var1 + b * var2 + c * var3 + d * var4 + e * var1 * var2 * var3 * var4 + f
        linear_table = pd.DataFrame(
        linear_fit(var1, var2, var3, var4).reshape(binned_table.shape), index=binned_table.index, columns=binned_table.columns)

    if len(ingested_variables)==5:
        interaction_term = var1_clean * var2_clean * var3_clean * var4_clean * var5_clean
        x_matrix = np.vstack([var1_clean, var2_clean, var3_clean, var4_clean, var5_clean, interaction_term]).T
        reg = LinearRegression()
        reg.fit(x_matrix, z_clean)
        a, b, c, d, e, f = reg.coef_
        g = reg.intercept_
        def linear_fit(var1, var2, var3, var4, var5):
            return a * var1 + b * var2 + c * var3 + d * var4 + e * var5 + f * var1 * var2 * var3 * var4 * var5 + g
        linear_table = pd.DataFrame(
        linear_fit(var1, var2, var3, var4, var5).reshape(binned_table.shape), index=binned_table.index, columns=binned_table.columns)

    # Replace negative values with values from the original table
    if len(ingested_variables)==1:
        linear_table = linear_table.where(linear_table >= 0, binned_table, axis=0)
    else:
        linear_table = linear_table.where(linear_table >= 0, binned_table)

    return linear_table


def save_values_as_pickle(values: list, filename: str) -> None:

    """Save the uncertainty values as a Pickle file.

    Args:
        values (list): The uncertainty values extracted from lookup table.
        filename (str): The path to the file where the values will be saved.
    """

    with open(filename, "wb") as pkl_wb_obj:
        pickle.dump(values, pkl_wb_obj)
    

def plot_2d_slices(pivot_table, outpath: str, plot_var: str, ingested_vars: list):

    """
    Plots all combinations of mean 2D slices.

    Args:
        pivot_table (np.ndarray): Filled uncertainty lookup table as a pivot table.
        outdir (str): The path to the file where the values will be saved.
        plot_var (str): Variable being plotted (look_up_table, count or stdev)
        ingested_vars (list): List of the ingested variable names

    """
    # lookup table input in pivot table format
    if plot_var == 'look_up_table':
        pivot_table = pd.melt(pivot_table, ignore_index=False)
        values = 'value'
    # count/stdev tables input as dataframes
    else:
        values = 'abs_delta_elevation'

    # if only one variable, plot a 1D visualisation
    if len(ingested_vars)==1:
        data = pd.pivot_table(pivot_table, index=[f'{ingested_vars[0]}_bin'],values=values, observed=False)
        _, ax = plt.subplots(figsize=(15, 6))
        sns.heatmap(data, ax=ax)
        filename = outpath + f'2dslice_{plot_var}_{ingested_vars[0]}.png'
        plt.savefig(filename, bbox_inches='tight')

    # if more than one variable
    else:
        # find all possible pair combinations
        combinations = list(itertools.combinations(ingested_vars, 2))
        for c in combinations:
            indices = []
            # find the indices of the combinations in input list of variables
            for i in range(2):
                index = ingested_vars.index(c[i])
                indices.append(index)
            
            # isolate a given pair of variables from the pivot table and plot as MEAN 2D slice
            data = pd.pivot_table(pivot_table, index=[f'{ingested_vars[indices[0]]}_bin'], columns=[f'{ingested_vars[indices[1]]}_bin'],values=values, observed=False)
            _, ax = plt.subplots(figsize=(15, 6))
            sns.heatmap(data, ax=ax)
            filename = outpath + f'2dslice_{plot_var}_{ingested_vars[indices[0]]}{ingested_vars[indices[1]]}.png'
            plt.savefig(filename, bbox_inches='tight')


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
        "-dh_dir",
        "-dh",
        help=(
            "path to directory of elevation difference npz files"
        ),
        type=str,
    )

    parser.add_argument(
        "--variables",
        "-v",
        default="slope",
        help=("choose the co-variates: 'slope' (default) or 'roughness' or 'power' or 'coherence' or 'poca_distance'"),
    )

    parser.add_argument(
        "-outdir",
        "-o",
        help=(
            "path of output directory"
        ),
        type=str,
    )

    # read arguments from the command line
    args = parser.parse_args()

    if args.area != 'antarctica_icesheets':
        sys.exit("Must specify 'antarctica_icesheets'")
    
    if args.variables:
        variables = args.variables.split(",")

    for variable in variables:
        if variable not in ["slope", "roughness", "power", "coherence", "poca_distance"]:
            sys.exit(
                f"{variable} not a valid varuable. Must be one of 'slope' (default) or 'roughness' or 'power' or 'coherence' or 'poca_distance")
    
    if args.area == 'antarctica_icesheets':
        this_slope = Slopes("rema_100m_900ws_slopes_zarr")
        this_roughness = Roughness("rema_100m_900ws_roughness_zarr")

    # set variables to None
    slope = None
    roughness = None
    power = None
    coherence = None
    poca_distance = None

    dh_all = []
    lats_all = []
    lons_all = []
    power_all = []
    coherence_all = []
    poca_distance_all = []

    # Read npz file to get dh,lat,lon values
    for path in glob.glob(f'{args.dh_dir}/**/*.npz', recursive=True):
        
        # extract data from each monthly file    
        print("Path: ", path)
        dh_data = np.load(path, allow_pickle=True)
        dh = dh_data['dh']
        lats = dh_data['lats']
        lons = dh_data['lons']

        # append monthly data to all data
        dh_all.extend(dh)
        lats_all.extend(lats)
        lons_all.extend(lons)

        if 'power' in args.variables:
            power = dh_data.get("pow")
            power_all.extend(power)
        if 'coherence' in args.variables:
            coherence = dh_data.get("coh")
            coherence_all.extend(coherence)
        if 'poca_distance' in args.variables:
            poca_distance = dh_data.get("dis_poca")
            poca_distance_all.extend(poca_distance)
    
    dh_all = np.asarray(dh_all)
    lats_all = np.asarray(lats_all)
    lons_all = np.asarray(lons_all)

    # Define bins 
    # the_slope_bins = np.arange(0, 2.1, 0.1)  # Define slope bins in degrees (0.1 degree steps from 0 to 2 degrees)
    # the_roughness_bins = np.arange(0, 2.1, 0.1)  # Define roughness bins in meters (0.1 m steps from 0 to 2 meters)
    # the_power_bins = np.arange(0, 10, 1)  # Define power bins (1 steps from 0 to 9)
    # the_coherence_bins = np.arange(0, 1.1, 0.1)  # Define coherence bins (0.1 steps from 0 to 1)
    # the_poca_distance_bins = np.arange(0, 100, 10)  # Define distance bins (10 steps from 0 to 100)

    the_slope_bins = None
    the_roughness_bins = None
    the_power_bins = None
    the_coherence_bins = None
    the_poca_distance_bins = None
    
    # Extract variables
    if 'slope' in args.variables:
        print('Fetching slope data')
        slope = this_slope.interp_slopes(lats_all, lons_all, method="linear", xy_is_latlon=True)
        _, the_slope_bins = pd.qcut(slope, 10, labels=False, retbins=True)
    if 'roughness' in args.variables:
        print('Fetching roughness data')
        roughness = this_roughness.interp_roughness(lats_all, lons_all, method="linear", xy_is_latlon=True)
        _, the_roughness_bins = pd.qcut(roughness, 10, labels=False, retbins=True)
    if 'power' in args.variables:
        print('Fetching power data')
        power = np.asarray(power_all)
        _, the_power_bins = pd.qcut(power, 10, labels=False, retbins=True)
    if 'coherence' in args.variables:
        print('Fetching coherence data')
        coherence = np.asarray(coherence_all)
        _, the_coherence_bins = pd.qcut(coherence, 10, labels=False, retbins=True)
    if 'poca_distance' in args.variables:
        print('Fetching poca distance data')
        poca_distance = np.asarray(poca_distance_all)
        _, the_poca_distance_bins = pd.qcut(poca_distance, 10, labels=False, retbins=True)

    # calculate uncertainty lookup table
    print("calculating uncertainty table... ")
    binned_table, binned_count, binned_stdev = calc_uncertainty_table(
        dh_all,
        slope,
        roughness,
        power,
        coherence,
        poca_distance,
        the_slope_bins,
        the_roughness_bins,
        the_power_bins,
        the_coherence_bins,
        the_poca_distance_bins,
        method=args.method
    )
    print(binned_table)

    # plot the binned count as MEAN 2D slices
    outpath = args.outdir + f'/{args.variables}_{args.method}_'
    plot_2d_slices(binned_count, outpath, 'count', variables)

    # fill the nans using linear interpolation
    binned_table_filled = fit_linear_model(
        binned_table,
        the_slope_bins,
        the_roughness_bins,
        the_power_bins,
        the_coherence_bins,
        the_poca_distance_bins,
        variables
    )
    print(binned_table_filled)

    # plot the plane fitted lookup table as 2D slices
    outpath = args.outdir + f'/{args.variables}_{args.method}_'
    plot_2d_slices(binned_table_filled, outpath, 'look_up_table', variables)

    slope_values = None
    roughness_values = None
    power_values = None
    coherence_values = None
    poca_distance_values = None
    
    #### TEST UNCERTAINTY EXTRACTION ###

    # Test extraction from lookup table
    if 'slope' in args.variables:
        slope_values = [0.15, 0.4]  
    if 'roughness' in args.variables:
        roughness_values = [0.25, 0.3]  
    if 'power' in args.variables:
        power_values = [-1, 6]  
    if 'coherence' in args.variables:
        coherence_values = [0.2, 0.8]
    if 'poca_distance' in args.variables:
        poca_distance_values = [37, 82]

    print("Getting uncertainty values from table...")
    # Extract the corresponding value from the binned_table_filled for test values
    values = get_binned_values(
        slope_values, roughness_values, power_values, coherence_values, poca_distance_values, binned_table_filled, 
        the_slope_bins, the_roughness_bins, the_power_bins, the_coherence_bins, the_poca_distance_bins
    )
    print(
        f"Uncertainty for slope {slope_values} and roughness {roughness_values} and power" +
        f"{power_values} and coherence {coherence_values} and poca distance {poca_distance_values}: {values} m")

    #### EXTRACT UNCERTAINTIES ###

    # Extract the corresponding value from the binned_table
    values = get_binned_values(
        slope, roughness, power, coherence, poca_distance, binned_table_filled, 
        the_slope_bins, the_roughness_bins, the_power_bins, the_coherence_bins, the_poca_distance_bins
        )

    # save the final uncertainties to a pickle table
    outpath = args.outdir + f'/{args.variables}_{args.method}_uncertainties.pickle'
    save_values_as_pickle(values, outpath)
    

if __name__ == "__main__":
    main()


















# EXTRA CODE

        # print(binned_metric_stats)

        # print(binned_metric_stats.pivot(
        # index=grouping_bins[0], columns=grouping_bins[1:], values="abs_delta_elevation"))

        # print(binned_metric_stats[binned_metric_stats['power_bin'] == 0])
        # print(binned_metric_stats[binned_metric_stats['coherence_bin'] == 0])
        # print(binned_metric_stats[binned_metric_stats['poca_distance_bin'] == 0])

        # df = binned_metric_stats[binned_metric_stats['poca_distance_bin'] == 0]
        # data = pd.pivot_table(df, index=['power_bin'], columns=['coherence_bin'],values='abs_delta_elevation')
        # print(data)
        # f, ax = plt.subplots(figsize=(15, 6))
        # sns.heatmap(data, ax=ax)
        # plt.savefig('/media/luna/boxallk/clev2er/uncertainty_assessment/uncertainty_tables/test/test_2dslice_count.png', bbox_inches='tight')

        # df = binned_metric_stats[binned_metric_stats['coherence_bin'] == 0]
        # data = pd.pivot_table(df, index=['power_bin'], columns=['poca_distance_bin'],values='abs_delta_elevation')
        # print(data)
        # f, ax = plt.subplots(figsize=(15, 6))
        # sns.heatmap(data, ax=ax)
        # plt.savefig('/media/luna/boxallk/clev2er/uncertainty_assessment/uncertainty_tables/test/test_2dslice_count2.png', bbox_inches='tight')

        # df = binned_metric_stats[binned_metric_stats['power_bin'] == 0]
        # data = pd.pivot_table(df, index=['coherence_bin'], columns=['poca_distance_bin'],values='abs_delta_elevation')
        # print(data)
        # f, ax = plt.subplots(figsize=(15, 6))
        # sns.heatmap(data, ax=ax)
        # plt.savefig('/media/luna/boxallk/clev2er/uncertainty_assessment/uncertainty_tables/test/test_2dslice_count3.png', bbox_inches='tight')


    # print(linear_table.loc[:][0])
    # data = linear_table.loc[:][0] # so nearly there - just need to figure out how to select other combinations
    # f, ax = plt.subplots(figsize=(15, 6))
    # sns.heatmap(data, ax=ax)
    # plt.savefig('/media/luna/boxallk/clev2er/uncertainty_assessment/uncertainty_tables/test/test_2dslice2.png')

    # data = pd.pivot_table(binned_metric, index=['power_bin'], columns=['coherence_bin'],values='abs_delta_elevation')
    # print(data)
    # f, ax = plt.subplots(figsize=(15, 6))
    # sns.heatmap(data, ax=ax)
    # plt.savefig('/media/luna/boxallk/clev2er/uncertainty_assessment/uncertainty_tables/test/test_2dslice2.png')

    # index (power), col1 (coherence), col2 (poca)
    # linear_table.loc[:, (0.1,10)]  # index
    # linear_table.loc[0, :]  # col1 (coherence) vs col2 (poca)
    # linear_table.loc[:,0]  # index (power) vs col2 (poca)
    # index (power) vs col2 (coherence)  THIS IS WRONG - NEED TO WORK THIS OUT

    # code that works for power,coherence and poca distance

    #     if plot_var == 'look_up_table':
    #     df = pivot_table[pivot_table.index == 0]
    #     data = pd.pivot_table(df, index=['coherence_bin'], columns=['poca_distance_bin'],values=values, observed=False)
    #     _, ax = plt.subplots(figsize=(15, 6))
    #     sns.heatmap(data, ax=ax)
    #     filename = outpath + f'_2dslice1_{plot_var}.png'
    #     plt.savefig(filename, bbox_inches='tight')

    # elif plot_var == 'count' or plot_var == 'stdev':
    #     df = pivot_table[pivot_table['power_bin'] == 0]
    #     data = pd.pivot_table(df, index=['coherence_bin'], columns=['poca_distance_bin'],values=values, observed=False)
    #     _, ax = plt.subplots(figsize=(15, 6))
    #     sns.heatmap(data, ax=ax)
    #     filename = outpath + f'_2dslice1_{plot_var}.png'
    #     plt.savefig(filename, bbox_inches='tight')

    # df = pivot_table[pivot_table['poca_distance_bin'] == 0]
    # data = pd.pivot_table(df, index=['power_bin'], columns=['coherence_bin'],values=values, observed=False)
    # _, ax = plt.subplots(figsize=(15, 6))
    # sns.heatmap(data, ax=ax)
    # filename = outpath + f'_2dslice2_{plot_var}.png'
    # plt.savefig(filename, bbox_inches='tight')

    # df = pivot_table[pivot_table['coherence_bin'] == 0]
    # data = pd.pivot_table(df, index=['power_bin'], columns=['poca_distance_bin'],values=values, observed=False)
    # _, ax = plt.subplots(figsize=(15, 6))
    # sns.heatmap(data, ax=ax)
    # filename = outpath + f'_2dslice3_{plot_var}.png'
    # plt.savefig(filename, bbox_inches='tight')

    # plot 2d slice other code

        # if len(ingested_vars)==2:
    #     data = pd.pivot_table(pivot_table, index=[f'{ingested_vars[0]}_bin'], columns=[f'{ingested_vars[1]}_bin'],values=values, observed=False)
    #     _, ax = plt.subplots(figsize=(15, 6))
    #     sns.heatmap(data, ax=ax)
    #     filename = outpath + f'2dslice_{plot_var}_{ingested_vars[0]}{ingested_vars[1]}.png'
    #     plt.savefig(filename, bbox_inches='tight')
    
    # if len(ingested_vars)>=3:
    #     if plot_var == 'look_up_table':
    #         df = pivot_table[pivot_table.index == 0]
    #         data = pd.pivot_table(df, index=[f'{ingested_vars[1]}_bin'], columns=[f'{ingested_vars[2]}_bin'],values=values, observed=False)
    #         _, ax = plt.subplots(figsize=(15, 6))
    #         sns.heatmap(data, ax=ax)
    #         filename = outpath + f'2dslice_{plot_var}_{ingested_vars[1]}{ingested_vars[2]}.png'
    #         plt.savefig(filename, bbox_inches='tight')
    #     else:
    #         df = pivot_table[pivot_table[f'{ingested_vars[0]}_bin'] == 0]
    #         data = pd.pivot_table(df, index=[f'{ingested_vars[1]}_bin'], columns=[f'{ingested_vars[2]}_bin'],values=values, observed=False)
    #         _, ax = plt.subplots(figsize=(15, 6))
    #         sns.heatmap(data, ax=ax)
    #         filename = outpath + f'2dslice_{plot_var}_{ingested_vars[1]}{ingested_vars[2]}.png'
    #         plt.savefig(filename, bbox_inches='tight')

    #     df = pivot_table[pivot_table[f'{ingested_vars[2]}_bin'] == 0]
    #     data = pd.pivot_table(df, index=[f'{ingested_vars[0]}_bin'], columns=[f'{ingested_vars[1]}_bin'],values=values, observed=False)
    #     _, ax = plt.subplots(figsize=(15, 6))
    #     sns.heatmap(data, ax=ax)
    #     filename = outpath + f'2dslice_{plot_var}_{ingested_vars[0]}{ingested_vars[1]}.png'
    #     plt.savefig(filename, bbox_inches='tight')

    #     df = pivot_table[pivot_table[f'{ingested_vars[1]}_bin'] == 0]
    #     data = pd.pivot_table(df, index=[f'{ingested_vars[0]}_bin'], columns=[f'{ingested_vars[2]}_bin'],values=values, observed=False)
    #     _, ax = plt.subplots(figsize=(15, 6))
    #     sns.heatmap(data, ax=ax)
    #     filename = outpath + f'2dslice_{plot_var}_{ingested_vars[0]}{ingested_vars[2]}.png'
    #     plt.savefig(filename, bbox_inches='tight')


        # if plot_var == 'look_up_table':


        # elif plot_var == 'count' or plot_var == 'stdev':
        #     df = pivot_table[pivot_table['power_bin'] == 0]
        #     data = pd.pivot_table(df, index=['coherence_bin'], columns=['poca_distance_bin'],values=values, observed=False)
        #     _, ax = plt.subplots(figsize=(15, 6))
        #     sns.heatmap(data, ax=ax)
        #     filename = outpath + f'_2dslice1_{plot_var}.png'
        #     plt.savefig(filename, bbox_inches='tight')

    # df = pivot_table[pivot_table['poca_distance_bin'] == 0]
    # data = pd.pivot_table(df, index=['power_bin'], columns=['coherence_bin'],values=values, observed=False)
    # _, ax = plt.subplots(figsize=(15, 6))
    # sns.heatmap(data, ax=ax)
    # filename = outpath + f'_2dslice2_{plot_var}.png'
    # plt.savefig(filename, bbox_inches='tight')

    # df = pivot_table[pivot_table['coherence_bin'] == 0]
    # data = pd.pivot_table(df, index=['power_bin'], columns=['poca_distance_bin'],values=values, observed=False)
    # _, ax = plt.subplots(figsize=(15, 6))
    # sns.heatmap(data, ax=ax)
    # filename = outpath + f'_2dslice3_{plot_var}.png'
    # plt.savefig(filename, bbox_inches='tight')