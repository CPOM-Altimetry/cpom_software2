"""
Script to assess which variables should be used to generate uncertainty lookup table.

Multi-linear model is built using all combinations of variables.
    - R2 - X% of the dependent variable can be explained using our indepedent variables 
    (higher = better)
    - F test - determine whether our complex model performs better than a simpler model 
    (i.e. model with only one independent variable) (if the F p value is less than 0.05, 
    the model performs better than other simple models)
    - t test - measure of the precision with which the regression coefficient is measured 
    (if p value is less than 0.05 for a given indepedent variable, there is sufficient 
    evidence that that indepedent variable affects the dependent variable)

Only available for Antarctic currently.

Compatible for the following variables:
- Slope: interpolated from Slopes("rema_100m_900ws_slopes_zarr")
- Roughness: interpolated from Roughness ("rema_100m_900ws_roughness_zarr)
- Power: sigma 0 from L2i product
- Coherence: coherence from L2i product
- Distance to POCA: distance between nadir and POCA

Input is dh values, lat, lon, power, coherence and POCA distance from CS2-IS2 differences 
npz files, for example: cs2_minus_is2_gt2lgt2r_p2p_diffs_antarctica_icesheets.npz

example usage: 
###

Author: Karla Boxall

"""
# ---------------------------------------------------------------------------------------------------------------------
# Package Imports
# ---------------------------------------------------------------------------------------------------------------------
import argparse

import numpy as np
import pandas as pd
import statsmodels.api as sm

from cpom.roughness.roughness import Roughness
from cpom.slopes.slopes import Slopes

# ---------------------------------------------------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------------------------------------------------

def import_variables(dh_file, area):

    """
    Import the variable data associated with joined elevation difference values.

    Args:
        dh_file (str): filepath to elevation difference file
        area (np.str): choice of ice sheet

    Returns:
        dh (np.ndarray): Array of elevation difference values
        slope (np.ndarray): Array of slope values
        roughness (np.ndarray): Array of roughness values
        power (np.ndarray): Array of power values
        coherence (np.ndarray): Array of coherence values

    """

    dh_data = np.load(dh_file, allow_pickle=True)

    # Extract variables directly from difference dataset
    lats = dh_data.get("lats")
    lons = dh_data.get("lons")
    dh = dh_data.get("dh")
    power = dh_data.get("pow")
    coherence = dh_data.get("coh")

    # Extract slope and roughness values from zarr files
    if area == 'antarctica_icesheets':
        this_slope = Slopes("rema_100m_900ws_slopes_zarr")
        this_roughness = Roughness("rema_100m_900ws_roughness_zarr")
    slope = this_slope.interp_slopes(lats, lons, method="linear", xy_is_latlon=True)
    roughness = this_roughness.interp_roughness(lats, lons, method="linear", xy_is_latlon=True)

    return dh, slope, roughness, power, coherence


def fit_multilinear_regression(dh, slope, roughness, power, coherence):

    """
    Docstring here
    
    """

    dict = {'difference_abs': abs(dh), 'slope': slope, 'roughness': roughness, 'power': power, 'coherence': coherence}
    df = pd.DataFrame(dict)

    x = df[['slope', 'roughness', 'power', 'coherence']]
    y = df['difference']

    olsmod = sm.OLS(y, x).fit()

    r2 = olsmod.rsquared
    F_stat = olsmod.fvalue
    F_pval = olsmod.f_pvalue
    pvals = olsmod.pvalues

    return r2, F_stat, F_pval, pvals

# -------------------------------------------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------------------------------------------

def main():

    """main function for command line tool"""

    # initiate the command line parser
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--area",
        "-a",
        choices=['antarctica_icesheets'],
        help=("choose the ice sheet: 'median' (default)"),
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

    dh, slope, roughness, power, coherence = import_variables(args.dh_file, args.area)

