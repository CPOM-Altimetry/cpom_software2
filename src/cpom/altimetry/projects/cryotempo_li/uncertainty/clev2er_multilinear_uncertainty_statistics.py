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
python clev2er_multilinear_uncertainty_statistics.py -a antarctica_icesheets \
    -dh_file /media/luna/boxallk/clev2er/uncertainty_assessment/uncertainty_tables/test/2020/03/cs2_minus_is2_gt2r_p2p_diffs_ais_zwally_21.npz \
        -v slope,roughness,power,coherence,poca_distance

Author: Karla Boxall

"""
# ---------------------------------------------------------------------------------------------------------------------
# Package Imports
# ---------------------------------------------------------------------------------------------------------------------
import argparse
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm

from cpom.roughness.roughness import Roughness
from cpom.slopes.slopes import Slopes

# ---------------------------------------------------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------------------------------------------------

def import_variables(dh_file, area, vars):

    """
    Import the variable data associated with joined elevation difference values.

    Args:
        dh_file (str): filepath to elevation difference file
        area (str): choice of ice sheet
        vars (list): list of variables to import

    Returns:
        dh (np.ndarray): Array of elevation difference values
        slope (np.ndarray): Array of slope values
        roughness (np.ndarray): Array of roughness values
        power (np.ndarray): Array of power values
        coherence (np.ndarray): Array of coherence values
        distance (np.ndarray): Array of distance values

    """

    slope=None
    roughness=None
    power=None
    coherence=None
    poca_distance=None
    
    dh_data = np.load(dh_file, allow_pickle=True)

    # Extract variables directly from difference dataset
    lats = dh_data.get("lats")
    lons = dh_data.get("lons")
    dh = dh_data.get("dh")

    if "power" in vars:
        power = dh_data.get("pow")
    if "coherence" in vars:
        coherence = dh_data.get("coh")
    if "poca_distance" in vars:
        poca_distance = dh_data.get("dis_poca")

    # Extract slope and roughness values from zarr files
    if area == 'antarctica_icesheets':
        if "slope" in vars:
            this_slope = Slopes("rema_100m_900ws_slopes_zarr")
            slope = this_slope.interp_slopes(lats, lons, method="linear", xy_is_latlon=True)
        if "roughness" in vars:
            this_roughness = Roughness("rema_100m_900ws_roughness_zarr")
            roughness = this_roughness.interp_roughness(lats, lons, method="linear", xy_is_latlon=True)

    return dh, slope, roughness, power, coherence, poca_distance


def fit_multilinear_regression(dh, stats_vars):

    """
    Fit multi-linear regression to input variables. 
    Elevation difference (dh) as dependent variable
    Slope, roughness, power, coherence and distance as independent variables

    Args:
        dh (np.ndarray): Array of elevation difference values
        stats_vars (list): List of variables to assess statistically

    Returns:
        r2: R2 value (% of dependent variable explained by independent variables)
        F_pval: F test p value (complex model performs better than simple models; if less than 0.05)
        pvals: p values for independent variables (independent variable affects dependent variable; if less than 0.05)

    
    """
    
    if len(stats_vars) == 1:

        dict = {'difference_abs': abs(dh), 'var1': stats_vars[0]}
        df = pd.DataFrame(dict)

        x = df[['var1']]
        y = df['difference_abs']
    
    elif len(stats_vars) == 2:

        dict = {'difference_abs': abs(dh), 'var1': stats_vars[0], 'var2': stats_vars[1]}
        df = pd.DataFrame(dict)

        x = df[['var1','var2']]
        y = df['difference_abs']
    
    elif len(stats_vars) == 3: 

        dict = {'difference_abs': abs(dh), 'var1': stats_vars[0], 'var2': stats_vars[1], 'var3': stats_vars[2]}
        df = pd.DataFrame(dict)

        x = df[['var1','var2', 'var3']]
        y = df['difference_abs']
    
    elif len(stats_vars) == 4:

        dict = {'difference_abs': abs(dh), 'var1': stats_vars[0], 'var2': stats_vars[1], 'var3': stats_vars[2], 'var4': stats_vars[3]}
        df = pd.DataFrame(dict)

        x = df[['var1','var2', 'var3', 'var4']]
        y = df['difference_abs']
    
    elif len(stats_vars) == 5:

        dict = {'difference_abs': abs(dh), 'var1': stats_vars[0], 'var2': stats_vars[1], 'var3': stats_vars[2], 'var4': stats_vars[3], 'var5': stats_vars[4]}
        df = pd.DataFrame(dict)

        x = df[['var1','var2', 'var3', 'var4', 'var5']]
        y = df['difference_abs']


    olsmod = sm.OLS(y, x).fit()

    r2 = olsmod.rsquared
    F_pval = olsmod.f_pvalue
    pvals = olsmod.pvalues

    return r2, F_pval, pvals


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

    parser.add_argument(
    "--vars",
    "-v",
    help="[optional, default=all] Variables to assess. Comma separated list of: all, slope, roughness, power, coherence, poca_distance",
)

    # read arguments from the command line
    args = parser.parse_args()

    if args.vars:
        vars = args.vars.split(",")

    for var in vars:
        if var not in ["slope", "roughness", "power", "coherence", "poca_distance"]:
            sys.exit(
                "{} not a valid variable. Must be one of slope, roughness, power, coherence, poca_distance".format(
                    var
                )
            )

    # import variables
    dh, slope, roughness, power, coherence, poca_distance = import_variables(args.dh_file, args.area, args.vars)

    stats_vars = []
    for v in [slope, roughness, power, coherence, poca_distance]:
        if v is not None: 
            stats_vars.append(v)

    # carry out statistics on variables
    # r2, F_stat, F_pval, pvals = fit_multilinear_regression(dh, slope, roughness, power, coherence, poca_distance)
    r2, F_pval, pvals = fit_multilinear_regression(dh, stats_vars)

    print(f'{np.round(r2*100,0)}% explained')

    if F_pval < 0.05:
        print('Complex model performs better than simple model')
    else:
        print('Complex model does NOT perform better than simple model')
    
    for i in range(len(pvals)):
        if pvals.iloc[i] < 0.05:
            print(f'{vars[i]} affects elevation difference; p value: {np.round(pvals.iloc[i], 3)}')

if __name__ == "__main__":
    main()



# EXTRA

    # for var in vars: 
    #     if var == 'slope':
    #         dh, slope = import_variables(args.dh_file, args.area)
    #     if var == 'roughness':
    #         dh, roughness = import_variables(args.dh_file, args.area)
    #     if var == 'power':
    #         dh, power = import_variables(args.dh_file, args.area)
    #     if var == 'coherence':
    #         dh, coherence = import_variables(args.dh_file, args.area)
    #     if var == 'poca_distance':
    #         dh, poca_distance = import_variables(args.dh_file, args.area)


    # def fit_multilinear_regression(dh, slope, roughness, power, coherence, distance):

    # """
    # Fit multi-linear regression to input variables. 
    # Elevation difference (dh) as dependent variable
    # Slope, roughness, power, coherence and distance as independent variables

    # Args:
    #     dh (np.ndarray): Array of elevation difference values
    #     slope (np.ndarray): Array of slope values
    #     roughness (np.ndarray): Array of roughness values
    #     power (np.ndarray): Array of power values
    #     coherence (np.ndarray): Array of coherence values
    #     distance (np.ndarray): Array of distance values

    # Returns:
    #     r2: R2 value (% of dependent variable explained by independent variables)
    #     F_pval: F test p value (complex model performs better than simple models; if less than 0.05)
    #     pvals: p values for independent variables (independent variable affects dependent variable; if less than 0.05)

    
    # """

    # dict = {'difference_abs': abs(dh), 'slope': slope, 'roughness': roughness, 'power': power, 'coherence': coherence, 'distance': distance}
    # df = pd.DataFrame(dict)

    # x = df[['slope', 'roughness', 'power', 'coherence', 'distance']]
    # y = df['difference']

    # olsmod = sm.OLS(y, x).fit()

    # r2 = olsmod.rsquared
    # F_pval = olsmod.f_pvalue
    # pvals = olsmod.pvalues

    # return r2, F_pval, pvals
