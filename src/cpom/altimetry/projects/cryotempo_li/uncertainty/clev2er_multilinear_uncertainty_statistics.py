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
import itertools
import glob

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

from cpom.roughness.roughness import Roughness
from cpom.slopes.slopes import Slopes

# ---------------------------------------------------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------------------------------------------------

def import_variables(indir, area):

    """
    Import the variable data associated with joined elevation difference values.

    Args:
        indir (str): filepath to input directory with elevation difference files
        area (str): choice of ice sheet

    Returns:
        dh_all (np.ndarray): Array of elevation difference values
        slope_all (np.ndarray): Array of slope values
        roughness_all (np.ndarray): Array of roughness values
        power_all (np.ndarray): Array of power values
        coherence_all (np.ndarray): Array of coherence values
        poca_distance_all (np.ndarray): Array of distance values

    """
    dh_all = []
    lats_all = []
    lons_all = []
    power_all = []
    coherence_all = []
    poca_distance_all = []

    for path in glob.glob(f'{indir}/**/*.npz', recursive=True):
        
        # extract data from each monthly file    
        print("Path: ", path)
        dh_data = np.load(path, allow_pickle=True)
        dh = dh_data['dh']
        lats = dh_data['lats']
        lons = dh_data['lons']
        power = dh_data.get("pow")
        coherence = dh_data.get("coh")
        poca_distance = dh_data.get("dis_poca")

        # append monthly data to all data
        dh_all.extend(dh)
        lats_all.extend(lats)
        lons_all.extend(lons)
        power_all.extend(power)
        coherence_all.extend(coherence)
        poca_distance_all.extend(poca_distance)
    
    dh_all = np.asarray(dh_all)
    power_all = np.asarray(power_all)
    coherence_all = np.asarray(coherence_all)
    poca_distance_all = np.asarray(poca_distance_all)
    lats_all = np.asarray(lats_all)
    lons_all = np.asarray(lons_all)

    # dh_data = np.load(dh_file, allow_pickle=True)

    # Extract variables directly from difference dataset
    # lats = dh_data.get("lats")
    # lons = dh_data.get("lons")
    # dh = dh_data.get("dh")

    # power = dh_data.get("pow")
    # coherence = dh_data.get("coh")
    # poca_distance = dh_data.get("dis_poca")

    # Extract slope and roughness values from zarr files
    if area == 'antarctica_icesheets':
        this_slope = Slopes("rema_100m_900ws_slopes_zarr")
        this_roughness = Roughness("rema_100m_900ws_roughness_zarr")
    
    print('interpolating slopes')
    slope = this_slope.interp_slopes(lats_all, lons_all, method="linear", xy_is_latlon=True)
    print('interpolating roughness')
    roughness = this_roughness.interp_roughness(lats_all, lons_all, method="linear", xy_is_latlon=True)

    mask = ~np.isnan(slope)
    slope = slope[mask]
    roughness = roughness[mask]
    power_all = power_all[mask]
    coherence_all = coherence_all[mask]
    poca_distance_all = poca_distance_all[mask]
    dh_all = dh_all[mask]

    return dh_all, slope, roughness, power_all, coherence_all, poca_distance_all


def correlation(dh, independent_vars, var_names):

    """
    Calculate correlation between elevation difference and each variable. 

    Args:
        dh (np.ndarray): Array of elevation difference values
        independent_vars (list): List of variables to assess statistically
        var_names (list): List of variable names

    Returns:
        corr_coef_dict (Dictionary): Correlation coefficient, for each variable
    
    """
    corr_coef_dict = {}

    # for each variable, calculate the correlation coefficient with elevation difference
    for i in range(len(independent_vars)):
        corr_coef = np.corrcoef(dh, independent_vars[i])  
        print(f'{var_names[i]}: {np.round(corr_coef[0,1],3)}')  # print true correlation coefficients
        corr_coef_dict[var_names[i]] = abs(corr_coef[0,1])  # add absolute correlation coefficients to dict for sorting
    
    return corr_coef_dict


def fit_multilinear_regression(dh, independent_vars, var_names):

    """
    Fit multi-linear regression to input variables. 
    Elevation difference (dh) as dependent variable
    Slope, roughness, power, coherence and distance as independent variables

    Args:
        dh (np.ndarray): Array of elevation difference values
        independent_vars (list): List of variables to assess statistically
        var_names (list): List of variable names

    Returns:
        r2: R2 value (% of dependent variable explained by independent variables)
        F_pval: F test p value (complex model performs better than simple models; if less than 0.05)
        pvals: p values for independent variables (independent variable affects dependent variable; if less than 0.05)
    
    """
    # organise independent and dependent variables, for a range of total independent variables
    if len(independent_vars) == 1:

        dict = {'difference_abs': abs(dh), 'var1': independent_vars[0]}
        df = pd.DataFrame(dict)

        x = df[['var1']]
        y = df['difference_abs']
    
    elif len(independent_vars) == 2:

        dict = {'difference_abs': abs(dh), 'var1': independent_vars[0], 'var2': independent_vars[1]}
        df = pd.DataFrame(dict)

        x = df[['var1','var2']]
        y = df['difference_abs']
    
    elif len(independent_vars) == 3: 

        dict = {'difference_abs': abs(dh), 'var1': independent_vars[0], 'var2': independent_vars[1], 'var3': independent_vars[2]}
        df = pd.DataFrame(dict)

        x = df[['var1','var2', 'var3']]
        y = df['difference_abs']
    
    elif len(independent_vars) == 4:

        dict = {'difference_abs': abs(dh), 'var1': independent_vars[0], 'var2': independent_vars[1], 'var3': independent_vars[2], 'var4': independent_vars[3]}
        df = pd.DataFrame(dict)

        x = df[['var1','var2', 'var3', 'var4']]
        y = df['difference_abs']
    
    elif len(independent_vars) == 5:

        dict = {'difference_abs': abs(dh), 'var1': independent_vars[0], 'var2': independent_vars[1], 'var3': independent_vars[2], 'var4': independent_vars[3], 'var5': independent_vars[4]}
        df = pd.DataFrame(dict)

        x = df[['var1','var2', 'var3', 'var4', 'var5']]
        y = df['difference_abs']

    # fit regression 
    olsmod = sm.OLS(y, x).fit()

    # output statistics to assess regression
    r2 = olsmod.rsquared
    F_pval = olsmod.f_pvalue
    pvals = olsmod.pvalues

    # print output
    print(f'R2 = {np.round(r2,2)}')

    if F_pval > 0.05:
        print('WARNING: Complex model does NOT perform better than simple model')
    
    for i in range(len(pvals)):
        if pvals.iloc[i] < 0.05:
            print(f'{var_names[i]} SIGNIFICANT; p value: {np.round(pvals.iloc[i], 3)}')
        else: 
            print(f'{var_names[i]} NOT SIGNIFICANT; p value: {np.round(pvals.iloc[i], 3)}')

    return r2, F_pval, pvals


def plot_r2(df, var_names, outdir):

    """
    Plot the R2 values for all multi-linear regressions as individual box plots
    Plot by colour (variable) and size (number of variables)

    Args:
        df (pd.DataFrame): DataFrame of R2 values, variable combinations and variable combination lengths
        var_names (list): List of named variables to assess statistically
        outdir (str): Output directory for figure
    
    """
    _, ax = plt.subplots()

    filtered_r2_list = []

    for i, var in enumerate(var_names):
        df_filter = df[df.combinations.str.contains(var)]
        ax.scatter([i+1]*len(df_filter['r2']), df_filter['r2'], s=df_filter['comb_length']**3, edgecolors='k', label=var_names[i])
        filtered_r2_list.append(df_filter['r2'])
    
    filtered_r2_array = np.transpose(np.array(filtered_r2_list), (1, 0))
    ax.boxplot(filtered_r2_array)

    ax.set_xlabel('Variable')
    ax.set_ylabel('R2')

    ax.legend()

    outpath = outdir + '/r2_box_plots.png'

    plt.savefig(outpath)


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
        "-dh_dir",
        "-dh",
        help=(
            "directory of elevation difference npz file"
        ),
        type=str,
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

    # import variables
    dh, slope, roughness, power, coherence, poca_distance = import_variables(args.dh_dir, args.area)
    var_names_all = ['slope', 'roughness', 'power', 'coherence'] #, 'poca_distance']
    independent_vars_all = [slope, roughness, power, coherence] # poca_distance]

    # calculate correlation coefficient for each variable with elevation difference
    correlation(abs(dh), independent_vars_all, var_names_all)

    # set up empty lists
    combinations_list = []
    combinations_length_list = []
    r2_list = []

    # for each combination of variables, calculate multi-linear regression
    # for each possible length of combination (ie 1,2,3,4 or 5)
    for v_no in range(1, len(var_names_all)+1):
        # find all the possible combinations of variables for a given length of combination
        combinations = list(itertools.combinations(var_names_all, v_no))
        # for each possible combination
        for c in combinations:
            independent_vars = []
            var_names = []
            # collect the dataset corresponding to the variable combination
            for i in range(v_no):
                index = var_names_all.index(c[i])
                independent_vars.append(independent_vars_all[index])
                var_names.append(var_names_all[index])
            print(var_names)
            # carry out multi-linear regression using this combination of variables
            r2, _, _ = fit_multilinear_regression(dh, independent_vars, var_names)
            r2_list.append(r2)
            combinations_list.append(str(var_names))
            combinations_length_list.append(len(var_names))
    
    # build and sort dataframe of R2 values for all possible combinations
    r2_dict = {'r2':r2_list, 'combinations':combinations_list, 'comb_length':combinations_length_list}
    df = pd.DataFrame(r2_dict)
    df = df.sort_values(by=['r2'], ascending=False)
    print(df)

    # save dataframe to csv
    df.to_csv(args.outdir + '/r2.csv')

    # plot scatter and boxplots for R2 values, split by variables, sized by number of dimensions
    plot_r2(df, var_names_all, args.outdir)


if __name__ == "__main__":
    main()


# EXTRA

# PLOT SEPARATE SCATTER FOR EACH VARIABLE (X DIMS, Y R2)

    # fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(10,6))

    # for var, ax in zip(var_names, axs.ravel()):
    #     df_filter = df[df.combinations.str.contains(var)]
    #     ax.scatter(df_filter['comb_length'], df_filter['r2'])

    #     ax.set_xlim(0,5)
    #     ax.set_ylim(0,1)

    #     ax.grid()

# BUILD MULTI LINEAR REGRESSION MODELS FOLLOWING THE ORDER OF THE CORRELATION STRENGTH

    # # iterate through multi-linear regressions (add variables in order of correlation strength)
    # # sort dictionary by strongest correlation
    # independent_vars = []
    # var_names = []

    # corr_coef_dict_sorted_keys = sorted(corr_coef_dict, key=corr_coef_dict.get, reverse=True) 

    # for each variable, in order, use dataset in multilinear regression, before adding next variable
    # for i in range(len(corr_coef_dict_sorted_keys)):
    #     index = var_names_all.index(corr_coef_dict_sorted_keys[i])
    #     independent_vars.append(independent_vars_all[index])
    #     var_names.append(corr_coef_dict_sorted_keys[i])
    #     fit_multilinear_regression(dh, independent_vars, var_names)
