#!/usr/bin/env python3
"""
plot_diff.py: plot two NetCDF variables (possibly in nested groups) and their difference,
with optional range controls, units labeling, NaN-aware min/max annotation placed below,
stats (min, max, mean, std, MAD, trimmed mean, trimmed std), and difference histograms.

Usage:
    plot_diff.py -f1 FILE1 -f2 FILE2 -p1 PARAM1 -p2 PARAM2 \
                 [--range1 MIN MAX] [--range2 MIN MAX] [--units UNIT_STRING]

Example:
    plot_diff.py -f1 file1.nc -f2 file2.nc \
                 -p1 data/ku/elevation -p2 elevation \
                 --range1 0 1000 --range2 -50 50 --units m
"""
import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset  # pylint: disable=no-name-in-module


def get_variable(ncfile, param_path):
    """
    Open ncfile, navigate to nested groups per param_path (e.g. 'data/ku/elevation'),
    and return the variable's data as a NumPy array, with all FillValue (and/or missing_value)
    replaced by np.nan.
    """
    ds = Dataset(ncfile, "r")
    parts = param_path.strip("/").split("/")
    *groups, varname = parts

    # Drill into nested groups
    grp = ds
    for g in groups:
        if g in grp.groups:
            grp = grp.groups[g]
        else:
            ds.close()
            sys.exit(f"Error: group '{g}' not found in '{ncfile}'")

    # Find the variable
    if varname not in grp.variables:
        ds.close()
        sys.exit(
            f"Error: variable '{varname}' not found in "
            f"'{ncfile}' (inside group '{'/'.join(groups)}')"
        )
    varobj = grp.variables[varname]

    # Read data as float so we can assign np.nan
    data = varobj[:].astype(float)

    # Try to get the FillValue or missing_value attribute
    fill_value = None
    if "_FillValue" in varobj.ncattrs():
        fill_value = varobj.getncattr("_FillValue")
        print(f"{param_path} FillValue found {fill_value}")
    elif "missing_value" in varobj.ncattrs():
        fill_value = varobj.getncattr("missing_value")

    data = np.array(data)
    # Replace all occurrences of fill_value with np.nan
    if fill_value is not None:
        data[np.isclose(data, fill_value.astype(float), atol=0.1)] = np.nan

    ds.close()
    return data


def main():
    """main tool function"""
    p = argparse.ArgumentParser(
        description="Plot two NetCDF variables and their difference "
        "with ranges, units, stats, and histograms"
    )
    p.add_argument(
        "-l",
        "--list",
        required=False,
        help="List differences line by line to stdout",
        action="store_true",
    )

    p.add_argument("-f1", "--file1", required=True, help="First NetCDF file")
    p.add_argument("-f2", "--file2", required=True, help="Second NetCDF file")
    p.add_argument(
        "-p1", "--param1", required=True, help="Path to first variable (e.g. data/ku/elevation)"
    )
    p.add_argument("-p2", "--param2", required=True, help="Path to second variable")
    p.add_argument(
        "--range1",
        nargs=2,
        type=float,
        metavar=("MIN", "MAX"),
        help="Value range [min max] for the parameter plots",
    )
    p.add_argument(
        "--range2",
        nargs=2,
        type=float,
        metavar=("MIN", "MAX"),
        help="Value range [min max] for the difference plot and histograms",
    )

    p.add_argument(
        "--subset",
        nargs=2,
        type=int,
        metavar=("MIN", "MAX"),
        help="subset indices of input [min max] to plot data",
        required=False,
    )

    p.add_argument("-u", "--units", type=str, default="", help="Units string to append to labels")
    args = p.parse_args()

    units_label = f" ({args.units})" if args.units else ""

    v1 = get_variable(args.file1, args.param1)
    v2 = get_variable(args.file2, args.param2)

    if args.subset:
        index1, index2 = args.subset
        v1 = np.array(v1)[index1:index2]
        v2 = np.array(v2)[index1:index2]
    else:
        v1 = np.array(v1)
        v2 = np.array(v2)

    if v1.shape != v2.shape:
        sys.exit(f"Error: shapes do not match: {v1.shape} vs {v2.shape}")

    v1_n_finite = np.isfinite(v1).sum()
    v2_n_finite = np.isfinite(v2).sum()

    ok = np.isfinite(v1) & np.isfinite(v2)
    v1 = v1[ok]
    v2 = v2[ok]
    diff = v1 - v2

    if args.list:
        for index, idiff in enumerate(diff):
            if np.abs(idiff) > 1.0:
                print(f"{index} : diff {idiff} : {v1[index]} {v2[index]}")
    diff_min = np.nanmin(diff)
    diff_max = np.nanmax(diff)

    # --- compute statistics of the differences ---
    mean_diff = np.nanmean(diff)
    std_diff = np.nanstd(diff)
    mad_diff = np.nanmedian(np.abs(diff - np.nanmedian(diff)))

    # --- compute a 10%-trimmed (outlier-resistant) mean and std ---
    flat = diff.ravel()
    flat_nonan = flat[~np.isnan(flat)]
    p_low, p_high = np.percentile(flat_nonan, [5, 95])
    mask_trim = (flat_nonan >= p_low) & (flat_nonan <= p_high)
    trim_mean = np.nanmean(flat_nonan[mask_trim])
    trim_std = np.nanstd(flat_nonan[mask_trim])

    ndim = diff.ndim
    if ndim not in (1, 2):
        sys.exit(f"Error: can only plot 1D or 2D variables (got {ndim}D)")

    # Create figure and layout
    fig = plt.figure(figsize=(10, 10))
    fig.subplots_adjust(top=0.95, bottom=0.1, hspace=0.5, wspace=0.4)
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1])

    # Top: original variables
    if ndim == 1:
        ax1 = fig.add_subplot(gs[0, :])
        x_vals = np.array(range(len(v1)))
        if args.subset:
            x_vals += index1
        ax1.plot(x_vals, v1, label=args.param1)
        ax1.plot(x_vals, v2, label=args.param2)
        ax1.set_ylabel(f"Value{units_label}")
        ax1.set_title("Variables")
        ax1.legend()
        if args.range1:
            ax1.set_ylim(args.range1)
    else:
        ax1 = fig.add_subplot(gs[0, 0])
        im1 = ax1.imshow(
            v1,
            aspect="auto",
            vmin=args.range1[0] if args.range1 else None,
            vmax=args.range1[1] if args.range1 else None,
        )
        ax1.set_title(args.param1)
        c1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        if args.units:
            c1.set_label(args.units)

        ax2 = fig.add_subplot(gs[0, 1])
        im2 = ax2.imshow(
            v2,
            aspect="auto",
            vmin=args.range1[0] if args.range1 else None,
            vmax=args.range1[1] if args.range1 else None,
        )
        ax2.set_title(args.param2)
        c2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        if args.units:
            c2.set_label(args.units)

    # Middle: difference plot
    axdiff = fig.add_subplot(gs[1, :])
    if ndim == 1:
        axdiff.plot(x_vals, diff, color="k")
        axdiff.set_ylabel(f"Difference{units_label}")
        axdiff.set_xlabel("Index")
        axdiff.set_title(f"{args.param1} - {args.param2}")
    else:
        imd = axdiff.imshow(
            diff,
            aspect="auto",
            vmin=args.range2[0] if args.range2 else None,
            vmax=args.range2[1] if args.range2 else None,
        )
        axdiff.set_title(f"{args.param1} - {args.param2}")
        cd = plt.colorbar(imd, ax=axdiff, fraction=0.046, pad=0.04)
        if args.units:
            cd.set_label(args.units)
    if args.range2 and ndim == 1:
        axdiff.set_ylim(args.range2)

    # Bottom: histograms
    dif_flat = flat_nonan
    bins = 50
    if args.range2:
        axh1 = fig.add_subplot(gs[2, 0])
        axh2 = fig.add_subplot(gs[2, 1])
    else:
        axh1 = fig.add_subplot(gs[2, :])
        axh2 = None

    axh1.hist(dif_flat, bins=bins)
    axh1.set_title("Histogram of all differences")
    axh1.set_xlabel(f"Difference{units_label}")
    axh1.set_ylabel("Count")

    if axh2 is not None:
        mask2 = (dif_flat >= args.range2[0]) & (dif_flat <= args.range2[1])
        axh2.hist(dif_flat[mask2], bins=bins)
        axh2.set_title(f"Differences within [{args.range2[0]}, {args.range2[1]}]")
        axh2.set_xlabel(f"Difference{units_label}")
        axh2.set_ylabel("Count")

    # Footer: all stats
    fig.text(
        0.5,
        0.02,
        f"min={diff_min:.3g}    max={diff_max:.3g}    "
        f"mean={mean_diff:.3g}    std={std_diff:.3g}    "
        f"MAD={mad_diff:.3g}    trim_mean={trim_mean:.3g}    "
        f"trim_std={trim_std:.3g}",
        ha="center",
        va="bottom",
    )

    if v1_n_finite != v2_n_finite:
        print(f"Number not nan mismatch {v1_n_finite} {v2_n_finite}")

    plt.show()


if __name__ == "__main__":
    main()
