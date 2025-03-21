#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""cpom.altimetry.tools.nc_vals.py

Tool to print out NetCDF values with more control than ncdump.

By default it prints scaled values (scale_factor, add_offset) unless disabled with `-r`.

# Command Line Options

  -h, --help              Show this help message and exit
  -d, --describe          Print only the specified parameter definition and attributes (like ncdump)
  -g, --global-atts       Print only global attributes from the file
  -p PARAM, --param PARAM Parameter path (e.g., data/ku/power_noise_floor)
  -r, --raw               Print raw values without applying scale_factor or add_offset
  -s, --show_every_index  Print every value with its index (one per line)
  -n PRECISION, --precision PRECISION
                          Number of decimal places to print for float values

# Example

    nc_vals.py -s -p data/ku/power_noise_floor yourfile.nc
"""

import argparse
import os
import sys

import numpy as np
from netCDF4 import Dataset, Variable  # pylint: disable=no-name-in-module


def print_variable_description(var: Variable, varname: str) -> None:
    """Print the NetCDF variable's group, type, dimensions, and attributes.

    Args:
        var: The NetCDF variable object.
        varname: The name of the variable (leaf name only).
    """
    # Reconstruct group path
    group_path = ""
    grp = var.group()
    while grp.parent is not None:
        group_path = "/" + grp.name.lstrip("/") + group_path
        grp = grp.parent
    if group_path:
        print(f"group: {group_path}")

    dtype = var.dtype
    dims = ", ".join(var.dimensions)
    print(f"{dtype} {varname}({dims}) ;")

    for attr in var.ncattrs():
        val = getattr(var, attr)
        if isinstance(val, str):
            print(f'    {varname}:{attr} = "{val}" ;')
        elif isinstance(val, (int, np.integer)):
            suffix = "LL" if "int64" in str(dtype) else ""
            print(f"    {varname}:{attr} = {val}{suffix} ;")
        elif isinstance(val, float):
            print(f"    {varname}:{attr} = {val} ;")
        elif isinstance(val, (list, np.ndarray)):
            formatted = ", ".join(map(str, val))
            print(f"    {varname}:{attr} = {formatted} ;")
        else:
            print(f"    {varname}:{attr} = {val} ;  // unhandled type")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(
        description="Print values of a NetCDF parameter, with optional scaling and formatting."
    )
    parser.add_argument(
        "-f",
        "--filename",
        required=not any(arg in sys.argv for arg in ["-g", "--global-atts", "-h", "--help"]),
        help="Path to the NetCDF file",
    )

    parser.add_argument(
        "-d",
        "--describe",
        action="store_true",
        help="Print only the specified parameter definition and attributes (like ncdump)",
    )
    parser.add_argument(
        "-g",
        "--global-atts",
        action="store_true",
        help="Print only global attributes from the file",
    )
    parser.add_argument(
        "-p",
        "--param",
        required=False,
        help=(
            "Parameter path (e.g., data/ku/power_noise_floor). "
            "By default, values are scaled unless --raw is passed."
        ),
    )
    parser.add_argument(
        "-r",
        "--raw",
        action="store_true",
        help="Print raw values without applying scale_factor or add_offset",
    )
    parser.add_argument(
        "-s",
        "--show_every_index",
        action="store_true",
        help="Print every value with its index (one per line)",
    )
    parser.add_argument(
        "-n",
        "--precision",
        type=int,
        default=None,
        help="Number of decimal places to print for float values",
    )
    return parser.parse_args()


def get_variable(dataset: Dataset, var_path: str) -> Variable:
    """Retrieve a variable from the dataset given a path.

    Args:
        dataset: The open NetCDF dataset.
        var_path: Slash-separated variable path (e.g., data/ku/var).

    Returns:
        The NetCDF variable object.

    Raises:
        KeyError: If the group or variable is not found.
    """
    parts = var_path.strip("/").split("/")
    grp = dataset
    for part in parts[:-1]:
        if part not in grp.groups:
            raise KeyError(f"Group '{part}' not found in path '{var_path}'")
        grp = grp.groups[part]
    varname = parts[-1]
    if varname not in grp.variables:
        raise KeyError(f"Variable '{varname}' not found in path '{var_path}'")
    return grp.variables[varname]


def apply_scaling(var: Variable, data: np.ndarray) -> np.ndarray:
    """Apply scale_factor and add_offset to data.

    Args:
        var: The NetCDF variable.
        data: The data array (possibly masked).

    Returns:
        Scaled NumPy array.
    """
    scale = getattr(var, "scale_factor", 1.0)
    offset = getattr(var, "add_offset", 0.0)
    return data.astype(np.float64) * scale + offset


def is_float_dtype(dtype: np.dtype) -> bool:
    """Check if dtype is floating-point.

    Args:
        dtype: NumPy data type.

    Returns:
        True if float, else False.
    """
    return np.issubdtype(dtype, np.floating)


def print_indexed(data: np.ndarray, precision: int | None = None) -> None:
    """Print each value with its index. Masked values are shown as '--FV--'.

    Args:
        data: Data array (masked or regular).
        precision: Decimal precision for float values.
    """
    flat = data.flatten()
    for i, val in enumerate(flat):
        if np.ma.is_masked(val):
            print(f"{i}: --FV--")
        else:
            if isinstance(val, float) and precision is not None:
                print(f"{i}: {val:.{precision}f}")
            else:
                print(f"{i}: {val}")


def print_global_attributes(dataset: Dataset) -> None:
    """Print global attributes of the dataset.

    Args:
        dataset: The open NetCDF dataset.
    """
    print("// global attributes:")
    for attr in dataset.ncattrs():
        val = getattr(dataset, attr)
        if isinstance(val, str):
            print(f'    :{attr} = "{val}" ;')
        elif isinstance(val, (int, np.integer)):
            print(f"    :{attr} = {val} ;")
        elif isinstance(val, float):
            print(f"    :{attr} = {val} ;")
        elif isinstance(val, (list, np.ndarray)):
            formatted = ", ".join(map(str, val))
            print(f"    :{attr} = {formatted} ;")
        else:
            print(f"    :{attr} = {val} ;  // unhandled type")


def main() -> None:
    """Main function to drive CLI behavior."""
    args = parse_args()

    if not os.path.exists(args.filename):
        print(f"Error: File '{args.filename}' not found.", file=sys.stderr)
        sys.exit(1)

    try:
        with Dataset(args.filename, "r") as ds:
            if args.global_atts:
                print_global_attributes(ds)
                return

            if not args.param:
                print("Error: --param is required unless using --global-atts", file=sys.stderr)
                sys.exit(1)

            var = get_variable(ds, args.param)
            fill_value = getattr(var, "_FillValue", None)
            var.set_auto_maskandscale(False)

            if args.describe:
                print_variable_description(var, args.param.split("/")[-1])
                return

            raw_data = var[:]

            # Mask fill values before scaling
            if fill_value is not None:
                data = np.ma.masked_equal(raw_data, fill_value)
            else:
                data = raw_data

            if not args.raw:
                data = apply_scaling(var, data)

            if args.show_every_index:
                print_indexed(data, precision=args.precision)
            else:
                output_array = data.filled(np.nan) if np.ma.isMaskedArray(data) else data

                np.set_printoptions(
                    precision=args.precision if args.precision is not None else 8,
                    threshold=1000,
                    linewidth=120,
                    floatmode="fixed",
                )
                print(output_array)

    except FileNotFoundError:
        print(f"Error: File '{args.filename}' not found.", file=sys.stderr)
        sys.exit(1)
    except KeyError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except OSError as exc:
        print(f"OS error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
