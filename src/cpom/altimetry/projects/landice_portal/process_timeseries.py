#!/usr/bin/env python3
"""
process_timeseries.py

Pre-processing tool to optimize timeseries CSV files for the land ice web portal.
Takes the aggregate dh and uncertainty CSVs and splits them into individual JSON
files per basin (e.g., basin_1.json, basin_AIS.json) for O(1) loading in the browser.
"""

import argparse
import json
import os

import pandas as pd

# Mapping of Rignot 2016 basin strings (as seen in the CSV) to their integer index.
# Aggregate basins are kept as their string representation.
BASIN_MAPPING = {
    "H-Hp": 1,
    "F-G": 2,
    "E-Ep": 3,
    "D-Dp": 4,
    "Cp-D": 5,
    "B-C": 6,
    "A-Ap": 7,
    "Jpp-K": 8,
    "G-H": 9,
    "Dp-E": 10,
    "Ap-B": 11,
    "C-Cp": 12,
    "K-A": 13,
    "J-Jpp": 14,
    "Ipp-J": 15,
    "I-Ipp": 16,
    "Hp-I": 17,
    "Ep-F": 18,
    # Aggregates
    "AIS": "AIS",
    "WAIS": "WAIS",
    "EAIS": "EAIS",
    "APIS": "APIS",
}


def process_dh(args):
    """Process height change (dh) timeseries"""
    if not (os.path.exists(args.dh) and os.path.exists(args.unc)):
        print("Skipping DH processing (one or both files not found).")
        return

    print(f"Reading DH CSV: {args.dh}")
    df_dh = pd.read_csv(args.dh)

    print(f"Reading DH Uncertainty CSV: {args.unc}")
    df_unc = pd.read_csv(args.unc)

    if "time" not in df_dh.columns or "time" not in df_unc.columns:
        print("Error: 'time' column missing from one or both DH CSV files.")
        return

    times = df_dh["time"].tolist()
    generated_dh_files = 0

    for basin_str, basin_idx in BASIN_MAPPING.items():
        dh_col = f"{basin_str}_height_change"
        unc_col = f"{basin_str}_dh_uncertainty"

        if dh_col in df_dh.columns and unc_col in df_unc.columns:
            dh_data = df_dh[dh_col].tolist()
            unc_data = df_unc[unc_col].tolist()

            records = []
            for i, time_val in enumerate(times):
                dh_val = dh_data[i]
                unc_val = unc_data[i]
                if pd.isna(dh_val):
                    dh_val = None
                if pd.isna(unc_val):
                    unc_val = None

                records.append({"time": time_val, "dh": dh_val, "uncertainty": unc_val})

            out_file = os.path.join(args.outdir, f"basin_{basin_idx}.json")
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(records, f, indent=2)
            print(f"Generated {out_file} (from {basin_str})")
            generated_dh_files += 1
        else:
            print(
                f"Warning: Missing data columns for DH basin"
                f" '{basin_str}'. Looked for '{dh_col}' and '{unc_col}'."
            )
    print(f"Successfully generated {generated_dh_files} DH JSON timeseries files.")


def process_dm(args):
    """Process mass change (dm) timeseries"""
    if not (os.path.exists(args.dm) and os.path.exists(args.dm_unc)):
        print("Skipping DM processing (one or both files not found).")
        return

    print(f"\nReading DM CSV: {args.dm}")
    df_dm = pd.read_csv(args.dm)

    print(f"Reading DM Uncertainty CSV: {args.dm_unc}")
    df_dm_unc = pd.read_csv(args.dm_unc)

    if "time" not in df_dm.columns or "time" not in df_dm_unc.columns:
        print("Error: 'time' column missing from one or both DM CSV files.")
        return

    times = df_dm["time"].tolist()
    generated_dm_files = 0

    for basin_str, basin_idx in BASIN_MAPPING.items():
        dm_col = f"{basin_str}_mass_change_Gt"
        unc_col = f"{basin_str}_uncertainty_Gt"

        if dm_col in df_dm.columns and unc_col in df_dm_unc.columns:
            dm_data = df_dm[dm_col].tolist()
            unc_data = df_dm_unc[unc_col].tolist()

            records = []
            for i, time_val in enumerate(times):
                dm_val = dm_data[i]
                unc_val = unc_data[i]
                if pd.isna(dm_val):
                    dm_val = None
                if pd.isna(unc_val):
                    unc_val = None

                records.append({"time": time_val, "dm": dm_val, "uncertainty": unc_val})

            out_file = os.path.join(args.outdir, f"basin_dm_{basin_idx}.json")
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(records, f, indent=2)
            print(f"Generated {out_file} (from {basin_str})")
            generated_dm_files += 1
        else:
            print(
                f"Warning: Missing data columns for DM basin '{basin_str}'."
                f" Looked for '{dm_col}' and '{unc_col}'."
            )
    print(f"Successfully generated {generated_dm_files} DM JSON timeseries files.")


def main():
    """Main entry point for processing timeseries CSVs."""
    parser = argparse.ArgumentParser(
        description="Process and split timeseries CSVs for the land ice portal."
    )
    parser.add_argument(
        "--dh",
        type=str,
        default=(
            "/cpnet/altimetry/landice/cpom/product_files/multi_mission_dh/"
            "basin_height_change_1995.4_2026.1.csv"
        ),
        help="Path to the height change CSV file.",
    )
    parser.add_argument(
        "--unc",
        type=str,
        default=(
            "/cpnet/altimetry/landice/cpom/product_files/multi_mission_dh/"
            "basin_height_change_uncertainty_1995.4_2026.1.csv"
        ),
        help="Path to the uncertainty CSV file.",
    )
    parser.add_argument(
        "--dm",
        type=str,
        default=(
            "/cpnet/altimetry/landice/cpom/product_files/mass_change/"
            "basin_mass_change_1995.4_2026.1.csv"
        ),
        help="Path to the mass change CSV file.",
    )
    parser.add_argument(
        "--dm_unc",
        type=str,
        default=(
            "/cpnet/altimetry/landice/cpom/product_files/mass_change/"
            "basin_mass_change_uncertainty_1995.4_2026.1.csv"
        ),
        help="Path to the mass change uncertainty CSV file.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="src/cpom/altimetry/projects/landice_portal/www/data/timeseries",
        help="Output directory for the JSON files.",
    )

    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    process_dh(args)
    process_dm(args)


if __name__ == "__main__":
    main()
