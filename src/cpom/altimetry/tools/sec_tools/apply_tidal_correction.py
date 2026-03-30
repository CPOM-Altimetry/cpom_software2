import os

import numpy as np
import polars as pl

from cpom.masks.masks import Mask

INPUT_DIR = "/home/willisc3/luna/CPOM/willisc3/SEC/petermann_paper/greenland_is2_5km/gridded_altimetry_with_correction_cols"
OUTPUT_DIR = "/home/willisc3/luna/CPOM/willisc3/SEC/petermann_paper/greenland_is2_5km/gridded_altimetry_dac_corrected"
YEARS = [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]
DAC_FILL = 3.4028235e38
PARTITION_COLS = ["year", "x_part", "y_part"]
THIS_MASK = Mask("greenland_bedmachine_v3_grid_mask", basin_numbers=[3])


def apply_dac_correction(df):
    return df.filter(  # Drop points that don't have valid correction
        # pl.col("tide_ocean").is_not_null()
        pl.col("dac").is_not_null()
        # & ~pl.col("tide_ocean").is_nan()
        & ~pl.col("dac").is_nan()
    ).with_columns(  # Apply correction
        (pl.col("elevation") - pl.col("dac")).alias("elevation")  # - pl.col("tide_ocean")
    )


def process_partition(year, x_part):
    gridded_x_part = pl.scan_parquet(
        f"{INPUT_DIR}/year={year}/x_part={x_part}/**/*.parquet"
    ).with_columns(
        [
            # pl.col("tide_ocean").replace(tide_ocean_fill, np.nan),
            pl.col("dac").replace(DAC_FILL, np.nan),
        ]
    )

    floating_x_part = THIS_MASK.points_inside_polars(gridded_x_part)
    corrected_floating_x_part = apply_dac_correction(floating_x_part)
    # add a column to indicate these points have been corrected.
    corrected_floating_x_part = corrected_floating_x_part.with_columns(
        pl.lit(True).alias("corrected")
    )

    # keep non-floating rows
    non_floating = gridded_x_part.join(
        corrected_floating_x_part.select(["x", "y"]).unique(),
        on=["x", "y"],
        how="anti",
    ).with_columns(pl.lit(False).alias("corrected"))

    # combine non-floating + corrected floating
    chunk = pl.concat(
        [non_floating, corrected_floating_x_part],
    ).collect()

    for partition_key, group in chunk.partition_by(PARTITION_COLS, as_dict=True).items():
        subdir = os.path.join(
            OUTPUT_DIR,
            *[
                f"{col}={value}"
                for col, value in zip(PARTITION_COLS, partition_key, strict=True)
            ],
        )
        if not os.path.isdir(subdir):
            os.makedirs(subdir, exist_ok=True)

        group.write_parquet(os.path.join(subdir, "data.parquet"), compression="zstd")

    del chunk
    # pq.write_table(
    #     pa.Table.from_pandas(group.to_pandas()),
    #     outfile,
    #     compression="zstd",
    # )


def tidal_correction():
    for year in YEARS:
        print(f"Processing year {year}...")
        x_parts = (
            pl.scan_parquet(f"{INPUT_DIR}/year={year}/**/*.parquet")
            .select("x_part")
            .unique()
            .collect()["x_part"]
            .to_list()
        )
        for x_part in x_parts:
            print(f"  x_part={x_part}")
            process_partition(year, x_part)

        print(f"Processed year {year}")


# def tidal_correction():
#     """
#     Apply tidal correction
#     """
#     for year in years:
#         print(f"Processing year {year}...")
#         # Load gridded altimetry data
#         gridded_altimetry = pl.scan_parquet(
#             f"{INPUT_DIR}/year={year}/**/*.parquet"
#         ).with_columns(
#             [
#                 # pl.col("tide_ocean").replace(tide_ocean_fill, np.nan),
#                 pl.col("dac").replace(dac_fill, np.nan),
#             ]
#         )

#         # Mask to the floating ice
#         this_mask = Mask("greenland_bedmachine_v3_grid_mask", basin_numbers=[3])
#         gridded_floating = this_mask.points_inside_polars(gridded_altimetry)

#         gridded_floating = (
#             gridded_floating
#             # Drop points that don't have valid correction
#             .filter(
#                 # pl.col("tide_ocean").is_not_null()
#                 pl.col("dac").is_not_null()
#                 # & ~pl.col("tide_ocean").is_nan()
#                 & ~pl.col("dac").is_nan()
#             )
#             # Apply correction
#             .with_columns(
#                 (pl.col("elevation") - pl.col("dac")).alias("elevation")  # - pl.col("tide_ocean")
#             )
#         )

#         # add a column to indicate these points have been corrected.
#         gridded_floating = gridded_floating.with_columns(pl.lit(True).alias("corrected"))

#         # keep non-floating rows
#         non_floating = gridded_altimetry.join(
#             gridded_floating.select(["x", "y"]).unique(),
#             on=["x", "y"],
#             how="anti",
#         ).with_columns(pl.lit(False).alias("corrected"))

#         # combine non-floating + corrected floating
#         gridded_altimetry_corrected = pl.concat(
#             [non_floating, gridded_floating],
#         )

#         partition_columns = ["year", "x_part", "y_part"]
#         partitions = gridded_altimetry_corrected.collect().partition_by(
#             partition_columns, as_dict=True
#         )
#         print(f"Writing partitions to disk for year {year}...")
#         for key, group in partitions.items():
#             subdir = os.path.join(
#                 "/home/willisc3/luna/CPOM/willisc3/SEC/greenland_is2_5km_paper/gridded_altimetry_floating_dac_corrected/",
#                 *[f"{group}={str(i)}" for group, i in zip(partition_columns, key)],
#             )
#             if not os.path.isdir(subdir):
#                 os.makedirs(subdir, exist_ok=True)

#             outfile = os.path.join(subdir, "data.parquet")
#             pq.write_table(
#                 pa.Table.from_pandas(group.to_pandas()),
#                 outfile,
#                 compression="zstd",
#             )


if __name__ == "__main__":

    tidal_correction()
