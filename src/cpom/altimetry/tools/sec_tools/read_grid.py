"""cpom.altimetry.tools.sec_tools.read_grid.py

# Purpose

Example of reading a single grid cell from Parquet grid directory

"""

import duckdb
import pandas as pd

from cpom.gridding.gridareas import GridArea


def query_parquet_for_xy(
    parquet_path: str,
    x: float,
    y: float,
    min_x: float,
    min_y: float,
    bin_size: float,
    partition_factor: int = 20,
) -> pd.DataFrame:
    """
    Query a partitioned Parquet dataset for all rows belonging to
    the single grid cell that contains the point (x, y).

    Assumes:
      - x_bin = floor( (x - min_x) / bin_size )
      - y_bin = floor( (y - min_y) / bin_size )
      - x_part = x_bin // partition_factor
      - y_part = y_bin // partition_factor
      - The Parquet dataset is partitioned by (x_part, y_part).
    """
    # 1) Convert (x, y) to bin indices
    x_bin = int((x - min_x) // bin_size)
    y_bin = int((y - min_y) // bin_size)

    # 2) Compute coarse partition indices
    x_part = x_bin // partition_factor
    y_part = y_bin // partition_factor

    # 3) Build a DuckDB query that:
    #    - Restricts to the partition (x_part, y_part)
    #    - Further filters by the exact x_bin, y_bin
    #    - Use a glob so DuckDB finds the real .parquet files in subdirs.
    parquet_glob = f"{parquet_path}/**/*.parquet"
    query = f"""
        SELECT *
        FROM parquet_scan('{parquet_glob}')
        WHERE x_part = {x_part}
          AND y_part = {y_part}
          AND x_bin = {x_bin}
          AND y_bin = {y_bin}
    """

    # 4) Execute query
    con = duckdb.connect()
    df = con.execute(query).df()  # returns a pandas DataFrame
    con.close()

    return df


# Example usage
if __name__ == "__main__":
    grid = GridArea("greenland", 5000)

    df_cell = query_parquet_for_xy(
        parquet_path=(
            "/Users/alanmuir/software/cpom_software2/"
            "src/cpom/altimetry/tools/sec_tools/test.parquet"
        ),
        x=-608028.57826279,
        y=-1285783.27183183,
        min_x=grid.minxm,
        min_y=grid.minym,
        bin_size=grid.binsize,
        partition_factor=20,
    )

    print(df_cell)
