#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compact_year_partitioned_data.py

Purpose:
    Take a Parquet dataset that is currently partitioned by:
      year=YYYY/[month=MM/] x_part=NNN/ y_part=MMM/ *.parquet
    and produce a new "compacted" dataset partitioned only by:
      x_part=NNN/ y_part=MMM/ data.parquet
    so that each (x_part, y_part) chunk contains all years in a single file.

Usage example:
    python compact_year_partitioned_data.py \
        --input_dir /path/to/year_partitioned \
        --output_dir /path/to/compacted_output
"""

import argparse
import logging
import os
import shutil
import sys
import time

import duckdb

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s [%(asctime)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

log = logging.getLogger(__name__)


def compact_partitions(input_dir: str, output_dir: str):
    """
    Perform compaction from a layout partitioned by year=..., x_part=..., y_part=...
    (optionally month=...) into a new layout partitioned only by x_part=..., y_part=....

    1) Copy 'grid_meta.json' from input_dir to output_dir.
    2) Find all (x_part, y_part) subdirectories in input_dir.
    3) For each partition:
       - Glob ALL matching parquet files under year=*/(month=*/)?/x_part=?/y_part=?
       - Load them in memory with DuckDB
       - Write a single 'data.parquet' in x_part=N/y_part=M/ in output_dir
       - Record this in the partition_index.json
    """
    start_time = time.time()

    # 1) Copy meta-data JSON if present
    meta_src = os.path.join(input_dir, "grid_meta.json")
    meta_dest = os.path.join(output_dir, "grid_meta.json")
    if os.path.exists(meta_src):
        os.makedirs(output_dir, exist_ok=True)
        shutil.copy2(meta_src, meta_dest)
        log.info("Copied meta-data JSON from %s to %s", meta_src, meta_dest)
    else:
        log.warning("No grid_meta.json found in %s (continuing anyway)", input_dir)

    # 2) Discover all x_part,y_part combos
    con = duckdb.connect()

    df = con.execute(f"SELECT * FROM parquet_scan('{ f"{input_dir}/year=*/**/*.parquet"}')").pl()
    con.register("compact_grid_tbl", df)

    combos = con.execute(
        """
        SELECT DISTINCT x_part, y_part FROM compact_grid_tbl
    """
    ).fetchall()

    # Iterate over each partition
    for xp, yp in sorted(combos):
        # Create directory for partition
        chunk_out_dir = os.path.join(output_dir, f"x_part={xp}", f"y_part={yp}")
        os.makedirs(chunk_out_dir, exist_ok=True)

        # Now df has the entire set of rows for that chunk, across all years
        # Write out as a single data.parquet
        con.execute(
            f"""
            COPY (
                SELECT * FROM compact_grid_tbl
                WHERE x_part = {xp} AND y_part = {yp}
            ) TO '{chunk_out_dir}/data.parquet'
            (FORMAT parquet, OVERWRITE, COMPRESSION gzip)
        """
        )

    con.close()
    log.info("Compaction complete. Elapsed time: %.1f seconds", time.time() - start_time)


def main(args):
    """Tool entry point"""
    parser = argparse.ArgumentParser(
        description=(
            "Compact a year-partitioned altimetry dataset into x_part=y_part-only partition, "
            "writing one data.parquet per chunk and a partition_index.json with all file paths."
        )
    )
    parser.add_argument(
        "--input_dir",
        "-i",
        required=True,
        help="Path to the existing Parquet dataset partitioned by year=... etc.",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        required=True,
        help="Where to write the new compacted dataset (x_part=..., y_part=...).",
    )
    args = parser.parse_args(args)

    input_dir = args.input_dir
    output_dir = args.output_dir

    if not os.path.exists(input_dir):
        log.error("Input directory does not exist: %s", input_dir)
        sys.exit(1)

    log.info("Starting compaction from '%s' to '%s'", input_dir, output_dir)
    compact_partitions(input_dir, output_dir)
    log.info("Done.")


if __name__ == "__main__":
    main(sys.argv[1:])
