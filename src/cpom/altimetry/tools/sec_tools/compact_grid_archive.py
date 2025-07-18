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

    Additionally, writes a 'partition_index.json' at the top level of the
    compacted dataset. This file lists all (x_part, y_part) chunks and their
    single Parquet file path, enabling direct reads without globbing.

Usage example:
    python compact_year_partitioned_data.py \
        --input_dir /path/to/year_partitioned \
        --output_dir /path/to/compacted_output

Example partition_index.json structure:

    {
      "entries": [
        {
          "x_part": 12,
          "y_part": 7,
          "file_path": "x_part=12/y_part=7/data.parquet"
        },
        ...
      ]
    }

where file_path is relative to the top-level compacted directory.

"""

import argparse
import glob
import json
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


def update_partition_index_file(
    index_path: str,
    new_entries: list[dict],
) -> None:
    """
    Update or create the 'partition_index.json' file to store the final layout of
    (x_part, y_part) => file_path.

    new_entries is a list of dicts, e.g.:
      [
        {
          "x_part": 12,
          "y_part": 7,
          "file_path": "x_part=12/y_part=7/data.parquet"
        },
        ...
      ]

    We'll append these to an existing "entries" list in index_path, deduplicate,
    and write back out.
    """
    if os.path.isfile(index_path):
        with open(index_path, "r", encoding="utf-8") as f_idx:
            index_data = json.load(f_idx)
    else:
        index_data = {"entries": []}

    existing = set()
    for e in index_data["entries"]:
        # We'll define a key for dedup: (x_part, y_part, file_path)
        xp = e.get("x_part", None)
        yp = e.get("y_part", None)
        fp = e.get("file_path", None)
        existing.add((xp, yp, fp))

    additions = 0
    for e in new_entries:
        xp = e["x_part"]
        yp = e["y_part"]
        fp = e["file_path"]
        key = (xp, yp, fp)
        if key not in existing:
            index_data["entries"].append(e)
            additions += 1
            existing.add(key)

    if additions > 0:
        with open(index_path, "w", encoding="utf-8") as f_idx:
            json.dump(index_data, f_idx, indent=2)
        log.info("Appended %d new partition-file entries into partition_index.json", additions)
    else:
        log.info("No new entries to add to partition_index.json (all duplicates).")


def compact_partitions(input_dir: str, output_dir: str):
    """
    Perform compaction from a layout partitioned by year=..., x_part=..., y_part=...
    (optionally month=...) into a new layout partitioned only by x_part=..., y_part=....

    1) Copy 'grid_meta.json' (if present) from input_dir to output_dir.
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
    parquet_glob = f"{input_dir}/year=*/**/*.parquet"
    con = duckdb.connect()

    # Register parquet files as a virtual table
    df = con.execute(f"SELECT * FROM parquet_scan('{parquet_glob}')").df()
    con.register("compact_grid_tbl", df)

    combos = con.execute(
        f"""
        SELECT DISTINCT x_part, y_part FROM compact_grid_tbl
    """
    ).fetchall()

    index_entries = []
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

        # Add an entry for partition_index.json
        # file_path is stored relative to output_dir
        relative_path = os.path.relpath(os.path.join(chunk_out_dir, "data.parquet"), output_dir)
        index_entries.append(
            {
                "x_part": xp,
                "y_part": yp,
                "file_path": relative_path,  # e.g. "x_part=12/y_part=7/data.parquet"
            }
        )
    con.close()

    # Update the partition_index.json with all the new entries
    index_path = os.path.join(output_dir, "partition_index.json")
    update_partition_index_file(index_path, index_entries)

    elapsed = time.time() - start_time
    log.info("Compaction complete. Elapsed time: %.1f seconds", elapsed)


def main():
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
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir

    if not os.path.exists(input_dir):
        log.error("Input directory does not exist: %s", input_dir)
        sys.exit(1)

    log.info("Starting compaction from '%s' to '%s'", input_dir, output_dir)
    compact_partitions(input_dir, output_dir)
    log.info("Done.")


if __name__ == "__main__":
    main()
    # check whether compacted archive should be deleted first if exists
