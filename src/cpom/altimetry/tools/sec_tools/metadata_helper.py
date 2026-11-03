"""
cpom.altimetry.tools.sec_tools.metadata_helper

Purpose:
    Utilities for loading, resolving, and writing SEC algorithm run metadata.

    Metadata is stored as a JSON entry store keyed by algorithm name and UTC timestamp:
    {
        "algo1_20260414T1200": {"param1": val1, "param2": val2, ...},
        "algo2_20260414T1300": {"param1": val1, ...},
    }
"""

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def elapsed(t0: float) -> str:
    """
    Format elapsed time since t0 as HH:MM:SS.
    Args:
        t0 (float): Start time in seconds.
    Returns:
        str: Elapsed time formatted as "HH:MM:SS".
    """
    h, rem = divmod(int(time.time() - t0), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02}:{m:02}:{s:02}"


# --------------------------------
# Track metadata sourced fields #
# --------------------------------
def _add_sourced_field(params: argparse.Namespace, field: str) -> None:
    """Mark a field as sourced from upstream metadata."""
    sourced: set[str] = set(getattr(params, "_grid_sourced_fields", set()))
    setattr(params, "_grid_sourced_fields", sourced | {field})


# ------------------
# Path resolution
# ------------------
def get_metadata_path(
    params: argparse.Namespace,
    basin_name: str | None,
    logger: logging.Logger | None = None,
) -> str | None:
    """
    Resolve the metadata file path from params.

    Args:
        params (argparse.Namespace): Command line parameters
            (includes: in_dir, in_step, basin_structure).
        basin_name (str | None):  Basin subdirectory name, inserted into the path
            when params.basin_structure is True.
        logger (logging.Logger | None): Logger Object

    Returns:
        str | None: Resolved metadata path, or None.
            For root data : <in_dir>/<in_step>_meta.json
            For basin data : <in_dir>/<basin_name>/<in_step>_meta.json
    """

    in_dir = getattr(params, "in_dir", None)
    in_step = getattr(params, "in_step", None)

    if in_dir is None or in_step is None:
        return None

    base_path = Path(in_dir)
    filename = f"{in_step}_meta.json"

    # Attempt to derive metadata path from in_dir and in_step
    if getattr(params, "basin_structure", False) and basin_name:
        if logger:
            logger.info(
                "Using basin-structured metadata path '%s'", base_path / basin_name / filename
            )
        return str(base_path / basin_name / filename)

    # Root-level mode: metadata in <in_dir>/<in_step>_meta.json
    if logger:
        logger.info("Using root-level metadata path '%s'", base_path / filename)
    return str(base_path / filename)


def get_algo_name(script_path: str | Path) -> str:
    """Get algorithm name from a script file path.
    Args:
        script_path (str | Path): Path to the script file (e.g., '/path/to/clip_to_basins.py').
    Returns:
        str: Filename stem without extension (e.g., 'clip_to_basins').
    """
    return Path(script_path).stem


# ---------------
# Entry sorting
# ---------------
def _entry_sort_key(entry_key: str) -> tuple[int, float | str]:
    """
    Generate sort key for metadata entries.Timestamped entries (algo_YYYYMMDDTHHMM)
    sort above untimestamped ones,with most recent timestamps first.

    Returns:
        tuple[int, float | str]: (1, -unix_timestamp) for timestamped entries,
            (0, entry_key) otherwise.
    """
    try:
        _, timestamp = entry_key.rsplit("_", 1)
        parsed = datetime.strptime(timestamp, "%Y%m%dT%H%M")
        return (1, -parsed.timestamp())
    except ValueError:
        return (0, entry_key)


def _get_latest_entry(entry_store: dict[str, Any], algo_name: str) -> dict[str, Any]:
    """
    Get the latest metadata entry for a given algorithm.
    Args:
        entry_store (dict[str, Any]): Full metadata entry store.
        algo_name (str): Algorithm prefix to filter by.
    Returns:
        dict[str, Any]: Parameter dict from the latest matching entry.
    """
    matching_keys = [k for k in entry_store if k.startswith(f"{algo_name}_")]
    if not matching_keys:
        raise KeyError(f"No metadata found for algorithm '{algo_name}'")

    latest_key = max(matching_keys, key=_entry_sort_key)
    return entry_store[latest_key]


# -----------------------------------------------
# Loading metadata
# -----------------------------------------------
def get_metadata(
    params=None,
    basin_name: str | None = None,
    algo_name: str | None = None,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """
    Load metadata from a JSON entry store, optionally filtering to a single algorithm.

    Args:
        params: Algorithm command line parameters.
        basin_name (str | None): Basin name.
        algo_name (str | None): If provided, returns only the latest entry for this
            algorithm. If None, the full entry store is returned.
        logger (logging.Logger | None): Logger Object

    Returns:
        dict[str, Any]: Full entry store if algo_name is None, otherwise the parameter
            dict for the latest matching entry.
    """

    resolved_path = get_metadata_path(
        params=params,
        basin_name=basin_name,
        logger=logger,
    )
    if logger:
        logger.info("Resolved metadata path: %s", resolved_path)
    if resolved_path is None:
        raise ValueError(
            f"Could not resolve metadata path: {resolved_path}. Provide metadata_path or ensure "
            "in_dir and in_step are set (and basin_name if using basin_structure)."
        )

    # Load JSON file
    path = Path(resolved_path)
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        if logger:
            logger.error("Metadata file not found: %s", path)
        raise

    # Validate JSON is a dict
    if not isinstance(data, dict):
        raise TypeError(f"Metadata file must contain a JSON object, got {type(data)}")

    # Validate entry-store format: all values must be dicts
    if not data or not all(isinstance(v, dict) for v in data.values()):
        raise ValueError("Metadata must be an entry store: {algo_timestamp: {params}}")

    # If algo_name specified, return only the latest entry for that algorithm
    if algo_name is not None:
        return _get_latest_entry(data, algo_name)

    return data


# -----------------------------------------------
# Resolving parameters
# -----------------------------------------------
def get_metadata_params(
    params,
    fields: list[str] | str = "all",
    algo_name: str = "grid_for_elev_change",
    basin_name: str | None = None,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """
    Resolve parameter values from command-line arguments, falling back to metadata.

    Params attributes take precedence over metadata values. Fields sourced from
    metadata are tracked via _add_sourced_field so they can be excluded from output metadata.

    Args:
        params: Algorithm command line parameters.
        fields (list[str] | str): Field names to retrieve or 'all' to retrieve all fields.
        algo_name (str, optional): Algorithm name to lookup. Defaults to "grid_for_elev_change".
        basin_name (str | None): Basin name.
        logger (logging.Logger | None): Logger Object.

    Returns:
        dict[str, Any]: Resolved field values keyed by field name.
    """
    try:
        grid_meta = get_metadata(
            params=params,
            basin_name=basin_name,
            algo_name=algo_name,
            logger=logger,
        )
    except (ValueError, KeyError, FileNotFoundError):
        # No metadata available
        grid_meta = {}
    resolved = {}

    if fields == "all":
        return grid_meta

    for field in fields:
        # check parameter first
        param_value = getattr(params, field, None)
        if param_value is not None:
            resolved[field] = param_value
            continue

        # Fall back to metadata
        meta_value = grid_meta.get(field)
        if meta_value is None:
            raise ValueError(f"Missing required parameter: {field}")

        resolved[field] = meta_value
        _add_sourced_field(params, field)

    return resolved


# -----------------------------------------------
# Writing metadata
# -----------------------------------------------
def write_metadata(
    params,
    algo_name: str,
    out_meta_path: str | Path,
    metadata: dict[str, Any],
    basin_name: str | None = None,
    logger: logging.Logger | None = None,
) -> None:
    """
    Write a timestamped metadata entry to <out_meta_path>/<algo_name>_meta.json.
    Merges with any existing upstream metadata. Fields that were sourced from
    upstream metadata (tracked via _add_sourced_field) are excluded from the new
    entry to prevent re-propagating inherited values.

    Args:
        params: Algorithm command line parameters.
        algo_name (str): Algorithm name for entry key and filename.
        out_meta_path (str | Path): Output directory for metadata JSON file.
        metadata (dict[str, Any]): Metadata content to write.
        basin_name (str | None): Basin name.
        logger (logging.Logger | None): Logger Object.
    """

    # Build output path and timestamped entry key
    out_path = Path(out_meta_path) / f"{algo_name}_meta.json"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M")
    entry_key = f"{algo_name}_{timestamp}"

    # Exclude fields sourced from upstream
    sourced_fields: set[str] = set(getattr(params, "_grid_sourced_fields", set()))
    filtered_metadata = {
        k: v for k, v in metadata.items() if k not in sourced_fields and k != "_grid_sourced_fields"
    }

    # Start with new entry
    to_write: dict[str, Any] = {entry_key: filtered_metadata}
    try:
        existing = get_metadata(
            params=params,
            basin_name=basin_name,
            algo_name=None,
            logger=logger,
        )
        if logger:
            logger.info(
                "Existing metadata found with %d entries, merging new entry.", len(existing)
            )
        to_write = {**existing, **to_write}
        if logger:
            logger.info("Merged metadata now has %d entries.", len(to_write))
    except (ValueError, KeyError, FileNotFoundError):
        # No existing metadata, will write new file
        if logger:
            logger.info("No existing metadata found, writing new file at %s", out_path)

    # Write merged entry store
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(to_write, f, indent=2)
        f.write("\n")
