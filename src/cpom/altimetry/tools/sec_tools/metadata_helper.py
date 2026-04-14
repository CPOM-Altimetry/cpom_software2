"""
cpom.altimetry.tools.sec_tools.metadata_helper
Helper functions for metadata handling in SEC tools.
Tracks parameter provenance across processing algorithms using timestamped metadata entries in json
format.

Provides utilities to:
- Load metadata from JSON files, enforcing an entry-store format.
- Extract the latest metadata entry, or the latest entry for a specific algorithm.
- Merge new metadata entries with existing metadata stores.
- Resolve grid parameters from metadata with fallback to CLI arguments.
- Write updated metadata back to JSON files with proper entry naming.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# -----------------------------
# Generic helper utilities  #
# ----------------------------


def load_metadata(metadata_path: str | Path) -> dict[str, Any]:
    """Load metadata into a dictionary from a JSON filepath.
    Enforces an entry-store format.
    Args:
        metadata_path (str | Path): The file path to the metadata JSON file.
    Returns:
        dict[str, Any]: The metadata loaded from the JSON file as a dictionary.
    """

    with open(metadata_path, "r", encoding="utf-8") as file_obj:
        data = json.load(file_obj)

    if not isinstance(data, dict):
        raise TypeError(f"Metadata file must contain a JSON object, got {type(data)}")

    if not data or not all(isinstance(v, dict) for v in data.values()):
        raise ValueError("Metadata must be an entry store: {algo_timestamp: {params}}")

    return data


def get_algo_name(script_path: str | Path) -> str:
    """Return algorithm name from a script file path."""
    return Path(script_path).stem


def get_basin_in_meta(
    base_in_meta: str | None,
    basin_name: str,
    basin_in_dir: Path,
    logger: logging.Logger,
) -> str | None:
    """Resolve metadata path for a basin, preferring basin-specific files."""
    candidates: list[Path] = []

    if base_in_meta:
        in_meta_path = Path(base_in_meta)
        if in_meta_path.is_dir():
            candidates.extend(sorted((in_meta_path / basin_name).glob("*_meta.json")))
        elif in_meta_path.is_file():
            basin_sibling = in_meta_path.parent / basin_name / in_meta_path.name
            if basin_sibling.exists():
                candidates.append(basin_sibling)
            candidates.append(in_meta_path)

    candidates.extend(sorted(basin_in_dir.glob("*_meta.json")))

    for candidate in candidates:
        if candidate.exists():
            logger.info("Using basin metadata: %s", candidate)
            return str(candidate)
    return None


def _entry_sort_key(entry_key: str) -> tuple[int, str]:
    """Sort metadata entries by timestamp if available.
    Args:
        entry_key (str): Metadata entry key, in the format "algo_YYYYMMDDTHHMM".
    Returns:
        tuple[int, str]: A tuple used for sorting metadata entries.
    """
    try:
        _, timestamp = entry_key.rsplit("_", 1)
        parsed = datetime.strptime(timestamp, "%Y%m%dT%H%M")
        return (1, parsed.isoformat())
    except ValueError:
        return (0, entry_key)


def get_metadata_for_algo(metadata_path: str | Path, algo_name: str) -> dict[str, Any]:
    """Filters file metadata to a specific algorithm.
    Returns the latest entry for that algorithm if multiple are present."""

    incoming = load_metadata(metadata_path)
    matching_keys = [key for key in incoming if key.startswith(f"{algo_name}_")]

    if not matching_keys:
        raise KeyError(f"No metadata found for algorithm '{algo_name}'")

    return incoming[max(matching_keys, key=_entry_sort_key)]


def merge_metadata(
    in_step_meta_path: str | Path,
    entry_key: str,
    new_meta: dict[str, Any],
) -> dict[str, Any]:
    """Merge prior metadata store with the new metadata entry."""

    incoming = load_metadata(in_step_meta_path)
    merged = dict(incoming)
    merged[entry_key] = new_meta
    return merged


def get_grid_params(
    params,
    fields: list[str],
    algo_name: str = "grid_for_elev_change",
) -> dict[str, Any]:
    """Retrieve grid parameters from metadata or fallback to params.

    Args:
        params (_type_): _description_
        fields (list[str]): _description_
        algo_name (str, optional): _description_. Defaults to "grid_for_elev_change".

    Raises:
        ValueError: _description_

    Returns:
        dict[str, Any]: _description_
    """
    grid_meta = get_metadata_for_algo(params.in_meta, algo_name)
    resolved: dict[str, Any] = {}

    for field in fields:
        value = grid_meta.get(field, getattr(params, field, None))
        if value is None:
            raise ValueError(f"Missing required parameter: {field}")
        if field in grid_meta:
            setattr(params, field, None)
            # Track which fields were sourced from upstream metadata so write_metadata
            # can strip them automatically.
            existing: set[str] = getattr(params, "_grid_sourced_fields", set())
            setattr(params, "_grid_sourced_fields", existing | {field})
        resolved[field] = value

    return resolved


def write_metadata(
    params,
    algo_name: str,
    out_meta_path: str | Path,
    metadata: dict[str, Any],
) -> None:
    """Write metadata to a JSON file."""
    out_path = Path(out_meta_path) / f"{algo_name}_meta.json"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M")
    entry_key = f"{algo_name}_{timestamp}"

    grid_sourced = getattr(params, "_grid_sourced_fields", set())
    exclude = grid_sourced | {"_grid_sourced_fields"}
    metadata = {k: v for k, v in metadata.items() if k not in exclude}

    to_write: dict[str, Any] = {entry_key: metadata}

    if getattr(params, "in_meta", None):
        in_path = Path(params.in_meta)
        if in_path.exists():
            to_write = merge_metadata(in_path, entry_key, metadata)

    with open(out_path, "w", encoding="utf-8") as f_meta:
        json.dump(to_write, f_meta, indent=2)
        f_meta.write("\n")
