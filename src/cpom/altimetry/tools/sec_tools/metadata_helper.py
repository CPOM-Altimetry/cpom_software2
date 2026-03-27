"""Helpers for SEC metadata persistence and provenance stamping."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def load_metadata(metadata_path: str | Path) -> dict[str, Any]:
    """Load metadata from a JSON file."""

    with open(metadata_path, "r", encoding="utf-8") as file_obj:
        data = json.load(file_obj)

    if not isinstance(data, dict):
        raise TypeError(f"Metadata file must contain a JSON object, got {type(data)}")

    return data


def _is_entry_store(metadata: dict[str, Any]) -> bool:
    """Return True when metadata is already a map of entry_key -> metadata dict."""

    return bool(metadata) and all(isinstance(value, dict) for value in metadata.values())


def _infer_legacy_entry_key(
    metadata_path: str | Path,
    metadata: dict[str, Any],
) -> str:
    """Infer an algo_timestamp-style key for legacy flat metadata."""

    path = Path(metadata_path)
    algo_name = str(metadata.get("algo") or path.stem.replace("_meta", "") or "unknown_algo")
    timestamp = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).strftime(
        "%Y%m%dT%H%M"
    )
    return f"{algo_name}_{timestamp}"


def _entry_sort_key(entry_key: str) -> tuple[int, str]:
    """Sort metadata entry keys by embedded timestamp when available."""

    try:
        _, timestamp = entry_key.rsplit("_", 1)
        parsed = datetime.strptime(timestamp, "%Y%m%dT%H%M")
        return (1, parsed.isoformat())
    except ValueError:
        return (0, entry_key)


def merge_metadata(
    in_step_meta_path: str | Path,
    entry_key: str,
    new_meta: dict[str, Any],
) -> dict[str, Any]:
    """Merge prior metadata store with the new metadata entry."""

    incoming = load_metadata(in_step_meta_path)

    if _is_entry_store(incoming):
        merged = dict(incoming)
    elif incoming:
        merged = {_infer_legacy_entry_key(in_step_meta_path, incoming): incoming}
    else:
        merged = {}

    merged[entry_key] = new_meta
    return merged


def extract_latest_metadata(metadata_path: str | Path) -> dict[str, Any]:
    """Return the latest inner metadata entry from wrapped or legacy metadata."""

    incoming = load_metadata(metadata_path)

    if _is_entry_store(incoming):
        latest_key = max(incoming, key=_entry_sort_key)
        return incoming[latest_key]

    return incoming


def extract_all_metadata(metadata_path: str | Path) -> dict[str, Any]:
    """Return metadata merged across all entries from oldest to newest."""

    incoming = load_metadata(metadata_path)

    if not _is_entry_store(incoming):
        return incoming

    merged: dict[str, Any] = {}
    for entry_key in sorted(incoming, key=_entry_sort_key):
        entry = incoming.get(entry_key)
        if isinstance(entry, dict):
            merged.update(entry)

    return merged


def extract_metadata_for_algo(metadata_path: str | Path, algo_name: str) -> dict[str, Any]:
    """Return the latest metadata entry for a specific algorithm key prefix."""

    incoming = load_metadata(metadata_path)

    if _is_entry_store(incoming):
        prefix = f"{algo_name}_"
        matching_keys = [key for key in incoming if key.startswith(prefix)]
        if matching_keys:
            latest_key = max(matching_keys, key=_entry_sort_key)
            return incoming[latest_key]
        return extract_latest_metadata(metadata_path)

    if incoming.get("algo") == algo_name:
        return incoming

    return incoming


def write_metadata(
    params,
    out_meta_path: str | Path,
    metadata: dict[str, Any],
) -> None:
    """Write metadata to a JSON file."""
    out_path = Path(out_meta_path)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M")
    algo_name = getattr(params, "algo", None)
    if not algo_name:
        raise AttributeError("params.algo is required for metadata entry naming")
    entry_key = f"{algo_name}_{timestamp}"

    in_meta = getattr(params, "in_meta", None)

    to_write: dict[str, Any] = {entry_key: metadata}
    if in_meta:
        in_path = Path(in_meta)
        if in_path.exists():
            to_write = merge_metadata(in_path, entry_key, metadata)

    try:
        with open(out_path, "w", encoding="utf-8") as f_meta:
            json.dump(to_write, f_meta, indent=2)
            f_meta.write("\n")
    except OSError:
        pass


def _resolve_basin_in_meta(
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


def _resolve_grid_params(params, fields: list[str]) -> list[Any]:
    """Resolve fields from grid metadata first, then fall back to CLI args.

    Supports metadata aliases used across SEC stages:
    - grid_area <- grid_area | gridarea
    """

    grid_meta: dict[str, Any] = {}
    if getattr(params, "in_meta", None):
        grid_meta = extract_metadata_for_algo(params.in_meta, "grid_for_elev_change")

    resolved: list[Any] = []
    missing: list[str] = []

    for field in fields:
        value = None
        from_meta = False
        candidate = grid_meta.get(field)
        if candidate is not None:
            value = candidate
            from_meta = True

        if value is None:
            value = getattr(params, field, None)

        if value is None:
            missing.append(field)

        if from_meta:
            # Clear from params so it is excluded from output metadata
            setattr(params, field, None)

        resolved.append(value)

    if missing:
        raise ValueError(
            "Missing required parameter(s): "
            + ", ".join(missing)
            + ". Provide --in_meta with grid metadata or pass these on the command line."
        )

    return resolved
