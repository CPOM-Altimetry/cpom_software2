"""cpom.altimetry.tools.sec_tools.grid_for_elev_change_corrections.

Utility functions for applying mission-specific quality and elevation corrections
within the SEC gridding pipeline.

Corrections are either loaded dynamically from names/dotted paths specified in the
dataset YAML config, or a default correction function can be applied if passed
to params.correction_function within the gridding script's run control file.
"""

import importlib
import inspect
import typing
from typing import Callable, Union

import numpy as np
from pyproj import CRS, Transformer

# ----------------------------
# Helper methods
# ---------------------------


def _apply_mask_to_dict(variable_dict: dict, mask: np.ndarray) -> dict:
    """Apply a boolean mask to all non-null arrays in variable_dict."""
    return {k: v[mask] if v is not None else v for k, v in variable_dict.items()}


def _resolve_callable(name: Union[str, Callable]) -> Callable:
    """
    Resolve a string function name into a callable function.
        A) If a callable is passed directly, return it.
        B) Plain name lookup in the current global scope.
        C) Dotted path lookup for functions in other modules.
    """
    if callable(name):
        return name
    local = globals().get(name)
    if callable(local):
        return local
    if "." in name:
        module_name, attr_name = name.rsplit(".", 1)
        fn = getattr(importlib.import_module(module_name), attr_name, None)
        if callable(fn):
            return fn
    raise ValueError(f"Correction function '{name}' not found.")


def _call_with_accepted_kwargs(func: Callable, **kwargs) -> typing.Any:
    """Call func with only the kwargs its signature actually declares."""
    accepted = inspect.signature(func).parameters
    return func(**{k: v for k, v in kwargs.items() if k in accepted})


# ----------------------------
# Mission Specific Corrections
# ----------------------------


def _transform_is1_to_wgs84(variable_dict: dict) -> dict:
    """Convert ICESat-1 TOPEX lat/lon/height coordinates in-place to WGS84."""
    topex_crs = CRS.from_proj4("+proj=latlong +a=6378136.300 +rf=298.257 +no_defs")
    ecef_crs = CRS.from_proj4("+proj=geocent +a=6378136.300 +rf=298.257 +no_defs")
    topex_to_ecef = Transformer.from_crs(topex_crs, ecef_crs, always_xy=True)
    ecef_to_wgs = Transformer.from_crs(ecef_crs, CRS.from_epsg(4979), always_xy=True)

    x_ecef, y_ecef, h_ecef = topex_to_ecef.transform(  # type: ignore[misc]
        variable_dict["longitude"],
        variable_dict["latitude"],
        variable_dict["elevation"],
    )
    longitude, latitude, elevation = ecef_to_wgs.transform(
        x_ecef, y_ecef, h_ecef
    )  # type: ignore[misc]
    variable_dict["longitude"] = longitude
    variable_dict["latitude"] = latitude
    variable_dict["elevation"] = elevation
    return variable_dict


# ----------------------------
# Default Corrections
# ---------------------------


def default_corrections(dataset, nc, input_mask, variable_dict, params) -> dict:
    """
    Run the qual/elev corrections defined in the dataset YAML config.

      1. Quality filter drops rows from the variable_dict.
      2. Elevation correction. Corrected elevation replaces the original
      'elevation' in variable_dict.

    ICESat-1 is transformed from TOPEX to WGS84.
    """
    surviving_mask = input_mask

    strict_missing = getattr(params, "strict_missing", False) or getattr(params, "debug", False)

    qual_name = getattr(dataset, "default_qual_correction", None)
    if qual_name:
        qual_mask = _resolve_callable(qual_name)(dataset, nc, input_mask, strict_missing)
        variable_dict = _apply_mask_to_dict(variable_dict, qual_mask)
        surviving_mask = input_mask[qual_mask]

    elev_name = getattr(dataset, "default_elev_correction", None)
    if elev_name:
        variable_dict["elevation"] = _call_with_accepted_kwargs(
            _resolve_callable(elev_name),
            dataset=dataset,
            nc=nc,
            input_mask=surviving_mask,
            elevation=variable_dict["elevation"],
            time=variable_dict.get("time"),
            standard_epoch=params.standard_epoch,
            strict_missing=strict_missing,
        )
        if dataset.mission == "is1":
            variable_dict = _transform_is1_to_wgs84(variable_dict)

    return variable_dict


def apply_corrections(dataset, nc, input_mask, variable_dict, params) -> dict:
    """
    Entry point for corrections to be applied.
    If correction_function != "default_corrections",
    then that function is called directly with all available kwargs.
    Otherwise, run default_corrections is called.
    """
    custom_fn = getattr(params, "correction_function", None)
    user_supplied = custom_fn not in (None, "default_corrections")

    if user_supplied:
        if not isinstance(custom_fn, str) and not callable(custom_fn):
            raise TypeError("params.correction_function must be a dotted-path string or callable")
        return _call_with_accepted_kwargs(
            _resolve_callable(custom_fn),
            dataset=dataset,
            nc=nc,
            input_mask=input_mask,
            variable_dict=variable_dict,
            params=params,
        )

    return default_corrections(dataset, nc, input_mask, variable_dict, params)
