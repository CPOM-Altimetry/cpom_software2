"""cpom.altimetry.tools.sec_tools.grid_for_elev_change_corrections.

Utility functions for applying mission-specific quality and elevation corrections
within the SEC gridding pipeline.

Corrections are either loaded dynamically from names/dotted paths specified in the
dataset YAML config. A custom correction function can also be supplied via
params.correction_function within the gridding script's run control file.
"""

import importlib
import inspect
import typing
from typing import Callable, Union

import numpy as np
from pyproj import CRS, Transformer

from cpom.altimetry.datasets.dataset_helper import DatasetHelper

# ----------------------------
# Helper methods
# ---------------------------


def _apply_mask_to_dict(variable_dict: dict, mask: np.ndarray) -> dict:
    """Apply a boolean mask to all non-null arrays in variable_dict."""
    return {k: v[mask] if v is not None else v for k, v in variable_dict.items()}


def _resolve_callable(name: Union[str, Callable]) -> Callable:
    """
    Resolve a string function name or callable into a callable.

        A) If a callable is passed directly, return it.
        B) Plain name lookup in the current modules global scope.
        C) Import via dotted path (e.g. 'my.module.func')

    Args:
        name: A callable, plain function name , or dotted import path
    Returns
        Callable: Resolved function
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
    """Call func with only the kwargs its signature declares."""
    accepted = inspect.signature(func).parameters
    return func(**{k: v for k, v in kwargs.items() if k in accepted})


# ----------------------------
# Mission Specific Corrections
# ----------------------------


def _transform_is1_to_wgs84(variable_dict: dict) -> dict:
    """Convert ICESat-1 lat/lon/elevation from TOPEX ellipsoid to WGS84 in-place.
    Args:
        variable_dict: Dict containing 'latitude', 'longitude', and 'elevation' arrays.

    Returns:
        dict: The same dict with latitude, longitude, and elevation replaced
              by their WGS84 equivalents.
    """
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


def default_corrections(
    dataset: DatasetHelper, nc: object, input_mask: np.ndarray, variable_dict: dict, params: object
) -> dict:
    """
    Apply quality and elevation corrections defined in the dataset config YAML.

    Steps:
        1. Quality filter: applies the dataset's default_qual_correction function,
           masking out rows that fail quality checks.
        2. Elevation correction: applies the dataset's default_elev_correction function,
           replacing the 'elevation' entry in variable_dict.
        3. For ICESat-1 data, converts corrected coordinates from TOPEX to WGS84.

    Args:
        dataset (DatasetHelper): Dataset configuration object with default_qual_correction,
                 default_elev_correction, and mission attributes.
        nc (object): Open NetCDF file.
        input_mask (np.ndarray): Boolean array selecting valid observations before correction.
        variable_dict (dict): Dictionary of variable arrays
            (e.g. latitude, longitude, elevation, time).
        params (object): Command line arguments from gridding script.

    Returns:
        dict: Updated variable_dict with quality-filtered and corrected values.
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
            standard_epoch=getattr(params, "standard_epoch", None),
            strict_missing=strict_missing,
        )
        if dataset.mission == "is1":
            variable_dict = _transform_is1_to_wgs84(variable_dict)

    return variable_dict


def apply_corrections(
    dataset: DatasetHelper, nc: object, input_mask: np.ndarray, variable_dict: dict, params: object
) -> dict:
    """
    Entry point for corrections to be applied.

    If a custom correction function is supplied via params.correction_function,
    it will be called with all available kwargs.
    Otherwise, the default_corrections function will be applied.

    Args:
        dataset (DatasetHelper): Dataset configuration object.
        nc (object):  Open NetCDF file.
        input_mask (np.ndarray): Boolean array selecting valid observations before correction.
        variable_dict (dict): Dictionary of variable arrays
            (e.g. latitude, longitude, elevation, time).
        params (object): Command line arguments from gridding script.
    Returns:
        dict: Updated variable_dict with quality-filtered and corrected values.
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
