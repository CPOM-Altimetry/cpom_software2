"""
cpom.altimetry.datasets.correction_pipeline.py
"""

from __future__ import annotations

import importlib
import inspect
import typing
from typing import Callable, Protocol, Union

import numpy as np
from pyproj import CRS, Transformer

# ----------------------------
# Helper functions
# ----------------------------


def _apply_mask(variable_dict: dict, mask: np.ndarray) -> dict:
    return {k: v[mask] if v is not None else v for k, v in variable_dict.items()}


def _resolve_callable(name: Union[str, Callable]) -> Callable:
    """
    Take a string and return a callable function.
    - If the input is already a callable, return it.
    - If the input is a string, first check if it's a local function name.
    - If not found locally, treat it as a dotted import path and attempt to import it

    Args:
        name: A callable, plain function name, or dotted import path.

    Returns:
        Callable: Resolved function.
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


def _describe_callable(func: Callable | None) -> dict | None:
    """
    Summarize a resolved correction callable for metadata/logging.

    Args:
        func: Resolved callable, or None if that correction stage is disabled.

    Returns:
        dict | None: {"name": "module.qualname", "summary": first docstring line},
        or None if func is None.
    """
    if func is None:
        return None
    doc_lines = (inspect.getdoc(func) or "").strip().splitlines()
    return {
        "name": f"{func.__module__}.{func.__qualname__}",
        "summary": doc_lines if doc_lines else "",
    }


def topex_to_wgs84(variable_dict: dict) -> dict:
    """Convert lat/lon/elevation from TOPEX ellipsoid to WGS84."""
    topex_crs = CRS.from_proj4("+proj=latlong +a=6378136.300 +rf=298.257 +no_defs")
    ecef_crs = CRS.from_proj4("+proj=geocent +a=6378136.300 +rf=298.257 +no_defs")
    to_ecef = Transformer.from_crs(topex_crs, ecef_crs, always_xy=True)
    to_wgs = Transformer.from_crs(ecef_crs, CRS.from_epsg(4979), always_xy=True)

    x, y, h = typing.cast(
        tuple[typing.Any, typing.Any, typing.Any],
        to_ecef.transform(
            variable_dict["longitude"], variable_dict["latitude"], variable_dict["elevation"]
        ),
    )
    longitude, latitude, elevation = to_wgs.transform(x, y, h)
    variable_dict["longitude"] = longitude
    variable_dict["latitude"] = latitude
    variable_dict["elevation"] = elevation
    return variable_dict


class CorrectionDataset(Protocol):  # pylint: disable=too-few-public-methods
    """Dataset attributes used to configure correction stages."""

    mission: str
    default_qual_correction: str | None
    default_elev_correction: str | None


# ----------------------------
# CorrectionPipeline
# ----------------------------
class CorrectionPipeline:
    """
    Ordered pipeline to apply corrections to a dataset.

    Prefer constructing via CorrectionPipeline.from_args() when building from
    CLI arguments or other optional inputs — it handles falling back to dataset
    defaults when values are not provided.

    Use the constructor directly when you have already-resolved callables and
    want full explicit control (e.g. in tests or programmatic use).

    quality_filter must return (mask, extra_params) and elevation_correction
    must return (elevation, extra_params) — extra_params is a dict of column
    name -> array (pass {} if there's nothing to add). This lets a
    mission-specific correction surface the parameters it applied (e.g. a
    per-record correction magnitude); the extras are merged into
    variable_dict and so flow straight through to the output Parquet.
    custom_correction already receives and returns the full variable_dict,
    so it can add columns directly and doesn't need this convention.

    Attributes:
        dataset:               DatasetHelper instance.
        quality_filter:        Optional callable matching QualityFilter protocol.
        elevation_correction:  Optional callable matching ElevationCorrection protocol.
        custom_correction:     Optional callable for additional corrections.
        coordinate_transforms: Ordered list of post-correction coordinate transforms.
    """

    def __init__(
        self,
        dataset: CorrectionDataset,
        quality_filter: Callable | str | None = None,
        elevation_correction: Callable | str | None = None,
        custom_correction: Callable | str | None = None,
        coordinate_transforms: list[Callable] | None = None,
    ):
        """
        Construct a pipeline from already-resolved values.

        All correction arguments are taken as-is: None means disabled,
        and a callable or import path is resolved for use. No fallback to
        dataset defaults occurs here.

        Args:
            dataset:               DatasetHelper instance.
            quality_filter:        Callable/import path or None to disable.
            elevation_correction:  Callable/import path or None to disable.
            custom_correction:     Callable/import path or None to disable.
            coordinate_transforms: Explicit list of transforms, or None to
                                   infer from dataset mission (is1 → topex_to_wgs84).
        """
        self.dataset = dataset
        self.quality_filter = _resolve_callable(quality_filter) if quality_filter else None
        self.elevation_correction = (
            _resolve_callable(elevation_correction) if elevation_correction else None
        )
        self.custom_correction = _resolve_callable(custom_correction) if custom_correction else None
        self.coordinate_transforms = (
            coordinate_transforms
            if coordinate_transforms is not None
            else [topex_to_wgs84] if getattr(dataset, "mission", None) == "is1" else []
        )

    @classmethod
    def from_args(
        cls,
        dataset: CorrectionDataset,
        quality_filter: Callable | str | None = None,
        elevation_correction: Callable | str | None = None,
        custom_correction: Callable | str | None = None,
        coordinate_transforms: list[Callable] | None = None,
    ) -> "CorrectionPipeline":
        """
        Build a pipeline, falling back to dataset defaults when args are None.

        This is the preferred constructor when inputs come from CLI arguments
        or other optional sources. None means "not provided" here, so the
        dataset default is used. To explicitly disable a correction that has
        a dataset default, pass the constructor directly with None.

        Args:
            dataset:               DatasetHelper instance.
            quality_filter:        Callable/string to use, or None to use
                                   dataset.default_qual_correction.
            elevation_correction:  Callable/string to use, or None to use
                                   dataset.default_elev_correction.
            custom_correction:     Callable/string to use, or None to disable.
            coordinate_transforms: Explicit list of transforms, or None to
                                   infer from dataset mission.

        Returns:
            CorrectionPipeline: Constructed pipeline.
        """
        return cls(
            dataset=dataset,
            quality_filter=(
                quality_filter if quality_filter is not None else dataset.default_qual_correction
            ),
            elevation_correction=(
                elevation_correction
                if elevation_correction is not None
                else dataset.default_elev_correction
            ),
            custom_correction=custom_correction,
            coordinate_transforms=coordinate_transforms,
        )

    def run(
        self,
        nc: object,
        input_mask: np.ndarray,
        variable_dict: dict,
        strict_missing: bool,
    ) -> dict:
        """
        Apply corrections in order:
            1. Quality filter  — masks out low-quality records.
            2. Elevation correction — adjusts elevation values.
            3. Custom correction — any additional dataset-specific correction.
            4. Coordinate transforms — e.g. TOPEX → WGS84 for ICESat-1.

        quality_filter must return (mask, extra_params) and elevation_correction
        must return (elevation, extra_params), where extra_params is a dict of
        column name -> array aligned to the returned value (pass {} if there's
        nothing to add). extra_params are merged into variable_dict and so
        become columns in the output Parquet, e.g. to record the per-record
        correction applied.

        Args:
            nc:             Open netCDF4 dataset.
            input_mask:     Boolean array of initially valid records.
            variable_dict:  Dict of variable name → numpy array.
            strict_missing: Raise on missing expected variables rather than warn.

        Returns:
            dict: Corrected variable_dict.
        """
        surviving_mask = input_mask if input_mask is not None else slice(None)

        if self.quality_filter is not None:
            qual_mask, qual_extras = self.quality_filter(
                self.dataset, nc, surviving_mask, strict_missing
            )
            if input_mask is not None:
                masked_variable_dict = _apply_mask(variable_dict, input_mask)
                surviving_mask = np.array(input_mask, copy=True)
                surviving_mask[input_mask] = qual_mask
                masked_variable_dict.update(qual_extras)
                variable_dict = _apply_mask(masked_variable_dict, qual_mask)
            else:
                surviving_mask = qual_mask
                variable_dict.update(qual_extras)
                variable_dict = _apply_mask(variable_dict, surviving_mask)
        elif input_mask is not None:
            variable_dict = _apply_mask(variable_dict, surviving_mask)

        if self.elevation_correction is not None:
            elevation, elev_extras = _call_with_accepted_kwargs(
                self.elevation_correction,
                dataset=self.dataset,
                nc=nc,
                input_mask=surviving_mask,
                elevation=variable_dict["elevation"],
                strict_missing=strict_missing,
            )
            variable_dict["elevation"] = elevation
            variable_dict.update(elev_extras)

        if self.custom_correction is not None:
            variable_dict = _call_with_accepted_kwargs(
                self.custom_correction,
                dataset=self.dataset,
                nc=nc,
                input_mask=surviving_mask,
                variable_dict=variable_dict,
                strict_missing=strict_missing,
            )

        for transform in self.coordinate_transforms:
            variable_dict = transform(variable_dict)

        return variable_dict

    def describe(self) -> dict:
        """
        Summarize the resolved corrections in this pipeline for metadata/logging.

        Reports what actually ran (including any CLI overrides of the dataset
        defaults), unlike the raw dataset YAML which only records the configured
        defaults. Static only — no per-run statistics; the columns added via
        extra_params are already visible in the output Parquet schema.

        Returns:
            dict: {"quality_filter": {...} | None, "elevation_correction": {...} | None,
            "custom_correction": {...} | None, "coordinate_transforms": [...]}.
        """
        return {
            "quality_filter": _describe_callable(self.quality_filter),
            "elevation_correction": _describe_callable(self.elevation_correction),
            "custom_correction": _describe_callable(self.custom_correction),
            "coordinate_transforms": [_describe_callable(t) for t in self.coordinate_transforms],
        }
