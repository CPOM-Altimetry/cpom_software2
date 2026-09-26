"""cpom.altimetry.projects.csqa.loader

Load the values, locations and acquisition modes of a CSQA parameter from a set of product
files, keeping only the records within a cycle's time range.

Record times are read from the coordinate variable of the parameter variable's dimension
(ie time_20_ku) and compared with the cycle bounds converted with that variable's units.
Note that CryoSat-2 product times are TAI in some baselines, so records within ~37 s of a cycle
boundary may be assigned to the adjacent cycle.

Locations are taken from the variable's 'coordinates' attribute (ie "lon_poca_20_ku
lat_poca_20_ku"), or the configured default coordinates when that is missing or unusable.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import cast

import numpy as np
from netCDF4 import Dataset, date2num, num2date  # pylint: disable=no-name-in-module

from cpom.altimetry.projects.csqa.csqa_config import CsqaConfig, ParameterConfig
from cpom.altimetry.projects.csqa.product_files import ProductFile

log = logging.getLogger(__name__)

MODE_FILL = -1


@dataclass
class ParameterData:  # pylint: disable=too-many-instance-attributes
    """Values of a parameter's variants for all records of a cycle"""

    n_records: int = 0
    values: dict[str, np.ndarray] = field(default_factory=dict)  # variant id -> values
    coord_names: dict[str, tuple[str, str]] = field(default_factory=dict)  # variant -> lat,lon
    lats: dict[tuple[str, str], np.ndarray] = field(default_factory=dict)
    lons: dict[tuple[str, str], np.ndarray] = field(default_factory=dict)
    modes: np.ndarray | None = None  # acquisition mode flag per record (MODE_FILL if unknown)
    files_used: list[str] = field(default_factory=list)  # files containing cycle records
    bad_files: list[str] = field(default_factory=list)  # files that could not be read
    missing_variables: dict[str, int] = field(default_factory=dict)  # var -> n files missing
    first_time: datetime | None = None
    last_time: datetime | None = None

    def lat(self, variant: str) -> np.ndarray:
        """latitudes of a variant's values"""
        return self.lats[self.coord_names[variant]]

    def lon(self, variant: str) -> np.ndarray:
        """longitudes of a variant's values"""
        return self.lons[self.coord_names[variant]]


def resolve_coordinates(nc: Dataset, var_name: str, cfg: CsqaConfig) -> tuple[str, str]:
    """Find the latitude and longitude variables used as a variable's coordinates

    Args:
        nc (Dataset): open netCDF dataset
        var_name (str): variable name
        cfg (CsqaConfig): CSQA config (for the default coordinates)

    Returns:
        tuple[str, str]: (lat variable name, lon variable name)
    """
    var = nc.variables.get(var_name)
    lat_name = lon_name = None
    if var is not None:
        for coord in str(getattr(var, "coordinates", "")).split():
            if coord not in nc.variables:
                continue
            std_name = getattr(nc.variables[coord], "standard_name", "")
            if std_name == "latitude" or (not std_name and coord.startswith("lat")):
                lat_name = coord
            elif std_name == "longitude" or (not std_name and coord.startswith("lon")):
                lon_name = coord
    if lat_name is None or lon_name is None:
        return cfg.default_lat, cfg.default_lon
    return lat_name, lon_name


def _read(nc: Dataset, name: str, sel: np.ndarray, dtype, fill) -> np.ndarray:
    """read the selected records of a variable, replacing missing/fill values with fill"""
    data = np.ma.asarray(nc.variables[name][:])[sel]
    return np.ma.filled(data.astype(dtype), fill)


def _time_bounds(time_var, start: datetime, end: datetime) -> tuple[float, float]:
    """cycle bounds in the units of a time variable"""
    calendar = getattr(time_var, "calendar", "standard")
    return (
        float(date2num(start, time_var.units, calendar)),
        float(date2num(end, time_var.units, calendar)),
    )


def _num2datetime(value: float, time_var) -> datetime:
    """convert a time value to a naive datetime"""
    when = num2date(
        np.array([value]),
        time_var.units,
        getattr(time_var, "calendar", "standard"),
        only_use_cftime_datetimes=False,
        only_use_python_datetimes=True,
    )
    return cast(datetime, np.asarray(when)[0]).replace(tzinfo=None, microsecond=0)


def load_parameter_data(  # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    files: list[ProductFile],
    param: ParameterConfig,
    cfg: CsqaConfig,
    start: datetime,
    end: datetime,
) -> ParameterData:
    """Load a parameter's values for all records within [start, end)

    Args:
        files (list[ProductFile]): product files to read
        param (ParameterConfig): parameter to load
        cfg (CsqaConfig): CSQA config
        start (datetime): start of cycle (inclusive)
        end (datetime): end of cycle (exclusive)

    Returns:
        ParameterData
    """
    data = ParameterData()
    dtype = np.dtype(param.dtype)

    values: dict[str, list[np.ndarray]] = {v.id: [] for v in param.variants}
    lats: dict[tuple[str, str], list[np.ndarray]] = {}
    lons: dict[tuple[str, str], list[np.ndarray]] = {}
    modes: list[np.ndarray] = []
    need_modes = bool(param.modes)

    # coordinates of each variant, from the first file containing the variant's variable
    for pfile in files:
        unresolved = [v for v in param.variants if v.id not in data.coord_names]
        if not unresolved:
            break
        try:
            with Dataset(pfile.path) as nc:
                for variant in unresolved:
                    if variant.variable in nc.variables:
                        data.coord_names[variant.id] = resolve_coordinates(
                            nc, variant.variable, cfg
                        )
        except OSError:
            continue
    for variant in param.variants:
        data.coord_names.setdefault(variant.id, (cfg.default_lat, cfg.default_lon))
    coord_keys = set(data.coord_names.values())

    for pfile in files:
        try:
            with Dataset(pfile.path) as nc:
                # the time coordinate is the dimension of the first available variant
                ref_var = next(
                    (nc.variables[v] for v in param.variables if v in nc.variables), None
                )
                if ref_var is None:
                    log.warning("none of %s found in %s", param.variables, pfile.name)
                    for var_name in param.variables:
                        data.missing_variables[var_name] = (
                            data.missing_variables.get(var_name, 0) + 1
                        )
                    continue
                time_name = ref_var.dimensions[0]
                if time_name not in nc.variables:
                    log.error("no time coordinate %s in %s", time_name, pfile.name)
                    data.bad_files.append(pfile.name)
                    continue
                time_var = nc.variables[time_name]
                times = np.ma.filled(np.ma.asarray(time_var[:]).astype(np.float64), np.nan)
                t_start, t_end = _time_bounds(time_var, start, end)
                sel = (times >= t_start) & (times < t_end)
                n_sel = int(np.count_nonzero(sel))
                if n_sel == 0:
                    continue

                sel_times = times[sel]
                first = _num2datetime(float(np.min(sel_times)), time_var)
                last = _num2datetime(float(np.max(sel_times)), time_var)
                data.first_time = first if data.first_time is None else min(data.first_time, first)
                data.last_time = last if data.last_time is None else max(data.last_time, last)

                # parameter values of each variant
                for variant in param.variants:
                    var = nc.variables.get(variant.variable)
                    if var is None or var.dimensions[0] != time_name:
                        data.missing_variables[variant.variable] = (
                            data.missing_variables.get(variant.variable, 0) + 1
                        )
                        values[variant.id].append(np.full(n_sel, np.nan, dtype=dtype))
                    else:
                        values[variant.id].append(_read(nc, variant.variable, sel, dtype, np.nan))

                # locations (read once per coordinate pair)
                for key in coord_keys:
                    lat_name, lon_name = key
                    if lat_name not in nc.variables or lon_name not in nc.variables:
                        log.warning(
                            "coordinates %s,%s not in %s, using %s,%s",
                            lat_name,
                            lon_name,
                            pfile.name,
                            cfg.default_lat,
                            cfg.default_lon,
                        )
                        lat_name, lon_name = cfg.default_lat, cfg.default_lon
                    lats.setdefault(key, []).append(_read(nc, lat_name, sel, np.float32, np.nan))
                    lons.setdefault(key, []).append(_read(nc, lon_name, sel, np.float32, np.nan))

                # acquisition modes
                if need_modes:
                    mode_var = nc.variables.get(cfg.mode_variable)
                    if mode_var is None or mode_var.dimensions[0] != time_name:
                        data.missing_variables[cfg.mode_variable] = (
                            data.missing_variables.get(cfg.mode_variable, 0) + 1
                        )
                        modes.append(np.full(n_sel, MODE_FILL, dtype=np.int8))
                    else:
                        modes.append(_read(nc, cfg.mode_variable, sel, np.int8, MODE_FILL))

                data.n_records += n_sel
                data.files_used.append(pfile.name)
        except (OSError, RuntimeError, KeyError, IndexError, ValueError) as exc:
            log.error("failed to read %s: %s", pfile.path, exc)
            data.bad_files.append(pfile.name)

    data.values = {
        vid: np.concatenate(arrs) if arrs else np.array([], dtype=dtype)
        for vid, arrs in values.items()
    }
    for key in coord_keys:
        data.lats[key] = np.concatenate(lats[key]) if key in lats else np.array([], np.float32)
        data.lons[key] = np.concatenate(lons[key]) if key in lons else np.array([], np.float32)
    if need_modes:
        data.modes = np.concatenate(modes) if modes else np.array([], dtype=np.int8)

    for var_name, n_missing in data.missing_variables.items():
        log.warning("%s missing in %d of %d files", var_name, n_missing, len(files))

    return data
