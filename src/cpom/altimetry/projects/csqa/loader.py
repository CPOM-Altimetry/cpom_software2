"""cpom.altimetry.projects.csqa.loader

Load the values, locations and acquisition modes of a CSQA parameter from a set of product
files, keeping only the records within a cycle's time range.

Record times are read from the coordinate variable of the parameter variable's dimension
(ie time_20_ku) and compared with the cycle bounds converted with that variable's units.
Note that CryoSat-2 product times are TAI in some baselines, so records within ~37 s of a cycle
boundary may be assigned to the adjacent cycle.

Locations are taken from the variable's 'coordinates' attribute (ie "lon_poca_20_ku
lat_poca_20_ku"), or the configured default coordinates when that is missing or unusable.

Product files are read with ncreader.NcDataset (h5py), which opens files ~10x faster than
netCDF4 while masking and scaling values in the same way.

For bit flag parameters the flag word is read once and each bit's values (1 set, 0 not set)
are derived from it on demand (ParameterData.variant_values).

Derived variants (ie the mispointing angle) are computed from their input variables. Values
equal to a variant's invalid_values (ie 0 where a variable is unused) are set to NaN, and
values of variants with a value_scale are then scaled (ie degrees to millidegrees).

Variants with a reject bit (ie freeboard_error of flag_prod_status_20_ku) have their values set
to NaN where the bit is set, so only unflagged values are counted, mapped and gridded. Likewise
the values of parameters with valid_modes are set to NaN in the other acquisition modes.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import cast

import numpy as np
from netCDF4 import date2num, num2date  # pylint: disable=no-name-in-module

from cpom.altimetry.projects.csqa.csqa_config import (
    CsqaConfig,
    ParameterConfig,
    VariantDef,
)
from cpom.altimetry.projects.csqa.derived import DERIVED_VARIABLES
from cpom.altimetry.projects.csqa.ncreader import NcDataset
from cpom.altimetry.projects.csqa.product_files import ProductFile

log = logging.getLogger(__name__)

MODE_FILL = -1


@dataclass
class ParameterData:  # pylint: disable=too-many-instance-attributes
    """Values of a parameter's variants for all records of a cycle"""

    n_records: int = 0
    values: dict[str, np.ndarray] = field(default_factory=dict)  # variant id -> values
    bit_words: dict[str, np.ndarray] = field(default_factory=dict)  # variable -> flag words
    coord_names: dict[str, tuple[str, str]] = field(default_factory=dict)  # variant -> lat,lon
    lats: dict[tuple[str, str], np.ndarray] = field(default_factory=dict)
    lons: dict[tuple[str, str], np.ndarray] = field(default_factory=dict)
    modes: np.ndarray | None = None  # acquisition mode flag per record (MODE_FILL if unknown)
    # surface type mask per record (MODE_FILL if unknown), for mode surface selections
    surfaces: np.ndarray | None = None
    # pass direction per record of each coordinate pair: 1 ascending, -1 descending, 0 unknown
    directions: dict[tuple[str, str], np.ndarray] = field(default_factory=dict)
    # modes of the records of a coordinate pair, when its records are not those of 'modes'
    # (ie the crossovers of each variant)
    modes_by_key: dict[tuple[str, str], np.ndarray] = field(default_factory=dict)
    files_used: list[str] = field(default_factory=list)  # files containing cycle records
    bad_files: list[str] = field(default_factory=list)  # files that could not be read
    missing_variables: dict[str, int] = field(default_factory=dict)  # var -> n files missing
    first_time: datetime | None = None
    last_time: datetime | None = None

    def variant_values(self, variant: VariantDef) -> np.ndarray:
        """values of a variant: for a bit of a flag word 1 (set), 0 (not set) or NaN (fill)"""
        if variant.bit_mask is None:
            return self.values[variant.id]
        return bit_values(self.bit_words[variant.variable], variant.bit_mask)

    def lat(self, variant: str) -> np.ndarray:
        """latitudes of a variant's values"""
        return self.lats[self.coord_names[variant]]

    def lon(self, variant: str) -> np.ndarray:
        """longitudes of a variant's values"""
        return self.lons[self.coord_names[variant]]

    def mode_array(self, variant: str) -> np.ndarray | None:
        """acquisition modes of a variant's values (None if not loaded)"""
        return self.modes_by_key.get(self.coord_names[variant], self.modes)

    def direction(self, variant: str) -> np.ndarray:
        """pass directions of a variant's values (1 ascending, -1 descending, 0 unknown)"""
        return self.directions[self.coord_names[variant]]


def pass_directions(lats: np.ndarray) -> np.ndarray:
    """Pass direction of time ordered records of a product file, from the rate of change of
    their latitude: 1 ascending (latitude increasing), -1 descending, 0 unknown (ie a single
    record, or a missing latitude; missing latitudes are skipped when taking the rate). Meant
    for nadir latitudes (ie lat_01): POCA latitudes of sloping surfaces can jump between
    neighbouring records

    Args:
        lats (np.ndarray): latitudes of consecutive records (in time order)

    Returns:
        np.ndarray: int8 direction of each record
    """
    directions = np.zeros(lats.size, dtype=np.int8)
    finite = np.flatnonzero(np.isfinite(lats))
    if finite.size >= 2:
        directions[finite] = np.sign(np.gradient(lats[finite].astype(np.float64)))
    return directions


def resolve_coordinates(nc: NcDataset, var_name: str, cfg: CsqaConfig) -> tuple[str, str]:
    """Find the latitude and longitude variables used as a variable's coordinates

    Args:
        nc (NcDataset): open netCDF dataset
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


def _read(nc: NcDataset, name: str, sel: np.ndarray, dtype, fill) -> np.ndarray:
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


def bit_values(words: np.ndarray, mask: int) -> np.ndarray:
    """values of one bit of flag words: 1 (set), 0 (not set) or NaN (missing, words < 0)"""
    vals = ((words & mask) != 0).astype(np.float32)
    vals[words < 0] = np.nan
    return vals


def check_bit_meanings(nc: NcDataset, param: ParameterConfig, file_name: str):
    """warn when a configured bit's name differs from the flag_meanings of the product"""
    var = nc.variables.get(param.variants[0].variable)
    if var is None or not hasattr(var, "flag_masks") or not hasattr(var, "flag_meanings"):
        return
    meanings = dict(zip((int(m) for m in np.atleast_1d(var.flag_masks)), var.flag_meanings.split()))
    for variant in param.variants:
        name = meanings.get(int(variant.bit_mask or 0))
        if name is not None and variant.bit_name and name != variant.bit_name:
            log.warning(
                "%s bit %d is %s in %s, not %s as configured",
                variant.variable,
                variant.bit_mask,
                name,
                file_name,
                variant.bit_name,
            )


def _read_variant(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    nc: NcDataset,
    variant: VariantDef,
    sel: np.ndarray,
    time_name: str,
    dtype,
    data: "ParameterData",
) -> np.ndarray | None:
    """Read (or derive from its input variables) the selected records of a variant's values

    Args:
        nc (NcDataset): open product file
        variant (VariantDef): variant (not a flag word bit)
        sel (np.ndarray): selected records of the file
        time_name (str): time dimension of the parameter
        dtype: numpy dtype of the values
        data (ParameterData): loaded data (for the missing variable counts)

    Returns:
        np.ndarray|None: values (NaN where missing or fill), or None if a variable is missing
    """
    names = variant.inputs if variant.derived else (variant.variable,)
    missing = [
        name
        for name in names
        if name not in nc.variables or nc.variables[name].dimensions[0] != time_name
    ]
    for name in missing:
        data.missing_variables[name] = data.missing_variables.get(name, 0) + 1
    if missing:
        return None
    if not variant.derived:
        return _read(nc, variant.variable, sel, dtype, np.nan)
    inputs = [_read(nc, name, sel, np.float64, np.nan) for name in names]
    return DERIVED_VARIABLES[variant.derived][1](*inputs).astype(dtype)


def _reject(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    nc: NcDataset,
    variant: VariantDef,
    vals: np.ndarray,
    sel: np.ndarray,
    time_name: str,
    words_cache: dict[str, np.ndarray | None],
    data: "ParameterData",
    file_name: str,
):
    """Set a variant's values to NaN where its reject bit is set (every value if the flag
    word variable is missing)

    Args:
        nc (NcDataset): open product file
        variant (VariantDef): variant with a reject bit
        vals (np.ndarray): the variant's values of the selected records, updated in place
        sel (np.ndarray): selected records of the file
        time_name (str): time dimension of the values
        words_cache (dict): flag words read from this file, by variable name
        data (ParameterData): loaded data (for the missing variable counts)
        file_name (str): name of the file (for messages)
    """
    var_name = variant.reject_variable
    if var_name not in words_cache:
        var = nc.variables.get(var_name)
        if var is None or var.dimensions[0] != time_name:
            data.missing_variables[var_name] = data.missing_variables.get(var_name, 0) + 1
            words_cache[var_name] = None
        else:
            meanings = dict(
                zip(
                    (int(m) for m in np.atleast_1d(getattr(var, "flag_masks", []))),
                    str(getattr(var, "flag_meanings", "")).split(),
                )
            )
            name = meanings.get(int(variant.reject_mask or 0))
            if name is not None and variant.reject_name and name != variant.reject_name:
                log.warning(
                    "%s bit %d is %s in %s, not %s as configured",
                    var_name,
                    variant.reject_mask,
                    name,
                    file_name,
                    variant.reject_name,
                )
            words_cache[var_name] = _read(nc, var_name, sel, np.int64, 0)
    words = words_cache[var_name]
    if words is None:
        vals[:] = np.nan
    else:
        vals[(words & int(variant.reject_mask or 0)) != 0] = np.nan


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

    values: dict[str, list[np.ndarray]] = {v.id: [] for v in param.variants if v.bit_mask is None}
    # flag words of bit flag parameters, read once per variable
    bit_words: dict[str, list[np.ndarray]] = {
        v.variable: [] for v in param.variants if v.bit_mask is not None
    }
    lats: dict[tuple[str, str], list[np.ndarray]] = {}
    lons: dict[tuple[str, str], list[np.ndarray]] = {}
    modes: list[np.ndarray] = []
    # acquisition modes are needed for mode (and mode surface) selections and valid modes
    need_modes = any(v.valid_modes for v in param.variants) or any(
        m in cfg.mode_values or m in cfg.mode_surfaces for m in param.modes
    )
    surfaces: list[np.ndarray] = []
    need_surfaces = any(m in cfg.mode_surfaces for m in param.modes)
    # pass directions of each coordinate pair (for ascending/descending pass selections)
    directions: dict[tuple[str, str], list[np.ndarray]] = {}
    need_directions = any(m in cfg.pass_selections for m in param.modes)

    # coordinates of each variant, from the first file containing the variant's variable
    for pfile in files:
        unresolved = [v for v in param.variants if v.id not in data.coord_names]
        if not unresolved:
            break
        try:
            with NcDataset(pfile.path) as nc:
                for variant in unresolved:
                    if variant.variable in nc.variables:
                        data.coord_names[variant.id] = resolve_coordinates(
                            nc, variant.variable, cfg
                        )
        except OSError:
            continue
    # a variant missing from every file uses the coordinates of another variant (which are on
    # the parameter's time dimension), rather than the default coordinates (which may not be)
    fallback = next(iter(data.coord_names.values()), (cfg.default_lat, cfg.default_lon))
    for variant in param.variants:
        data.coord_names.setdefault(variant.id, fallback)
    coord_keys = set(data.coord_names.values())

    for pfile in files:
        try:
            with NcDataset(pfile.path) as nc:
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

                # flag words of bit flag parameters (-1 where missing or fill)
                for var_name, words in bit_words.items():
                    var = nc.variables.get(var_name)
                    if var is None or var.dimensions[0] != time_name:
                        data.missing_variables[var_name] = (
                            data.missing_variables.get(var_name, 0) + 1
                        )
                        words.append(np.full(n_sel, -1, dtype=np.int64))
                    else:
                        if not words:
                            check_bit_meanings(nc, param, pfile.name)
                        words.append(_read(nc, var_name, sel, np.int64, -1))

                # parameter values of each variant
                reject_words: dict[str, np.ndarray | None] = {}
                for variant in param.variants:
                    if variant.bit_mask is not None:
                        continue
                    vals = _read_variant(nc, variant, sel, time_name, dtype, data)
                    if vals is None:
                        values[variant.id].append(np.full(n_sel, np.nan, dtype=dtype))
                        continue
                    if variant.invalid_values:
                        vals[np.isin(vals, variant.invalid_values)] = np.nan
                    if variant.value_scale not in (None, 1.0):
                        vals *= variant.value_scale
                    if variant.reject_mask is not None:
                        _reject(nc, variant, vals, sel, time_name, reject_words, data, pfile.name)
                    values[variant.id].append(vals)

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
                    if need_directions:
                        directions.setdefault(key, []).append(pass_directions(lats[key][-1]))

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

                # surface types (for mode surface selections, ie LRM over ice)
                if need_surfaces:
                    surf_var = nc.variables.get(cfg.surface_variable)
                    if surf_var is None or surf_var.dimensions[0] != time_name:
                        data.missing_variables[cfg.surface_variable] = (
                            data.missing_variables.get(cfg.surface_variable, 0) + 1
                        )
                        surfaces.append(np.full(n_sel, MODE_FILL, dtype=np.int8))
                    else:
                        surfaces.append(_read(nc, cfg.surface_variable, sel, np.int8, MODE_FILL))

                data.n_records += n_sel
                data.files_used.append(pfile.name)
        except (OSError, RuntimeError, KeyError, IndexError, ValueError) as exc:
            log.error("failed to read %s: %s", pfile.path, exc)
            data.bad_files.append(pfile.name)

    data.values = {
        vid: np.concatenate(arrs) if arrs else np.array([], dtype=dtype)
        for vid, arrs in values.items()
    }
    data.bit_words = {
        var_name: np.concatenate(arrs) if arrs else np.array([], dtype=np.int64)
        for var_name, arrs in bit_words.items()
    }
    for key in coord_keys:
        data.lats[key] = np.concatenate(lats[key]) if lats.get(key) else np.array([], np.float32)
        data.lons[key] = np.concatenate(lons[key]) if lons.get(key) else np.array([], np.float32)
        if need_directions:
            data.directions[key] = (
                np.concatenate(directions[key]) if directions.get(key) else np.array([], np.int8)
            )
    if need_modes:
        data.modes = np.concatenate(modes) if modes else np.array([], dtype=np.int8)
    if need_surfaces:
        data.surfaces = np.concatenate(surfaces) if surfaces else np.array([], dtype=np.int8)
    for variant in param.variants:
        if variant.valid_modes and data.modes is not None and variant.id in data.values:
            # values of other acquisition modes are rejected (ie 0 rather than fill where unused)
            other_mode = ~np.isin(data.modes, [cfg.mode_values[m] for m in variant.valid_modes])
            data.values[variant.id][other_mode] = np.nan

    for var_name, n_missing in data.missing_variables.items():
        log.warning("%s missing in %d of %d files", var_name, n_missing, len(files))

    return data
