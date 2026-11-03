"""cpom.altimetry.projects.csqa.crossovers

Single cycle crossover differences: the differences of a parameter's values (ie surface
heights) where the ascending and descending ground tracks of a cycle cross.

For each area (with a mask, ie an ice sheet's grounded ice, and a polar stereographic
grid_area), variant (ie retracker height) and acquisition mode:

1. the measurements of each product file in the mode, with valid values, are joined into a
   track of arcs between the (POCA) locations of consecutive measurements (configured lat, lon;
   in the polar stereographic projection of the area's grid), omitting arcs longer than
   max_arc_length_m (data gaps). Arcs are ascending or descending according to the change of
   the 1 Hz nadir latitude (POCA locations can move back and forth between neighbouring
   measurements, so do not give the pass direction)
2. the intersections of ascending and descending arcs are found with a shapely STRtree, and
   the values and times at each intersection are interpolated linearly along both arcs
3. intersections of the same orbit (less than min_time_separation_s apart, near the orbit's
   turning latitude) and outside the area's mask are removed
4. the crossover difference is the ascending minus the descending value
5. a pair of ascending and descending passes gives one crossover (one_per_pass_pair): tracks
   of POCA locations zig-zag (ie LRM slope corrected POCAs move independently), so can cross
   the other pass several times, giving several noisy crossovers. The median crossing of a
   pass pair is kept (for an even number, the mean of the two middle crossings)
6. differences larger than max_abs_difference are outliers: counted, but not valid (NaN)

This is a vectorised version of the cpom (v1) xo_process.py single cycle crossover search.
The crossovers are returned as a loader.ParameterData whose values (per variant) are the
crossover differences, so they are summarised, mapped and gridded like other parameters.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import shapely
from pyproj import Transformer

from cpom.altimetry.projects.csqa.csqa_config import (
    AreaConfig,
    CrossoverConfig,
    CsqaConfig,
    ParameterConfig,
)
from cpom.altimetry.projects.csqa.gridding import get_grid_area, grid_epsg
from cpom.altimetry.projects.csqa.loader import (
    MODE_FILL,
    ParameterData,
    _num2datetime,
    _read,
    _read_variant,
    _reject,
    _time_bounds,
)
from cpom.altimetry.projects.csqa.ncreader import NcDataset
from cpom.altimetry.projects.csqa.product_files import ProductFile
from cpom.masks.masks import Mask

log = logging.getLogger(__name__)


@dataclass
class Arcs:  # pylint: disable=too-many-instance-attributes
    """track arcs between consecutive measurements: start (1) and end (2) points, values and
    times, and whether each arc is ascending"""

    x1: np.ndarray
    y1: np.ndarray
    x2: np.ndarray
    y2: np.ndarray
    v1: np.ndarray
    v2: np.ndarray
    t1: np.ndarray
    t2: np.ndarray
    ascending: np.ndarray
    pass_id: np.ndarray  # pass (track and direction) of each arc, unique within a search

    @classmethod
    def concatenate(cls, arcs: list["Arcs"]) -> "Arcs":
        """arcs of several tracks"""
        names = ("x1", "y1", "x2", "y2", "v1", "v2", "t1", "t2", "ascending", "pass_id")
        dtypes = {"ascending": bool, "pass_id": np.int64}
        if not arcs:
            return cls(**{n: np.array([], dtype=dtypes.get(n, float)) for n in names})
        return cls(**{n: np.concatenate([getattr(a, n) for a in arcs]) for n in names})


@dataclass
class Crossovers:
    """crossover points: location (projection x, y), ascending minus descending value, and
    the times of the ascending and descending measurements"""

    x: np.ndarray
    y: np.ndarray
    difference: np.ndarray
    t_asc: np.ndarray
    t_desc: np.ndarray


def track_arcs(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    times: np.ndarray,
    nadir_lat: np.ndarray,
    max_arc_length: float,
    track_id: int = 0,
) -> Arcs:
    """Arcs between consecutive measurements of a track (in time order)

    Args:
        x (np.ndarray): projection x of the measurements (m)
        y (np.ndarray): projection y of the measurements (m)
        values (np.ndarray): measurement values
        times (np.ndarray): measurement times (s)
        nadir_lat (np.ndarray): nadir latitude of each measurement (gives the pass direction)
        max_arc_length (float): longest arc (m): longer arcs (data gaps) are omitted
        track_id (int): identifier of the track (ie product file) in a crossover search

    Returns:
        Arcs: the passes of the track (runs of arcs in the same direction) have pass_id
              track_id * 1000 + their number in the track
    """
    length = np.hypot(np.diff(x), np.diff(y))
    dlat = np.diff(nadir_lat)
    keep = (length > 0) & (length <= max_arc_length) & (dlat != 0)
    ascending = dlat[keep] > 0
    pass_number = np.concatenate(([0], np.cumsum(ascending[1:] != ascending[:-1])))
    return Arcs(
        x1=x[:-1][keep],
        y1=y[:-1][keep],
        x2=x[1:][keep],
        y2=y[1:][keep],
        v1=values[:-1][keep],
        v2=values[1:][keep],
        t1=times[:-1][keep],
        t2=times[1:][keep],
        ascending=ascending,
        pass_id=track_id * 1000 + pass_number[: ascending.size].astype(np.int64),
    )


def _lines(arcs: Arcs, index: np.ndarray) -> np.ndarray:
    """shapely LineStrings of arcs"""
    start = np.stack([arcs.x1[index], arcs.y1[index]], axis=-1)
    end = np.stack([arcs.x2[index], arcs.y2[index]], axis=-1)
    return shapely.linestrings(np.stack([start, end], axis=1))


def find_crossovers(
    arcs: Arcs, min_time_separation: float = 0.0, one_per_pass_pair: bool = False
) -> Crossovers:
    """Crossovers of the ascending and descending arcs

    Args:
        arcs (Arcs): track arcs
        min_time_separation (float): minimum time between the crossing measurements (s)
        one_per_pass_pair (bool): one crossover for each pair of ascending and descending
                                  passes: the median of their crossings

    Returns:
        Crossovers (an intersection at an arc end point is only counted once)
    """
    asc = np.flatnonzero(arcs.ascending)
    desc = np.flatnonzero(~arcs.ascending)
    if asc.size == 0 or desc.size == 0:
        empty = np.array([])
        return Crossovers(empty, empty, empty, empty, empty)
    tree = shapely.STRtree(_lines(arcs, desc))
    i_asc, i_desc = tree.query(_lines(arcs, asc), predicate="intersects")
    i, j = asc[i_asc], desc[i_desc]

    # intersection p + s r = q + u w of the ascending (p, r) and descending (q, w) arcs
    px, py = arcs.x1[i], arcs.y1[i]
    rx, ry = arcs.x2[i] - px, arcs.y2[i] - py
    qx, qy = arcs.x1[j], arcs.y1[j]
    wx, wy = arcs.x2[j] - qx, arcs.y2[j] - qy
    cross = rx * wy - ry * wx
    with np.errstate(divide="ignore", invalid="ignore"):
        s = ((qx - px) * wy - (qy - py) * wx) / cross
        u = ((qx - px) * ry - (qy - py) * rx) / cross
    # [0, 1): an intersection at the joint of two arcs belongs to the second
    keep = (cross != 0) & (s >= 0) & (s < 1) & (u >= 0) & (u < 1)
    i, j, s, u = i[keep], j[keep], s[keep], u[keep]

    t_asc = arcs.t1[i] + s * (arcs.t2[i] - arcs.t1[i])
    t_desc = arcs.t1[j] + u * (arcs.t2[j] - arcs.t1[j])
    keep = np.abs(t_asc - t_desc) >= min_time_separation
    i, j, s, u = i[keep], j[keep], s[keep], u[keep]
    v_asc = arcs.v1[i] + s * (arcs.v2[i] - arcs.v1[i])
    v_desc = arcs.v1[j] + u * (arcs.v2[j] - arcs.v1[j])
    crossovers = Crossovers(
        x=arcs.x1[i] + s * (arcs.x2[i] - arcs.x1[i]),
        y=arcs.y1[i] + s * (arcs.y2[i] - arcs.y1[i]),
        difference=v_asc - v_desc,
        t_asc=t_asc[keep],
        t_desc=t_desc[keep],
    )
    if one_per_pass_pair:
        crossovers = _median_per_pass_pair(crossovers, arcs.pass_id[i], arcs.pass_id[j])
    return crossovers


def _median_per_pass_pair(
    crossovers: Crossovers, pass_asc: np.ndarray, pass_desc: np.ndarray
) -> Crossovers:
    """one crossover per pair of passes: the crossing with the median difference (for an even
    number of crossings, the mean of the two middle crossings)"""
    if crossovers.difference.size == 0:
        return crossovers
    pair = np.unique(np.stack([pass_asc, pass_desc], axis=1), axis=0, return_inverse=True)[1]
    pair = np.asarray(pair).ravel()
    order = np.lexsort((crossovers.difference, pair))
    _, starts, counts = np.unique(pair[order], return_index=True, return_counts=True)
    lower = order[starts + (counts - 1) // 2]
    upper = order[starts + counts // 2]

    def middle(values: np.ndarray) -> np.ndarray:
        return 0.5 * (values[lower] + values[upper])

    return Crossovers(
        x=middle(crossovers.x),
        y=middle(crossovers.y),
        difference=middle(crossovers.difference),
        t_asc=middle(crossovers.t_asc),
        t_desc=middle(crossovers.t_desc),
    )


@dataclass
class _FileTrack:
    """measurements of a product file within the crossover areas' latitudes"""

    lat: np.ndarray
    lon: np.ndarray
    times: np.ndarray
    nadir_lat: np.ndarray
    modes: np.ndarray
    values: dict[str, np.ndarray] = field(default_factory=dict)  # variant id -> values


def _read_file(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    nc: NcDataset,
    pfile: ProductFile,
    param: ParameterConfig,
    cfg: CsqaConfig,
    start: datetime,
    end: datetime,
    lat_bands: list[tuple[float, float]],
    data: ParameterData,
) -> _FileTrack | None:
    """the measurements of a file in the cycle and the areas' latitude bands"""
    xcfg = param.crossover
    assert xcfg is not None
    ref_var = next((nc.variables[v] for v in param.variables if v in nc.variables), None)
    if ref_var is None:
        for name in param.variables:
            data.missing_variables[name] = data.missing_variables.get(name, 0) + 1
        return None
    time_name = ref_var.dimensions[0]
    time_var = nc.variables[time_name]
    times = np.ma.filled(np.ma.asarray(time_var[:]).astype(np.float64), np.nan)
    t_start, t_end = _time_bounds(time_var, start, end)
    in_cycle = (times >= t_start) & (times < t_end)
    if not np.any(in_cycle):
        return None
    sel_times = times[in_cycle]
    first = _num2datetime(float(np.min(sel_times)), time_var)
    last = _num2datetime(float(np.max(sel_times)), time_var)
    data.first_time = first if data.first_time is None else min(data.first_time, first)
    data.last_time = last if data.last_time is None else max(data.last_time, last)
    data.n_records += int(np.count_nonzero(in_cycle))
    data.files_used.append(pfile.name)

    # crossover locations: the measurement (POCA) locations
    lat_name, lon_name = xcfg.lat, xcfg.lon
    if lat_name not in nc.variables or lon_name not in nc.variables:
        for name in (lat_name, lon_name):
            data.missing_variables[name] = data.missing_variables.get(name, 0) + 1
        return None
    lat = _read(nc, lat_name, in_cycle, np.float64, np.nan)
    keep = np.zeros(lat.size, dtype=bool)
    for lat_min, lat_max in lat_bands:
        keep |= (lat >= lat_min) & (lat <= lat_max)
    if not np.any(keep):
        return None
    sel = np.flatnonzero(in_cycle)[keep]
    sel_mask = np.zeros(times.size, dtype=bool)
    sel_mask[sel] = True

    # pass direction: the 1 Hz nadir latitude interpolated to the measurement times
    if xcfg.nadir_lat in nc.variables and xcfg.nadir_time in nc.variables:
        t_1hz = np.ma.filled(nc[xcfg.nadir_time][:].astype(np.float64), np.nan)
        lat_1hz = np.ma.filled(nc[xcfg.nadir_lat][:].astype(np.float64), np.nan)
        ok = np.isfinite(t_1hz) & np.isfinite(lat_1hz)
        nadir_lat = np.interp(times[sel], t_1hz[ok], lat_1hz[ok]) if ok.sum() > 1 else lat[keep]
    else:
        data.missing_variables[xcfg.nadir_lat] = data.missing_variables.get(xcfg.nadir_lat, 0) + 1
        nadir_lat = lat[keep]

    mode_var = nc.variables.get(cfg.mode_variable)
    track = _FileTrack(
        lat=lat[keep],
        lon=_read(nc, lon_name, sel_mask, np.float64, np.nan),
        times=times[sel],
        nadir_lat=nadir_lat,
        modes=(
            _read(nc, cfg.mode_variable, sel_mask, np.int8, MODE_FILL)
            if mode_var is not None
            else np.full(sel.size, MODE_FILL, dtype=np.int8)
        ),
    )
    reject_words: dict[str, np.ndarray | None] = {}
    for variant in param.variants:
        vals = _read_variant(nc, variant, sel_mask, time_name, np.float64, data)
        if vals is None:
            vals = np.full(sel.size, np.nan)
        else:
            if variant.invalid_values:
                vals[np.isin(vals, variant.invalid_values)] = np.nan
            if variant.value_scale not in (None, 1.0):
                vals *= variant.value_scale
            if variant.reject_mask is not None:
                _reject(nc, variant, vals, sel_mask, time_name, reject_words, data, pfile.name)
        track.values[variant.id] = vals
    return track


def _area_crossovers(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    tracks: list[_FileTrack],
    area: AreaConfig,
    mask: Mask,
    variant_id: str,
    mode_value: int,
    xcfg: CrossoverConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """crossovers of a variant in a mode within an area: latitudes, longitudes, differences"""
    epsg = grid_epsg(get_grid_area(area.grid_area, 10000))
    to_xy = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    arcs = []
    for track_id, track in enumerate(tracks):
        ok = (
            (track.modes == mode_value)
            & np.isfinite(track.values[variant_id])
            & (track.lat >= area.lat_min)
            & (track.lat <= area.lat_max)
        )
        if np.count_nonzero(ok) < 2:
            continue
        x, y = to_xy.transform(track.lon[ok], track.lat[ok])
        arcs.append(
            track_arcs(
                np.asarray(x),
                np.asarray(y),
                track.values[variant_id][ok],
                track.times[ok],
                track.nadir_lat[ok],
                xcfg.max_arc_length_m,
                track_id,
            )
        )
    xovers = find_crossovers(
        Arcs.concatenate(arcs), xcfg.min_time_separation_s, xcfg.one_per_pass_pair
    )
    lons, lats = to_xy.transform(xovers.x, xovers.y, direction="INVERSE")
    lats, lons = np.asarray(lats), np.mod(np.asarray(lons), 360.0)
    if lats.size:
        inside, _ = mask.points_inside(lats, lons, basin_numbers=list(area.mask_basins) or None)
        lats, lons, diffs = lats[inside], lons[inside], xovers.difference[inside]
    else:
        diffs = xovers.difference
    return lats, lons, diffs


def load_crossover_data(  # pylint: disable=too-many-locals
    files: list[ProductFile],
    param: ParameterConfig,
    cfg: CsqaConfig,
    start: datetime,
    end: datetime,
) -> ParameterData:
    """Single cycle crossover differences of a parameter (see module docstring)

    Args:
        files (list[ProductFile]): product files of the cycle
        param (ParameterConfig): parameter with a crossover definition
        cfg (CsqaConfig): CSQA config
        start (datetime): start of cycle (inclusive)
        end (datetime): end of cycle (exclusive)

    Returns:
        ParameterData: crossovers of each variant (its own coordinates and modes), with the
        crossover differences as the variant values. n_records is the number of
        measurements read
    """
    xcfg = param.crossover
    assert xcfg is not None
    data = ParameterData()
    areas = [cfg.areas[a] for a in param.areas]
    lat_bands = [(a.lat_min, a.lat_max) for a in areas]

    tracks = []
    for pfile in files:
        try:
            with NcDataset(pfile.path) as nc:
                track = _read_file(nc, pfile, param, cfg, start, end, lat_bands, data)
        except (OSError, RuntimeError, KeyError, IndexError, ValueError) as exc:
            log.error("failed to read %s: %s", pfile.path, exc)
            data.bad_files.append(pfile.name)
            continue
        if track is not None:
            tracks.append(track)

    masks = {a.id: Mask(a.mask_name) for a in areas}
    modes = [m for m in param.modes if m in cfg.mode_values]
    for variant in param.variants:
        lats, lons, diffs, xo_modes = [], [], [], []
        for area in areas:
            for mode in modes:
                if variant.valid_modes and mode not in variant.valid_modes:
                    continue
                a_lats, a_lons, a_diffs = _area_crossovers(
                    tracks, area, masks[area.id], variant.id, cfg.mode_values[mode], xcfg
                )
                log.info(
                    "%s %s %s %s: %d crossovers", param.id, variant.id, area.id, mode, a_diffs.size
                )
                lats.append(a_lats)
                lons.append(a_lons)
                diffs.append(a_diffs)
                xo_modes.append(np.full(a_diffs.size, cfg.mode_values[mode], dtype=np.int8))
        key = (f"crossover_lat:{variant.id}", f"crossover_lon:{variant.id}")
        data.coord_names[variant.id] = key
        data.lats[key] = np.concatenate(lats).astype(np.float32) if lats else np.array([])
        data.lons[key] = np.concatenate(lons).astype(np.float32) if lons else np.array([])
        data.modes_by_key[key] = np.concatenate(xo_modes) if xo_modes else np.array([], np.int8)
        values = np.concatenate(diffs) if diffs else np.array([])
        # outliers are counted, but not valid
        values[np.abs(values) > xcfg.max_abs_difference] = np.nan
        data.values[variant.id] = values.astype(np.dtype(param.dtype))

    for var_name, n_missing in data.missing_variables.items():
        log.warning("%s missing in %d of %d files", var_name, n_missing, len(files))
    return data
