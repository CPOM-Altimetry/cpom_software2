"""cpom.altimetry.projects.csqa.plotting

Map plots of CSQA parameters using cpom.areas.area_plot.Polarplot.

Large selections are drawn as a regular along-track subsample (every Nth valid record) limited
to the configured maximum number of points, while the statistics drawn on the plot (and the flag
percentages) are always those of every record. Records without a valid value are only drawn on
the bad data mini-map, so are subsampled to at most MAX_MISSING_POINTS (with the % valid / % NaN
drawn being those of all records). Parameters only valid in a few % of records (ie freeboard)
are therefore not thinned out by their missing values.

Plots are prepared (prepare_plot_job) in the process holding the cycle's data, then rendered
(render_plot_job) either in the same process or in a pool of plot worker processes.

Gridded maps (prepare_grid_plot_job) draw a statistic of the measurements in each cell of a
polar stereographic grid, with the histograms and statistics of the cell values.

A thumbnail of each plot is saved in a 'thumbs' sub-directory of the plot's directory.
"""

import contextlib
import io
import logging
import math
import os
from dataclasses import dataclass
from datetime import datetime, timedelta

import matplotlib
import numpy as np
from PIL import Image

from cpom.altimetry.projects.csqa.csqa_config import (
    AreaConfig,
    ColourScale,
    CsqaConfig,
    GridStatistic,
    ParameterConfig,
    VariantDef,
)
from cpom.altimetry.projects.csqa.gridding import (
    GriddedSelection,
    cell_centres_latlon,
    grid_image,
)

matplotlib.use("Agg")  # non-interactive backend for batch processing

# pylint: disable=wrong-import-position
from cpom.areas.area_plot import Annotation, Polarplot  # noqa: E402

log = logging.getLogger(__name__)

THUMBNAIL_WIDTH = 360  # pixels
MAX_MISSING_POINTS = 200_000  # missing values drawn on the bad data mini-map


def plot_filename(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    param_id: str, variant_id: str, mode: str, area_id: str, fmt: str, scale_suffix: str = ""
) -> str:
    """Name of a parameter's map plot file, ie backscatter_rtk1_sar_global.webp or
    height_rtk1_all_global_ocean.webp

    Args:
        param_id (str): parameter id
        variant_id (str): variant id ('' if none)
        mode (str): mode selection ('' if none)
        area_id (str): area id
        fmt (str): image format / file extension
        scale_suffix (str): colour scale file suffix ('' for the default colour scale)

    Returns:
        str: file name
    """
    parts = (param_id, variant_id, mode, area_id, scale_suffix)
    return "_".join(part for part in parts if part) + f".{fmt}"


def thumbnail_path(plot_path: str) -> str:
    """path of a plot's thumbnail image"""
    return os.path.join(os.path.dirname(plot_path), "thumbs", os.path.basename(plot_path))


def save_thumbnail(plot_path: str, quality: int = 75):
    """save a thumbnail (THUMBNAIL_WIDTH pixels wide) of a plot image"""
    thumb = thumbnail_path(plot_path)
    os.makedirs(os.path.dirname(thumb), exist_ok=True)
    with Image.open(plot_path) as img:
        fmt = img.format or "WEBP"
        img.thumbnail((THUMBNAIL_WIDTH, THUMBNAIL_WIDTH * 4), Image.Resampling.LANCZOS)
        tmp_thumb = f"{thumb}.tmp"
        img.save(tmp_thumb, format=fmt, quality=quality)
    os.replace(tmp_thumb, thumb)


def decimation_step(n_points: int, max_points: int) -> int:
    """step N so that every Nth point gives at most max_points points"""
    if max_points <= 0 or n_points <= max_points:
        return 1
    return math.ceil(n_points / max_points)


def plot_title(param: ParameterConfig, variant: VariantDef, mode: str, cfg: CsqaConfig) -> str:
    """title of a map plot, ie 'Backscatter (Sigma0): Retracker 1, SAR mode'"""
    title = param.long_name
    if param.has_variants:
        title += f": {variant.name}"
    if mode:
        # ie ', All modes', ', SAR mode', ', LRM Ice', ', Ascending passes'
        title += f", {cfg.mode_text(param, mode)}"
    return title


@dataclass
class PlotJob:  # pylint: disable=too-many-instance-attributes
    """A map plot, prepared in the cycle process and rendered by render_plot_job(), possibly in
    a separate worker process. It only holds the (decimated) points plotted, so is cheap to
    send to a worker."""

    out_path: str
    polarplot_area: str
    data_set: dict
    annotations: list[Annotation]
    image_format: str
    dpi: int
    webp_quality: int
    step: int


def _annotations(
    title: str, cycle: int, cycle_bounds: tuple[datetime, datetime], baseline: str, note: str
) -> list[Annotation]:
    """title, cycle and note lines of a map plot"""
    start, end = cycle_bounds
    last_day = end - timedelta(seconds=1)
    annotations = [
        Annotation(0.22, 0.972, title, fontsize=13, fontweight="bold"),
        Annotation(
            0.22,
            0.948,
            f"CryoSat-2 Baseline-{baseline}   Cycle {cycle}: "
            f"{start:%d-%b-%Y} to {last_day:%d-%b-%Y}",
            fontsize=10,
        ),
    ]
    if note:
        annotations.append(Annotation(0.22, 0.928, note, fontsize=8, color="dimgray"))
    return annotations


def _display_stats(stats: dict, n_key: str, vals: np.ndarray) -> dict:
    """statistics drawn on a float map: those of a statistics row (of all the values)"""
    valid = vals[np.isfinite(vals)].astype(np.float64)
    return {
        "nvals": stats[n_key],
        "min": stats["min"],
        "max": stats["max"],
        "mean": stats["mean"],
        "median": stats["median"],
        "std": stats["std"],
        "mad": float(np.mean(np.abs(valid - stats["mean"]))) if valid.size else 0.0,
    }


def prepare_plot_job(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    cfg: CsqaConfig,
    param: ParameterConfig,
    variant: VariantDef,
    mode: str,
    area: AreaConfig,
    lats: np.ndarray,
    lons: np.ndarray,
    vals: np.ndarray,
    stats: dict,
    cycle: int,
    cycle_bounds: tuple[datetime, datetime],
    baseline: str,
    out_path: str,
    scale: ColourScale | None = None,
) -> PlotJob:
    """Prepare the map plot of a parameter selection

    Args:
        cfg (CsqaConfig): CSQA config
        param (ParameterConfig): parameter
        variant (VariantDef): parameter variant plotted
        mode (str): acquisition mode selection ('' if none)
        area (AreaConfig): area plotted
        lats (np.ndarray): latitudes of the selected records
        lons (np.ndarray): longitudes of the selected records
        vals (np.ndarray): values of the selected records (NaN for missing)
        stats (dict): statistics of the selection, from stats.float_stats() or
                      stats.flag_stats(), drawn on the map
        cycle (int): cycle number
        cycle_bounds (tuple[datetime,datetime]): cycle start (inclusive) and end (exclusive)
        baseline (str): product baseline
        out_path (str): output plot file path
        scale (ColourScale|None): colour scale of float parameters (default: the parameter's
                                  default colour scale)

    Returns:
        PlotJob
    """
    finite = np.isfinite(vals)
    n_valid = int(np.count_nonzero(finite))
    n_missing = vals.size - n_valid
    step = decimation_step(n_valid, cfg.max_points)
    missing_step = decimation_step(n_missing, MAX_MISSING_POINTS)
    if step == missing_step:
        points: slice | np.ndarray = slice(None, None, step)
    else:
        points = np.concatenate(
            (np.flatnonzero(finite)[::step], np.flatnonzero(~finite)[::missing_step])
        )

    data_set: dict = {
        "name": variant.display_variable,
        # compact copies, so a job does not keep the full resolution arrays alive
        "lats": np.ascontiguousarray(lats[points]),
        "lons": np.ascontiguousarray(lons[points]),
        "vals": np.ascontiguousarray(vals[points]),
        "units": param.variant_units(variant) or "no units",
    }
    if vals.size and missing_step > 1:
        # % of all records (the missing values plotted are a sparser subsample)
        data_set["bad_data_percents"] = {
            "valid": 100.0 * n_valid / vals.size,
            "nan": 100.0 * n_missing / vals.size,
        }
    if param.type == "flag":
        data_set["flag_values"] = [f.value for f in param.flags]
        data_set["flag_names"] = [f.name for f in param.flags]
        if all(f.color for f in param.flags):
            data_set["flag_colors"] = [f.color for f in param.flags]
        # percentages of all records, not just those plotted
        data_set["flag_percents"] = [stats["pct"][f.key] for f in param.flags]
    else:
        if scale is None:
            scale = param.colour_scales[0]
        # a variant's own colour range / colormap override the parameter's
        plot_range = variant.plot_range or scale.range
        data_set["cmap_name"] = variant.cmap or scale.cmap
        data_set["cmap_log"] = scale.log if variant.plot_log is None else variant.plot_log
        if plot_range is not None:
            data_set["min_plot_range"] = plot_range[0]
            data_set["max_plot_range"] = plot_range[1]
        # statistics of all records, not just those plotted
        data_set["stats"] = _display_stats(stats, "n_valid", vals)

    # note line: the colour scale (when not the default) and any subsampling of the map
    notes = []
    if scale is not None and scale.file_suffix:
        notes.append(f"{scale.name} colour scale.")
    if step > 1:
        notes.append(
            f"Map shows 1 in {step} valid records. Statistics use all {n_valid:,} valid records"
        )
    annotations = _annotations(
        plot_title(param, variant, mode, cfg), cycle, cycle_bounds, baseline, " ".join(notes)
    )

    return PlotJob(
        out_path=out_path,
        polarplot_area=area.polarplot_area,
        data_set=data_set,
        annotations=annotations,
        image_format=cfg.image_format,
        dpi=cfg.dpi,
        webp_quality=cfg.webp_quality,
        step=step,
    )


def prepare_grid_plot_job(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    cfg: CsqaConfig,
    param: ParameterConfig,
    variant: VariantDef,
    mode: str,
    area: AreaConfig,
    gridded: GriddedSelection,
    stat: GridStatistic,
    stats: dict,
    cycle: int,
    cycle_bounds: tuple[datetime, datetime],
    baseline: str,
    out_path: str,
) -> PlotJob:
    """Prepare the gridded map of a statistic of a parameter selection's grid cells

    Args:
        cfg (CsqaConfig): CSQA config
        param (ParameterConfig): parameter (with a grid)
        variant (VariantDef): parameter variant gridded
        mode (str): acquisition mode selection ('' if none)
        area (AreaConfig): area gridded
        gridded (GriddedSelection): the selection's gridded measurements (with data)
        stat (GridStatistic): statistic of each cell mapped
        stats (dict): statistics of the cell values (a grid statistics row)
        cycle (int): cycle number
        cycle_bounds (tuple[datetime,datetime]): cycle start (inclusive) and end (exclusive)
        baseline (str): product baseline
        out_path (str): output plot file path

    Returns:
        PlotJob
    """
    assert param.grid is not None
    lats, lons = cell_centres_latlon(gridded)
    vals = gridded.values[stat.id].astype(np.float32)
    data_set: dict = {
        # ie radar_freeboard_20_ku cell median
        "name": f"{variant.display_variable} cell {stat.id}",
        "lats": lats.astype(np.float32),
        "lons": lons.astype(np.float32),
        "vals": vals,
        "units": stat.units if stat.units else ("count" if stat.id == "count" else "no units"),
        "grid": grid_image(gridded, stat.id),
        "cmap_name": stat.cmap,
        "cmap_log": stat.log,
        # the cells cover the area: no other area mask
        "apply_area_mask_to_data": False,
        # statistics of the cell values
        "stats": _display_stats(stats, "n_cells", vals),
    }
    if stat.range is not None:
        data_set["min_plot_range"] = stat.range[0]
        data_set["max_plot_range"] = stat.range[1]
    what = (
        "number of measurements"
        if stat.id == "count"
        else f"{stat.name.lower()} of the measurements"
    )
    min_count = param.grid.min_count
    note = (
        f"{param.grid.label}: {what} in each cell"
        f"{f' (cells with {min_count}+)' if min_count > 1 else ''}. "
        f"{stats['n_cells']:,} cells, {gridded.n_records:,} measurements"
    )
    return PlotJob(
        out_path=out_path,
        polarplot_area=area.polarplot_area,
        data_set=data_set,
        annotations=_annotations(
            plot_title(param, variant, mode, cfg), cycle, cycle_bounds, baseline, note
        ),
        image_format=cfg.image_format,
        dpi=cfg.dpi,
        webp_quality=cfg.webp_quality,
        step=1,
    )


def render_plot_job(job: PlotJob) -> int:
    """Render a map plot and its thumbnail

    Args:
        job (PlotJob): prepared plot

    Returns:
        int: decimation step used for the plotted points (1 = every record plotted)
    """
    os.makedirs(os.path.dirname(job.out_path), exist_ok=True)
    tmp_path = f"{job.out_path}.tmp.{job.image_format}"
    # Polarplot prints progress messages to stdout: keep batch logs readable
    with contextlib.redirect_stdout(io.StringIO()):
        Polarplot(job.polarplot_area).plot_points(
            job.data_set,
            annotation_list=job.annotations,
            output_file=tmp_path,
            image_format=job.image_format,
            dpi=job.dpi,
            webp_settings=(job.webp_quality, 6),
        )
    os.replace(tmp_path, job.out_path)
    save_thumbnail(job.out_path)
    log.info(
        "saved %s (%d points plotted, step %d)", job.out_path, job.data_set["vals"].size, job.step
    )
    return job.step
