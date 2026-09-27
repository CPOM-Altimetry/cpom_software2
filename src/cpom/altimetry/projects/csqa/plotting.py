"""cpom.altimetry.projects.csqa.plotting

Map plots of CSQA parameters using cpom.areas.area_plot.Polarplot.

Large selections are drawn as a regular along-track subsample (every Nth record) limited to
the configured maximum number of points, while the statistics drawn on the plot (and the flag
percentages) are always those of every record.

Plots are prepared (prepare_plot_job) in the process holding the cycle's data, then rendered
(render_plot_job) either in the same process or in a pool of plot worker processes.

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
    ParameterConfig,
    VariantDef,
)

matplotlib.use("Agg")  # non-interactive backend for batch processing

# pylint: disable=wrong-import-position
from cpom.areas.area_plot import Annotation, Polarplot  # noqa: E402

log = logging.getLogger(__name__)

THUMBNAIL_WIDTH = 360  # pixels


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
        label = cfg.mode_labels.get(mode, mode)
        title += f", {label}" if mode == "all" else f", {label} mode"
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
    step = decimation_step(vals.size, cfg.max_points)

    data_set: dict = {
        "name": variant.variable,
        # compact copies, so a job does not keep the full resolution arrays alive
        "lats": np.ascontiguousarray(lats[::step]),
        "lons": np.ascontiguousarray(lons[::step]),
        "vals": np.ascontiguousarray(vals[::step]),
        "units": param.units if param.units else "no units",
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
        data_set["cmap_name"] = scale.cmap
        data_set["cmap_log"] = scale.log
        if scale.range is not None:
            data_set["min_plot_range"] = scale.range[0]
            data_set["max_plot_range"] = scale.range[1]
        # statistics of all records, not just those plotted
        valid = vals[np.isfinite(vals)].astype(np.float64)
        data_set["stats"] = {
            "nvals": stats["n_valid"],
            "min": stats["min"],
            "max": stats["max"],
            "mean": stats["mean"],
            "median": stats["median"],
            "std": stats["std"],
            "mad": float(np.mean(np.abs(valid - stats["mean"]))),
        }

    start, end = cycle_bounds
    last_day = end - timedelta(seconds=1)
    annotations = [
        Annotation(
            0.22,
            0.972,
            plot_title(param, variant, mode, cfg),
            fontsize=13,
            fontweight="bold",
        ),
        Annotation(
            0.22,
            0.948,
            f"CryoSat-2 Baseline-{baseline}   Cycle {cycle}: "
            f"{start:%d-%b-%Y} to {last_day:%d-%b-%Y}",
            fontsize=10,
        ),
    ]
    # note line: the colour scale (when not the default) and any subsampling of the map
    notes = []
    if scale is not None and scale.file_suffix:
        notes.append(f"{scale.name} colour scale.")
    if step > 1:
        notes.append(
            f"Map shows 1 in {step} records. Statistics use all {stats['n_valid']:,} valid records"
        )
    if notes:
        annotations.append(Annotation(0.22, 0.928, " ".join(notes), fontsize=8, color="dimgray"))

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
