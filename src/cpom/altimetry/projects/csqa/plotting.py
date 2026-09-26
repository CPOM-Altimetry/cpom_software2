"""cpom.altimetry.projects.csqa.plotting

Map plots of CSQA parameters using cpom.areas.area_plot.Polarplot.

Large selections are drawn as a regular along-track subsample (every Nth record) limited to
the configured maximum number of points, while the statistics drawn on the plot (and the flag
percentages) are always calculated from every record.

A thumbnail of each plot is saved in a 'thumbs' sub-directory of the plot's directory.
"""

import contextlib
import io
import logging
import math
import os
from datetime import datetime, timedelta

import matplotlib
import numpy as np
from PIL import Image

from cpom.altimetry.projects.csqa.csqa_config import (
    AreaConfig,
    CsqaConfig,
    ParameterConfig,
    VariantDef,
)

matplotlib.use("Agg")  # non-interactive backend for batch processing

# pylint: disable=wrong-import-position
from cpom.areas.area_plot import Annotation, Polarplot  # noqa: E402

log = logging.getLogger(__name__)

THUMBNAIL_WIDTH = 360  # pixels


def plot_filename(param_id: str, variant_id: str, mode: str, area_id: str, fmt: str) -> str:
    """Name of a parameter's map plot file, ie backscatter_rtk1_sar_global.webp

    Args:
        param_id (str): parameter id
        variant_id (str): variant id ('' if none)
        mode (str): mode selection ('' if none)
        area_id (str): area id
        fmt (str): image format / file extension

    Returns:
        str: file name
    """
    return "_".join(part for part in (param_id, variant_id, mode, area_id) if part) + f".{fmt}"


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


def plot_parameter_map(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    cfg: CsqaConfig,
    param: ParameterConfig,
    variant: VariantDef,
    mode: str,
    area: AreaConfig,
    lats: np.ndarray,
    lons: np.ndarray,
    vals: np.ndarray,
    cycle: int,
    cycle_bounds: tuple[datetime, datetime],
    baseline: str,
    out_path: str,
) -> int:
    """Plot a map of a parameter selection and save it to out_path

    Args:
        cfg (CsqaConfig): CSQA config
        param (ParameterConfig): parameter
        variant (VariantDef): parameter variant plotted
        mode (str): acquisition mode selection ('' if none)
        area (AreaConfig): area plotted
        lats (np.ndarray): latitudes of the selected records
        lons (np.ndarray): longitudes of the selected records
        vals (np.ndarray): values of the selected records (NaN for missing)
        cycle (int): cycle number
        cycle_bounds (tuple[datetime,datetime]): cycle start (inclusive) and end (exclusive)
        baseline (str): product baseline
        out_path (str): output plot file path

    Returns:
        int: decimation step used for the plotted points (1 = every record plotted)
    """
    step = decimation_step(vals.size, cfg.max_points)
    valid_vals = vals[np.isfinite(vals)]

    data_set: dict = {
        "name": variant.variable,
        "lats": lats[::step],
        "lons": lons[::step],
        "vals": vals[::step],
        "units": param.units if param.units else "no units",
        "stats_vals": valid_vals,
    }
    if param.type == "flag":
        data_set["flag_values"] = [f.value for f in param.flags]
        data_set["flag_names"] = [f.name for f in param.flags]
        if all(f.color for f in param.flags):
            data_set["flag_colors"] = [f.color for f in param.flags]
    else:
        data_set["cmap_name"] = param.cmap
        if param.plot_range is not None:
            data_set["min_plot_range"] = param.plot_range[0]
            data_set["max_plot_range"] = param.plot_range[1]

    start, end = cycle_bounds
    last_day = end - timedelta(seconds=1)
    annotations = [
        Annotation(
            0.22, 0.972, plot_title(param, variant, mode, cfg), fontsize=13, fontweight="bold"
        ),
        Annotation(
            0.22,
            0.948,
            f"CryoSat-2 Baseline-{baseline}   Cycle {cycle}: "
            f"{start:%d-%b-%Y} to {last_day:%d-%b-%Y}",
            fontsize=10,
        ),
    ]
    if step > 1:
        annotations.append(
            Annotation(
                0.22,
                0.928,
                f"Map shows 1 in {step} records. Statistics use all {valid_vals.size:,} "
                "valid records",
                fontsize=8,
                color="dimgray",
            )
        )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp_path = f"{out_path}.tmp.{cfg.image_format}"
    # Polarplot prints progress messages to stdout: keep batch logs readable
    with contextlib.redirect_stdout(io.StringIO()):
        Polarplot(area.polarplot_area).plot_points(
            data_set,
            annotation_list=annotations,
            output_file=tmp_path,
            image_format=cfg.image_format,
            dpi=cfg.dpi,
            webp_settings=(cfg.webp_quality, 6),
        )
    os.replace(tmp_path, out_path)
    save_thumbnail(out_path)
    log.info("saved %s (%d points plotted, step %d)", out_path, data_set["vals"].size, step)
    return step
