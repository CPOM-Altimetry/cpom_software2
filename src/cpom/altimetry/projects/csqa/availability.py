"""cpom.altimetry.projects.csqa.availability

Availability of the most recent input products, for the portal's Data Availability page: the
number of product files (and hours of data) per acquisition day over the most recent days of
data, and the latest data and file received.

For each product, the files of the configured baselines acquired in the last
availability:search_days days are found (the highest version of each granule), and the
availability:days days ending on the day of the latest acquisition are summarised. Products
with several file types (ie the L2i LRM, SAR and SARin products) are summarised per file type,
labelled by products:<id>:file_types.

Written to <output_dir>/availability.json by build_portal_index.
"""

import logging
import os
from datetime import datetime, timedelta, timezone

from cpom.altimetry.projects.csqa.csqa_config import CsqaConfig, ProductConfig
from cpom.altimetry.projects.csqa.product_files import (
    ProductFile,
    coverage_days,
    find_product_files,
)

log = logging.getLogger(__name__)

TIME_FMT = "%Y-%m-%dT%H:%M:%SZ"


def _series_label(product: ProductConfig, pfile: ProductFile, multi_baseline: bool) -> str:
    """name of the chart series of a file: its file type's label (and baseline if several)"""
    label = product.file_type_labels.get(pfile.file_type, pfile.file_type.strip("_"))
    return f"{label} (Baseline-{pfile.baseline})" if multi_baseline else label


def product_availability(cfg: CsqaConfig, product_id: str, now: datetime | None = None) -> dict:
    """Availability of a product's most recent files

    Args:
        cfg (CsqaConfig): CSQA config
        product_id (str): product id, ie 'GDR-A'
        now (datetime|None): current (UTC) time (default: now)

    Returns:
        dict: {id, long_name, latest: {file, start, stop, received} or None,
               window: {start, end} (dates) or None, series: [labels],
               daily: [{date, files: {label: count}, hours: {label: hours of data}}]}
    """
    product = cfg.products[product_id]
    now = now or datetime.now(timezone.utc).replace(tzinfo=None)
    search_start = now - timedelta(days=cfg.availability_search_days)
    files: list[ProductFile] = []
    for baseline in cfg.baselines:
        files += find_product_files(
            product, search_start, now + timedelta(days=1), baseline, cfg.stage_preference
        )
    if not files:
        log.info("availability: no %s files since %s", product_id, f"{search_start:%Y-%m-%d}")
        return {
            "id": product_id,
            "long_name": product.long_name,
            "latest": None,
            "window": None,
            "series": [],
            "daily": [],
        }

    latest = max(files, key=lambda f: (f.stop, f.start))
    try:
        received = datetime.fromtimestamp(os.path.getmtime(latest.path), timezone.utc)
        received_str: str | None = received.strftime(TIME_FMT)
    except OSError:
        received_str = None

    # the most recent days of data, ending on the day of the latest acquisition
    last_day = max(f.start for f in files).replace(hour=0, minute=0, second=0, microsecond=0)
    first_day = last_day - timedelta(days=cfg.availability_days - 1)
    window_files = [f for f in files if f.start >= first_day]
    multi_baseline = len({f.baseline for f in window_files}) > 1
    labels: dict[str, list[ProductFile]] = {}
    for pfile in sorted(window_files, key=lambda f: (f.file_type, f.baseline)):
        labels.setdefault(_series_label(product, pfile, multi_baseline), []).append(pfile)
    # series in the configured file type order
    order = list(product.file_type_labels.values())
    series = sorted(labels, key=lambda s: (order.index(s) if s in order else len(order), s))
    daily = []
    for n_day in range(cfg.availability_days):
        day = first_day + timedelta(days=n_day)
        next_day = day + timedelta(days=1)
        day_counts, day_hours = {}, {}
        for label in series:
            started = [f for f in labels[label] if day <= f.start < next_day]
            overlapping = [f for f in labels[label] if f.start < next_day and f.stop > day]
            day_counts[label] = len(started)
            day_hours[label] = round(24.0 * coverage_days(overlapping, day, next_day), 2)
        daily.append({"date": f"{day:%Y-%m-%d}", "files": day_counts, "hours": day_hours})
    return {
        "id": product_id,
        "long_name": product.long_name,
        "latest": {
            "file": latest.name,
            "start": latest.start.strftime(TIME_FMT),
            "stop": latest.stop.strftime(TIME_FMT),
            "received": received_str,
        },
        "window": {"start": f"{first_day:%Y-%m-%d}", "end": f"{last_day:%Y-%m-%d}"},
        "series": series,
        "daily": daily,
    }


def availability(cfg: CsqaConfig, now: datetime | None = None) -> dict:
    """Availability of every configured product (see product_availability)"""
    now = now or datetime.now(timezone.utc).replace(tzinfo=None)
    return {
        "generated_at": now.strftime(TIME_FMT),
        "days": cfg.availability_days,
        "data_latency_days": cfg.data_latency_days,
        "products": [product_availability(cfg, pid, now) for pid in cfg.products],
    }
