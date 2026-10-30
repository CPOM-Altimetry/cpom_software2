"""cpom.altimetry.projects.csqa.product_files

Find CryoSat-2 L2 (GDR-A) and L2i product files covering a time range for a product baseline.

Product filenames have the form:
    CS_<SSSS>_<TTTTTTTTTT>_<YYYYMMDDTHHMMSS>_<YYYYMMDDTHHMMSS>_<B><VVV>.nc
where SSSS is the processing stage (ie OFFL, LTA_), TTTTTTTTTT the file type
(ie SIR_GDR_2_, SIR_LRMI2_), B the baseline character and VVV the version, for example:
    CS_OFFL_SIR_GDR_2__20260801T002509_20260801T020423_F001.nc
    CS_LTA__SIR_SINI2__20110101T001140_20110101T001303_E001.nc

Archive directories are organised as <dir>/<YYYY>/<MM>/
"""

import glob
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta

from cpom.altimetry.projects.csqa.csqa_config import ProductConfig

log = logging.getLogger(__name__)

FILENAME_RE = re.compile(
    r"^CS_(?P<stage>[A-Z_]{4})_(?P<file_type>[A-Z0-9_]{10})_"
    r"(?P<start>\d{8}T\d{6})_(?P<stop>\d{8}T\d{6})_(?P<baseline>[A-Z])(?P<version>\d{3})\.nc$"
)


@dataclass(frozen=True)
class ProductFile:
    """A CryoSat-2 product file and the metadata parsed from its name"""

    path: str
    stage: str
    file_type: str
    start: datetime
    stop: datetime
    baseline: str
    version: int

    @property
    def name(self) -> str:
        """file name without directory"""
        return os.path.basename(self.path)


def parse_product_filename(path: str) -> ProductFile | None:
    """Parse a CryoSat-2 product file name

    Args:
        path (str): path of product file

    Returns:
        ProductFile | None: None if the name is not a CryoSat-2 product name
    """
    match = FILENAME_RE.match(os.path.basename(path))
    if match is None:
        return None
    return ProductFile(
        path=path,
        stage=match["stage"],
        file_type=match["file_type"],
        start=datetime.strptime(match["start"], "%Y%m%dT%H%M%S"),
        stop=datetime.strptime(match["stop"], "%Y%m%dT%H%M%S"),
        baseline=match["baseline"],
        version=int(match["version"]),
    )


def months_between(start: datetime, end: datetime) -> list[tuple[int, int]]:
    """(year, month) of every month overlapping [start, end]"""
    months = []
    year, month = start.year, start.month
    while (year, month) <= (end.year, end.month):
        months.append((year, month))
        month += 1
        if month > 12:
            year, month = year + 1, 1
    return months


def select_latest_versions(
    files: list[ProductFile], stage_preference: list[str] | None = None
) -> list[ProductFile]:
    """Remove duplicate granules, keeping the highest version of each.

    A granule is identified by its file type and start time. For equal versions the
    processing stage earliest in stage_preference is preferred.

    Args:
        files (list[ProductFile]): product files (of a single baseline)
        stage_preference (list[str]|None): preferred processing stages, ie ["LTA_", "OFFL"]

    Returns:
        list[ProductFile]: files sorted by start time
    """
    stage_preference = stage_preference or []

    def rank(pfile: ProductFile) -> tuple[int, int]:
        stage_rank = (
            len(stage_preference) - stage_preference.index(pfile.stage)
            if pfile.stage in stage_preference
            else 0
        )
        return pfile.version, stage_rank

    best: dict[tuple[str, datetime], ProductFile] = {}
    for pfile in files:
        key = (pfile.file_type, pfile.start)
        if key not in best:
            best[key] = pfile
            continue
        log.debug("duplicate granule %s and %s", best[key].name, pfile.name)
        if rank(pfile) > rank(best[key]):
            best[key] = pfile
    return sorted(best.values(), key=lambda f: (f.start, f.file_type))


def find_product_files(
    product: ProductConfig,
    start: datetime,
    end: datetime,
    baseline: str,
    stage_preference: list[str] | None = None,
) -> list[ProductFile]:
    """Find the product files of a baseline containing data within [start, end)

    Args:
        product (ProductConfig): product archive configuration
        start (datetime): start of time range (inclusive)
        end (datetime): end of time range (exclusive)
        baseline (str): baseline character, ie 'F'
        stage_preference (list[str]|None): preferred processing stages for duplicates

    Returns:
        list[ProductFile]: files overlapping the time range, sorted by start time
    """
    # a file starting before the range may be stored in the previous month's directory
    months = months_between(start - timedelta(days=1), end)

    found = []
    for product_dir in product.dirs:
        for year, month in months:
            month_dir = os.path.join(product_dir, f"{year:04d}", f"{month:02d}")
            if not os.path.isdir(month_dir):
                continue
            for path in glob.glob(os.path.join(month_dir, product.file_glob)):
                pfile = parse_product_filename(path)
                if pfile is None:
                    log.debug("ignoring unrecognised file name %s", path)
                    continue
                if pfile.baseline != baseline:
                    continue
                if pfile.stop < start or pfile.start >= end:
                    continue
                found.append(pfile)

    return select_latest_versions(found, stage_preference)


def coverage_days(files: list[ProductFile], start: datetime, end: datetime) -> float:
    """Number of days within [start, end) covered by the time spans of files
    (overlapping file time spans are only counted once)

    Args:
        files (list[ProductFile]): product files
        start (datetime): start of time range
        end (datetime): end of time range

    Returns:
        float: days covered
    """
    spans = sorted((max(f.start, start), min(f.stop, end)) for f in files)
    total = timedelta(0)
    cur_start: datetime | None = None
    cur_end: datetime | None = None
    for span_start, span_end in spans:
        if span_end <= span_start:
            continue
        if cur_end is None or span_start > cur_end:
            if cur_end is not None and cur_start is not None:
                total += cur_end - cur_start
            cur_start, cur_end = span_start, span_end
        else:
            cur_end = max(cur_end, span_end)
    if cur_end is not None and cur_start is not None:
        total += cur_end - cur_start
    return total / timedelta(days=1)
