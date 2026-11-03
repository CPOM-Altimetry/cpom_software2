"""pytests of cpom.altimetry.projects.csqa.product_files"""

from datetime import datetime

from cpom.altimetry.projects.csqa.csqa_config import ProductConfig
from cpom.altimetry.projects.csqa.product_files import (
    coverage_days,
    find_product_files,
    months_between,
    parse_product_filename,
    select_latest_versions,
)


def test_parse_product_filename():
    """GDR-A and L2i file names are parsed, others rejected"""
    pfile = parse_product_filename(
        "/a/b/CS_OFFL_SIR_GDR_2__20260630T222536_20260701T000451_E001.nc"
    )
    assert pfile is not None
    assert pfile.stage == "OFFL"
    assert pfile.file_type == "SIR_GDR_2_"
    assert pfile.start == datetime(2026, 6, 30, 22, 25, 36)
    assert pfile.stop == datetime(2026, 7, 1, 0, 4, 51)
    assert (pfile.baseline, pfile.version) == ("E", 1)

    pfile = parse_product_filename("CS_LTA__SIR_SINI2__20110101T001140_20110101T001303_F012.nc")
    assert pfile is not None
    assert (pfile.stage, pfile.file_type, pfile.baseline, pfile.version) == (
        "LTA_",
        "SIR_SINI2_",
        "F",
        12,
    )
    assert (
        parse_product_filename("CS_OFFL_SIR_LRMI2__20260801T000030_20260801T000038_F001.HDR")
        is None
    )
    assert parse_product_filename("notes.txt") is None


def test_months_between():
    """month directories spanning a time range"""
    assert months_between(datetime(2010, 11, 30), datetime(2011, 1, 2)) == [
        (2010, 11),
        (2010, 12),
        (2011, 1),
    ]


def test_select_latest_versions():
    """highest version, then preferred stage, is kept for duplicate granules"""
    names = [
        "CS_OFFL_SIR_GDR_2__20260801T002509_20260801T020423_F001.nc",
        "CS_OFFL_SIR_GDR_2__20260801T002509_20260801T020423_F002.nc",
        "CS_OFFL_SIR_GDR_2__20260801T020423_20260801T034338_F001.nc",
        "CS_LTA__SIR_GDR_2__20260801T020423_20260801T034338_F001.nc",
    ]
    files = [parse_product_filename(n) for n in names]
    selected = select_latest_versions(files, ["LTA_", "OFFL"])  # type: ignore[arg-type]
    assert [f.name for f in selected] == [names[1], names[3]]


def test_find_product_files(tmp_path):
    """files are found across month directories, filtered by baseline and cycle overlap"""
    names = {
        "2026/06": ["CS_OFFL_SIR_GDR_2__20260630T222536_20260701T000451_F001.nc"],
        "2026/07": [
            "CS_OFFL_SIR_GDR_2__20260701T000451_20260701T014406_F001.nc",
            "CS_OFFL_SIR_GDR_2__20260701T000451_20260701T014406_E001.nc",
            "CS_OFFL_SIR_GDR_2__20260731T235000_20260801T013000_F001.nc",
        ],
        "2026/08": ["CS_OFFL_SIR_GDR_2__20260801T013000_20260801T031000_F001.nc"],
    }
    for month, month_names in names.items():
        (tmp_path / month).mkdir(parents=True)
        for name in month_names:
            (tmp_path / month / name).touch()
    product = ProductConfig("GDR-A", "GDR-A", [str(tmp_path)], "CS_*_SIR_GDR_2__*.nc")

    files = find_product_files(product, datetime(2026, 7, 1), datetime(2026, 8, 1), "F")
    assert [f.name for f in files] == [
        "CS_OFFL_SIR_GDR_2__20260630T222536_20260701T000451_F001.nc",
        "CS_OFFL_SIR_GDR_2__20260701T000451_20260701T014406_F001.nc",
        "CS_OFFL_SIR_GDR_2__20260731T235000_20260801T013000_F001.nc",
    ]
    files = find_product_files(product, datetime(2026, 7, 1), datetime(2026, 8, 1), "E")
    assert len(files) == 1


def test_coverage_days():
    """overlapping file spans are counted once and clipped to the range"""
    files = [
        parse_product_filename(n)
        for n in (
            "CS_OFFL_SIR_GDR_2__20260731T120000_20260801T120000_F001.nc",
            "CS_OFFL_SIR_GDR_2__20260801T060000_20260802T000000_F001.nc",
            "CS_OFFL_SIR_GDR_2__20260803T000000_20260803T120000_F001.nc",
        )
    ]
    days = coverage_days(files, datetime(2026, 8, 1), datetime(2026, 9, 1))  # type: ignore
    assert abs(days - 1.5) < 1e-9
