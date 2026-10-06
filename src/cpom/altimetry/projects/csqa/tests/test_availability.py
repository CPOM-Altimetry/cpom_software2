"""pytests of cpom.altimetry.projects.csqa.availability: files per day of the most recent
input products"""

from datetime import datetime

from cpom.altimetry.projects.csqa.availability import availability, product_availability
from cpom.altimetry.projects.csqa.csqa_config import load_config
from cpom.altimetry.projects.csqa.tests.conftest import write_test_config


def _touch(directory, name):
    """create an empty product file in its <YYYY>/<MM> directory"""
    month_dir = directory / name[19:23] / name[23:25]
    month_dir.mkdir(parents=True, exist_ok=True)
    (month_dir / name).touch()


def test_product_availability(tmp_path):
    """files are counted per acquisition day over the days ending with the latest data"""
    gdr = tmp_path / "GDR-A"
    for name in (
        "CS_OFFL_SIR_GDR_2__20260801T002509_20260801T020423_F001.nc",
        "CS_OFFL_SIR_GDR_2__20260801T020423_20260801T034338_F001.nc",
        # a later version of the same granule is counted once
        "CS_OFFL_SIR_GDR_2__20260801T020423_20260801T034338_F002.nc",
        # spans midnight: counted on its start day, its hours shared between the days
        "CS_OFFL_SIR_GDR_2__20260803T230000_20260804T010000_F001.nc",
        # another baseline (not configured) is ignored
        "CS_OFFL_SIR_GDR_2__20260805T000000_20260805T010000_D001.nc",
    ):
        _touch(gdr, name)
    l2i = tmp_path / "L2I_SIN"
    _touch(l2i, "CS_OFFL_SIR_SINI2__20260802T010000_20260802T013000_F001.nc")
    _touch(l2i, "CS_OFFL_SIR_LRMI2__20260802T020000_20260802T030000_F001.nc")
    cfg = load_config(write_test_config(tmp_path, {"GDR-A": [str(gdr)], "L2I": [str(l2i)]}))
    now = datetime(2026, 9, 10)

    gdr_avail = product_availability(cfg, "GDR-A", now)
    assert gdr_avail["latest"]["file"].startswith("CS_OFFL_SIR_GDR_2__20260803T230000")
    assert gdr_avail["latest"]["stop"] == "2026-08-04T01:00:00Z"
    assert gdr_avail["latest"]["received"] is not None
    assert gdr_avail["window"] == {"start": "2026-07-05", "end": "2026-08-03"}
    assert gdr_avail["series"] == ["GDR-A"]
    daily = {d["date"]: d for d in gdr_avail["daily"]}
    assert len(daily) == 30
    assert daily["2026-08-01"]["files"] == {"GDR-A": 2}
    assert daily["2026-08-03"]["files"] == {"GDR-A": 1}
    assert daily["2026-08-03"]["hours"] == {"GDR-A": 1.0}
    assert daily["2026-08-02"]["files"] == {"GDR-A": 0}

    # L2i: a series per file type (mode), in the configured order
    l2i_avail = product_availability(cfg, "L2I", now)
    assert l2i_avail["series"] == ["LRM", "SARin"]
    last = l2i_avail["daily"][-1]
    assert last["date"] == "2026-08-02" and last["files"] == {"LRM": 1, "SARin": 1}
    assert last["hours"] == {"LRM": 1.0, "SARin": 0.5}

    # nothing in the search period
    assert product_availability(cfg, "GDR-A", datetime(2027, 6, 1))["latest"] is None
    summary = availability(cfg, now)
    assert summary["days"] == 30 and [p["id"] for p in summary["products"]] == ["GDR-A", "L2I"]
