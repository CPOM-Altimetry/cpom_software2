"""pytests of CSQA cycle processing (cpom.altimetry.projects.csqa.processing,
process_cycles and build_portal_index) using a real GDR-A product"""

import csv
import glob
import json
import os

import pytest

from cpom.altimetry.projects.csqa.csqa_config import load_config
from cpom.altimetry.projects.csqa.process_cycles import allocate_workers
from cpom.altimetry.projects.csqa.process_cycles import main as process_cycles_main
from cpom.altimetry.projects.csqa.tests.conftest import write_test_config

GDR_A_DIR = "/raid6/cpdata/SATS/RA/CRY/L2/GDR-A"
GDR_A_FILES = sorted(glob.glob(f"{GDR_A_DIR}/2026/08/CS_*_SIR_GDR_2__20260801T*_F001.nc"))

pytestmark = [
    pytest.mark.requires_external_data,
    pytest.mark.skipif(not GDR_A_FILES, reason="GDR-A test products not available"),
]


@pytest.fixture
def one_file_config(tmp_path):
    """config reading a GDR-A archive containing a single product file"""
    month_dir = tmp_path / "GDR-A" / "2026" / "08"
    month_dir.mkdir(parents=True)
    os.symlink(GDR_A_FILES[0], month_dir / os.path.basename(GDR_A_FILES[0]))
    return write_test_config(tmp_path, {"GDR-A": [str(tmp_path / "GDR-A")]})


def test_process_cycle(one_file_config):  # pylint: disable=redefined-outer-name
    """statistics, plots and the portal index are produced for a cycle"""
    cfg = load_config(one_file_config)
    # maps rendered by a pool of 2 plot worker processes
    status = process_cycles_main(
        ["-c", "193", "-b", "F", "--config", one_file_config, "--areas", "north_polar"]
        + ["south_polar", "-p", "acquisition_mode", "--plot_workers", "2"]
    )
    assert status == 0
    status = process_cycles_main(
        ["-c", "193", "-b", "F", "--config", one_file_config, "-p", "backscatter", "--no_plots"]
    )
    assert status == 0

    cdir = os.path.join(cfg.output_dir, "baseline_F", "cycles", "cycle_193")
    with open(os.path.join(cdir, "cycle_info.json"), encoding="utf-8") as fh:
        info = json.load(fh)
    assert info["products"]["GDR-A"]["n_files"] == 1
    assert set(info["parameters"]) == {"acquisition_mode", "backscatter"}

    # flag statistics and plot of the processed area only
    with open(os.path.join(cdir, "stats", "acquisition_mode.json"), encoding="utf-8") as fh:
        mode_stats = json.load(fh)
    assert [r["area"] for r in mode_stats["rows"]] == ["north_polar", "south_polar"]
    for row in mode_stats["rows"]:
        assert row["n_valid"] > 0
        assert abs(sum(row["pct"].values()) - 100.0) < 0.01
        assert row["plot_step"] == 1
        assert os.path.isfile(os.path.join(cdir, "plots", "acquisition_mode", row["plot"]))
        assert os.path.isfile(
            os.path.join(cdir, "plots", "acquisition_mode", "thumbs", row["plot"])
        )

    # float statistics per retracker, mode and area. Retracker 2 is only used in LRM mode
    with open(os.path.join(cdir, "stats", "backscatter.json"), encoding="utf-8") as fh:
        sig0_stats = json.load(fh)
    rows = {(r["area"], r["variant"], r["mode"]): r for r in sig0_stats["rows"]}
    assert len(rows) == 3 * 3 * 4
    assert rows[("global", "rtk1", "all")]["n_valid"] > 0
    assert rows[("global", "rtk2", "sar")]["n_valid"] == 0
    assert rows[("global", "rtk2", "sar")]["mean"] is None
    assert rows[("global", "rtk1", "all")]["n_records"] == sum(
        rows[("global", "rtk1", m)]["n_records"] for m in ("lrm", "sar", "sarin")
    )
    assert all(r["plot"] is None for r in rows.values())

    # portal index
    with open(os.path.join(cfg.output_dir, "manifest.json"), encoding="utf-8") as fh:
        manifest = json.load(fh)
    assert [b["id"] for b in manifest["baselines"]] == ["F"]
    sig0 = next(p for p in manifest["parameters"] if p["id"] == "backscatter")
    assert sig0["variants"][2]["mode_descriptions"]["lrm"] == "OCOG retracker"
    assert manifest["baselines"][0]["cycles"][0]["cycle"] == 193
    with open(
        os.path.join(cfg.output_dir, "baseline_F", "timeseries", "backscatter.csv"),
        encoding="utf-8",
    ) as fh:
        ts_rows = list(csv.DictReader(fh))
    assert len(ts_rows) == 36
    assert ts_rows[0]["cycle"] == "193" and ts_rows[0]["start_date"] == "2026-07-26"

    # maps no configured selection or colour scale produces are removed when re-plotting
    mode_plots = os.path.join(cdir, "plots", "acquisition_mode")
    stale = "acquisition_mode_north_polar_oldscale.webp"
    for directory in (mode_plots, os.path.join(mode_plots, "thumbs")):
        with open(os.path.join(directory, stale), "wb"):
            pass
    status = process_cycles_main(
        ["-c", "193", "-b", "F", "--config", one_file_config, "--areas", "north_polar"]
        + ["-p", "acquisition_mode"]
    )
    assert status == 0
    assert not os.path.exists(os.path.join(mode_plots, stale))
    assert not os.path.exists(os.path.join(mode_plots, "thumbs", stale))
    assert os.path.isfile(os.path.join(mode_plots, "acquisition_mode_south_polar.webp"))

    # quality flag word: statistics of every bit, in every mode (no maps)
    status = process_cycles_main(
        ["-c", "193", "-b", "F", "--config", one_file_config, "--areas", "north_polar"]
        + ["-p", "quality_flags", "--no_plots"]
    )
    assert status == 0
    with open(os.path.join(cdir, "stats", "quality_flags.json"), encoding="utf-8") as fh:
        qf_rows = json.load(fh)["rows"]
    assert len(qf_rows) == 31 * 4
    for row in qf_rows:
        assert abs(row["pct"]["set"] + row["pct"]["not_set"] - 100.0) < 1e-3
    qf = {(r["variant"], r["mode"]): r for r in qf_rows}
    assert qf[("b21", "sar")]["pct"]["set"] < 100.0  # backscatter (retracker 1) error
    assert qf[("b19", "sar")]["pct"]["set"] == 100.0  # retracker 3 backscatter unused in SAR

    # update mode skips unchanged inputs
    status = process_cycles_main(
        ["-c", "193", "-b", "F", "--config", one_file_config, "-p", "backscatter", "--no_plots"]
        + ["--update"]
    )
    assert status == 0
    with open(os.path.join(cdir, "stats", "backscatter.json"), encoding="utf-8") as fh:
        assert json.load(fh)["processed_at"] == sig0_stats["processed_at"]


def test_no_data(one_file_config):  # pylint: disable=redefined-outer-name
    """cycles without input files produce no outputs"""
    cfg = load_config(one_file_config)
    assert process_cycles_main(["-c", "100", "--config", one_file_config, "--no_index"]) == 0
    assert not os.path.exists(cfg.output_dir)


def test_allocate_workers():
    """worker processes are shared between the cycles to process and their plot workers"""
    # few cycles: many plot workers each (limited by the plots of a cycle)
    assert allocate_workers(64, 2, None, 42) == (2, 32)
    assert allocate_workers(64, 1, None, 42) == (1, 42)
    # many cycles: one process per cycle
    assert allocate_workers(64, 390, None, 42) == (64, 1)
    assert allocate_workers(8, 3, None, 42) == (3, 2)
    # explicit plot workers, and nothing to process
    assert allocate_workers(64, 390, 4, 42) == (64, 4)
    assert allocate_workers(64, 0, None, 42) == (1, 42)
