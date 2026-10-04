"""pytests of CSQA cycle processing (cpom.altimetry.projects.csqa.processing,
process_cycles and build_portal_index) using a real GDR-A product"""

import csv
import glob
import json
import os

import numpy as np
import pytest
from netCDF4 import Dataset  # pylint: disable=no-name-in-module

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
    # 31 bits x (all + 3 modes + 6 mode surface selections)
    assert len(qf_rows) == 31 * 10
    for row in qf_rows:
        assert abs(row["pct"]["set"] + row["pct"]["not_set"] - 100.0) < 1e-3
    qf = {(r["variant"], r["mode"]): r for r in qf_rows}
    assert qf[("b21", "sar")]["pct"]["set"] < 100.0  # backscatter (retracker 1) error
    assert qf[("b19", "sar")]["pct"]["set"] == 100.0  # retracker 3 backscatter unused in SAR

    # mode surface selections: % set of the records of a mode over a surface type
    with Dataset(GDR_A_FILES[0]) as nc:
        words = np.ma.filled(nc["flag_prod_status_20_ku"][:], -1).astype(np.int64)
        modes = np.ma.filled(nc["flag_instr_mode_op_20_ku"][:], -1)
        surfaces = np.ma.filled(nc["surf_type_20_ku"][:], -1)
        lat = np.ma.filled(nc["lat_poca_20_ku"][:].astype(float), np.nan)
    for sel_id, mode, surface in (("lrm_land", 1, 3), ("lrm_ocean", 1, 0), ("sar_ocean", 2, 0)):
        sel = (modes == mode) & (surfaces == surface) & (lat >= 60) & (words >= 0)
        assert sel.any()
        expected = 100.0 * np.count_nonzero(words[sel] & 16777216) / np.count_nonzero(sel)
        assert qf[("b24", sel_id)]["pct"]["set"] == pytest.approx(expected, abs=1e-4)
        assert qf[("b24", sel_id)]["n_valid"] == np.count_nonzero(sel)

    # geophysical corrections (1 Hz): the sea state bias is not in Baseline-F products, which
    # must not stop the other corrections being read
    status = process_cycles_main(
        ["-c", "193", "-b", "F", "--config", one_file_config, "-p", "geophysical_corrections"]
        + ["--no_plots"]
    )
    assert status == 0
    with open(os.path.join(cdir, "stats", "geophysical_corrections.json"), encoding="utf-8") as fh:
        cor_stats = json.load(fh)
    assert not cor_stats["bad_files"]
    assert cor_stats["missing_variables"] == {"sea_state_bias_01_ku": 1}
    cors = {(r["area"], r["variant"]): r for r in cor_stats["rows"]}
    assert cors[("global", "ssb")]["n_valid"] == 0
    assert cors[("global", "dry")]["n_valid"] > 0
    assert -2.5 < cors[("global", "dry")]["median"] < -2.0
    # root mean square of the dry troposphere correction ~ its |mean|
    assert cors[("global", "dry")]["rms"] == pytest.approx(
        np.sqrt(cors[("global", "dry")]["mean"] ** 2 + cors[("global", "dry")]["std"] ** 2),
        rel=1e-5,
    )

    # update mode skips unchanged inputs
    status = process_cycles_main(
        ["-c", "193", "-b", "F", "--config", one_file_config, "-p", "backscatter", "--no_plots"]
        + ["--update"]
    )
    assert status == 0
    with open(os.path.join(cdir, "stats", "backscatter.json"), encoding="utf-8") as fh:
        assert json.load(fh)["processed_at"] == sig0_stats["processed_at"]


def test_freeboard_grid(one_file_config):  # pylint: disable=redefined-outer-name
    """freeboard: values with the freeboard error bit set are rejected, and the measurements
    of the polar areas are gridded into 10 km cells with maps and statistics of the cells"""
    cfg = load_config(one_file_config)
    args = ["-c", "193", "-b", "F", "--config", one_file_config, "-p", "freeboard"]
    status = process_cycles_main(
        args + ["--areas", "north_polar", "south_polar", "--plot_workers", "2"]
    )
    assert status == 0

    # expected numbers of measurements, read directly from the product
    with Dataset(GDR_A_FILES[0]) as nc:
        fb = np.ma.filled(nc["radar_freeboard_20_ku"][:].astype(float), np.nan)
        lat = np.ma.filled(nc["lat_20_ku"][:].astype(float), np.nan)
        fb_error = (np.ma.filled(nc["flag_prod_status_20_ku"][:], 0) & 65536) != 0
    south = (lat <= -60) & np.isfinite(fb)
    n_south, n_south_unflagged = int(south.sum()), int((south & ~fb_error).sum())
    assert 0 < n_south_unflagged < n_south

    cdir = os.path.join(cfg.output_dir, "baseline_F", "cycles", "cycle_193")
    pdir = os.path.join(cdir, "plots", "freeboard")
    with open(os.path.join(cdir, "stats", "freeboard.json"), encoding="utf-8") as fh:
        fb_stats = json.load(fh)
    rows = {(r["area"], r["variant"], r["mode"]): r for r in fb_stats["rows"]}
    assert rows[("south_polar", "unfiltered", "all")]["n_valid"] == n_south
    assert rows[("south_polar", "filtered", "all")]["n_valid"] == n_south_unflagged
    assert rows[("south_polar", "filtered", "all")]["min"] >= -0.3
    # every north polar freeboard value of this file has the freeboard error bit set
    assert rows[("north_polar", "filtered", "all")]["n_valid"] == 0
    assert rows[("north_polar", "unfiltered", "all")]["n_valid"] > 0

    assert fb_stats["grid"]["grid_areas"] == {
        "north_polar": "arctic",
        "south_polar": "antarctic_ocean",
    }
    grid_rows = {
        (r["area"], r["variant"], r["mode"], r["statistic"]): r for r in fb_stats["grid_rows"]
    }
    assert len(grid_rows) == 2 * 2 * 5
    count = grid_rows[("south_polar", "filtered", "all", "count")]
    median = grid_rows[("south_polar", "filtered", "all", "median")]
    assert count["n_records"] == median["n_records"] == n_south_unflagged
    # the mean number of measurements per cell x the number of cells = number of measurements
    assert count["mean"] * count["n_cells"] == pytest.approx(n_south_unflagged, rel=1e-5)
    assert 0 < median["n_cells"] < n_south_unflagged
    assert median["plot"] == "freeboard_filtered_all_south_polar_grid10km_median.webp"
    assert os.path.isfile(os.path.join(pdir, median["plot"]))
    assert os.path.isfile(os.path.join(pdir, "thumbs", median["plot"]))
    empty = grid_rows[("north_polar", "filtered", "all", "median")]
    assert empty["n_cells"] == 0 and empty["plot"] is None
    assert not os.path.exists(
        os.path.join(pdir, "freeboard_filtered_all_north_polar_grid10km_median.webp")
    )

    # gridded statistics timeseries and portal description of the grid
    with open(
        os.path.join(cfg.output_dir, "baseline_F", "timeseries", "freeboard_grid.csv"),
        encoding="utf-8",
    ) as fh:
        ts_rows = list(csv.DictReader(fh))
    assert len(ts_rows) == 20
    assert int(ts_rows[0]["n_cells"]) == 0 and ts_rows[0]["statistic"] == "median"
    with open(os.path.join(cfg.output_dir, "manifest.json"), encoding="utf-8") as fh:
        manifest = json.load(fh)
    fb_manifest = next(p for p in manifest["parameters"] if p["id"] == "freeboard")
    assert fb_manifest["grid"]["statistics"][0]["file_suffix"] == "grid10km_median"
    assert fb_manifest["variants"][0]["reject_bit"]["name"] == "freeboard_error"

    # update mode: nothing to do, unless the statistics predate the grid
    status = process_cycles_main(args + ["--areas", "north_polar", "south_polar", "--update"])
    assert status == 0
    with open(os.path.join(cdir, "stats", "freeboard.json"), encoding="utf-8") as fh:
        assert json.load(fh)["processed_at"] == fb_stats["processed_at"]
    del fb_stats["grid_rows"]
    with open(os.path.join(cdir, "stats", "freeboard.json"), "w", encoding="utf-8") as fh:
        json.dump(fb_stats, fh)
    status = process_cycles_main(
        args + ["--areas", "north_polar", "south_polar", "--update", "--no_plots"]
    )
    assert status == 0
    with open(os.path.join(cdir, "stats", "freeboard.json"), encoding="utf-8") as fh:
        updated = json.load(fh)
    assert len(updated["grid_rows"]) == 20
    # maps are kept when not plotting
    assert {(r["area"], r["variant"], r["statistic"]): r["plot"] for r in updated["grid_rows"]} == {
        (r["area"], r["variant"], r["statistic"]): r["plot"] for r in grid_rows.values()
    }


def test_sea_ice_thickness(one_file_config):  # pylint: disable=redefined-outer-name
    """sea ice thickness: only SAR/SARin values without the freeboard error bit are used (the
    products contain 0 in LRM mode and where the bit is set)"""
    cfg = load_config(one_file_config)
    args = ["-c", "193", "-b", "F", "--config", one_file_config, "-p", "sea_ice_thickness"]
    assert process_cycles_main(args + ["--no_plots"]) == 0

    with Dataset(GDR_A_FILES[0]) as nc:
        thk = np.ma.filled(nc["sea_ice_thickness_20_ku"][:].astype(float), np.nan)
        modes = np.ma.filled(nc["flag_instr_mode_op_20_ku"][:], 0)
        fb_error = (np.ma.filled(nc["flag_prod_status_20_ku"][:], 0) & 65536) != 0
    # the zeros excluded
    assert np.all(thk[modes == 1] == 0.0)
    assert np.all(thk[fb_error & np.isfinite(thk)] == 0.0)
    expected = np.isfinite(thk) & ~fb_error & np.isin(modes, [2, 3])

    stats_file = os.path.join(
        cfg.output_dir, "baseline_F", "cycles", "cycle_193", "stats", "sea_ice_thickness.json"
    )
    with open(stats_file, encoding="utf-8") as fh:
        thk_stats = json.load(fh)
    rows = {(r["area"], r["mode"]): r for r in thk_stats["rows"]}
    assert rows[("global", "all")]["n_valid"] == int(np.count_nonzero(expected))
    assert rows[("global", "all")]["n_valid"] == (
        rows[("global", "sar")]["n_valid"] + rows[("global", "sarin")]["n_valid"]
    )
    assert rows[("global", "all")]["median"] > 0.5
    assert len(thk_stats["grid_rows"]) == 2 * 5


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
