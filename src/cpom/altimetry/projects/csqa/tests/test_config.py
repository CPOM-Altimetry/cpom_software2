"""pytests of cpom.altimetry.projects.csqa.csqa_config"""

from datetime import datetime

import pytest
import yaml  # type: ignore[import-untyped]

from cpom.altimetry.projects.csqa.csqa_config import load_config, sanitize_key
from cpom.altimetry.projects.csqa.plotting import plot_filename
from cpom.altimetry.projects.csqa.processing import has_maps
from cpom.areas.area_plot import log_scale_ticks


def test_default_config(default_config):
    """the default config defines the initial parameters and areas"""
    cfg = default_config
    assert cfg.mission_start_date == datetime(2010, 10, 18)
    assert cfg.cycle_length_days == 30
    assert cfg.data_latency_days == 35
    assert cfg.calendar().data_latency.days == 35
    assert set(cfg.areas) == {"global", "north_polar", "south_polar"}
    assert {"acquisition_mode", "surface_type", "backscatter"} <= set(cfg.parameters)

    mode = cfg.parameters["acquisition_mode"]
    assert mode.type == "flag" and not mode.has_variants
    assert mode.mode_options == [""]
    assert [f.key for f in mode.flags] == ["lrm", "sar", "sarin"]

    sig0 = cfg.parameters["backscatter"]
    assert sig0.type == "float" and sig0.has_variants
    assert sig0.variables == ["sig0_1_20_ku", "sig0_2_20_ku", "sig0_3_20_ku"]
    assert sig0.mode_options == ["all", "lrm", "sar", "sarin"]
    # retracker used in each acquisition mode (None where the retracker is not used)
    assert sig0.variants[0].mode_descriptions == {
        "lrm": "Ocean CFI retracker",
        "sar": "UCL sea-ice retracker",
        "sarin": "UCL margins retracker",
    }
    assert sig0.variants[1].mode_descriptions["sar"] is None
    assert mode.variants[0].mode_descriptions == {}


def test_colour_scales(default_config):
    """height maps have a default and an ocean only colour scale, other parameters one"""
    scales = default_config.parameters["height"].colour_scales
    assert [(s.id, s.range, s.file_suffix) for s in scales] == [
        ("landocean", (-120.0, 4300.0), ""),
        ("ocean", (-120.0, 80.0), "ocean"),
    ]
    assert default_config.parameters["height"].plot_range == (-120.0, 4300.0)
    sig0 = default_config.parameters["backscatter"]
    assert [(s.range, s.file_suffix) for s in sig0.colour_scales] == [((0.0, 35.0), "")]
    # the default colour scale's maps have no file name suffix
    assert plot_filename("height", "rtk1", "all", "global", "webp") == (
        "height_rtk1_all_global.webp"
    )
    assert plot_filename("height", "rtk1", "all", "global", "webp", "ocean") == (
        "height_rtk1_all_global_ocean.webp"
    )
    assert plot_filename("surface_type", "", "", "north_polar", "webp") == (
        "surface_type_north_polar.webp"
    )


def test_log_colour_scale(tmp_path, default_config):
    """peakiness has a default log colour scale, and log scales need a range above 0"""
    peakiness = default_config.parameters["peakiness"]
    assert [(s.id, s.log, s.file_suffix) for s in peakiness.colour_scales] == [
        ("log", True, ""),
        ("low", False, "low"),
        ("high", False, "high"),
    ]
    assert log_scale_ticks(0.5, 200) == [0.5, 1, 2, 5, 10, 20, 50, 100, 200]
    assert log_scale_ticks(0.5, 200, decades_only=True) == [1, 10, 100]

    params = {
        "parameters": {
            "bad": {
                "source": "GDR-A",
                "type": "float",
                "variable": "x",
                "plot": {"scales": [{"id": "log", "range": [0.0, 10.0], "log": True}]},
            }
        }
    }
    params_file = tmp_path / "params.yaml"
    params_file.write_text(yaml.safe_dump(params), encoding="utf-8")
    with open(default_config.config_file, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["parameters_file"] = str(params_file)
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    with pytest.raises(ValueError, match="needs a range above 0"):
        load_config(str(config_file))


def test_bit_flag_parameter(default_config):
    """the quality flag word is a bit flag parameter with maps for all modes only"""
    qflags = default_config.parameters["quality_flags"]
    assert qflags.is_bit_flag and qflags.type == "flag"
    assert len(qflags.variants) == 31
    assert {v.variable for v in qflags.variants} == {"flag_prod_status_20_ku"}
    height_1 = next(v for v in qflags.variants if v.bit_name == "height_1_error")
    assert (height_1.id, height_1.bit_mask) == ("b24", 16777216)
    assert qflags.default_variant == "b24"
    assert qflags.map_modes == ["all"]
    assert [(f.value, f.key) for f in qflags.flags] == [(0, "not_set"), (1, "set")]
    # other parameters have maps for every mode
    assert default_config.parameters["backscatter"].map_modes == ["all", "lrm", "sar", "sarin"]
    assert default_config.parameters["surface_type"].map_modes == [""]

    # maps only where the mode has maps, and a bit is set
    row = {"mode": "all", "n_valid": 10, "counts": {"set": 2, "not_set": 8}}
    assert has_maps(qflags, row)
    assert not has_maps(qflags, {**row, "mode": "sar"})
    assert not has_maps(qflags, {**row, "counts": {"set": 0, "not_set": 10}})
    assert has_maps(default_config.parameters["backscatter"], {"mode": "sar", "n_valid": 5})


def test_sanitize_key():
    """display names are converted to keys"""
    assert sanitize_key("Lake/Enclosed Sea") == "lake_enclosed_sea"
    assert sanitize_key(" SARin ") == "sarin"


def test_invalid_parameter(tmp_path, default_config):
    """invalid parameter definitions are rejected"""
    params = {
        "parameters": {
            "bad": {"source": "GDR-A", "type": "float", "variable": "x", "areas": ["arctic"]}
        }
    }
    params_file = tmp_path / "params.yaml"
    params_file.write_text(yaml.safe_dump(params), encoding="utf-8")
    with open(default_config.config_file, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["parameters_file"] = str(params_file)
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    with pytest.raises(ValueError, match="area arctic"):
        load_config(str(config_file))
