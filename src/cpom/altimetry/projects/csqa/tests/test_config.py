"""pytests of cpom.altimetry.projects.csqa.csqa_config"""

import os
from datetime import datetime

import pytest
import yaml  # type: ignore[import-untyped]

from cpom.altimetry.projects.csqa.csqa_config import load_config, sanitize_key
from cpom.altimetry.projects.csqa.plotting import plot_filename
from cpom.altimetry.projects.csqa.processing import has_maps, params_to_process
from cpom.altimetry.projects.csqa.product_files import parse_product_filename
from cpom.altimetry.projects.csqa.tests.conftest import write_test_config
from cpom.areas.area_plot import log_scale_ticks


def test_default_config(default_config):
    """the default config defines the initial parameters and areas"""
    cfg = default_config
    assert cfg.mission_start_date == datetime(2010, 10, 18)
    assert cfg.cycle_length_days == 30
    assert cfg.data_latency_days == 35
    assert cfg.calendar().data_latency.days == 35
    assert set(cfg.areas) == {"global", "north_polar", "south_polar", "antarctica", "greenland"}
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


def test_variant_plot_ranges(default_config):
    """each geophysical correction has its own colour range and colormap"""
    cors = default_config.parameters["geophysical_corrections"]
    assert cors.mode_options == [""]
    by_id = {v.id: v for v in cors.variants}
    assert by_id["dry"].plot_range == (-2.4, -1.4) and by_id["dry"].cmap == "viridis"
    assert by_id["pt"].plot_range == (-0.015, 0.015) and by_id["pt"].cmap == "coolwarm"
    # variants without their own range use the parameter's
    assert default_config.parameters["backscatter"].variants[0].plot_range is None


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


def _config_with_parameters(tmp_path, default_config, params: dict) -> str:
    """write a config using the default config with other parameter definitions"""
    params_file = tmp_path / "params.yaml"
    params_file.write_text(yaml.safe_dump({"parameters": params}), encoding="utf-8")
    with open(default_config.config_file, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["parameters_file"] = str(params_file)
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return str(config_file)


def test_freeboard_grid(default_config):
    """freeboard has filtered and unfiltered variants, and 10 km gridded maps of the polar
    areas"""
    fb = default_config.parameters["freeboard"]
    filtered, unfiltered = fb.variants
    assert (filtered.id, filtered.variable) == ("filtered", "radar_freeboard_20_ku")
    assert (filtered.reject_variable, filtered.reject_mask, filtered.reject_name) == (
        "flag_prod_status_20_ku",
        65536,
        "freeboard_error",
    )
    assert unfiltered.reject_mask is None
    assert default_config.areas["north_polar"].grid_area == "arctic"
    assert default_config.areas["global"].grid_area == ""

    grid = fb.grid
    assert grid is not None
    assert (grid.binsize_km, grid.areas, grid.modes) == (
        10.0,
        ["north_polar", "south_polar"],
        ["all"],
    )
    assert grid.label == "10 km grid"
    assert grid.file_suffix("median") == "grid10km_median"
    stats = {s.id: s for s in grid.statistics}
    assert list(stats) == ["median", "mean", "max", "std", "count"]
    # values use the parameter's colour scale unless configured, counts have no units
    assert stats["median"].range == fb.plot_range and stats["median"].units == "m"
    assert stats["max"].range == (-0.3, 1.5)
    assert stats["count"].log and stats["count"].units == ""
    assert default_config.parameters["backscatter"].grid is None


@pytest.mark.parametrize(
    "grid, message",
    [
        ({"binsize_km": 10, "areas": ["global"], "statistics": [{"id": "mean"}]}, "grid_area"),
        ({"binsize_km": 10, "areas": ["north_polar"], "statistics": [{"id": "mode"}]}, "unknown"),
        (
            {
                "binsize_km": 10,
                "areas": ["north_polar"],
                "statistics": [{"id": "count", "log": True}],
            },
            "range above 0",
        ),
        (
            {
                "binsize_km": 10,
                "areas": ["north_polar"],
                "modes": ["lrm"],
                "statistics": [{"id": "mean"}],
            },
            "mode lrm",
        ),
    ],
)
def test_invalid_grid(tmp_path, default_config, grid, message):
    """invalid grid definitions are rejected"""
    params = {"bad": {"source": "GDR-A", "type": "float", "variable": "x", "grid": grid}}
    with pytest.raises(ValueError, match=message):
        load_config(_config_with_parameters(tmp_path, default_config, params))


def test_invalid_reject_bit(tmp_path, default_config):
    """a reject bit must be a single bit, and flag parameters can not be gridded"""
    params = {
        "bad": {
            "source": "GDR-A",
            "type": "float",
            "variable": "x",
            "reject_bit": {"variable": "flags", "mask": 3},
        }
    }
    with pytest.raises(ValueError, match="not a single bit"):
        load_config(_config_with_parameters(tmp_path, default_config, params))
    params = {
        "bad": {
            "source": "GDR-A",
            "type": "flag",
            "variable": "x",
            "flags": [{"value": 1, "name": "one"}],
            "grid": {"binsize_km": 10, "areas": ["north_polar"], "statistics": [{"id": "mean"}]},
        }
    }
    with pytest.raises(ValueError, match="only float parameters"):
        load_config(_config_with_parameters(tmp_path, default_config, params))


def test_mode_surfaces(tmp_path, default_config):
    """mode surface selections (ie LRM over ice) are selectable like modes"""
    cfg = default_config
    assert cfg.surface_variable == "surf_type_20_ku"
    assert cfg.surface_values == {"ocean": 0, "lake": 1, "ice": 2, "land": 3}
    lrm_ice = cfg.mode_surfaces["lrm_ice"]
    assert (lrm_ice.mode, lrm_ice.surfaces, lrm_ice.label) == ("lrm", ("ice",), "LRM Ice")
    assert cfg.mode_labels["sar_ocean"] == "SAR Ocean"
    qflags = cfg.parameters["quality_flags"]
    assert qflags.modes[4:] == [
        "lrm_ice",
        "lrm_land",
        "lrm_ocean",
        "sarin_ice",
        "sarin_land",
        "sar_ocean",
    ]
    assert qflags.map_modes == ["all"]

    # selections must be of a configured mode and surface types
    with open(cfg.config_file, encoding="utf-8") as fh:
        cfg_yaml = yaml.safe_load(fh)
    cfg_yaml["parameters_file"] = os.path.join(
        os.path.dirname(cfg.config_file), cfg_yaml["parameters_file"]
    )
    cfg_yaml["mode_surfaces"]["bad"] = {"mode": "lrm", "surfaces": ["snow"]}
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.safe_dump(cfg_yaml), encoding="utf-8")
    with pytest.raises(ValueError, match="surface_types"):
        load_config(str(config_file))
    # mode surface selections are not acquisition modes
    params = {
        "bad": {"source": "GDR-A", "type": "float", "variable": "x", "valid_modes": ["lrm_ice"]}
    }
    with pytest.raises(ValueError, match="valid mode"):
        load_config(_config_with_parameters(tmp_path, cfg, params))


def test_mispointing(tmp_path, default_config):
    """the mispointing angle is derived from the roll and pitch angles, in millidegrees, and
    selectable by pass direction"""
    cfg = default_config
    mis = cfg.parameters["mispointing"]
    assert [v.id for v in mis.variants] == ["mispointing", "roll", "pitch", "yaw"]
    derived = mis.variants[0]
    assert derived.derived == "mispointing_angle"
    assert derived.inputs == ("off_nadir_roll_angle_str_01", "off_nadir_pitch_angle_str_01")
    assert derived.variable == "off_nadir_roll_angle_str_01"  # its dimension and coordinates
    assert derived.display_variable == "mispointing_angle"
    assert mis.variants[1].display_variable == "off_nadir_roll_angle_str_01"
    assert mis.value_scale == 1000.0 and mis.units == "mdeg"
    assert mis.modes == mis.map_modes == ["all", "asc", "desc"]
    assert cfg.pass_selections["asc"].ascending and not cfg.pass_selections["desc"].ascending
    assert [cfg.mode_text(mis, m) for m in mis.modes] == [
        "All passes",
        "Ascending passes",
        "Descending passes",
    ]
    assert cfg.mode_text(cfg.parameters["height"], "all") == "All modes"
    assert cfg.mode_text(cfg.parameters["height"], "sar") == "SAR mode"
    assert cfg.mode_text(cfg.parameters["quality_flags"], "lrm_ice") == "LRM Ice"

    # derived variables must be known, with the right number of inputs
    for variant, message in (
        ({"id": "a", "derived": "unknown", "inputs": ["x", "y"]}, "unknown derived"),
        ({"id": "a", "derived": "mispointing_angle", "inputs": ["x"]}, "needs 2 input"),
    ):
        params = {"bad": {"source": "GDR-A", "type": "float", "variants": {"options": [variant]}}}
        with pytest.raises(ValueError, match=message):
            load_config(_config_with_parameters(tmp_path, cfg, params))


def test_crossover_parameter(tmp_path, default_config):
    """crossovers of the retracker heights over the ice sheet areas (with masks), with only
    smoothed gridded maps"""
    cfg = default_config
    xo = cfg.parameters["crossovers"]
    assert xo.crossover is not None and xo.record_name == "crossover"
    assert xo.crossover.max_abs_difference == 5.0 and xo.crossover.nadir_lat == "lat_01"
    assert xo.crossover.one_per_pass_pair
    # crossovers of the POCA locations; OCOG is the default LRM retracker
    assert (xo.crossover.lat, xo.crossover.lon) == ("lat_poca_20_ku", "lon_poca_20_ku")
    assert xo.default_variant == "rtk3"
    assert xo.mode_default_variants == {"lrm": "rtk3", "sarin": "rtk1"}
    assert cfg.parameters["height"].mode_default_variants == {}
    assert xo.modes == ["lrm", "sarin"] and xo.map_modes == []
    assert xo.areas == ["antarctica", "greenland"]
    assert cfg.areas["antarctica"].mask_name == "antarctica_bedmachine_v2_grid_mask"
    assert cfg.areas["antarctica"].mask_basins == (2, 4)
    assert cfg.areas["greenland"].grid_area == "greenland"
    assert xo.variants[1].valid_modes == ("lrm",) and xo.variants[1].reject_mask == 8388608
    assert xo.grid is not None and xo.grid.smooth_radius_km == 20.0
    assert xo.grid.cell_text == "within 20 km of each cell"
    assert [s.id for s in xo.grid.statistics] == ["mean", "std", "count"]

    # crossover areas need a mask, smoothed grids only mean / std / count
    params = {
        "bad": {"source": "GDR-A", "type": "float", "variable": "x", "areas": ["global"]}
        | {"crossover": {}}
    }
    with pytest.raises(ValueError, match="needs a mask"):
        load_config(_config_with_parameters(tmp_path, cfg, params))
    grid = {"binsize_km": 10, "smooth_radius_km": 20, "areas": ["antarctica"]}
    params = {
        "bad": {"source": "GDR-A", "type": "float", "variable": "x", "areas": ["antarctica"]}
        | {"grid": {**grid, "statistics": [{"id": "median"}]}}
    }
    with pytest.raises(ValueError, match="smoothed grids"):
        load_config(_config_with_parameters(tmp_path, cfg, params))

    # a mode's default variant must be used in the mode
    variants = {
        "options": [
            {"id": "a", "variable": "x"},
            {"id": "b", "variable": "y", "valid_modes": ["lrm"]},
        ]
    }
    params = {
        "bad": {"source": "GDR-A", "type": "float", "variants": variants}
        | {"modes": ["lrm", "sarin"], "default_variant": {"lrm": "b", "sarin": "b"}}
    }
    with pytest.raises(ValueError, match="b is not used in sarin"):
        load_config(_config_with_parameters(tmp_path, cfg, params))


def test_l2i_parameters(default_config):
    """L2i parameters, with per variant valid modes, invalid values, scales and units"""
    cfg = default_config
    l2i = [p for p in cfg.parameters.values() if p.source == "L2I"]
    assert [p.id for p in l2i] == [
        "l2i_retracker_correction",
        "l2i_retracker_flags",
        "l2i_surface_class",
        "l2i_sarin_discriminator",
        "l2i_sarin_ambiguity",
        "l2i_stack",
        "l2i_doppler_correction",
        "l2i_slope_attitude",
    ]
    rtk = {v.id: v for v in cfg.parameters["l2i_retracker_correction"].variants}
    assert rtk["rtk1"].valid_modes == () and rtk["rtk1"].reject_mask == 4
    assert rtk["rtk2"].valid_modes == ("lrm",) and rtk["rtk2"].reject_mask == 2
    assert rtk["rtk3"].reject_variable == "flag_retracker_20_ku"

    disc = {v.id: v for v in cfg.parameters["l2i_sarin_discriminator"].variants}
    # parameter settings apply to every variant, unless overridden
    assert all(v.valid_modes == ("sarin",) and v.invalid_values == (0.0,) for v in disc.values())
    assert disc["totalpower"].plot_log and disc["totalpower"].value_scale == 1.0
    assert disc["maxpowerbin"].value_scale == 0.001
    disc_param = cfg.parameters["l2i_sarin_discriminator"]
    assert disc_param.variant_units(disc["maxpowerbin"]) == "bin"
    assert disc_param.variant_units(disc["totalpower"]) == ""

    stack = cfg.parameters["l2i_stack"]
    assert stack.modes == stack.map_modes == ["sar", "sarin"]
    assert stack.variant_units(stack.variants[0]) == "beams"
    doppler = {v.id: v for v in cfg.parameters["l2i_doppler_correction"].variants}
    assert doppler["doppler"].valid_modes == () and doppler["doppler"].invalid_values == ()
    assert doppler["slope"].valid_modes == ("lrm",) and doppler["slope"].invalid_values == (0.0,)
    flags = cfg.parameters["l2i_retracker_flags"]
    assert flags.is_bit_flag and len(flags.variants) == 18 and flags.default_variant == "b2"


def test_sea_ice_parameters(tmp_path, default_config):
    """sea ice freeboard and thickness are Baseline-F onwards, SAR/SARin only, without values
    with the freeboard error bit set, and gridded like the radar freeboard"""
    radar_fb = default_config.parameters["freeboard"]
    for pid in ("sea_ice_freeboard", "sea_ice_thickness"):
        param = default_config.parameters[pid]
        assert not param.has_variants
        assert param.variants[0].reject_mask == 65536
        assert param.variants[0].reject_variable == "flag_prod_status_20_ku"
        assert param.valid_modes == ["sar", "sarin"]
        assert param.first_baseline == "F"
        assert not param.in_baseline("E")
        assert param.in_baseline("F") and param.in_baseline("G")
        assert param.grid is not None and radar_fb.grid is not None
        assert (param.grid.binsize_km, param.grid.areas) == (10.0, radar_fb.grid.areas)
    assert radar_fb.in_baseline("E") and radar_fb.valid_modes == []
    thickness_grid = default_config.parameters["sea_ice_thickness"].grid
    assert thickness_grid is not None
    assert thickness_grid.statistics[0].range == (-1.0, 5.0)

    # parameters are not processed for baselines before their first baseline
    cfg = load_config(write_test_config(tmp_path))
    e_file = parse_product_filename(
        "/data/CS_LTA__SIR_GDR_2__20110101T011919_20110101T025833_E001.nc"
    )
    assert e_file is not None
    needed = params_to_process(
        cfg, 3, "E", ["freeboard", "sea_ice_freeboard"], False, False, {"GDR-A": [e_file]}
    )
    assert needed == ["freeboard"]


@pytest.mark.parametrize(
    "extra, message",
    [({"valid_modes": ["all"]}, "valid mode all"), ({"first_baseline": "F1"}, "baseline letter")],
)
def test_invalid_valid_modes(tmp_path, default_config, extra, message):
    """valid modes must be acquisition modes, and the first baseline a letter"""
    params = {"bad": {"source": "GDR-A", "type": "float", "variable": "x", **extra}}
    with pytest.raises(ValueError, match=message):
        load_config(_config_with_parameters(tmp_path, default_config, params))
