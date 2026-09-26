"""pytests of cpom.altimetry.projects.csqa.csqa_config"""

from datetime import datetime

import pytest
import yaml  # type: ignore[import-untyped]

from cpom.altimetry.projects.csqa.csqa_config import load_config, sanitize_key


def test_default_config(default_config):
    """the default config defines the initial parameters and areas"""
    cfg = default_config
    assert cfg.mission_start_date == datetime(2010, 10, 18)
    assert cfg.cycle_length_days == 30
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
