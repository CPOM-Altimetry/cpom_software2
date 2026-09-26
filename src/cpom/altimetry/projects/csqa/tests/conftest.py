"""pytest fixtures for the CSQA tools"""

import os

import pytest
import yaml  # type: ignore[import-untyped]

from cpom.altimetry.projects.csqa.csqa_config import DEFAULT_CONFIG_FILE, load_config


def write_test_config(tmp_path, product_dirs: dict[str, list[str]] | None = None) -> str:
    """write a copy of the default CSQA config with output (and optionally input) directories
    in tmp_path

    Args:
        tmp_path (Path): pytest tmp_path
        product_dirs (dict|None): {product id: [dirs]} replacing the configured input dirs

    Returns:
        str: path of the test config file
    """
    with open(DEFAULT_CONFIG_FILE, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["output_dir"] = str(tmp_path / "csqa_out")
    cfg["parameters_file"] = os.path.join(
        os.path.dirname(DEFAULT_CONFIG_FILE), cfg["parameters_file"]
    )
    for prod_id, dirs in (product_dirs or {}).items():
        cfg["products"][prod_id]["dirs"] = dirs
    config_file = tmp_path / "csqa_config.yaml"
    config_file.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return str(config_file)


@pytest.fixture
def default_config():
    """the default CSQA config"""
    return load_config(DEFAULT_CONFIG_FILE)
