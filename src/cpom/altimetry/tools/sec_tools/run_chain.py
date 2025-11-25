"""
src.cpom.altimetry.tools.sec_tools.run_chain

Run the sec processing chain based on a configuration file.

This script reads a configuration file that specifies
the algorithms to run, along with their parameters.
It then executes each algorithm in sequence, passing
the appropriate command-line arguments.

Supports two types of configuration files:

1. Single-Mission Config:
   - Has 'missions_to_run' list at root level
   - Each mission has its own section with 'algorithm_list'
   - Example: greenland_single_mission.yml

2. Multi-Mission Config:
   - Has 'algorithm_list' at root level (no missions)
   - Used for cross-calibration and multi-mission analysis
   - Example: greenland_multimission.yml

Args:
    --config (Path): Path to the configuration file.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml  # type: ignore[import-untyped]


def dict_to_cli_args(config):
    """
    Convert a dictionary of configuration parameters to command-line arguments.

    Skips internal keys like 'out_folder_name' and 'in_step' which are
    used for path construction but not passed to scripts.

    Args:
        config (dict): Configuration parameters dictionary.

    Returns:
        list: List of command-line argument strings.
    """
    args = []
    for key, value in config.items():
        # Skip internal configuration keys
        if key in ["out_folder_name", "in_step"]:
            continue

        # Handle different value types
        if key == "dataset" and isinstance(value, dict):
            args.extend(["--dataset", json.dumps(value)])
        if key == "mission_mapper" and isinstance(value, dict):
            args.extend(["--mission_mapper", json.dumps(value)])
        elif isinstance(value, bool):
            if value:
                args.append(f"--{key}")
        elif isinstance(value, list):
            args.append(f"--{key}")
            args.extend(str(item) for item in value)
        else:
            args.extend([f"--{key}", str(value)])

    return args


def build_path(base_out, base_folder, algo_config):
    """
    Build directory path from configuration.

    Args:
        base_out (str): Base output directory.
        base_folder (str): Mission-specific base folder (or None for multi-mission).
        algo_config (dict): Algorithm configuration with 'out_folder_name'.

    Returns:
        Path: Complete directory path.
    """
    if base_folder:
        return Path(base_out) / base_folder / algo_config["out_folder_name"]
    return Path(base_out) / algo_config["out_folder_name"]


def requires_grid_metadata(algo):
    """Check if algorithm requires grid metadata JSON file."""
    return algo in [
        "grid_for_elev_change_update_year",
        "surface_fit",
        "surface_fit_plots",
        "epoch_average",
        "interpolate_grids_of_dh",
        "single_mission_dhdt",
        "clip_to_glaciers",
    ]


def get_auto_path(algo):
    """Check if algorithm needs automatic in_dir/out_dir handling."""
    return algo not in ["grid_for_elev_change_update_year"]


def build_args(algo, algo_config, config, mission=None):
    """
    Build complete command-line arguments for an algorithm.

    Args:
        algo (str): Algorithm name.
        algo_config (dict): Algorithm configuration.
        config (dict): Full configuration dictionary.
        mission (str, optional): Mission name for single-mission processing.

    Returns:
        list: Complete list of command-line arguments.
    """
    args = dict_to_cli_args(algo_config)
    base_folder = config[mission]["base_folder"] if mission else None

    if get_auto_path(algo):
        # Input directory
        if "in_dir" not in algo_config and "in_step" in algo_config:
            in_dir = build_path(config["base_out"], base_folder, config[algo_config["in_step"]])
            args.extend(["--in_dir", str(in_dir)])

        # Output directory
        if "out_folder_name" in algo_config:
            out_dir = build_path(config["base_out"], base_folder, algo_config)
            args.extend(["--out_dir", str(out_dir)])

    # Grid metadata
    if "grid_info_json" not in algo_config and requires_grid_metadata(algo):
        if "in_dir" in algo_config:
            metadata = Path(algo_config["in_dir"]) / "metadata.json"
        else:
            grid_path = build_path(
                config["base_out"], config[mission]["base_folder"], config["grid_for_elev_change"]
            )
            metadata = grid_path / "metadata.json"
        args.extend(["--grid_info_json", str(metadata)])

    return args


def run_algorithm(algo, config, mission=None):
    """
    Execute a single algorithm.

    Args:
        algo (str): Algorithm name.
        config (dict): Full configuration dictionary.
        mission (str, optional): Mission name for single-mission processing.
    """
    print(f"Starting {algo}" + (f" for mission {mission}" if mission else ""))

    # Get algorithm configuration with mission overrides
    algo_config = config[algo].copy()
    if mission and algo in config[mission]:
        algo_config.update(config[mission][algo])

    # Build and execute
    args = build_args(algo, algo_config, config, mission)
    print(f"Running {algo}.py with args: {args}")
    subprocess.run(["python", f"{algo}.py"] + args, check=True)


def main(args):
    """
    Run the processing chain.

    Parses command-line arguments, loads configuration, and executes
    the appropriate processing workflow (single-mission or multi-mission).

    Args:
        args (list): Command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Run the processing chain")
    parser.add_argument("--config", type=Path, required=True, help="Path to the config file")
    parsed_args = parser.parse_args(args)

    with open(parsed_args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Single-mission or multi-mission processing
    if "missions_to_run" in config:
        for mission in config["missions_to_run"]:
            for algo in config[mission]["algorithm_list"]:
                run_algorithm(algo, config, mission)
    else:
        for algo in config["algorithm_list"]:
            run_algorithm(algo, config)


if __name__ == "__main__":
    main(sys.argv[1:])
