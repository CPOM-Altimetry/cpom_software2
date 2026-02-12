"""
Run SEC processing chain from YAML configuration.

Executes multi-step workflow: reads config → runs algorithms in sequence → passes outputs as inputs.

CONFIGURATION MODES:
    Single-mission:
        Set 'missions_to_run' + 'algorithm_list' per mission
            e.g. example_single_mission_icesheet_wide.yml
    Multi-mission:
        Set 'algorithm_list' at root
            e.g. example_multi_mission_icesheet_wide.yml

PARAMETER PRIORITY (low → high):
  1. Algorithm defaults (config[algo])
  2. Mission overrides (config[mission][algo])

AUTO PATH HANDLING:
  - Input:  auto-detected from previous step's 'out_folder_name' (or use 'in_dir')
  - Output: auto-built from 'base_out' + 'base_folder' + 'out_folder_name' (or use 'out_dir')
  - Grid metadata: priority order (high → low):
      1. Algorithm params: config[mission][algo]['grid_info_json'] or config[algo]['grid_info_json']
      2. Mission level: config[mission]['grid_info_json']
      3. Auto-computed: grid_for_elev_change output or in_dir/metadata.json

Usage: python run_chain.py --config path/to/config.yml
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from envyaml import EnvYAML


def dict_to_cli_args(config):
    """
    Convert config dict to CLI argument list.

    Skips internal keys: 'out_folder_name', 'in_step', 'grid_info_json', 'base_folder'
    Serializes to JSON:
        'dataset' : See grid_for_elev_change.py
        'mission_mapper' : See multi_mission_cross_cal.py
    Example: {'verbose': True, 'bands': [1, 2]} → ['--verbose', '--bands', '1', '2']
    """
    args = []
    for key, value in config.items():
        # Skip internal configuration keys
        if key in ["out_folder_name", "in_step", "base_folder"]:
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
    Build complete directory path for algorithm output.

    Constructs: base_out/[base_folder]/algo_config[out_folder_name]

    Args:
        base_out (str): Base output directory path.
        base_folder (str): Mission-specific base folder (or None for multi-mission).
        algo_config (dict): Algorithm configuration dictionary containing 'out_folder_name' key.

    Returns:
        Path: Complete directory path for algorithm output.

    Example:
        build_path('/output', 'CS2', {'out_folder_name': 'grid_300m'})
        → /output/CS2/grid_300m
    """
    if base_folder:
        return Path(base_out) / base_folder / algo_config["out_folder_name"]
    return Path(base_out) / algo_config["out_folder_name"]


def requires_grid_metadata(algo):
    """Return True if algorithm needs grid metadata.json."""
    return algo in [
        "grid_for_elev_change_update_year",
        "surface_fit",
        "surface_fit_plots",
        "epoch_average",
        "epoch_average_plots",
        "interpolate_grids_of_dh",
        "dhdt_plots",
        "clip_to_basins",
        "clip_to_basins_from_shapefile",
    ]


def get_auto_path(algo):
    """Return True if algorithm uses automatic in_dir/out_dir."""
    return algo not in ["grid_for_elev_change_update_year"]


def build_args(algo, config, mission=None):
    """
    Build complete command-line arguments for an algorithm from merged configuration.

    Merges configuration parameters with priority order, then converts to CLI args.
    Automatically handles input/output directory paths and grid metadata if needed.

    Configurations (lowest → highest priority):
    1. Algorithm defaults from config[algo] section
    2. Mission-specific overrides from config[mission][algo] (single-mission mode only)

    Automatic path management:
    - Input directory: Built from previous algorithm's output ('in_step' reference), or
                        use 'in_dir' if specified.
    - Output directory: Built from 'out_folder_name' and base_folder structure or
                        use 'out_dir' if specified.
    - Grid metadata: Automatically located and passed if algorithm requires it,
                        unless 'grid_info_json' is explicitly provided to the mission
                        or algorithm configuration.

    Args:
        algo (str): Algorithm name (must match key in config file).
        config (dict): Full configuration dictionary (root level + all sections).
        mission (str, optional): Mission name for single-mission processing.
                               If None, assumes multi-mission mode.

    Returns:
        list: Complete list of command-line argument strings.
    """

    # Get algorithm configuration
    algo_config = config[algo].copy()
    # Get the mission overrides for the algorithm if they exist
    if mission and algo in config[mission]:
        # Add / Replace with mission-specific parameters
        algo_config.update(config[mission][algo])
        # Add grid_info_json if specified at mission level
        if "grid_info_json" in config[mission]:
            algo_config["grid_info_json"] = config[mission]["grid_info_json"]

    # Convert to CLI args
    args = dict_to_cli_args(algo_config)

    base_folder = config[mission]["base_folder"] if mission else None
    if get_auto_path(algo):
        # Input directory
        if "in_dir" not in algo_config and "in_step" in algo_config:
            in_dir = build_path(config["base_out"], base_folder, config[algo_config["in_step"]])
            args.extend(["--in_dir", str(in_dir)])

        # Output directory
        if "out_dir" not in algo_config and "out_folder_name" in algo_config:
            out_dir = build_path(config["base_out"], base_folder, algo_config)
            args.extend(["--out_dir", str(out_dir)])

    # Grid metadata
    if "grid_info_json" not in algo_config and requires_grid_metadata(algo):
        metadata = None
        if "in_dir" in algo_config:
            metadata = Path(algo_config["in_dir"]) / "metadata.json"
        elif mission and "grid_for_elev_change" in config:
            grid_path = build_path(
                config["base_out"], config[mission]["base_folder"], config["grid_for_elev_change"]
            )
            metadata = grid_path / "metadata.json"

        # Only add if metadata file exists
        if metadata and metadata.exists():
            args.extend(["--grid_info_json", str(metadata)])

    return args


def main(args):
    """
    Run the processing chain.

    STEPS:
    1. Parse --config and optional --debug command-line arguments
    2. Load YAML configuration file
    3. Detect processing mode:
       - Single-mission: 'missions_to_run' key exists in config
       - Multi-mission: No 'missions_to_run' key, 'algorithm_list' at root
    4. Execute algorithms:
       - Single-mission: For each mission in missions_to_run:
         For each algorithm in mission['algorithm_list']: run_algorithm()
       - Multi-mission: For each algorithm in root-level algorithm_list: run_algorithm()

    Args:
        args (list): Command-line arguments (from sys.argv[1:]).
                    Must contain: ['--config', '/path/to/config.yml']
                    Optionally: ['--debug'] to enable DEBUG level logging in child processes
    """
    parser = argparse.ArgumentParser(description="Run the processing chain")
    parser.add_argument("--config", type=Path, required=True, help="Path to the config file")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable DEBUG level logging in all child processes",
    )
    parsed_args = parser.parse_args(args)

    yml = EnvYAML(str(parsed_args.config))  # read the YML and parse environment variables
    config = yml.export()

    # Build debug flag to pass to child processes
    debug_args = ["--debug"] if parsed_args.debug else []

    # Single-mission or multi-mission processing
    if "missions_to_run" in config:
        for mission in config["missions_to_run"]:
            for algo in config[mission]["algorithm_list"]:
                print(f"Starting {algo}" + (f" for mission {mission}" if mission else ""))
                args = build_args(algo, config, mission)
                print(f"Running {algo}.py with args: {args}")
                subprocess.run(["python", f"{algo}.py"] + args + debug_args, check=True)
    else:
        for algo in config["algorithm_list"]:
            args = build_args(algo, config)
            print(f"Running {algo}.py with args: {args}")
            subprocess.run(["python", f"{algo}.py"] + args + debug_args, check=True)


if __name__ == "__main__":
    main(sys.argv[1:])
