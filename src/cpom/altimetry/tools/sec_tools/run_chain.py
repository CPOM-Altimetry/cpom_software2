#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run SEC processing chain from YAML configuration.

Executes a multi-step workflow: reads config → runs algorithms in sequence →
passes outputs as inputs to the next step.

CONFIGURATION MODES:
    Single-mission:  set 'missions_to_run' + per-mission 'algorithm_list'
    Multi-mission:   set 'algorithm_list' at root level

PARAMETER PRIORITY (low → high):
    1. Algorithm defaults (config[algo])
    2. Mission overrides (config[mission][algo])

AUTO PATH HANDLING:
    Input: previous step's 'out_folder_name', or explicit 'in_dir'
    Output: base_out' / 'base_folder' / 'out_folder_name', or explicit 'out_dir'

USEAGE:
    python run_chain.py --config path/to/config.yml
"""

import argparse
import importlib
import json
import sys
from pathlib import Path

from envyaml import EnvYAML


def parse_args(args):
    """Parse command-line arguments for the SEC main processing chain."""
    parser = argparse.ArgumentParser(description="Run the processing chain")
    parser.add_argument("--config", "-c", type=Path, required=True, help="Path to config file")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable DEBUG level logging in all child processes",
    )
    parser.add_argument(
        "--missions",
        "-m",
        help=(
            "Override missions_to_run in config (for single-mission configs, useful for testing"
            " specific missions). Comma-separated list of missions to run, e.g. --mission cs2,env"
        ),
    )
    parser.add_argument(
        "--start_alg",
        "-sa",
        help=(
            "Optional algorithm name to start processing from. Any earlier algorithms in the "
            "configured algorithm list are skipped."
        ),
    )
    parser.add_argument(
        "--end_alg",
        "-ea",
        help=(
            "Optional algorithm name to stop processing at. Any later algorithms in the "
            "configured algorithm list are skipped."
        ),
    )
    return parser.parse_args(args)


def dict_to_cli_args(config):
    """
    Convert a config dict to a flat CLI argument list.

    Internal keys 'out_folder_name' and 'base_folder' are skipped.
    'dataset' and 'mission_mapper' dicts are JSON-serialised.

    Example:
        {'verbose': True, 'bands': [1, 2]} → ['--verbose', '--bands', '1', '2']
    """
    args = []
    for key, value in config.items():
        # Skip internal configuration keys
        if key in ["out_folder_name", "base_folder"]:
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
    Build an output directory path for an algorithm.

    Returns 'base_out/base_folder/out_folder_name' when base_folder is set,
    otherwise 'base_out/out_folder_name'.

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


def get_algorithms_to_run(algorithm_list, start_alg=None, end_alg=None):
    """
    Return the sub-list of algorithms bounded by start_alg and end_alg (both inclusive).

    Raises:
        ValueError: If start_alg or end_alg are not in algorithm_list, or if
            start_alg comes after end_alg.
    """

    start_index = 0
    end_index = len(algorithm_list)

    if start_alg is not None:
        if start_alg not in algorithm_list:
            raise ValueError(
                f"Start algorithm '{start_alg}' not found in algorithm list: "
                f"{', '.join(algorithm_list)}"
            )
        start_index = algorithm_list.index(start_alg)

    if end_alg is not None:
        if end_alg not in algorithm_list:
            raise ValueError(
                f"End algorithm '{end_alg}' not found in algorithm list: "
                f"{', '.join(algorithm_list)}"
            )
        end_index = algorithm_list.index(end_alg) + 1

    if start_index >= end_index:
        raise ValueError(
            f"Start algorithm '{start_alg}' must come before or equal to end algorithm "
            f"'{end_alg}' in the algorithm list."
        )

    return list(algorithm_list[start_index:end_index])


def get_merged_algo_config(config, algo, mission=None):
    """Return root config merged with any mission-specific override for one algorithm.
    Mission values take precedence over root-level values.

    Raises:
        TypeError: If any of the relevant config sections are not dictionaries.
    """

    root_algo_config = config.get(algo) or {}
    if not isinstance(root_algo_config, dict):
        raise TypeError(f"Config section '{algo}' must be a mapping, got {type(root_algo_config)}")
    algo_config = root_algo_config.copy()

    mission_config = config.get(mission) if mission else None
    if mission_config is None:
        mission_config = {}
    if not isinstance(mission_config, dict):
        raise TypeError(f"Mission config '{mission}' must be a mapping, got {type(mission_config)}")

    mission_algo_config = mission_config.get(algo) or {}
    if not isinstance(mission_algo_config, dict):
        raise TypeError(
            f"Mission override section '{mission}.{algo}' must be a mapping, "
            f"got {type(mission_algo_config)}"
        )
    algo_config.update(mission_algo_config)
    return algo_config, mission_config


def build_args(
    algo,
    config,
    mission=None,
):
    """
    Build the CLI argument list for an algorithm from merged configuration.

    Merges root and mission-specific config, then auto-resolves input/output paths
    when 'in_step'/'out_folder_name' are present and explicit 'in_dir'/'out_dir'
    are not set.

    Args:
        algo (str): Algorithm name (must match key in config file).
        config (dict): Full configuration dictionary (root level + all sections).
        mission: Mission name for single-mission mode; None for multi-mission.

    Returns:
        list: Complete list of command-line argument strings.
    """
    algo_config, mission_config = get_merged_algo_config(config, algo, mission)
    # Convert to CLI args
    args = dict_to_cli_args(algo_config)
    base_folder = mission_config.get("base_folder") if mission else None

    # Input directory
    if "in_dir" not in algo_config and "in_step" in algo_config:
        in_step_config, _ = get_merged_algo_config(config, algo_config["in_step"], mission)
        in_dir = build_path(config["base_out"], base_folder, in_step_config)
        args.extend(["--in_dir", str(in_dir)])
        print(f"Auto-detected input for {algo}: {in_dir} from in_step:{algo_config['in_step']}")

    # Output directory
    if "out_dir" not in algo_config and "out_folder_name" in algo_config:
        out_dir = build_path(config["base_out"], base_folder, algo_config)
        print(
            f"Built output for {algo}:{out_dir} from base_out: {config['base_out']} \n"
            f"and out_folder_name: {algo_config['out_folder_name']}"
        )
        args.extend(["--out_dir", str(out_dir)])
    return args


def run_algorithm(
    algo: str,
    args: list[str],
    debug_args: list[str],
):
    """
    Run an algorithm by importing its module and calling the eponymous function.

    Falls back with an error message if the module cannot be imported or the
    callable is not found.
    """
    cli_args = args + debug_args
    try:
        module = importlib.import_module(algo)
        main_func = getattr(module, algo, None)

        if callable(main_func):
            print(f"Running {algo}.{algo} with args: {cli_args}")
            main_func(cli_args)
        else:
            print(f"No callable main function for {algo}")
    except ImportError as e:
        print(f"Failed to import module {algo}: {e}")


def main(args):
    """
    Main entry point for running the SEC processing chain based on a YAML configuration file.

    Detects single-mission mode ('missions_to_run' key present) or multi-mission
    mode ('algorithm_list' at root), then runs each algorithm in sequence.

        STEPS:
        1. Parse command line arguments.
        2. Load YAML configuration file
        3. Detect processing mode.
        4. Execute algorithms.

    Args:
        args (list): Command-line arguments (from sys.argv[1:]).
                    Must contain: ['--config', '/path/to/config.yml']
                    Optionally: ['--debug'] to enable DEBUG level logging in child processes
    """

    parsed_args = parse_args(args)

    yml = EnvYAML(str(parsed_args.config))  # read the YML and parse environment variables
    config = yml.export()

    # Build debug flag to pass to child processes
    debug_args = ["--debug"] if parsed_args.debug else []

    # Single-mission or multi-mission processing
    if "missions_to_run" in config:
        if parsed_args.missions:
            # Override missions_to_run with command-line argument
            config["missions_to_run"] = [m.strip() for m in parsed_args.missions.split(",")]
            print(f"Overriding missions_to_run with: {config['missions_to_run']}")
        for mission in config["missions_to_run"]:
            try:
                algorithms_to_run = get_algorithms_to_run(
                    config[mission]["algorithm_list"], parsed_args.start_alg, parsed_args.end_alg
                )
            except ValueError as exc:
                sys.exit(f"{exc} (mission: {mission})")
            for algo in algorithms_to_run:
                print(f"Starting {algo}" + (f" for mission {mission}" if mission else ""))
                args = build_args(
                    algo,
                    config,
                    mission=mission,
                )
                run_algorithm(algo, args, debug_args)
    else:
        try:
            algorithms_to_run = get_algorithms_to_run(
                config["algorithm_list"], parsed_args.start_alg, parsed_args.end_alg
            )
        except ValueError as exc:
            sys.exit(str(exc))
        for algo in algorithms_to_run:
            args = build_args(
                algo,
                config,
            )
            run_algorithm(algo, args, debug_args)


if __name__ == "__main__":
    main(sys.argv[1:])
