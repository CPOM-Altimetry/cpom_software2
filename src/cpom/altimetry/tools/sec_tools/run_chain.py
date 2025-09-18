"""
src.cpom.altimetry.tools.sec_tools.run_chain

Run a processing chain based on a configuration file.
"""

# Load config file
import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

import yaml

from cpom.logging_funcs.logging import set_loggers

print("Running run_chain.py")


def dict_to_cli_args(config):
    """
    Convert a dictionary of configuration parameters
    to a list of command-line arguments.

    Handles special case for 'dataset' key. 'dataset' can
    be either a string (file path to a dataset config file)
    or a dictionary (inline config).

    Args:
        config (str): Path to the configuration file.

    Returns:
        list: List of command-line arguments.
    """
    args = []
    for k, v in config.items():
        # Handle dataset key
        if k == "dataset" and isinstance(v, dict):
            args.append("--dataset")
            args.append(json.dumps(v))
        elif isinstance(v, bool):
            if v:
                args.append(f"--{k}")
        elif isinstance(v, list):
            args.append(f"--{k}")
            args.extend(map(str, v))
        else:
            args.append(f"--{k}")
            args.append(str(v))
    return args


def main(args):
    """
    Run the processing chain.

    Args:
        args (list): Command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Run the processing chain")
    parser.add_argument("--config", type=Path, required=True, help="Path to the config file")
    args = parser.parse_args()

    # Setup logging
    default_log_level = logging.INFO
    # if args.debug:
    #     default_log_level = logging.DEBUG

    logfile = "/tmp/grid.log"
    set_loggers(
        log_file_info=logfile[:-3] + "info.log",
        log_file_warning=logfile[:-3] + "warning.log",
        log_file_error=logfile[:-3] + "errors.log",
        log_file_debug=logfile[:-3] + "debug.log",
        log_format="%(levelname)s : %(asctime)s %(name)s : %(message)s",
        default_log_level=default_log_level,
    )

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    print(config["algorithm_list"])

    for algo in config["algorithm_list"]:
        algo_config = config[algo]
        script = f"{algo}.py"
        args = dict_to_cli_args(algo_config)

        print(f"Running {script} with args: {args}")

        subprocess.run(
            [
                "python",
                script,
            ]
            + args,
            check=True,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
