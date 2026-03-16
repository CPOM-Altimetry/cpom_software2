"""
Check S3 archive for corrupted files

This script will take as input a base directory and a YAML dataset definition
as per src/cpom/altimetry/datasets/definitions/s3<a,b>_thematic_bc005.yml

It will find all files that match the search pattern and check if they are
valid NetCDF files and contain the parameters specified in the YAML file.

It will print out the path of any files that do not meet these criteria.

"""

import argparse
import logging
import sys
from pathlib import Path

from netCDF4 import Dataset  # pylint: disable=E0611

from cpom.altimetry.datasets.dataset_helper import DatasetHelper

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def check_file(file_path: Path, required_params: list[str]) -> bool:
    """Check if a NetCDF file is valid and contains all required parameters.

    Args:
        file_path (Path): Path to the NetCDF file.
        required_params (list[str]): List of required parameter names.

    Returns:
        bool: True if the file is valid and contains all parameters, False otherwise.
    """
    try:
        with Dataset(file_path, "r") as nc:
            for param in required_params:
                if param is None:
                    continue
                # Split param by '/' to handle nested groups
                parts = param.split("/")
                current_group = nc
                found = True
                for part in parts:
                    if part in current_group.groups:
                        current_group = current_group.groups[part]
                    elif part in current_group.variables:
                        # Reached the variable
                        break
                    else:
                        found = False
                        break

                if not found:
                    log.error("File %s is missing parameter: %s", file_path, param)
                    return False
        return True
    except (OSError, RuntimeError) as e:
        log.error("File %s is corrupted or cannot be opened: %s", file_path, e)
        return False


def main():
    """Main function to parse arguments and run the archive check."""
    parser = argparse.ArgumentParser(
        description="Check S3 archive for corrupted or incomplete NetCDF files."
    )
    parser.add_argument(
        "-d", "--base_dir", required=True, help="Base directory for the S3 archive."
    )
    parser.add_argument(
        "-y",
        "--dataset_yaml",
        required=True,
        help="Path to the YAML dataset definition file.",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose output (DEBUG level)."
    )

    args = parser.parse_args()

    if args.verbose:
        log.setLevel(logging.DEBUG)

    base_dir = Path(args.base_dir)
    dataset_yaml = Path(args.dataset_yaml)

    if not base_dir.is_dir():
        log.error("Base directory not found: %s", base_dir)
        sys.exit(1)

    if not dataset_yaml.is_file():
        log.error("Dataset YAML file not found: %s", dataset_yaml)
        sys.exit(1)

    try:
        # Load dataset configuration
        dataset = DatasetHelper(data_dir=str(base_dir), dataset_yaml=str(dataset_yaml))

        # Get list of required parameters from the YAML
        # These fields match the ones in DatasetConfig
        required_params = [
            dataset.latitude_param,
            dataset.longitude_param,
            dataset.elevation_param,
            dataset.time_param,
            dataset.power_param,
            dataset.mode_param,
            dataset.quality_param,
            dataset.uncertainty_param,
            dataset.latitude_nadir_param,
            dataset.longitude_nadir_param,
        ]
        # Filter out None values
        required_params = [p for p in required_params if p is not None]

        log.info("Searching for files matching pattern: %s", dataset.search_pattern)

        # Find all files matching the search pattern
        files = list(base_dir.rglob(dataset.search_pattern))
        log.info("Found %d files to check.", len(files))

        corrupted_files = []
        for file_path in files:
            log.debug("Checking file: %s", file_path)
            if not check_file(file_path, required_params):
                corrupted_files.append(file_path)

        if corrupted_files:
            print("\nFound corrupted or incomplete files:")
            for f in corrupted_files:
                print(f)
            sys.exit(1)
        else:
            log.info("All files are valid and complete.")
            sys.exit(0)

    except Exception as e:  # pylint: disable=broad-exception-caught:
        log.error("An error occurred during processing: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
