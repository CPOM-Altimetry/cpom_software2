"""
cpom.altimetry.tools.sec_tools.basin_selection_helper

Module with cental utilities for basin/region selection in SEC tools.

Functions:
    add_basin_selection_arguments: Add standard CLI args for basin/region selection.
    get_basins_to_process: Resolve which basins/regions to process given CLI params.

"""

import argparse
import logging
from pathlib import Path


def add_basin_selection_arguments(parser: argparse.ArgumentParser) -> None:
    """
    Add standard basin/region selection arguments to an ArgumentParser.
    Extends the given parser with two shared SEC arguments:

    --basin_structure: When set, treats immediate subdirectories
      of in_dir as basins to process. When omitted, processes root-level data only.
    - --region_selector: Basin/region names to include. Defaults to
      ["all"], which selects every available basin when basin_structure is set.

    Usage:
        parser = argparse.ArgumentParser()
        add_basin_selection_arguments(parser)
        params = parser.parse_args()
    """
    parser.add_argument(
        "--basin_structure",
        help=(
            "Enable basin-level processing: when set, treats immediate subdirectories "
            "of in_dir as basins to process. When not set, processes root-level data only."
        ),
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--region_selector",
        nargs="+",
        default=["all"],
        help="Select regions to process. Use 'all' to process all available regions. "
        "Ignored for root level data.",
    )


def get_basins_to_process(
    params: argparse.Namespace, directory: str | Path, logger: logging.Logger
) -> list[str]:
    """
    Determine which basins/regions should be processed.

    Modes:
    - Root mode (basin_structure=False): Returns ["None"] indicating root-level processing.
    - Basin mode (basin_structure=True): Basin/region names are determined from subdirectories
        of directory. Filtered based on params.regions_selector.
    Args:
        params (argparse.Namespace): Command Line Arguments. Includes
            (basin_structure and region_selector)
        directory (str | Path): Base directory to scan for basin subdirectories.
        logger (logging.Logger): Logger Object
    Returns:
        list[str]: Sorted basin names to process, or ``["None"]`` in root mode.
    """
    directory = Path(directory)

    # Root mode - no basin subdirectories
    if params.basin_structure is False:
        logger.info("Processing in root mode")
        return ["None"]

    # Single-tier or Two-tier: Get regions first
    if params.region_selector == ["all"]:
        logger.info("Finding all regions in %s", directory)
        basins_to_process = {
            subdir.name
            for subdir in directory.iterdir()
            if subdir.is_dir() and any(subdir.glob("*.parquet"))
        }
    else:
        basins_to_process = set(params.region_selector)

    logger.info("Basins/regions after region_selector: %s", sorted(basins_to_process))

    basins_sorted = sorted(basins_to_process)
    logger.info("Final basins to process (%d): %s", len(basins_sorted), basins_sorted)
    return basins_sorted
