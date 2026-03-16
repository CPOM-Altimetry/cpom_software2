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
import multiprocessing
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

from netCDF4 import Dataset  # pylint: disable=E0611

from cpom.altimetry.datasets.dataset_helper import DatasetHelper

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def check_file(file_path: str, required_params: list[str]) -> tuple[str, bool, str | None]:
    """Check if a NetCDF file is valid and contains all required parameters.

    Args:
        file_path (str): Path to the NetCDF file.
        required_params (list[str]): List of required parameter names.

    Returns:
        tuple[str, bool, str | None]: (file_path, is_valid, error_message)
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
                    return file_path, False, f"missing parameter: {param}"
        return file_path, True, None
    except (OSError, RuntimeError) as e:
        return file_path, False, f"corrupted or cannot be opened: {e}"
    except Exception as e:  # pylint: disable=broad-exception-caught
        return file_path, False, f"unexpected error: {e}"


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
        "-r",
        "--results_file",
        help="Optional path to save full paths of corrupted/incomplete files.",
    )
    parser.add_argument(
        "-j",
        "--max_workers",
        type=int,
        default=multiprocessing.cpu_count(),
        help="Number of worker processes (default: CPU count).",
    )
    parser.add_argument(
        "-c",
        "--chunk_size",
        type=int,
        default=2000,
        help="Number of files per processing chunk (default: 2000).",
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
        files = [str(f) for f in base_dir.rglob(dataset.search_pattern)]
        num_files = len(files)
        log.info("Found %d files to check.", num_files)

        if num_files == 0:
            log.info("No files found to check.")
            sys.exit(0)

        corrupted_files = []
        checked_count = 0

        # Process files in chunks for better robustness against pool crashes
        num_chunks = (num_files + args.chunk_size - 1) // args.chunk_size

        log.info(
            "Starting check with %d workers and chunk size %d...",
            args.max_workers,
            args.chunk_size,
        )

        for chunk_idx in range(num_chunks):
            start = chunk_idx * args.chunk_size
            end = min(start + args.chunk_size, num_files)
            chunk = files[start:end]
            finished_in_chunk = set()

            log.debug("Processing chunk %d/%d (%d files)", chunk_idx + 1, num_chunks, len(chunk))

            # Attempt parallel processing for the chunk
            try:
                with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
                    future_to_file = {
                        executor.submit(check_file, f, required_params): f for f in chunk
                    }

                    for future in as_completed(future_to_file):
                        file_path = future_to_file[future]
                        # Track that this file's future was processed (even if it crashed)
                        # so we don't triage it again sequentially if the pool breaks LATER.
                        finished_in_chunk.add(file_path)
                        checked_count += 1

                        try:
                            _, is_valid, err_msg = future.result()
                            if not is_valid:
                                log.error("File %s error: %s", file_path, err_msg)
                                corrupted_files.append(file_path)
                        except Exception as e:  # pylint: disable=broad-exception-caught
                            # This is a specific file error, not necessarily a pool crash
                            log.error("File %s caused an exception in worker: %s", file_path, e)
                            corrupted_files.append(file_path)

                        # Update progress indicator every 1%
                        if (
                            checked_count % max(1, num_files // 100) == 0
                            or checked_count == num_files
                        ):
                            percent = (checked_count / num_files) * 100
                            print(
                                f"\rProgress: {percent:.1f}% "
                                f"({checked_count}/{num_files} files checked)",
                                end="",
                                flush=True,
                            )

            except BrokenProcessPool as e:
                log.warning(
                    "\nProcess pool failed in chunk %d: %s. Triaging unchecked files...",
                    chunk_idx + 1,
                    e,
                )

                # Falling back to sequential triage ONLY for files not in finished_in_chunk
                remaining_in_chunk = [f for f in chunk if f not in finished_in_chunk]

                log.info("Checking %d remaining files sequentially...", len(remaining_in_chunk))

                for file_path in remaining_in_chunk:
                    checked_count += 1
                    _, is_valid, err_msg = check_file(file_path, required_params)
                    if not is_valid:
                        log.error("File %s error (sequential fallback): %s", file_path, err_msg)
                        corrupted_files.append(file_path)

                    # Update progress
                    if checked_count % max(1, num_files // 100) == 0 or checked_count == num_files:
                        percent = (checked_count / num_files) * 100
                        print(
                            f"\rProgress: {percent:.1f}% "
                            f"({checked_count}/{num_files} files checked)",
                            end="",
                            flush=True,
                        )

        print()  # New line after progress

        if corrupted_files:
            print(f"\nFound {len(corrupted_files)} corrupted or incomplete files.")
            if args.results_file:
                try:
                    results_path = Path(args.results_file)
                    results_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(results_path, "w", encoding="utf-8") as rf:
                        for corrupted_file in corrupted_files:
                            rf.write(f"{corrupted_file}\n")
                    log.info("List of bad files saved to: %s", args.results_file)
                except OSError as e:
                    log.error("Failed to write results file %s: %s", args.results_file, e)
            else:
                log.info("Bad files (first 10):")
                for f in corrupted_files[:10]:
                    print(f)
                if len(corrupted_files) > 10:
                    print("...")
            sys.exit(1)
        else:
            log.info("All files are valid and complete.")
            sys.exit(0)

    except Exception as e:  # pylint: disable=broad-exception-caught:
        log.error("An error occurred during processing: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
