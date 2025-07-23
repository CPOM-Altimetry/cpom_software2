"""cpom.altimetry.datasets.altdatasets.py

# Purpose
Module contains the AltDataset class to support Altimetry data sets.
It loads dataset definitions from JSON files.
Provides methods to access parameters and variables.

Supported Datasets:
- "cs2_l2i"
- "cryotempo_li"
- "ev_GDR"
- "e1_GDR"
- "e2_GDR"
- "s3a_l2"
- "s3b_l2"
- "e1_fdr4alt"
- "e2_fdr4alt"
- "ev_fdr4alt"

To add a dataset :
1. Create a new JSON file in the definitions directory with the dataset parameters.
2. Add the dataset name to the dataset_list list
"""

# pylint: disable=too-many-arguments
# pylint
# pylint: disable=no-member

import importlib
from datetime import datetime
from pathlib import Path
from typing import List
import numpy as np
from netCDF4 import Dataset, Variable  # pylint: disable=E0611,W0611

dataset_list = [
    "cryotempo_li",
    "ev_GDR",
    "e1_GDR",
    "e2_GDR",
    "cs2_l2i",
    "s3a_l2",
    "s3b_l2",
    "e1_fdr4alt",
    "e2_fdr4alt",
    "ev_fdr4alt",
]


class AltDataset:
    """Class to support Altimetry data sets."""

    def __init__(  # pylint :disable=R0917
        self,
        name: str,
        mission: str,
        level: str = "l2",
        overrides: dict | None = None,
        dataset_filename: str | None = None,
    ):
        """
        Class initialization.

        Args:
            name (str): Dataset name.
            **overrides: Optional keyword arguments to override default parameters.
        """
        self.name = name
        self.level = level
        self.mission = mission

        try:
            self.load_dataset(overrides, dataset_filename)
        except ImportError as exc:
            raise ImportError(f"Error loading dataset {name}") from exc

        if name not in dataset_list:
            raise ValueError(f"{name} is not a supported dataset in AltDataset class")

    def load_dataset(self, overrides: dict | None = None, dataset_filename: str | None = None):
        """Load dataset settings for current dataset name"""
        if dataset_filename is None:
            try:
                module = importlib.import_module(
                    f"cpom.altimetry.datasets.definitions.{self.level}.{self.mission}.{self.name}"
                )
            except ImportError as exc:
                raise ImportError(f"Could not load dataset definition {self.name}") from exc
        else:
            # Support passing a custom dataset definition file
            pass

        dataset_params = module.dataset_definition
        for k, v in dataset_params.items():
            if overrides:
                setattr(self, k, overrides.get(k, v))
            else:
                setattr(self, k, v)

        # Set any additional overrides not in the JSON
        if overrides:
            for k, v in overrides.items():
                if k not in dataset_params:
                    setattr(self, k, v)

    # ------------------------------------------#
    # Getters for L1 dirs, files and variables #
    # ------------------------------------------#

    # ------------------------------------------#
    # Getters for L2 dirs, files and variables #
    # ------------------------------------------#
    def get_files_dir(
        self, cyclenum: int | None = None, hemisphere: str | None = None, theme: str = "land_ice"
    ) -> str:
        """Get the L2 directory for a data product. If specified filters to specified cycle.
        For FDR4ALT data products filters to hemisphere, and theme.

        Args:
            cyclenum (int): Cycle number to filter by. Defaults to None.
            hemisphere (str): Hemisphere (FDR4ALT only). Defaults to None.
            theme (str): Theme (FDR4ALT only). Defaults to "land_ice".

        Returns:
            str: The L2 directory path.
        """
        if cyclenum is None:
            return self.l2_dir
        if "fdr4alt" in self.name:
            if hemisphere == "north":
                area = "/greenland"
            if hemisphere == "south":
                area = "/antarctica"
            else:
                area = "/*"
            return f"{self.l2_dir}/{area}/{theme}/Cycle_{cyclenum:03d}"

        if self.mission in ["ev", "s3a", "s3b"]:
            return f"{self.l2_dir}/cycle_{cyclenum:03d}"
        if self.mission in ["e1", "e2"]:
            return f"{self.l2_dir}/CYCLE{cyclenum:02d}"

    def get_files(
        self,
        min_dt_time=None,
        max_dt_time=None,
        cyclenum: int | None = None,
        modes: List = ["lrm", "sin"],
        hemisphere: str | None = None,  # Optional hemisphere filter for fdr4alt
        theme: str = "land_ice",  # Optional theme filter for fdr4alt
    ) -> List[Path]:  # pylint: disable=R0917
        """
        Returns an array of files for dataset <self.name>.
        Options:
            - Select by cycle number.
            - Select by date range (min_dt_time, max_dt_time).

        Args:
            min_dt_time (datetime | str, optional): Min time datetime object or "YYYYMMDD" string.
            max_dt_time (datetime | str, optional): Max time datetime object or "YYYYMMDD" string.
            cyclenum (int | None, optional): Cycle number. Defaults to None.
            modes (List[str], optional): cs2 modes to load(cs2 only). Defaults to ["lrm", "sin"].
            hemisphere (str | None, optional): Hemisphere (FDR4ALT only). Defaults to None.
            theme (str, optional): Theme (FDR4ALT only). Defaults to "land_ice".
        Returns:
            List[Path]: List of L2 files matching the search criteria.
        """

        def _get_files_by_date(
            search_dir, search_pattern, yyyymm_str_fname_indices, min_dt_time, max_dt_time
        ):
            valid_files = []
            for file in Path(search_dir).rglob(search_pattern):
                date_obj, _, _, _ = self.get_product_startdate_from_filename(
                    file, yyyymm_str_fname_indices
                )
                if date_obj is not None and min_dt_time <= date_obj <= max_dt_time:
                    valid_files.append(file)
            return valid_files

        base_dir = self.get_l2_dir(cyclenum, hemisphere, theme)
        if cyclenum is not None:
            return List[Path(search_dir).rglob(self.search_pattern)]

        if isinstance(min_dt_time, str):
            min_dt_time = datetime.strptime(min_dt_time, "%Y%m%d")
            max_dt_time = datetime.strptime(max_dt_time, "%Y%m%d")

        if min_dt_time.year == max_dt_time.year and min_dt_time.month == max_dt_time.month:
            search_dir = base_dir / f"{min_dt_time.year:04d}" / f"{min_dt_time.month:02d}"
        elif min_dt_time.year == max_dt_time.year:
            search_dir = base_dir / f"{min_dt_time.year:04d}"
        else:
            search_dir = base_dir

        if not search_dir.is_dir():
            search_dir = base_dir

        if self.name in ["cs2_l2i"]:
            valid_files = []
            for mode in modes:
                mode_config = getattr(self, mode)
                mode_files = _get_files_by_date(
                    Path(search_dir) / mode.upper(),
                    mode_config["search_pattern"],
                    mode_config["yyyymm_str_fname_indices"],
                    min_dt_time,
                    max_dt_time,
                )
                valid_files.extend(mode_files)
        else:
            valid_files = _get_files_by_date(
                search_dir,
                self.search_pattern,
                self.yyyymm_str_fname_indices,
                min_dt_time,
                max_dt_time,
            )

        return valid_files

    def get_product_startdate_from_filename(
        self, filename: str, yyyymm_str_fname_indices: list[int]
    ) -> tuple:
        """
        Extract L2 product start date from the filename
        filename is the full path of a L2 file
        returns datetime and  integer (year, month, day)
        """
        if ".SEN3" in filename.parent.name:
            fname = filename.parent.name
        else:
            fname = filename.name

        date_obj = datetime.strptime(
            fname[yyyymm_str_fname_indices[0] : yyyymm_str_fname_indices[1]], "%Y%m%d"
        )

        if date_obj is None:
            raise ValueError(f"Could not extract date from filename {filename}")

        return date_obj, date_obj.year, date_obj.month, date_obj.day

    def get_variables_from_file(self, nc: Dataset, nc_var_paths: str) -> np.ndarray:
        """Retrieve variable from NetCDF file, handling groups if necessary.

        Args:
            nc (Dataset): The dataset object
            nc_var_paths (str or list[str]): The path(s) to variable(s) within the file,
            with groups separated by '/'.

        Raises:
            KeyError: If the variable or group is not found in the file.

        Returns:
            np.array|List[np.array]: The retrieved variable(s) as array(s).
        """

        def get_single_var(nc, nc_var_path):
            parts = nc_var_path.split("/")
            var = nc
            for part in parts:
                var = var[part]
                if var is None:
                    raise IndexError(f"NetCDF parameter '{nc_var_path}' not found.")
            return var[:]

        if isinstance(nc_var_paths, str):
            return get_single_var(nc, nc_var_paths)
        elif isinstance(nc_var_paths, (list, tuple)):
            return [get_single_var(nc, path) for path in nc_var_paths]
        else:
            raise TypeError("nc_var_paths must be a string or a list/tuple of strings.")

    def get_unified_time_epoch_offset(
        self, goal_epoch_str: str = "1991-01-01", this_epoch_str: str = None
    ) -> float:
        """
        Convert a timestamp from one custom epoch to another.

        Parameters:
        - goal_epoch_str: str, the target epoch (default: "1991-01-01")
        - this_epoch_str: str, the original epoch (default: self.time_epoch)

        Returns:
        - float: the timestamp relative to `goal_epoch_str`
        """
        if this_epoch_str is None:
            this_epoch_str = self.time_epoch

        this_epoch = datetime.fromisoformat(this_epoch_str)  # + timedelta(days=1)
        goal_epoch = datetime.fromisoformat(goal_epoch_str)

        # Calculate offset between epochs in seconds
        offset = (this_epoch - goal_epoch).total_seconds()

        return offset

    def find_measurement_directions(self, nc: Dataset):
        """find the direction of each measurement (ie ascending or descending)

        Args:
            nc (Dataset): netCDF Dataset
        Returns:
            directions (np.ndarray[bool]) : array of bool directions (asc==True, desc==False)

        """
        if "cryotempo" in self.name:
            # CryoTEMPO L2 products include global attributes to identify
            # ascending and descending records for the track. These are
            # either set to "None" or the start record number

            # allocate an array to store directions

            ascending_start = nc.ascending_start_record
            descending_start = nc.descending_start_record

            if ascending_start == "None":
                asc_directions = np.zeros_like(nc["latitude"][:].data, dtype=bool)
            elif descending_start == "None":
                asc_directions = np.ones_like(nc["latitude"][:].data, dtype=bool)
            elif ascending_start == 0 and descending_start > 0:
                asc_directions = np.ones_like(nc["latitude"][:].data, dtype=bool)
                asc_directions[descending_start:] = False
            elif descending_start == 0 and ascending_start > 0:
                asc_directions = np.zeros_like(nc["latitude"][:].data, dtype=bool)
                asc_directions[ascending_start:] = True
            else:
                raise ValueError(
                    f"combination of nc.ascending_start_record {ascending_start}"
                    f"and nc.descending_start_record {descending_start} not"
                    " supported"
                )
            return asc_directions
        raise ValueError(f"{self.name} not a supported dataset type")
