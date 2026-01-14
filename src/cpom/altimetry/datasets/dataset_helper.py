"""
src.cpom.altimetry.datasets.datasetHelper

Helper class to load file paths and variables from altimetry datasets
in NetCDF or HDF5 format.

Supports two configuration methods:
1. YAML configuration file
    - To find example YAML configuration see src.cpom/altimetry/datasets/definitions/
2. Direct keyword arguments to the constructor
"""

import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import yaml  # type: ignore[import-untyped]
from netCDF4 import Dataset  # pylint: disable=E0611,W0611


# pylint: disable=R0902 # Too many instance attributes
@dataclass
class DatasetConfig:
    """Configuration parameters for altimetry datasets (YAML or constructor kwargs)."""

    data_dir: str  # Always required
    mission: Optional[str] = None  # e.g., 'is2', 's3', 'cs2'
    long_name: Optional[str] = None  # Name to describe the dataset
    # Pattern to match files (e.g., "**/CS_OFFL_SIR_TDP_LI*.nc" for cryotempo data)
    search_pattern: Optional[str] = None
    yyyymm_str_fname_indices: Optional[List[int]] = (
        None  # [start, end] indices for date in filename
    )
    dataset_epoch: Optional[str] = None  # The epoch the time variable is relative to
    latitude_param: Optional[str] = None  # latitude variable name (and groups if needed)
    longitude_param: Optional[str] = None  # longitude variable name (and groups if needed)
    elevation_param: Optional[str] = None  # elevation variable name (and groups if needed)
    time_param: Optional[str] = None  # time variable name (and groups if needed)
    # Options for particular datasets
    power_param: Optional[str] = None  # power variable name (and groups if needed)
    mode_param: Optional[str] = None  # mode variable name (and groups if needed)
    quality_param: Optional[str] = None  # quality variable name (and groups if needed)
    uncertainty_param: Optional[str] = None  # uncertainty variable name (and groups if needed)
    latitude_nadir_param: Optional[str] = (
        None  # nadir latitude variable name (and groups if needed)
    )
    longitude_nadir_param: Optional[str] = None  # nadir longitude variable
    beams: Optional[List[str]] = field(default_factory=list)
    #


class DatasetHelper(DatasetConfig):
    """
    Helper class to load file paths and variables from altimetry datasets
    in NetCDF or HDF5 format.

    Supports two configuration methods:
    1. YAML configuration file
    2. Direct keyword arguments to the constructor

    ---
    Parameters:
    - base_dir (str): Base directory for the dataset.
    - dataset_yaml (str | None): Optional path to a YAML configuration file.
    - **config_params: Additional configuration parameters (or alternative to YAML file).

    Example YAML configuration: datasets/definitions/cryotempo.yml
    """

    def __init__(
        self,
        data_dir: str,
        dataset_yaml: Optional[str] = None,
        **config_params: Any,
    ):
        config_dict: dict[str, Any] = {"data_dir": data_dir}
        if dataset_yaml:
            loaded = self._load_dataset_config(dataset_yaml)
            config_dict.update(loaded)
        config_dict.update(config_params)
        super().__init__(**config_dict)  # type: ignore[arg-type]

    def _load_dataset_config(self, dataset_yaml: str | Path) -> dict[str, Any]:
        """Load dataset configuration from YAML file.

        Args:
            dataset_yaml (str | Path): Path to YAML configuration file.

        Returns:
            dict[str, Any]: Configuration parameters.

        Raises:
            FileNotFoundError: If YAML file not found.
        """
        if isinstance(dataset_yaml, str):
            yaml_file = Path(dataset_yaml)
        else:
            yaml_file = dataset_yaml

        if yaml_file.is_file():
            with open(yaml_file, "r", encoding="utf-8") as f:
                result: dict[str, Any] = yaml.safe_load(f)
                return result
        raise FileNotFoundError(f"YAML config file not found: {yaml_file}")

    def get_files_dir(
        self,
        cyclenum: int | None = None,
    ) -> Path:
        """Return L2 directory path, optionally filtered by cycle number.

        Args:
            cyclenum (int | None): Cycle number to filter by.

        Returns:
            Path: L2 directory path.

        Raises:
            FileNotFoundError: If cycle directory not found.
        """
        cycle_format = [
            f"cycle_{cyclenum:03d}",
            f"Cycle_{cyclenum:03d}",
            f"CYCLE{cyclenum:02d}",
            f"Cycle{cyclenum:03d}",
        ]
        for f in cycle_format:
            directory = f"{self.data_dir}/{f}"
            if Path(directory).is_dir():
                return Path(directory)
        raise FileNotFoundError(
            f"Cycle directory for cycle {cyclenum} \
            not found in {directory} for formats {cycle_format}"
        )

    def get_product_startdate_from_filename(self, filename: Path) -> datetime:
        """Extract product start date from filename using configured indices.

        Args:
            filename (Path): Path to data file.

        Returns:
            datetime: Product start date.
        """
        if ".SEN3" in filename.parent.name:
            fname = filename.parent.name
        else:
            fname = filename.name

        assert self.yyyymm_str_fname_indices is not None
        date_obj = datetime.strptime(
            fname[self.yyyymm_str_fname_indices[0] : self.yyyymm_str_fname_indices[1]], "%Y%m%d"
        )
        return date_obj

    def get_files_and_dates(
        self,
        cycle: int | None = None,
        min_dt_time: str | datetime | None = None,
        max_dt_time: str | datetime | None = None,
        hemisphere: str | None = None,
    ) -> np.ndarray:
        """
        Get a structured numpy array of valid file and their dates from the filename.

        Options:
            Filter to cycle number.
            Filter to a min/max datetime range.
            Filter to hemisphere ('north' or 'south') for IS2 data.

        Args:
            cycle (int | None): Filter by cycle number.
            min_dt_time (str | datetime | None): Minimum date filter
                (formats: YYYYMMDD, YYYY/MM/DD, YYYY.MM.DD).
            max_dt_time (str | datetime | None): Maximum date filter (same formats).
            hemisphere (str | None): Filter to 'north' or 'south'.

        Returns:
            np.ndarray: Structured array with:
                'path', 'date', 'year', 'month' fields (and 'cycle' if filtered).
        """

        def _get_min_max_as_datetime(min_dt: str, max_dt: str) -> tuple[datetime, datetime]:
            if "/" in min_dt:
                fmt = "%Y/%m/%d"
            elif "." in min_dt:
                fmt = "%Y.%m.%d"
            else:
                fmt = "%Y%m%d"
            min_datetime = datetime.strptime(min_dt, fmt)
            max_datetime = datetime.strptime(max_dt, fmt)
            return min_datetime, max_datetime

        def _get_search_dir_by_date(l2_dir, min_dt, max_dt):
            if min_dt and max_dt:
                if min_dt.year == max_dt.year:
                    if min_dt.month == max_dt.month:
                        candidate = l2_dir / f"{min_dt.year:04d}" / f"{min_dt.month:02d}"
                    else:
                        candidate = l2_dir / f"{min_dt.year:04d}"
                    if candidate.is_dir():
                        return candidate
            return l2_dir

        def _get_file_by_hemisphere(self, hemisphere, file_and_dates):

            # Handle IS2 files
            if "ATL06" in self.search_pattern:
                if hemisphere.upper() == "NORTH":
                    region_codes = [3, 4, 5]
                else:
                    region_codes = [10, 11, 12]
                # Extract region code from filename and filter
                mask = [int(Path(f).name[-12:-10]) in region_codes for f in file_and_dates["path"]]
                return file_and_dates[mask]
            return file_and_dates

        def _get_cycle_number(file):
            cycle_patterns = [
                r"cycle_(\d{3})",
                r"Cycle_(\d{3})",
                r"CYCLE(\d{2,3})",
                r"Cycle(\d{3})",
            ]
            parent = str(Path(file).parent)
            for pattern in cycle_patterns:
                match = re.search(pattern, parent)
                if match:
                    return int(match.group(1))
            return None

        if isinstance(min_dt_time, str) and isinstance(max_dt_time, str):
            min_dt_time, max_dt_time = _get_min_max_as_datetime(min_dt_time, max_dt_time)

        # Filter search directory by cycle and date range
        search_dir = _get_search_dir_by_date(
            (Path(self.data_dir) if cycle is None else self.get_files_dir(cycle)),
            min_dt_time,
            max_dt_time,
        )

        if self.search_pattern is None:
            raise ValueError("search_pattern must be set")

        valid_files = []
        for file in Path(search_dir).rglob(self.search_pattern):
            date_obj = self.get_product_startdate_from_filename(file)
            if not isinstance(date_obj, datetime):
                continue
            if min_dt_time and max_dt_time:
                if isinstance(min_dt_time, datetime) and isinstance(max_dt_time, datetime):
                    if not min_dt_time <= date_obj <= max_dt_time:
                        continue
            row = [file, date_obj, date_obj.year, date_obj.month]
            if cycle is not None:
                row.append(_get_cycle_number(file))
            valid_files.append(tuple(row))

        dtype = [("path", object), ("date", "O"), ("year", int), ("month", int)]
        if cycle is not None:
            dtype.append(("cycle", object))

        if hemisphere:
            return _get_file_by_hemisphere(self, hemisphere, np.array(valid_files, dtype=dtype))

        return np.array(valid_files, dtype=dtype)

    def get_unified_time_epoch_offset(
        self, goal_epoch_str: str = "1991-01-01", this_epoch_str: str | None = None
    ) -> float:
        """Calculate offset in seconds between dataset epoch and target epoch.

        Args:
            goal_epoch_str (str): Target epoch (default: '1991-01-01').
            this_epoch_str (str | None): Dataset epoch; uses config if None.

        Returns:
            float: Offset in seconds from goal_epoch to dataset epoch.
        """
        if this_epoch_str is None:
            this_epoch_str = self.dataset_epoch
        assert this_epoch_str is not None
        this_epoch = datetime.fromisoformat(this_epoch_str)  # + timedelta(days=1)
        goal_epoch = datetime.fromisoformat(goal_epoch_str)
        # Calculate offset between epochs in seconds
        offset = (this_epoch - goal_epoch).total_seconds()

        return int(offset)

    def get_file_orbital_direction(
        self,
        nc: Dataset,
        latitude: np.ndarray | None = None,
    ) -> np.ndarray:
        """Determine satellite orbital direction (ascending/descending) per measurement.

        Tries CryoTEMPO method first, then infers from latitude monotonicity.

        Args:
            nc (Dataset): NetCDF dataset.
            latitude (np.ndarray | None): Latitude array, retrieved from config if None.

        Returns:
            np.ndarray: Boolean array where True=ascending, False=descending.
        """
        try:
            return self.get_measurement_directions_cryotempo(nc, latitude)
        except (AttributeError, KeyError, ValueError):
            pass

        if latitude is None:
            if self.latitude_param is not None and nc is not None:
                assert self.latitude_nadir_param is not None
                latitude = self.get_variable(
                    nc, self.latitude_nadir_param
                )  # type: ignore[assignment]
            else:
                raise ValueError("No latitude data provided or available.")

        assert latitude is not None
        asc_directions = np.zeros_like(latitude, dtype=bool)
        asc_directions[0] = latitude[1] > latitude[0]
        asc_directions[1:-1] = latitude[2:] > latitude[1:-1]
        asc_directions[-1] = latitude[-1] > latitude[-2]

        return asc_directions

    def get_measurement_directions_cryotempo(
        self, nc: Dataset, latitude: np.ndarray | None = None
    ) -> np.ndarray:
        """Get ascending/descending flags from CryoTEMPO product attributes.

        Uses ascending_start_record and descending_start_record global attributes.

        Args:
            nc (Dataset): NetCDF dataset.
            latitude (np.ndarray | None): Latitude array, retrieved from config if None.

        Returns:
            np.ndarray[bool]: True=ascending, False=descending for each measurement.

        Raises:
            ValueError: If attribute combination not supported.
        """

        ascending_start = nc.ascending_start_record
        descending_start = nc.descending_start_record

        if latitude is None:
            assert self.latitude_param is not None
            latitude = self.get_variable(nc, self.latitude_param, False)  # type: ignore[assignment]

        assert latitude is not None
        if ascending_start == "None":
            asc_directions = np.zeros_like(latitude, dtype=bool)
        elif descending_start == "None":
            asc_directions = np.ones_like(latitude, dtype=bool)
        elif ascending_start == 0 and descending_start > 0:
            asc_directions = np.ones_like(latitude, dtype=bool)
            asc_directions[descending_start:] = False
        elif descending_start == 0 and ascending_start > 0:
            asc_directions = np.zeros_like(latitude, dtype=bool)
            asc_directions[ascending_start:] = True
        else:
            raise ValueError(
                f"combination of nc.ascending_start_record {ascending_start}"
                f"and nc.descending_start_record {descending_start} not"
                " supported"
            )
        return asc_directions

    def get_variable(
        self, nc: Dataset, nc_var_path: str, return_beams: bool = False, replace_fill: bool = True
    ) -> np.ndarray:
        """Retrieve variable from NetCDF file, handling nested groups and beam data.

        For is2 data with beams, concatenates data across all configured beams.

        Args:
            nc (Dataset): NetCDF dataset.
            nc_var_path (str): Variable path with groups separated by '/' (e.g., 'beam1/heights').
            return_beams (bool): If True with beams, return beam identifiers instead of data.
            replace_fill (bool): Replace fill values (scaled) with NaN.

        Returns:
            np.ndarray: Variable data, or concatenated beam data, or beam identifiers.
        """

        def _get_var(dataset, path):
            try:
                var = dataset
                for part in path.split("/"):
                    var = var[part]
                    if var is None:
                        return np.array([])  # Variable not found
                return var[:]
            except KeyError:
                return np.array([])  # Variable not found

        if self.beams:
            data_list: list[np.ndarray] = []
            for beam in self.beams:
                var_data = _get_var(nc, f"{beam}/{nc_var_path}")
                if var_data.size > 0:
                    if return_beams:
                        data_list.append(np.full(var_data.shape[0], beam))
                    else:
                        data_list.append(var_data)

            data_array = np.concatenate(data_list, axis=0) if data_list else np.array([])
            return data_array

        # No beams, just return the variable directly
        var_data = _get_var(nc, nc_var_path)

        if replace_fill and var_data.size > 0:
            var = nc[nc_var_path]
            scale_factor = getattr(var, "scale_factor", 0.0)
            add_offset = getattr(var, "add_offset", 0.0)
            fill_value = getattr(var, "_FillValue", None)

            if fill_value is not None:
                scaled_fill = fill_value * scale_factor + add_offset
                var_data = np.where(
                    np.isclose(var_data, scaled_fill, atol=1e-6)
                    | np.isclose(var_data, -scaled_fill, atol=1e-6),
                    np.nan,
                    var_data,
                )

        return var_data
