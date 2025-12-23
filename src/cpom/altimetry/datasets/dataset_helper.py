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
    """
    List of possible configuration parameters for a dataset.
    These can be set in a YAML file or passed as keyword arguments to the constructor.
    This class is used as a base class for DatasetHelper.
    """

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
            dataset_yaml (str | Path): Path to the YAML configuration file of the dataset.

        Raises:
            FileNotFoundError: If the YAML configuration file is not found.

        Returns:
            dict[str, Any]: The loaded dataset configuration.
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
        """Get the L2 directory for a data product.
        If specified filter to specified cycle.

        Args:
            cyclenum (int): Cycle number to filter by. Defaults to None.
        Returns:
            str: The L2 directory path.
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
        """
        Extract L2 product start date from the filenam using the
        filename indicies defined in the configuration.

        Args:
            filename (Path): Full path to the L2 file.
        Returns :
            datetime: The file start date as a datetime object.
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

    # pylint: disable=R0913,R0914,R0915,R0917
    def get_files_and_dates(
        self,
        cycle: int | None = None,
        min_dt_time: str | datetime | None = None,
        max_dt_time: str | datetime | None = None,
        hemisphere: str | None = None,
        return_cycle_number: bool = False,
    ) -> np.ndarray:
        """
        Get a structured numpy array of valid files and their dates from the filename.
        Optionally include cycle numberin the output.

        Options:
            Filter to cycle number or a min/max datetime range.
            Filter to hemisphere if in the directory path or filename,
            in _get_file_by_hemisphere.

        Args:
            cycles (str | None, optional): Cycle number to filter files. Defaults to None.
            min_dt_time (str | datetime | None, optional):
            Minimum datetime to filter files. Defaults to None.
            max_dt_time (str | datetime | None, optional):
            Maximum datetime to filter files. Defaults to None.
            hemisphere (str, optional): Filter to hemisphere 'north' or 'south'. Defaults to None.
            return_cycle_number (bool, optional): Whether to include cycle number in the output.
            Defaults to False.

        Returns:
            search_dir: np.ndarray: Structured array with file paths and dates.
        """

        def _get_min_max_as_datetime(
            min_dt_time: str, max_dt_time: str
        ) -> tuple[datetime, datetime]:
            min_dt: datetime
            max_dt: datetime
            if "/" in min_dt_time:
                min_dt = datetime.strptime(min_dt_time, "%Y/%m/%d")
                max_dt = datetime.strptime(max_dt_time, "%Y/%m/%d")
            elif "." in min_dt_time:
                min_dt = datetime.strptime(min_dt_time, "%Y.%m.%d")
                max_dt = datetime.strptime(max_dt_time, "%Y.%m.%d")
            else:
                min_dt = datetime.strptime(min_dt_time, "%Y%m%d")
                max_dt = datetime.strptime(max_dt_time, "%Y%m%d")
            return min_dt, max_dt

        def _get_search_dir_by_date(l2_dir, min_dt_time, max_dt_time):
            if min_dt_time and max_dt_time:
                if min_dt_time.year == max_dt_time.year:
                    if min_dt_time.month == max_dt_time.month:
                        search_dir = l2_dir / f"{min_dt_time.year:04d}" / f"{min_dt_time.month:02d}"
                        if not search_dir.is_dir():
                            search_dir = l2_dir
                    else:
                        search_dir = l2_dir / f"{min_dt_time.year:04d}"
                        if not search_dir.is_dir():
                            search_dir = l2_dir
                search_dir = l2_dir
            else:
                search_dir = l2_dir
            return search_dir

        def _get_file_by_hemisphere(self, hemisphere, file_and_dates):

            # Handle IS2 files
            if "ATL06" in self.search_pattern:
                if hemisphere.upper() == "NORTH":
                    region_codes = [3, 4, 5]
                else:
                    region_codes = [10, 11, 12]
                # Extract region code from filename and filter
                mask = [int(Path(f).name[-12:-10]) in region_codes for f in file_and_dates['path']]
                file_and_dates = file_and_dates[mask]

            return file_and_dates

        def _get_cycle_number(file):
            cycle_patterns = [
                r"cycle_(\d{3})",
                r"Cycle_(\d{3})",
                r"CYCLE(\d{2,3})",
                r"Cycle(\d{3})",
            ]
            for pattern in cycle_patterns:
                match = re.search(pattern, str(Path(file).parent))
                if match:
                    return int(match.group(1))
            return None

        if isinstance(min_dt_time, str):
            assert isinstance(max_dt_time, str)
            min_dt_time, max_dt_time = _get_min_max_as_datetime(min_dt_time, max_dt_time)

        # Get narrow search directory based on min and max date if exists
        if cycle is None:
            l2_dir = Path(self.data_dir)
        else:
            l2_dir = self.get_files_dir(cycle)

        search_dir = _get_search_dir_by_date(l2_dir, min_dt_time, max_dt_time)

        assert self.search_pattern is not None
        valid_files = []
        for file in Path(search_dir).rglob(self.search_pattern):
            date_obj = self.get_product_startdate_from_filename(file)
            if isinstance(date_obj, datetime):
                if min_dt_time is not None and max_dt_time is not None:
                    assert isinstance(min_dt_time, datetime)
                    assert isinstance(max_dt_time, datetime)
                    if min_dt_time <= date_obj <= max_dt_time:
                        row = [file, date_obj, date_obj.year, date_obj.month]
                        if return_cycle_number:
                            row.append(_get_cycle_number(file))
                        valid_files.append(tuple(row))
                    else:
                        continue
                else:
                    row = [file, date_obj, date_obj.year, date_obj.month]
                    if return_cycle_number:
                        row.append(_get_cycle_number(file))
                    valid_files.append(tuple(row))
            else:
                continue

        dtype = [("path", object), ("date", "O"), ("year", int), ("month", int)]
        if return_cycle_number:
            dtype.append(("cycle", object))
            
        if hemisphere:
            valid_files = (
                _get_file_by_hemisphere(self, hemisphere, np.array(valid_files, dtype=dtype))
                if hemisphere
                else valid_files
            )

        dtype = [("path", object), ("date", "O"), ("year", int), ("month", int)]
        if return_cycle_number:
            dtype.append(("cycle", object))
        return np.array(valid_files, dtype=dtype)

    def get_unified_time_epoch_offset(
        self, goal_epoch_str: str = "1991-01-01", this_epoch_str: str | None = None
    ) -> float:
        """
        Convert a timestamp from one custom epoch to another.

        Args:
        - goal_epoch_str: str, the target epoch (default: "1991-01-01")
        - this_epoch_str: str, the original epoch of the dataset

        Returns:
        - float: the timestamp relative to `goal_epoch_str`
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
        """Get the orbital direction of the satellite based on latitude.
        For datasets with Nadir Latitude variables, checks if the latitudes are
        increasing or decreasing.

        For Cryotempo products calls `get_measurement_directions_cryotempo`.
        Which uses a global attribute to determine the direction of the track.

        Args:
            latitude (np.ndarray, optional): latitude array.
                    If None, will be retrieved from config.
            nc (Dataset, optional): netcdf Dataset.
        Returns:
            np.ndarray: array of bool track directions. Ascending or Descending.
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
        """Get the direction of each measurement (ie ascending or descending)
        as a numpy array of boolean values.

        This function is for CryoTEMPO products.
        CryoTEMPO L2 products include global attributes to identify
        ascending and descending records for the track.

        Args:
            nc (Dataset): netCDF Dataset
            latitude (np.ndarray, optional): latitude array. If None, will be retrieved from config.
        Returns:
            directions (np.ndarray[bool]) :
            array of bool track directions (asc==True, desc==False) of equal length
            to the input latitude array.
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
        """Retrieve a variable from a NetCDF file, handling groups if necessary.

        Can handle both regular variables and beam-specific variables for is2.
        If class has attribute `beams`, it will return a concatenated array of data in the
        passed beams.

        Args:
            nc (Dataset or hdf5.Dataset): The dataset object
            nc_var_path (str): The path to the variable within the file,
                with groups separated by '/'.
            return_beams (bool): Return an array of beam identifiers if True.
            replace_fill (bool): Whether to replace fill values with NaN.
        Raises:
            KeyError: If the variable or group is not found in the file.
        Returns:
            np.ndarray or tuple:
                - If use_beams=False: The retrieved variable as an array.
                - If use_beams=True and return_beams=False: Concatenated data from all beams.
                - If use_beams=True and return_beams=True: Tuple of (data, beam_ids).
        """
        

        def _get_var(dataset, path):
            try:
                var = dataset
                for part in path.split("/"):
                    var = var[part]
                    if var is None:
                        return np.array([])  # Variable not found
                data = var[:]  # automatically scaled & masked if applicable
                return data
            except KeyError:
                return np.array([])  # Variable not found

        if self.beams:
            data_list: list[np.ndarray] = []
            beam_list: list[np.ndarray] = []
            for beam in self.beams:
                var_data = _get_var(nc, f"{beam}/{nc_var_path}")
                if var_data is not None and len(var_data) > 0:
                    data_list.append(var_data)
                    if return_beams:
                        beam_list.append(np.full(var_data.shape[0], beam))

            data_array = np.concatenate(data_list, axis=0) if data_list else np.array([])
            if return_beams:
                beam_array = np.concatenate(beam_list, axis=0) if beam_list else np.array([])
                return beam_array
            return data_array

        # No beams, just return the variable directly
        var_data = _get_var(nc, nc_var_path)

        if replace_fill:
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

        if var_data is None:
            return np.array([])
        return var_data
