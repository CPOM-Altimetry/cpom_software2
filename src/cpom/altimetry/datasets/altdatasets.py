"""cpom.altimetry.datasets.altdatasets.py

# Purpose

class AltDataset to support Altimetry data sets

"""

import importlib
import json
import os

import numpy as np
from netCDF4 import Dataset, Variable  # pylint: disable=E0611

supported_datasets = ["cryotempo_li"]


class AltDataset:  # pylint: disable=too-few-public-methods
    """Class to support Altimetry data sets."""

    def __init__(
        self, name: str, overrides: dict | None = None, dataset_filename: str | None = None
    ):
        """
        Class initialization.

        Args:
            name (str): Dataset name.
            **overrides: Optional keyword arguments to override default parameters.
        """
        self.name = name

        try:
            self.load_dataset(overrides, dataset_filename)
        except ImportError as exc:
            raise ImportError(f"{name} not in supported area list") from exc

        if name not in supported_datasets:
            raise ValueError(f"{name} is not a supported dataset in AltDataset class")

    def load_dataset(self, overrides: dict | None = None, dataset_filename: str | None = None):
        """Load dataset settings for current dataset name"""
        if dataset_filename is None:
            try:
                module = importlib.import_module(f"cpom.altimetry.datasets.definitions.{self.name}")
            except ImportError as exc:
                raise ImportError(f"Could not load dataset definition {self.name}") from exc
        else:
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

    def get_variable(self, nc: Dataset, nc_var_path: str) -> Variable:
        """Retrieve variable from NetCDF file, handling groups if necessary.

        Args:
            nc (Dataset): The opened NetCDF dataset
            nc_var_path (str): Path to the variable (e.g. "data/ku/latitude")

        Returns:
            Variable: The NetCDF variable object
        """
        try:
            parts = nc_var_path.split("/")
            var = nc
            for part in parts:
                var = var[part]
                if var is None:
                    raise IndexError(f"NetCDF parameter '{nc_var_path}' not found.")
            return var
        except IndexError as err:
            raise IndexError("NetCDF parameter or group {err} not found") from err

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
