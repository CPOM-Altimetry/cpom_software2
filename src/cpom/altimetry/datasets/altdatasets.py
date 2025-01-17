"""cpom.altimetry.datasets.altdatasets.py

# Purpose

class AltDataset to support Altimetry data sets

"""

import numpy as np
from netCDF4 import Dataset  # pylint: disable=E0611

supported_datasets = ["cryotempo_li"]


class AltDataset:  # pylint: disable=too-few-public-methods
    """class to support Altimetry data sets"""

    def __init__(self, name: str):
        """class initialization

        Args:
            name (str): dataset name.
        """

        self.name = name

        if name not in supported_datasets:
            raise ValueError(f"{name} not a supported dataset in AltDataset class")

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
