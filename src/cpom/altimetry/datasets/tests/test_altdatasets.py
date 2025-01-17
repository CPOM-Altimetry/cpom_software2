"""pytests for cpom.altimetry.datasets.altdatasets.py"""

import os

import numpy as np
import pytest
from netCDF4 import Dataset  # pylint: disable=E0611

from cpom.altimetry.datasets.altdatasets import AltDataset


@pytest.mark.parametrize(
    "dir_path,file_name, ascending, both, dataset_name",
    [
        # Ascending CryoTEMPO file over Greenland,
        (
            f'{os.environ["CPOM_SOFTWARE_DIR"]}/testdata/cs2/l2/cryotempo/c/landice/greenl/',
            "CS_OFFL_SIR_TDP_LI_GREENL_20150131T182746_20150131T183002_07_00423_C001.nc",
            True,
            False,
            "cryotempo_li",
        ),
        # Ascending CryoTEMPO file over Antarctica,
        (
            f'{os.environ["CPOM_SOFTWARE_DIR"]}/testdata/cs2/l2/cryotempo/c/landice/antarc/',
            "CS_OFFL_SIR_TDP_LI_ANTARC_20100718T001450_20100718T001611_01_02650_B001.nc",
            True,
            False,
            "cryotempo_li",
        ),
        # Descending CryoTEMPO file over Greenland,
        (
            f'{os.environ["CPOM_SOFTWARE_DIR"]}/testdata/cs2/l2/cryotempo/c/landice/greenl/',
            "CS_OFFL_SIR_TDP_LI_GREENL_20111218T151752_20111218T151903_03_05253_C001.nc",
            False,
            False,
            "cryotempo_li",
        ),
        # Descending[0-614],Ascending[615..] CryoTEMPO file over Antarctica,
        (
            f'{os.environ["CPOM_SOFTWARE_DIR"]}/testdata/cs2/l2/cryotempo/c/landice/antarc/',
            "CS_OFFL_SIR_TDP_LI_ANTARC_20100731T203419_20100731T203857_02_03271_C001.nc",
            True,
            True,
            "cryotempo_li",
        ),
    ],
)
def test_find_measurement_directions(dir_path, file_name, ascending, both, dataset_name):
    """pytest for AltDatset.test_find_measurement_directions()"""

    this_ds = AltDataset(dataset_name)

    nc = Dataset(os.path.join(dir_path, file_name))

    ascending_locs = this_ds.find_measurement_directions(nc)

    if ascending and not both:
        assert np.all(ascending_locs), "should be an ascending track"
    elif not ascending and not both:
        assert np.all(~ascending_locs), "should be descending track"
    elif both:
        assert (
            np.any(ascending_locs) and not np.all(ascending_locs) and not ascending_locs[0]
        ), "should be mixed with descending first"
