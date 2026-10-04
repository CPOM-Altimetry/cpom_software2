"""pytests of cpom.altimetry.projects.csqa.ncreader: the h5py reader gives the same
dimensions, attributes and masked, scaled values as netCDF4"""

import glob

import numpy as np
import pytest
from netCDF4 import Dataset  # pylint: disable=no-name-in-module

from cpom.altimetry.projects.csqa.ncreader import NcDataset

PRODUCT_FILES = [
    files[0]
    for files in (
        sorted(glob.glob("/raid6/cpdata/SATS/RA/CRY/L2/GDR-A/2026/08/CS_*_F001.nc")),
        sorted(glob.glob("/raid6/cpdata/SATS/RA/CRY/L2I/SIN/2026/08/CS_*_F001.nc")),
        sorted(glob.glob("/raid6/cpdata/SATS/RA/CRY/L2I/LRM/2011/01/CS_*_E001.nc")),
    )
    if files
]

pytestmark = pytest.mark.requires_external_data


@pytest.mark.skipif(not PRODUCT_FILES, reason="test products not available")
@pytest.mark.parametrize("path", PRODUCT_FILES)
def test_same_as_netcdf4(path):
    """every variable of a product reads the same as with netCDF4"""
    with Dataset(path) as ref, NcDataset(path) as nc:
        assert set(ref.variables) == set(nc.variables)
        assert "no_such_variable" not in nc.variables
        assert nc.variables.get("no_such_variable") is None
        for name, ref_var in ref.variables.items():
            var = nc.variables[name]
            assert var.dimensions == tuple(ref_var.dimensions), name
            assert set(var.ncattrs()) == set(ref_var.ncattrs()), name
            for attr in ref_var.ncattrs():
                assert np.array_equal(
                    np.asarray(getattr(var, attr)), np.asarray(ref_var.getncattr(attr))
                ), (name, attr)
            ref_vals, vals = ref_var[:], var[:]
            ref_mask, mask = np.ma.getmaskarray(ref_vals), np.ma.getmaskarray(vals)
            assert np.array_equal(mask, ref_mask), name
            if vals.dtype.kind in "iuf":
                assert np.array_equal(
                    np.asarray(vals, dtype=float)[~mask], np.asarray(ref_vals, dtype=float)[~mask]
                ), name
        assert {k: str(nc.getncattr(k)) for k in nc.ncattrs()} == {
            k: str(ref.getncattr(k)) for k in ref.ncattrs()
        }
