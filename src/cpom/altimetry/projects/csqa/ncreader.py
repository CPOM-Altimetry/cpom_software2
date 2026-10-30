"""cpom.altimetry.projects.csqa.ncreader

A fast, read-only reader of netCDF-4 (HDF5) product files using h5py, providing the subset of
the netCDF4.Dataset interface used by the CSQA loader:

    with NcDataset(path) as nc:
        var = nc.variables.get("sig0_1_20_ku")     # or nc["sig0_1_20_ku"]
        var.dimensions                              # ('time_20_ku',)
        var.units, getattr(var, "coordinates", "")  # attributes
        values = var[:]                             # masked, scaled array as netCDF4

Opening a file with netCDF4 reads the metadata of every variable, which takes ~40-80 ms for
CryoSat-2 L2 / L2i products (L2i products have ~180 variables), while h5py reads metadata on
demand (~3 ms per file), so cycles of thousands of product files are read much faster.

Values are masked and unpacked as netCDF4-python does by default: values equal to _FillValue
(or missing_value) and outside valid_range / valid_min / valid_max (compared with the packed
values) are masked, then scale_factor and add_offset are applied.
"""

from collections.abc import Iterator, Mapping
from typing import Any

import h5py
import numpy as np

# netCDF default fill values, used to mask variables without a _FillValue attribute (except
# byte types, which netCDF4-python does not mask by default)
_DEFAULT_FILLVALS = {
    "i2": -32767,
    "u2": 65535,
    "i4": -2147483647,
    "u4": 4294967295,
    "i8": -9223372036854775806,
    "u8": 18446744073709551614,
    "f4": 9.969209968386869e36,
    "f8": 9.969209968386869e36,
}

# attributes of the HDF5 storage of netCDF-4 files, not netCDF attributes
_INTERNAL_ATTRS = {
    "DIMENSION_LIST",
    "REFERENCE_LIST",
    "CLASS",
    "NAME",
    "_Netcdf4Dimid",
    "_Netcdf4Coordinates",
    "_nc3_strict",
    "_NCProperties",
}


def _attr_value(value: Any) -> Any:
    """an attribute value as netCDF4-python returns it: str for text, a numpy scalar for a
    single value, otherwise an array"""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        if value.dtype.kind in "SO":
            items = [
                v.decode("utf-8", errors="replace") if isinstance(v, bytes) else v
                for v in value.ravel()
            ]
            return items[0] if len(items) == 1 else items
        if value.size == 1:
            return value.ravel()[0]
    return value


class NcVariable:
    """A netCDF variable of an open NcDataset"""

    def __init__(self, dataset: h5py.Dataset, name: str):
        self._ds = dataset
        self.name = name
        self._attrs = {
            key: _attr_value(val)
            for key, val in dataset.attrs.items()
            if key not in _INTERNAL_ATTRS
        }
        dims = []
        for i, dim in enumerate(dataset.dims):
            if len(dim):
                # the first dimension scale attached to the dimension (its netCDF dimension)
                dims.append(dim[0].name.lstrip("/"))
            elif dataset.is_scale and dataset.ndim == 1:
                dims.append(name)  # a coordinate variable: its own dimension
            else:
                dims.append(f"phony_dim_{i}")
        self.dimensions = tuple(dims)

    @property
    def shape(self) -> tuple[int, ...]:
        """shape of the variable"""
        return self._ds.shape

    @property
    def dtype(self) -> np.dtype:
        """data type of the stored (packed) values"""
        return self._ds.dtype

    def ncattrs(self) -> list[str]:
        """names of the variable's attributes"""
        return list(self._attrs)

    def getncattr(self, name: str) -> Any:
        """value of an attribute"""
        return self._attrs[name]

    def __getattr__(self, name: str) -> Any:
        try:
            return self.__dict__["_attrs"][name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def _mask(self, raw: np.ndarray) -> np.ndarray:
        """mask of the missing values of packed data, as netCDF4-python's default masking"""
        mask = np.zeros(raw.shape, dtype=bool)
        attrs = self._attrs
        for key in ("missing_value", "_FillValue"):
            if key in attrs:
                for value in np.atleast_1d(np.asarray(attrs[key], dtype=raw.dtype)):
                    mask |= np.isnan(raw) if np.isnan(value) else raw == value
        if "_FillValue" not in attrs:
            default = _DEFAULT_FILLVALS.get(raw.dtype.str[1:])
            if default is not None:
                mask |= raw == np.asarray(default).astype(raw.dtype)
        valid_min = valid_max = None
        if "valid_range" in attrs:
            valid_min, valid_max = np.asarray(attrs["valid_range"]).ravel()[:2]
        valid_min = attrs.get("valid_min", valid_min)
        valid_max = attrs.get("valid_max", valid_max)
        if valid_min is not None:
            mask |= raw < np.asarray(valid_min).astype(raw.dtype)
        if valid_max is not None:
            mask |= raw > np.asarray(valid_max).astype(raw.dtype)
        return mask

    def __getitem__(self, key) -> np.ma.MaskedArray:
        raw = np.asarray(self._ds[key])
        if raw.dtype.kind not in "iuf":
            return np.ma.masked_array(raw)
        mask = self._mask(raw)
        scale = self._attrs.get("scale_factor")
        offset = self._attrs.get("add_offset")
        data: np.ndarray = raw
        if scale is not None:
            data = data * scale
        if offset is not None:
            data = data + offset
        return np.ma.masked_array(data, mask=mask)


def _is_dimension_only(dataset: h5py.Dataset) -> bool:
    """True for the HDF5 dataset of a netCDF dimension without a coordinate variable"""
    name = dataset.attrs.get("NAME", b"")
    name = name.decode("utf-8", errors="replace") if isinstance(name, bytes) else str(name)
    return name.startswith("This is a netCDF dimension but not a netCDF variable")


class _Variables(Mapping):
    """the variables of a dataset. Variables are only inspected when used (checking every
    dataset of a product with ~180 variables would take most of the time to read a file)"""

    def __init__(self, h5file: h5py.File):
        self._file = h5file
        self._links = list(h5file.keys())  # names only: cheap
        self._link_set = set(self._links)
        self._is_variable: dict[str, bool] = {}
        self._variables: dict[str, NcVariable] = {}

    def _check(self, name: str) -> bool:
        """True if a link of the file is a netCDF variable (cached)"""
        if name not in self._is_variable:
            obj = self._file.get(name) if name in self._link_set else None
            self._is_variable[name] = isinstance(obj, h5py.Dataset) and not _is_dimension_only(obj)
        return self._is_variable[name]

    def __getitem__(self, name: str) -> NcVariable:
        if name not in self._variables:
            if not self._check(name):
                raise KeyError(name)
            self._variables[name] = NcVariable(self._file[name], name)
        return self._variables[name]

    def __contains__(self, name) -> bool:
        return isinstance(name, str) and self._check(name)

    def __iter__(self) -> Iterator[str]:
        return (name for name in self._links if self._check(name))

    def __len__(self) -> int:
        return sum(1 for _ in self)


class NcDataset:
    """A read-only netCDF-4 file, opened with h5py (see module docstring)"""

    def __init__(self, path: str):
        self.path = path
        self._file = h5py.File(path, "r")
        self.variables = _Variables(self._file)

    def __getitem__(self, name: str) -> NcVariable:
        return self.variables[name]

    def ncattrs(self) -> list[str]:
        """names of the global attributes"""
        return [k for k in self._file.attrs if k not in _INTERNAL_ATTRS]

    def getncattr(self, name: str) -> Any:
        """value of a global attribute"""
        return _attr_value(self._file.attrs[name])

    def close(self):
        """close the file"""
        self._file.close()

    def __enter__(self) -> "NcDataset":
        return self

    def __exit__(self, *exc):
        self.close()
