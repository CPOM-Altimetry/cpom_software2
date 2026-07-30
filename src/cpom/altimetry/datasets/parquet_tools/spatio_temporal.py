"""
cpom.altimetry.datasets.parquet_tools.spatio_temporal

General utility tool to run out of memory spatial and temporal
filters on large parquet altimetry archives via a small builder-pattern query ('ParquetFilter').

Two interchangeable backends implement the same builder interface:
    - DuckDBParquetFilter: SQL/DuckDB-backed. Supports polygon + H3 cell
      prefiltering and CPOM grid-mask filtering ,  bbox/time,
      plus arbitrary SQL predicates via 'where()'.
    - PolarsParquetFilter: native polars lazy-scan/streaming-sink backend.
      Faster for simple bbox/time/grid-mask filters that don't need
      polygon or H3 support. Polygon masking is based on a 2km grid.

Also provides a set of standalone helpers for building/reprojecting filter geometries from CPOM
Mask/Area instances or shapefiles.

Example :
    >>> from cpom.altimetry.datasets.parquet_tools.spatio_temporal import ParquetFilter,
        get_polygon_from_mask
    >>> mask = get_polygon_from_mask("antarctica", basin_numbers=[5, 6], target_crs=4326)
    >>> this_filter = ParquetFilter(input_path="data/*.parquet", engine = "duckdb", data_crs=4326)
    >>> this_filter.get_bounding_box(mask).get_time(
    ...     start="2020-01-01", end="2020-12-31").select(
    ...     ["x", "y", "datetime", "elevation"]).run(
    ...     output_table="filtered_data")
"""

from __future__ import annotations

import logging
import re
import shutil
import unicodedata
from datetime import date, datetime
from pathlib import Path
from typing import Any, Literal, Optional, Sequence, Union, cast, overload

import duckdb
import geopandas as gpd
import h3
import numpy as np
import pandas as pd
import polars as pl
from pyproj import Transformer
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as shp_transform
from shapely.ops import unary_union

from cpom.areas.areas import Area  # local import: optional dependency
from cpom.gridding.gridareas import GridArea  # local import: optional dependency
from cpom.masks.masks import Mask  # local import: optional dependency

Bounds = tuple[float, float, float, float]
Engine = Literal["duckdb", "polars"]
PolygonSource = Union[str, Path, gpd.GeoDataFrame]
MaskSource = Union[Mask, Area, str]

logger = logging.getLogger(__name__)


# ------------------------------#
# DuckDB connection helpers     #
# ------------------------------#
def get_db_connection(
    memory_limit: str = "150GB", temp_directory: str = ""
) -> duckdb.DuckDBPyConnection:
    """Get a DuckDB connection object.

    Args:
        memory_limit: The memory limit for the DuckDB connection.
        temp_directory: Optional path to a temporary directory for DuckDB spill files.
    Returns:
        A DuckDB connection object.
    """
    con = duckdb.connect()
    con.execute("INSTALL h3 FROM community; LOAD h3;")
    con.execute("INSTALL spatial; LOAD spatial;")
    con.execute(f"SET memory_limit='{memory_limit}'")
    if temp_directory:
        con.execute(f"SET temp_directory='{temp_directory}'")
    return con


def close_db_connection(con: duckdb.DuckDBPyConnection, temp_directory: str = "") -> None:
    """Close a DuckDB connection and delete temporary directory.

    Args:
        con: DuckDB connection object.
        temp_directory: Optional path to a temporary directory to delete.
    """
    con.close()
    if temp_directory:
        shutil.rmtree(temp_directory, ignore_errors=True)


# -------------------#
# Grid-mask helpers  #
# -------------------#
def to_mask(mask: MaskSource) -> Mask:
    """Convert a mask name, Mask instance, or Area instance to a Mask instance.

    Args:
        mask: A CPOM Mask instance, a CPOM Area instance (uses its underlying mask), or
            a string name of a supported mask.
    Returns:
        A CPOM Mask instance.
    """
    if isinstance(mask, Mask):
        return mask
    if isinstance(mask, Area):
        if mask.mask is None:
            raise ValueError(
                f"Area {mask.name!r} has no associated Mask (no maskname, or "
                "apply_area_mask_to_data=False)"
            )
        return mask.mask
    if isinstance(mask, str):
        return Mask(mask)
    raise TypeError(
        "mask must be a cpom.masks.masks.Mask, cpom.areas.areas.Area, "
        "or supported mask name string"
    )


def get_shapefile_column_and_values(
    mask: MaskSource,
    basin_numbers: list[int | str] | None = None,
) -> tuple[Optional[str], Optional[list[str]]]:
    """Resolve a basin selection on a CPOM Mask to the (column, values) pair
    needed to filter its underlying shapefile.

    Args:
        mask: A CPOM Mask instance, a CPOM Area instance (uses its underlying mask), or
            a string name of a supported mask.
        basin_numbers: Basin numbers or names to filter by, used only if the
            mask itself does not already define 'basin_numbers'.
    Returns:
        (filter_column, values): shapefile column name and the values to
        filter it by, or (None, None) if no basin filtering is needed.
    """
    resolved = to_mask(mask)
    values = resolved.resolve_basin_names(basin_numbers)
    if values is None:
        return None, None
    return resolved.shapefile_column_name, values


def get_mask_cell_indices(
    mask: Mask, basin_numbers: Optional[list[int | str]] = None
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve a basin selection on a grid-type CPOM Mask to (row, col) grid
    indices.

    Centralises the basin-resolution + numpy lookup that both
    'DuckDBParquetFilter.get_cpom_grid_mask' and
    'PolarsParquetFilter.get_cpom_grid_mask' need, so it isn't duplicated
    per backend.

    Args:
        mask: A CPOM Mask instance with mask_type == "grid".
        basin_numbers: Optional list of basin numbers or names to filter by.
            If None (and the mask has no stored default), all non-zero grid
            cells are selected.
    Returns:
        (rows, cols): integer numpy arrays of the grid row/column indices
        selected by basin_numbers. Empty arrays if nothing matches.
    """
    resolved_basins = mask.resolve_basin_numbers(basin_numbers)
    return np.where(
        mask.mask_grid > 0 if resolved_basins is None else np.isin(mask.mask_grid, resolved_basins)
    )


def _normalize_filter_text(value: object) -> str:
    """Standardise input text to remove case/whitespace/punctuation.

    E.g. "East", "east" and "East " all compare equal.
    Returns : Normalised string.
    """
    text = unicodedata.normalize("NFKC", str(value)).strip().casefold()
    text = re.sub(r"[\s\-_]+", "", text)
    return re.sub(r"[^0-9a-z]", "", text)


def _candidate_label_series(gdf: gpd.GeoDataFrame, filter_column: str) -> list[pd.Series]:
    """Get a series of labels in a GeoDataframe column.

    If exact matches on 'filter_column' fail, attempts to identify matches split across two columns.
    (e.g. "East" / "E-Ep") can still match a combined "East E-Ep" request.
    Args:
        gdf: GeoDataFrame to build series.
        filter_column: The column to match.
    Returns:
        A list of string-valued pandas Series, each a candidate to test for
        a match.
    """
    if filter_column not in gdf.columns:
        raise ValueError(
            f"Column {filter_column!r} not found in GeoDataFrame. "
            f"Available columns: {list(gdf.columns)}"
        )

    text_columns = list(gdf.select_dtypes(include=["object", "string"]).columns)
    series = [gdf[filter_column].astype(str)]
    if filter_column in text_columns:
        for other_col in text_columns:
            if other_col == filter_column:
                continue
            series.append(
                gdf[filter_column].astype(str).str.cat(gdf[other_col].astype(str), sep=" ")
            )
            series.append(
                gdf[other_col].astype(str).str.cat(gdf[filter_column].astype(str), sep=" ")
            )

    return series


def get_polygon_from_bb(minx: float, miny: float, maxx: float, maxy: float) -> BaseGeometry:
    """Build a shapely box from a bounding box.

    Args:
        minx: Minimum x-coordinate.
        miny: Minimum y-coordinate.
        maxx: Maximum x-coordinate.
        maxy: Maximum y-coordinate.
    Returns:
        A shapely.geometry.Polygon representing the bounding box.
    """
    return box(minx, miny, maxx, maxy)


def get_polygon_from_shapefile(
    source: Union[str, Path],
    target_crs: Optional[int] = None,
    filter_column: Optional[str] = None,
    values: Optional[Sequence[object]] = None,
) -> gpd.GeoDataFrame:
    """Read a shapefile into a GeoDataFrame, filter based on the values in
    filter_column, and reproject to target_crs if given.

    If an exact match on filter_column/values finds nothing, attempt to normalise input and to
    compare to matches split across columns ('_candidate_label_series').

    Args:
        source: Path to the shapefile.
        target_crs: Target coordinate reference system (EPSG code) to
            reproject to. Left unchanged if None.
        filter_column: Column name to filter features by.
        values: Values in filter_column to keep.
    Returns:
        A GeoDataFrame of the matching, reprojected geometries.
    Raises:
        ValueError: If filter_column/values are given but no feature matches.
    """
    gdf = gpd.read_file(str(source))
    gdf_all = gdf

    if filter_column is not None and values is not None:
        if filter_column not in gdf.columns:
            raise ValueError(
                f"Column {filter_column!r} not found in {source}. "
                f"Available columns: {list(gdf.columns)}"
            )

        requested = {str(v) for v in values}
        gdf = gdf_all[gdf_all[filter_column].astype(str).isin(requested)]

        if gdf.empty:
            requested_norm = {_normalize_filter_text(v) for v in requested}
            for label_series in _candidate_label_series(gdf_all, filter_column):
                label_norm = label_series.map(_normalize_filter_text)
                subset = gdf_all[label_norm.isin(requested_norm)]
                if not subset.empty:
                    gdf = subset
                    break

    if gdf.empty:
        available_values = []
        if filter_column in gdf_all.columns:
            available_values = sorted({str(v) for v in gdf_all[filter_column].dropna().unique()})
        raise ValueError(
            f"No matching geometry found in {source} for {filter_column} in {values}. "
            f"Available {filter_column} values: {available_values}"
        )

    return reproject(gdf, target_crs)


def get_polygon_from_mask(
    mask: MaskSource,
    basin_numbers: Optional[list[int | str]] = None,
    target_crs: Optional[int] = None,
) -> gpd.GeoDataFrame:
    """Build a polygon (GeoDataFrame) from a CPOM Mask or Area instance.

    Args:
        mask: A CPOM Mask instance, a CPOM Area instance (uses its underlying mask), or
            a string name of a supported mask.
        basin_numbers: Optional list of basin numbers or names to filter by.
        target_crs: Optional target coordinate reference system (EPSG code) to reproject to.
    Returns:
        A GeoDataFrame containing the polygon representing the mask.
    """
    mask = to_mask(mask)

    if hasattr(mask, "shapefile_path"):
        filter_column, values = get_shapefile_column_and_values(mask, basin_numbers)
        return get_polygon_from_shapefile(
            mask.shapefile_path, target_crs=target_crs, filter_column=filter_column, values=values
        )

    if mask.mask_type == "xylimits":
        if basin_numbers:
            logger.warning(
                "basin_numbers %s ignored: mask %r has mask_type='xylimits' "
                "and has no basin concept",
                basin_numbers,
                mask.mask_name,
            )
        polygon_xy = get_polygon_from_bb(
            mask.xlimits[0], mask.ylimits[0], mask.xlimits[1], mask.ylimits[1]
        )
        polygon_ll = shp_transform(
            lambda x, y, z=None: mask.xy_to_lonlat_transformer.transform(x, y), polygon_xy
        )
        gdf = gpd.GeoDataFrame({"geometry": [polygon_ll]}, crs="EPSG:4326")
        return reproject(gdf, target_crs)
    raise ValueError(f"mask has no shapefile and unsupported mask_type {mask.mask_type!r}")


def reproject(
    geom: Union[BaseGeometry, gpd.GeoDataFrame],
    target_crs: Optional[int],
    src_crs: Optional[int] = None,
) -> Union[BaseGeometry, gpd.GeoDataFrame]:
    """Reproject a shapely geometry or GeoDataFrame to the target CRS.

    Args:
        geom: A shapely geometry or GeoDataFrame to reproject.
        target_crs: The target coordinate reference system (EPSG code).
        src_crs (int): The source coordinate reference system (EPSG code).
    Returns:
        The reprojected shapely geometry or GeoDataFrame.
    """
    if isinstance(geom, gpd.GeoDataFrame):
        return geom.to_crs(epsg=target_crs) if target_crs is not None else geom
    if src_crs is None:
        raise ValueError("src_crs must be given when reprojecting a bare shapely geometry")
    return shp_transform(
        Transformer.from_crs(f"EPSG:{src_crs}", f"EPSG:{target_crs}", always_xy=True).transform,
        geom,
    )


def to_geometry(source: gpd.GeoDataFrame, target_crs: Optional[int] = None) -> BaseGeometry:
    """Merge a geometry source into one shapely geometry.

    Args:
        source: A shapely geometry, GeoDataFrame, or GeoSeries to merge.
    Returns:
        The merged shapely geometry, reprojected to target_crs if given.
    """
    if isinstance(source, BaseGeometry):
        raise ValueError("source must be a GeoDataFrame or GeoSeries, not a bare shapely geometry")
    if source.empty:
        raise ValueError("no geometry to merge (empty GeoDataFrame/GeoSeries)")
    values = source.geometry.values if isinstance(source, gpd.GeoDataFrame) else source.values
    crs = source.crs.to_epsg()
    if target_crs is not None and crs != target_crs:
        return cast(BaseGeometry, reproject(unary_union(values), target_crs, crs))
    return unary_union(values)


# --------------------------------------------------------------------------
# Handle returned by run(output_table=...)
# --------------------------------------------------------------------------
class TableHandle:
    """Class to represent a handle to a table materialized on a DuckDB
    connection.

    Enables querying the table after it has been created by
    DuckDBParquetFilter.run(output_table"").

    The table exists only on the connection that created it and will be dropped
    once the connection is closed.
    """

    def __init__(self, con: duckdb.DuckDBPyConnection, table_name: str):
        self.con = con
        self.table_name = table_name

    def sql(self, query: Optional[str] = None) -> duckdb.DuckDBPyRelation:
        """Run a query against this table (default 'SELECT * FROM <table>')."""
        return self.con.sql(query or f'SELECT * FROM "{self.table_name}"')

    def pl(self) -> pl.DataFrame:
        """Fetch the full table as a polars DataFrame."""
        return self.sql().pl()

    def df(self):
        """Fetch the full table as a pandas DataFrame."""
        return self.sql().df()

    def __repr__(self) -> str:
        return f"TableHandle(table_name={self.table_name!r})"


InputSource = Union[str, TableHandle, pl.DataFrame, pl.LazyFrame]


def _guard_once(obj: Any, flag_attr: str, method_name: str) -> None:
    """Helper function to prevent multiple calls to the same method on a filter
    object. Sets a flag to the function call, returns an Error if the flag is
    already set.

    Args:
        obj: Class object.
        flag_attr: The name of the flag attribute to check and set.
        method_name: The name of the method being called.
    """
    if getattr(obj, flag_attr):
        raise RuntimeError(
            f"{method_name}() already called on this filter; build a new one per query."
        )
    setattr(obj, flag_attr, True)


# --------------------------------------------------------------------------
# DuckDB Builder
# --------------------------------------------------------------------------
class DuckDBParquetFilter:
    """
    Filter a parquet store with duckdb backend.

    Functions:
        select(columns): Restrict the columns returned by the query.
        limit(n): Limit the number of rows returned.
        where(sql, params): Add an arbitrary SQL WHERE clause to the filter.
        order_by(sql): Add an ORDER BY clause to the filter.
        custom_clause(sql): Add one or more computed columns to the SELECT list.
        get_grid_cells_from_grid_area(this_grid): Add grid indices and cell-centre offsets.
        get_bounding_box(polygon, bounds): Filter to a bounding box.
        get_polygon(polygon, precise, h3_col, h3_res, h3_row_res, buffer_m):
            Polygon filter with optional H3 prefiltering and exact ST_Within test.
        get_time(time_col, start, end): Filter by time range and/or calendar components.
        get_cpom_grid_mask(mask, basin_numbers): Filter to points inside a CPOM grid mask.

    Example:
        >>> classInstance = DuckDBParquetFilter(input_path="data/*.parquet", data_crs=4326)
        >>> classInstance.get_bounding_box(polygon).get_time(start="2020-01-01",
                                                    end="2020-12-31").select(
                                                    ["x", "y", "datetime", "elevation"])
        >>> df = classInstancerun(output_table="filtered_data").pl()
    """

    # pylint: disable=too-many-instance-attributes

    _columns: Optional[list[str]]
    _limit: Optional[int]
    _order_by: Optional[str]
    _grid_mask_bound: bool

    # -----------------
    # Set up
    # -----------------
    def __init__(
        self,
        input_path: InputSource,
        data_crs: Optional[int] = None,
        lon_col: str = "x",
        lat_col: str = "y",
        memory_limit: str = "150GB",
        temp_directory: str = "",
        con: Optional[duckdb.DuckDBPyConnection] = None,
    ):
        """
        Initialize a DuckDBParquetFilter instance.

        Args:
            input_path: Path to parquet file(s) or a TableHandle from a previous run.
            data_crs: Optional EPSG code of the coordinate reference system of the data.
            lon_col: Name of the longitude column in the data.
            lat_col: Name of the latitude column in the data.
            memory_limit: Memory limit for DuckDB connection (default "150GB").
            temp_directory: Optional path to a temporary directory for DuckDB spill files.
            con: Optional existing DuckDB connection to use. If None, a new connection is created.

        Example:
            >>> classInstance = DuckDBParquetFilter(input_path="data/*.parquet", data_crs=4326)
        """
        self.input_path = input_path
        self.lon_col = lon_col
        self.lat_col = lat_col
        self.data_crs = data_crs
        self._temp_directory = temp_directory
        if isinstance(input_path, TableHandle):
            if con is not None and con is not input_path.con:
                raise ValueError(
                    "TableHandle belongs to a different DuckDB connection than 'con'; "
                    "pass con=input_path.con (or omit con) to reuse it."
                )
            con = input_path.con
        self._owns_connection = con is None
        self.con = con or get_db_connection(
            memory_limit=memory_limit, temp_directory=temp_directory
        )
        self._closed = False

        # Resolve the FROM clause based on the input source type
        if isinstance(input_path, TableHandle):
            self._from_clause = f'"{input_path.table_name}"'
        elif isinstance(input_path, (pl.DataFrame, pl.LazyFrame)):
            view_name = f"view_{id(self)}"
            self.con.register(
                view_name,
                input_path.collect() if isinstance(input_path, pl.LazyFrame) else input_path,
            )
            self._from_clause = f'"{view_name}"'
        else:
            self._from_clause = f"read_parquet('{input_path}', hive_partitioning=true)"

        self.reset()

    def close(self) -> None:
        """Close the DuckDB connection and delete the temporary directory.

        Example:
            >>> classInstance.close()
        """
        if self._closed:
            return
        if self._owns_connection:
            close_db_connection(self.con, self._temp_directory)
        self._closed = True

    def reset(self) -> "DuckDBParquetFilter":
        """Clear accumulated query state so this instance can build and run
        another query."""
        self._columns: Optional[list[str]] = None
        self._limit: Optional[int] = None
        self._clauses: list[tuple[str, list[Any]]] = []
        self._join_clauses: list[str] = []
        self._extra_columns: list[str] = []
        self._order_by: Optional[str] = None
        self._bbox_called = False
        self._time_called = False
        self._polygon_called = False
        self._grid_mask_bound = False
        return self

    def _add_clause(self, sql: str, params: Optional[list[Any]] = None) -> None:
        self._clauses.append((sql, params or []))

    # -----------------------------
    # Builder Methods
    # -----------------------------
    def select(self, columns: list[str]) -> "DuckDBParquetFilter":
        """Restrict the columns returned by the query.

        Example:
            >>> classInstance.select(["x", "y", "datetime", "elevation"])
        """
        self._columns = columns
        return self

    def limit(self, n: int) -> "DuckDBParquetFilter":
        """Limit the number of rows returned.

        Example:
            >>> classInstance.limit(1000)
        """
        self._limit = n
        return self

    def where(self, sql: str, params: Optional[list[Any]] = None) -> "DuckDBParquetFilter":
        """Add an arbitrary SQL WHERE clause to the filter. Use '?' placeholders for parameters.
        Example:
            >>> classInstance.where("elevation > ? AND quality_flag = ?", [0.0, 1])
        """
        self._add_clause(sql, params)
        return self

    def order_by(self, sql: str) -> "DuckDBParquetFilter":
        """Add an ORDER BY clause to the filter.

        Example:
            >>> classInstance.order_by("datetime ASC")
        """
        self._order_by = sql
        return self

    def custom_clause(self, sql: Union[str, list[str]]) -> "DuckDBParquetFilter":
        """Add one or more computed columns to the SELECT list.

        Accepts a single SQL string or a list of strings.

        Example:
            >>> classInstance.select(
            ["x", "y", "datetime", "elevation"]).custom_clause(
            "FLOOR(x / 100) AS x_bin")
        """
        self._extra_columns.extend([sql] if isinstance(sql, str) else sql)
        return self

    # -----------------------------
    # Temporal Methods
    # -----------------------------
    def get_time(
        self,
        time_col: str = "datetime",
        start: Optional[Union[str, date, datetime]] = None,
        end: Optional[Union[str, date, datetime]] = None,
    ) -> "DuckDBParquetFilter":
        """Filter by time range and/or calendar components.
        Args:
            time_col: Name of the time column to filter on. Default is "datetime".
            start: Start of the time range (inclusive). Can be a string, date, or datetime.
                Default is None (no lower bound).
            end: End of the time range (inclusive). Can be a string, date, or datetime.
                Default is None (no upper bound).

        Example:
            >>> classInstance.get_time(start="2020-01-01", end="2020-12-31")
        """
        _guard_once(self, "_time_called", "get_time")

        def _lit(value) -> str:
            if isinstance(value, datetime):
                value = value.isoformat(sep=" ")
            elif isinstance(value, date):
                value = value.isoformat()
            return f"'{str(value).replace(chr(39), chr(39) * 2)}'"

        clauses: list[str] = []
        if start is not None and end is not None:
            clauses.append(f"{time_col} BETWEEN {_lit(start)} AND {_lit(end)}")
        elif start is not None:
            clauses.append(f"{time_col} >= {_lit(start)}")
        elif end is not None:
            clauses.append(f"{time_col} <= {_lit(end)}")

        if clauses:
            self._add_clause(" AND ".join(clauses))
        return self

    # -----------------------------
    # GridArea methods
    # -----------------------------
    def get_grid_cells_from_grid_area(self, this_grid: GridArea) -> "DuckDBParquetFilter":
        """Adds x_bin/y_bin grid indices and x_cell_offset/y_cell_offset cell-
        centre offsets, computed the same way as GridArea.get_col_row_from_x_y.

        Example:
            >>> classInstance.get_grid_cells_from_grid_area(this_grid=my_grid_area)
        """
        x_bin = f"CAST(FLOOR(({self.lon_col} - {this_grid.minxm}) / {this_grid.binsize}) AS BIGINT)"
        y_bin = f"CAST(FLOOR(({self.lat_col} - {this_grid.minym}) / {this_grid.binsize}) AS BIGINT)"
        cell_x = f"({this_grid.minxm} + ({x_bin}) * {this_grid.binsize} + {this_grid.halfbinsize})"
        cell_y = f"({this_grid.minym} + ({y_bin}) * {this_grid.binsize} + {this_grid.halfbinsize})"
        return self.custom_clause(
            [
                f"{x_bin} AS x_bin",
                f"{y_bin} AS y_bin",
                f"({self.lon_col} - {cell_x}) AS x_cell_offset",
                f"({self.lat_col} - {cell_y}) AS y_cell_offset",
            ]
        )

    # -----------------------------
    # Spatial methods
    # -----------------------------

    def get_bounding_box(
        self, polygon: gpd.GeoDataFrame | BaseGeometry, bounds: Optional[Bounds] = None
    ) -> "DuckDBParquetFilter":
        """Filter to a bounding box.

        Accepts a raw (minx, miny, maxx, maxy) tuple,
        or a shapely geometry / GeoDataFrame / GeoSeries to extract the bounds.

        Args:
            polygon: A GeoDataFrame or shapely geometry to extract the bounding box from.
            bounds: Optional (minx, miny, maxx, maxy) tuple to use directly.

        Example:
            >>> classInstance.get_bounding_box(bounds=(-45.0, 60.0, -30.0, 70.0))
        """
        _guard_once(self, "_bbox_called", "get_bounding_box")
        if isinstance(polygon, gpd.GeoDataFrame):
            minx, miny, maxx, maxy = to_geometry(polygon, target_crs=self.data_crs).bounds
        elif isinstance(polygon, BaseGeometry):
            minx, miny, maxx, maxy = polygon.bounds
        elif bounds is not None:
            minx, miny, maxx, maxy = bounds
        else:
            raise ValueError(
                "Invalid input for get_bounding_box: must provide a "
                "GeoDataFrame, BaseGeometry, or bounds."
            )
        self._add_clause(
            f"{self.lon_col} BETWEEN {minx} AND {maxx} AND {self.lat_col} BETWEEN {miny} AND {maxy}"
        )
        return self

    def _h3_cell_expr(self, res: int) -> str:
        """Calculate the H3 cell id for each row on the fly from
        lon_col/lat_col.

        Used as a fallback, when no precomputed H3 index column is available.

        Args:
            res: H3 resolution to compute the cell id at.
        """
        if self.data_crs is None or self.data_crs == 4326:
            lon_expr, lat_expr = self.lon_col, self.lat_col
        else:
            point = (
                f"ST_Transform(ST_Point({self.lon_col}, {self.lat_col}), "
                f"'EPSG:{self.data_crs}', 'EPSG:4326', always_xy := true)"
            )
            lon_expr, lat_expr = f"ST_X({point})", f"ST_Y({point})"
        return f"h3_latlng_to_cell({lat_expr}, {lon_expr}, {res})"

    def get_polygon(
        self,
        polygon: gpd.GeoDataFrame,
        precise: bool = True,
        h3_col: Optional[str] = "h3_cell_id",
        h3_res: int = 7,
        h3_row_res: Optional[int] = 7,
        buffer_m: float = 5000,
    ) -> "DuckDBParquetFilter":
        """Filter to a polygon.

        Bounding box and H3 cell-membership prefiltering are always applied,
        optionally followed by an exact per-row 'ST_Within' test if precise=True.
        Args:
            polygon: A GeoDataFrame containing the polygon to filter by.
            precise: If True, adds an exact ST_Within test after H3 prefiltering. Default is False.
            h3_col: Name of a precomputed H3 index column. Default is "h3_cell_id".
            h3_res: H3 resolution to use for prefiltering. Default is 7.
            h3_row_res: H3 resolution of the precomputed h3_col, if different from h3_res.
                Default is 7.
            buffer_m: Buffer distance in meters to expand the polygon for H3 prefiltering when
            using precise=True. Default is 5000 meters.

        Example:
            >>> classInstance.get_polygon(polygon, precise=True)
        """
        _guard_once(self, "_polygon_called", "get_polygon")

        polygon_data_crs = reproject(polygon, target_crs=self.data_crs)

        if h3_col is not None:
            polygon_4326 = to_geometry(polygon_data_crs.buffer(buffer_m), target_crs=4326)
            cell_ids = [
                h3.str_to_int(c)
                for c in h3.geo_to_cells(polygon_4326.__geo_interface__, res=h3_res)
            ]
            if not cell_ids:
                self._add_clause("1 = 0")
            else:
                ids_sql = ",".join(str(c) for c in cell_ids)
                if h3_col in self.con.sql(f"SELECT * FROM {self._from_clause} LIMIT 0").columns:
                    if h3_row_res is None or h3_row_res == h3_res:
                        row_h3_expr = h3_col
                    else:
                        row_h3_expr = (
                            f"CAST(h3_cell_to_parent(CAST({h3_col} AS UBIGINT), "
                            f"{h3_res}) AS UBIGINT)"
                        )
                else:
                    row_h3_expr = self._h3_cell_expr(h3_res)
                self._add_clause(f"{row_h3_expr} IN ({ids_sql})")

        self.get_bounding_box(polygon_data_crs)
        if precise:
            wkt = to_geometry(polygon_data_crs).wkt.replace("'", "''")
            self._add_clause(
                f"ST_Within(ST_Point({self.lon_col}, {self.lat_col}), ST_GeomFromText('{wkt}'))"
            )

        return self

    def get_cpom_grid_mask(
        self, mask: MaskSource, basin_numbers: Optional[list[int | str]] = None
    ) -> "DuckDBParquetFilter":
        """Filter using the 2km grid method used by the Mask class.

        Filter to points falling inside a CPOM grid mask, optionally restricted to
        specific basin numbers/names.

        Args:
            mask: A CPOM Mask instance, a CPOM Area instance (uses its underlying mask), or
                a string name of a supported mask.
            basin_numbers: Optional list of basin numbers or names to filter by.

        Example:
            >>> classInstance.get_cpom_grid_mask("antarctica", basin_numbers=[5, 6])
        """
        self._grid_mask_bound = True
        mask = to_mask(mask)

        if mask.mask_type == "xylimits":
            if basin_numbers:
                logger.warning(
                    "basin_numbers %s ignored: mask %r has mask_type='xylimits' "
                    "and has no basin concept",
                    basin_numbers,
                    mask.mask_name,
                )
            self._add_clause(
                f"{self.lon_col} BETWEEN {mask.xlimits[0]} AND {mask.xlimits[1]} "
                f"AND {self.lat_col} BETWEEN {mask.ylimits[0]} AND {mask.ylimits[1]}"
            )
            return self

        rows, cols = get_mask_cell_indices(mask, basin_numbers)
        if rows.size == 0:
            self._add_clause("1 = 0")
            return self

        self.con.register(
            "tmp_grid_mask",
            pl.DataFrame({"ii": cols.astype(np.int32), "jj": rows.astype(np.int32)}),
        )

        ii = f"CAST(ROUND_EVEN(({self.lon_col} - {mask.minxm}) / {mask.binsize}, 0) AS BIGINT)"
        jj = f"CAST(ROUND_EVEN(({self.lat_col} - {mask.minym}) / {mask.binsize}, 0) AS BIGINT)"

        self._add_clause(
            " AND ".join(
                [
                    f"{ii} BETWEEN 0 AND {mask.num_x - 1}",
                    f"{jj} BETWEEN 0 AND {mask.num_y - 1}",
                ]
            )
        )

        self._join_clauses.append(f"SEMI JOIN tmp_grid_mask m ON {ii} = m.ii AND {jj} = m.jj")

        return self

    def _get_query(self) -> tuple[str, list[Any]]:
        """Compile the accumulated builder state into a single SQL SELECT
        statement and its positional parameters.

        Returns:
            tuple: (query, params) - the SQL text (with '?' placeholders) and
            the flattened list of parameter values, in the same order as
            their placeholders appear in 'query'. Pass both straight to
            'self.con.execute(query, params)'.
        """
        where_sql = "\n  AND ".join(sql for sql, _ in self._clauses) or "TRUE"
        params = [p for _, ps in self._clauses for p in ps]
        order_sql = f"\nORDER BY {self._order_by}" if self._order_by else ""
        limit_sql = f"\nLIMIT {self._limit}" if self._limit else ""
        cols_sql = ", ".join(self._columns) if self._columns else "*"
        if self._extra_columns:
            cols_sql = ", ".join([cols_sql, *self._extra_columns])
        join_sql = "\n " + "\n ".join(self._join_clauses) if self._join_clauses else ""
        query = f"""
            SELECT {cols_sql}
            FROM {self._from_clause}{join_sql}
            WHERE {where_sql}{order_sql}{limit_sql}
        """
        return query, params

    def run(
        self,
        output_path: Optional[str] = None,
        output_table: Optional[str] = None,
        if_exists: Literal["replace", "append", "fail"] = "replace",
        partition_by: Optional[list[str]] = None,
        verbose: bool = False,
        close_after_run: bool = False,
    ) -> Optional[Union[pl.DataFrame, TableHandle]]:
        """Compile filters and run against DuckDB.

        Modes:
            1. If output_path is given, writes the result to a parquet file at that path.
            2. If output_table is given, creates a table on the DuckDB connection with that name.
                The table exists only on the connection that created it and will be dropped once
                the connection is closed. It can be queried via the returned TableHandle.
            3. If neither is given, returns a polars DataFrame.

        Args:
            output_path: path to write the result to as a parquet file.
            output_table: name of a table to create on the DuckDB connection.
            if_exists: what to do if output_table already exists on the connection.
                replace, append or drop.
            partition_by: column names to hive-partition the output on. Only valid with output_path.
            verbose: if True, prints the query and parameters before executing.
            close_after_run: if True, closes the DuckDB connection after running.

        Example:
            >>> df = classInstance.get_bounding_box(polygon).run()
            >>> handle = classInstance.reset().get_time(start="2020-01-01").run(
            output_table="filtered_data")
        """
        if self._closed:
            raise RuntimeError(
                f"{type(self).__name__} for {self.input_path!r} is closed;"
                "build a new instance to run again."
            )
        if output_path is not None and output_table is not None:
            raise ValueError("Pass at most one of output_path, output_table.")
        if partition_by is not None and output_path is None:
            raise ValueError("partition_by requires output_path.")
        try:
            query, params = self._get_query()
            if verbose:
                print(f"Running query:\n{query}\nParams: {params}")
            if output_path is not None:
                if partition_by is not None:
                    partition_cols = ", ".join(f'"{col}"' for col in partition_by)
                    self.con.execute(
                        f"COPY ({query}) TO '{output_path}' "
                        f"(FORMAT PARQUET, PARTITION_BY ({partition_cols})"
                        "OVERWRITE_OR_IGNORE TRUE)",
                        params,
                    )
                    return None
                Path(output_path).unlink(missing_ok=True)
                self.con.execute(f"COPY ({query}) TO '{output_path}' (FORMAT PARQUET)", params)
                return None
            if output_table is not None:
                count_row = self.con.execute(
                    "SELECT count(*) FROM information_schema.tables WHERE table_name = ?",
                    [output_table],
                ).fetchone()
                exists = count_row is not None and count_row[0] > 0
                if exists and if_exists == "fail":
                    raise ValueError(f"Table {output_table!r} already exists.")
                if exists and if_exists == "replace":
                    self.con.execute(f'DROP TABLE "{output_table}"')
                    exists = False
                if exists:
                    self.con.execute(f'INSERT INTO "{output_table}" ({query})', params)
                else:
                    self.con.execute(f'CREATE TABLE "{output_table}" AS ({query})', params)
                return TableHandle(self.con, output_table)
            return self.con.execute(query, params).pl()
        finally:
            self.reset()
            if close_after_run:
                self.close()


# --------------------------------------------------------------------------
# Polars backend
# --------------------------------------------------------------------------
class PolarsParquetFilter:
    """
    Filter a parquet store with polars backend.

    Functions:
        select(columns): Restrict the columns returned by the query.
        limit(n): Limit the number of rows returned.
        where(expr): Add a filter predicate (polars expression or SQL-style string).
        order_by(expr): Add an ORDER BY clause to the filter.
        custom_clause(exprs): Add one or more computed columns via polars expressions.
        get_grid_cells_from_grid_area(this_grid): Add grid indices and cell-centre offsets.
        get_bounding_box(polygon, bounds): Filter to a bounding box.
        get_time(time_col, start, end): Filter by time range.
        get_cpom_grid_mask(mask, basin_numbers): Filter to points inside a CPOM grid mask.

    No polygon/H3 support here — those need duckdb's spatial extension.
    Use 'DuckDBParquetFilter' for geometry filtering. Clipping to a polygon is supported
    only through Mask class 2km grid.

    Example:
        >>> classInstance = PolarsParquetFilter(input_path="data/*.parquet", data_crs=4326)
        >>> classInstance.get_bounding_box(polygon).get_time(start="2020-01-01",
                                                    end="2020-12-31").select(
                                                    ["x", "y", "datetime", "elevation"])
        >>> df = classInstancerun(materialize=True)
    """

    # pylint: disable=too-many-instance-attributes

    # Declared here (rather than only in reset()) so pylint recognizes them
    # as defined by __init__, even though builder methods reassign them.
    _columns: Optional[list[str]]
    _limit: Optional[int]
    _order_by: Optional[pl.Expr]
    _grid_mask_join: Optional[dict[str, Any]]

    def __init__(
        self,
        input_path: Union[str, pl.DataFrame, pl.LazyFrame],
        lon_col: str = "x",
        lat_col: str = "y",
        data_crs: Optional[int] = None,
    ):
        """Initialize a PolarsParquetFilter instance.

        Args:
            input_path: Path to parquet file(s) or a polars DataFrame/LazyFrame.
            lon_col: Name of the longitude column in the data.
            lat_col: Name of the latitude column in the data.
            data_crs: Optional EPSG code of the coordinate reference system of the data.
        """
        if isinstance(input_path, TableHandle):
            raise TypeError(
                "PolarsParquetFilter has no DuckDB connection, so it can't take a "
                "TableHandle; use engine='duckdb', or pass input_path.pl() instead."
            )
        self.input_path = input_path
        self.lon_col = lon_col
        self.lat_col = lat_col
        self.data_crs = data_crs
        self.reset()

    def reset(self) -> "PolarsParquetFilter":
        """Clear accumulated query state so this instance can build another
        query.

        Example:
            >>> classInstancerun(materialize=True)
            >>> classInstancereset().get_bounding_box(other_polygon).run(materialize=True)
        """
        self._columns: Optional[list[str]] = None
        self._limit: Optional[int] = None
        self._filters: list[pl.Expr] = []
        self._extra_columns: list[pl.Expr] = []
        self._bbox_called = False
        self._time_called = False
        self._order_by: Optional[pl.Expr] = None
        self._grid_mask_called = False
        self._grid_mask_join: Optional[dict[str, Any]] = None
        return self

    # -----------------------------
    # Builder Methods
    # -----------------------------
    def select(self, columns: list[str]) -> "PolarsParquetFilter":
        """Restrict the columns returned by the query.

        Example:
            >>> classInstanceselect(["x", "y", "datetime", "elevation"])
        """
        self._columns = columns
        return self

    def limit(self, n: int) -> "PolarsParquetFilter":
        """Limit the number of rows returned.

        Example:
            >>> classInstancelimit(1000)
        """
        self._limit = n
        return self

    def where(self, expr: Union[pl.Expr, str]) -> "PolarsParquetFilter":
        """Add a filter predicate.
        Accepts either a native polars expression ('pl.col('x') > 10') or a
        SQL-style boolean condition as a string (e.g.
        '"x > 10 AND y < 5"')

        Example:
            >>> classInstancewhere("elevation > 0.0 AND quality_flag = 1")
        """
        if isinstance(expr, str):
            expr = pl.sql_expr(expr)
        self._filters.append(expr)
        return self

    def order_by(self, expr: pl.Expr) -> "PolarsParquetFilter":
        """Add an ORDER BY clause to the filter.

        Example:
            >>> classInstanceorder_by(pl.col("datetime"))
        """
        self._order_by = expr
        return self

    def custom_clause(self, exprs: Union[pl.Expr, list[pl.Expr]]) -> "PolarsParquetFilter":
        """Add one or more computed columns via 'with_columns', e.g.
        'pl.col('x').floor().alias('x_bin')'.

        Example:
            >>> classInstancecustom_clause(pl.col("x").floor().alias("x_bin"))
        """
        self._extra_columns.extend([exprs] if isinstance(exprs, pl.Expr) else exprs)
        return self

    # -----------------------------
    # Temporal Methods
    # -----------------------------
    def get_time(
        self,
        time_col: str = "datetime",
        start: Optional[Union[str, date, datetime]] = None,
        end: Optional[Union[str, date, datetime]] = None,
    ) -> "PolarsParquetFilter":
        """Filter by time range and/or calendar components.

        Args:
            time_col: Name of the time column to filter on. Default is "datetime".
            start: Start of the time range (inclusive). Can be a string, date, or datetime.
                Default is None (no lower bound).
            end: End of the time range (inclusive). Can be a string, date, or datetime.
                Default is None (no upper bound).
        Example:
            >>> classInstanceget_time(start="2020-01-01", end="2020-12-31")
        """
        _guard_once(self, "_time_called", "get_time")

        def _coerce_temporal_literal(value: Union[str, date, datetime]) -> Union[date, datetime]:
            if isinstance(value, (date, datetime)):
                return value
            text = str(value).strip()
            try:
                return datetime.fromisoformat(text)
            except ValueError:
                pass
            try:
                return date.fromisoformat(text)
            except ValueError as exc:
                raise ValueError(
                    f"Could not parse {value} as a date/datetime. Pass a date/datetime "
                    "object directly, or an ISO-8601 string (e.g. '2020-01-01' or "
                    "'2020-01-01 12:30:00')."
                ) from exc

        start = _coerce_temporal_literal(start) if start is not None else None
        end = _coerce_temporal_literal(end) if end is not None else None

        if start is not None and end is not None:
            self._filters.append(
                (pl.col(time_col) >= pl.lit(start)) & (pl.col(time_col) <= pl.lit(end))
            )
        elif start is not None:
            self._filters.append(pl.col(time_col) >= pl.lit(start))
        elif end is not None:
            self._filters.append(pl.col(time_col) <= pl.lit(end))
        return self

    # -----------------------------
    # GridArea methods
    # -----------------------------
    def get_grid_cells_from_grid_area(self, this_grid: GridArea) -> "PolarsParquetFilter":
        """Adds x_bin/y_bin grid indices and x_cell_offset/y_cell_offset cell-
        centre offsets, computed the same way as GridArea.get_col_row_from_x_y.

        Example:
            >>> classInstance.get_grid_cells_from_grid_area(this_grid=my_grid_area)
        """
        lon, lat = pl.col(self.lon_col), pl.col(self.lat_col)
        x_bin = ((lon - this_grid.minxm) / this_grid.binsize).floor().cast(pl.Int64)
        y_bin = ((lat - this_grid.minym) / this_grid.binsize).floor().cast(pl.Int64)
        cell_x = this_grid.minxm + x_bin * this_grid.binsize + this_grid.halfbinsize
        cell_y = this_grid.minym + y_bin * this_grid.binsize + this_grid.halfbinsize
        return self.custom_clause(
            [
                x_bin.alias("x_bin"),
                y_bin.alias("y_bin"),
                (lon - cell_x).alias("x_cell_offset"),
                (lat - cell_y).alias("y_cell_offset"),
            ]
        )

    def get_bounding_box(
        self,
        polygon: Optional[gpd.GeoDataFrame | BaseGeometry] = None,
        bounds: Optional[Bounds] = None,
    ) -> "PolarsParquetFilter":
        """Filter to a bounding box.

        Accepts a raw (minx, miny, maxx, maxy) tuple,
        or a shapely geometry / GeoDataFrame / GeoSeries to extract the bounds.

        Args:
            polygon: A GeoDataFrame or shapely geometry to extract the bounding box from.
            bounds: Optional (minx, miny, maxx, maxy) tuple to use directly.

        Example:
            >>> classInstance.get_bounding_box(bounds=(-45.0, 60.0, -30.0, 70.0))
        """
        _guard_once(self, "_bbox_called", "get_bounding_box")
        if isinstance(polygon, gpd.GeoDataFrame):
            minx, miny, maxx, maxy = to_geometry(polygon, target_crs=self.data_crs).bounds
        elif isinstance(polygon, BaseGeometry):
            minx, miny, maxx, maxy = polygon.bounds
        elif bounds is not None:
            minx, miny, maxx, maxy = bounds
        else:
            raise ValueError(
                "Invalid input for get_bounding_box: must provide a GeoDataFrame, BaseGeometry,"
                "or bounds."
            )
        self._filters.append(
            pl.col(self.lon_col).is_between(minx, maxx)
            & pl.col(self.lat_col).is_between(miny, maxy)
        )
        return self

    def get_cpom_grid_mask(
        self, mask: MaskSource, basin_numbers: Optional[list[int | str]] = None
    ) -> "PolarsParquetFilter":
        """Filter using the 2km grid method used by the Mask class.

        Filter to points falling inside a CPOM grid mask, optionally restricted to
        specific basin numbers/names.

        Args:
            mask: A CPOM Mask instance, a CPOM Area instance (uses its underlying mask),
                or a string name of a supported mask.
            basin_numbers: Optional list of basin numbers or names to filter by.
                If None, all basins are included.
        Example:
            >>> classInstance.get_cpom_grid_mask("antarctica", basin_numbers=[5, 6])
        """
        _guard_once(self, "_grid_mask_called", "get_cpom_grid_mask")

        mask = to_mask(mask)

        if mask.mask_type == "xylimits":
            if basin_numbers:
                logger.warning(
                    "basin_numbers %s ignored: mask %r has mask_type='xylimits' "
                    "and has no basin concept",
                    basin_numbers,
                    mask.mask_name,
                )
            self._filters.append(
                pl.col(self.lon_col).is_between(mask.xlimits[0], mask.xlimits[1])
                & pl.col(self.lat_col).is_between(mask.ylimits[0], mask.ylimits[1])
            )
            return self

        rows, cols = get_mask_cell_indices(mask, basin_numbers)
        if rows.size == 0:
            self._filters.append(pl.lit(False))
            return self

        ii_expr = ((pl.col(self.lon_col) - mask.minxm) / mask.binsize).round().cast(pl.Int64)
        jj_expr = ((pl.col(self.lat_col) - mask.minym) / mask.binsize).round().cast(pl.Int64)

        self._filters.append(ii_expr.is_between(0, mask.num_x) & jj_expr.is_between(0, mask.num_y))

        self._grid_mask_join = {
            "cells": pl.LazyFrame({"ii": cols.astype(np.int32), "jj": rows.astype(np.int32)}),
            "ii_expr": ii_expr.alias("ii"),
            "jj_expr": jj_expr.alias("jj"),
        }
        return self

    def run(
        self,
        output_path: Optional[str] = None,
        partition_by: Optional[Union[str, list[str]]] = None,
        materialize: bool = False,
        verbose: bool = False,
    ) -> Optional[Union[pl.DataFrame, pl.LazyFrame]]:
        """Compile filters and run via polars' lazy engine.

        Modes:
            1. If 'output_path' is given, writes the result to a parquet file at that path.
            2. If 'materialize' is True, returns a polars DataFrame with the filtered results.
            3. If neither is given, returns a polars LazyFrame with the filtered results.

        Args:
            output_path: path to write the result to as a parquet file.
            partition_by: column name(s) to hive-partition the output on.
                Only valid with 'output_path'.
            materialize: if True, returns a polars DataFrame with the filtered results.
                Default is False (returns a LazyFrame).
            verbose: if True, prints the query plan before executing.

        Example:
            >>> df = classInstance.get_bounding_box(polygon).run(materialize=True)
            >>> lf = classInstance.reset().get_time(start="2020-01-01").run()
        """
        if partition_by is not None and output_path is None:
            raise ValueError("partition_by requires output_path.")
        try:
            if isinstance(self.input_path, pl.LazyFrame):
                lf = self.input_path
            elif isinstance(self.input_path, pl.DataFrame):
                lf = self.input_path.lazy()
            else:
                lf = pl.scan_parquet(self.input_path, hive_partitioning=True)

            for expr in self._filters:
                lf = lf.filter(expr)

            if self._order_by is not None:
                lf = lf.sort(self._order_by)

            if self._limit is not None:
                lf = lf.limit(self._limit)

            if self._columns is not None or self._extra_columns:
                select_exprs: list[pl.Expr] = []
                if self._columns is not None:
                    select_exprs.extend(pl.col(name) for name in self._columns)
                else:
                    select_exprs.append(pl.all())
                select_exprs.extend(self._extra_columns)
                lf = lf.select(select_exprs)

            if self._grid_mask_join is not None:
                gm = self._grid_mask_join
                lf = lf.with_columns([gm["ii_expr"], gm["jj_expr"]])
                lf = lf.join(gm["cells"], on=["ii", "jj"], how="semi").drop(["ii", "jj"])

            if verbose:
                print("Running query (polars native lazy backend)")

            if output_path is not None:
                if partition_by is not None:
                    lf.sink_parquet(
                        pl.PartitionBy(output_path, key=partition_by, include_key=True)
                    )  # streaming write; never materializes in full
                    return None
                Path(output_path).unlink(missing_ok=True)
                lf.sink_parquet(output_path)  # streaming write; never materializes in full
                return None
            if materialize:
                return lf.collect()
            return lf
        finally:
            self.reset()


# -----------------#
# Factory          #
# -----------------#
@overload
def parquet_filter(
    input_path: InputSource,
    lon_col: str = "x",
    lat_col: str = "y",
    engine: Literal["duckdb"] = "duckdb",
    *,
    memory_limit: str = "150GB",
    temp_directory: str = "",
    con: Optional[duckdb.DuckDBPyConnection] = None,
    data_crs: Optional[int] = None,
) -> DuckDBParquetFilter: ...


@overload
def parquet_filter(
    input_path: InputSource,
    lon_col: str = "x",
    lat_col: str = "y",
    engine: Literal["polars"] = "polars",
    *,
    memory_limit: str = "150GB",
    temp_directory: str = "",
    con: Optional[duckdb.DuckDBPyConnection] = None,
    data_crs: Optional[int] = None,
) -> PolarsParquetFilter: ...


def parquet_filter(
    input_path: InputSource,
    lon_col: str = "x",
    lat_col: str = "y",
    engine: Engine = "duckdb",
    *,
    memory_limit: str = "150GB",
    temp_directory: str = "",
    con: Optional[duckdb.DuckDBPyConnection] = None,
    data_crs: Optional[int] = None,
) -> Union[DuckDBParquetFilter, PolarsParquetFilter]:
    """Factory that builds a chainable, engine-specific spatio-temporal filter on
    parquet data.

    Returns a 'DuckDBParquetFilter' or 'PolarsParquetFilter' depending on
    the specified engine.

    Args:
        input_path: Parquet source to filter. Accepts a path/glob ('str'),
            a polars 'DataFrame'/'LazyFrame', or (for 'engine="duckdb"'
            only) a 'TableHandle' returned from a previous
            'run(output_table="")' call.
        lon_col: Name of the longitude/x column used by bounding-box and
            polygon filters.
        lat_col: Name of the latitude/y column used by bounding-box and
            polygon filters.
        engine: Which backend to build, duckdb or polars.
        memory_limit: Maximum memory duckdb may use before
            spilling to 'temp_directory'. Ignored for 'engine="polars"'.
        temp_directory: Spill directory for out-of-core
            execution. Ignored for 'engine="polars"'.
        con: An existing 'duckdb.DuckDBPyConnection' to reuse
            (e.g. so a 'TableHandle' from an earlier filter stays valid).
            Ignored for 'engine="polars"'.
        data_crs: EPSG code that 'lon_col'/'lat_col' are stored
            in.

    Returns:
        A 'DuckDBParquetFilter' if 'engine="duckdb"', otherwise a
        'PolarsParquetFilter'.

    Raises:
        ValueError: If 'engine' is not '"duckdb"' or '"polars"'.

    Example:
        >>> f = parquet_filter("data/*.parquet", data_crs=3413)
        >>> df = f.get_bounding_box(polygon).get_time(start="2020-01-01").run(materialize=True)
    """
    if engine == "duckdb":
        return DuckDBParquetFilter(
            input_path=input_path,
            data_crs=data_crs,
            lon_col=lon_col,
            lat_col=lat_col,
            memory_limit=memory_limit,
            temp_directory=temp_directory,
            con=con,
        )
    if engine == "polars":
        if isinstance(input_path, TableHandle):
            raise ValueError("TableHandle input_path is only supported for engine='duckdb'.")
        return PolarsParquetFilter(input_path, lon_col, lat_col, data_crs=data_crs)
    raise ValueError(f"Unsupported engine {engine!r}; expected 'duckdb' or 'polars'.")
