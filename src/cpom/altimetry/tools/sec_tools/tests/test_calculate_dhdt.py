"""Regression tests for SEC dh/dt calculation helpers."""

import json
import logging
from datetime import datetime

import polars as pl
import pytest

from cpom.altimetry.tools.sec_tools.calculate_dhdt import (
    get_start_end_dates_for_calculation,
    resolve_input_parquet_path,
)


def test_resolve_input_parquet_path_uses_recursive_scan_for_partitioned_epoch_average(
    tmp_path,
) -> None:
    """Use partition metadata to expand a root parquet path into a recursive scan."""

    metadata_file = tmp_path / "epoch_avg_meta.json"
    metadata_file.write_text(json.dumps({"partitioned": True}), encoding="utf-8")

    resolved = resolve_input_parquet_path(
        tmp_path, "epoch_average.parquet", logging.getLogger("test")
    )

    assert resolved == str(tmp_path / "**" / "epoch_average.parquet")


def test_get_start_end_dates_for_calculation_raises_for_empty_input() -> None:
    """Raise a clear error when no datetime values are available for dh/dt bounds."""

    input_df = pl.DataFrame(schema={"epoch_midpoint_dt": pl.Datetime("us")}).lazy()

    with pytest.raises(ValueError, match="No non-null values found"):
        get_start_end_dates_for_calculation(input_df, None, None, "epoch_midpoint_dt")


def test_get_start_end_dates_for_calculation_uses_dataset_extent() -> None:
    """Fall back to the dataset min and max datetimes when no bounds are provided."""

    input_df = pl.DataFrame(
        {"epoch_midpoint_dt": [datetime(2020, 1, 1), datetime(2020, 6, 1), datetime(2021, 1, 1)]}
    ).lazy()

    start_time, end_time = get_start_end_dates_for_calculation(
        input_df, None, None, "epoch_midpoint_dt"
    )

    assert start_time == datetime(2020, 1, 1)
    assert end_time == datetime(2021, 1, 1)
