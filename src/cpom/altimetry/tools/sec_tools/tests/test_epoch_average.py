"""Regression tests for SEC epoch averaging helpers."""

from datetime import datetime

import polars as pl

from cpom.altimetry.tools.sec_tools.epoch_average import (
    assign_epochs_lf,
    infer_epoch_origin_dt,
)


def test_infer_epoch_origin_from_filtered_epochs() -> None:
    """Recover epoch zero from a filtered epoch table that starts at a later index."""

    epoch_lf = pl.LazyFrame(
        {
            "epoch_number": [2, 3],
            "epoch_lo_dt": [datetime(2000, 1, 21), datetime(2000, 1, 31)],
            "epoch_hi_dt": [datetime(2000, 1, 31), datetime(2000, 2, 10)],
            "epoch_midpoint_dt": [datetime(2000, 1, 26), datetime(2000, 2, 5)],
            "epoch_midpoint_fractional_yr": [2000.0, 2000.1],
        }
    )

    assert infer_epoch_origin_dt(epoch_lf, epoch_length_days=10) == datetime(2000, 1, 1)


def test_assign_epochs_uses_absolute_epoch_numbering() -> None:
    """Assign observations to the preserved absolute epoch numbers after filtering."""

    epoch_origin_dt = datetime(2000, 1, 1)
    epoch_lf = pl.LazyFrame(
        {
            "epoch_number": [2, 3],
            "epoch_lo_dt": [datetime(2000, 1, 21), datetime(2000, 1, 31)],
            "epoch_hi_dt": [datetime(2000, 1, 31), datetime(2000, 2, 10)],
            "epoch_midpoint_dt": [datetime(2000, 1, 26), datetime(2000, 2, 5)],
            "epoch_midpoint_fractional_yr": [2000.0, 2000.1],
        }
    )
    surface_fit_lf = pl.LazyFrame(
        {
            "x_bin": [10, 10],
            "y_bin": [20, 20],
            "time_dt": [datetime(2000, 1, 22), datetime(2000, 2, 2)],
            "time_years": [0.06, 0.09],
            "dh": [1.0, 2.0],
        }
    )

    assigned = assign_epochs_lf(
        surface_fit_lf, epoch_lf, epoch_origin_dt, epoch_length_days=10
    ).collect()

    assert assigned.shape[0] == 2
    assert assigned["epoch_number"].to_list() == [2, 3]
    assert assigned["epoch_lo_dt"].to_list() == [datetime(2000, 1, 21), datetime(2000, 1, 31)]
    assert assigned["time_delta_years"].to_list() == [0.0, 0.0]
