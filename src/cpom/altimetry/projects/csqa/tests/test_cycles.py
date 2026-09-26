"""pytests of cpom.altimetry.projects.csqa.cycles"""

from datetime import datetime

import pytest

from cpom.altimetry.projects.csqa.cycles import CycleCalendar

CAL = CycleCalendar(datetime(2010, 10, 18), 30)


def test_cycle_bounds():
    """cycle 1 starts at 00:00 on 18-Oct-2010 and cycles are 30 days"""
    assert CAL.cycle_bounds(1) == (datetime(2010, 10, 18), datetime(2010, 11, 17))
    assert CAL.cycle_start(2) == datetime(2010, 11, 17)
    assert CAL.cycle_bounds(3) == (datetime(2010, 12, 17), datetime(2011, 1, 16))
    assert CAL.cycle_bounds(193) == (datetime(2026, 7, 26), datetime(2026, 8, 25))
    with pytest.raises(ValueError):
        CAL.cycle_start(0)


def test_cycle_for_datetime():
    """times map to the cycle containing them (start inclusive, end exclusive)"""
    assert CAL.cycle_for_datetime(datetime(2010, 10, 18)) == 1
    assert CAL.cycle_for_datetime(datetime(2010, 11, 16, 23, 59, 59)) == 1
    assert CAL.cycle_for_datetime(datetime(2010, 11, 17)) == 2
    assert CAL.cycle_for_datetime(datetime(2011, 1, 1)) == 3
    assert CAL.cycle_for_datetime(datetime(2026, 8, 1)) == 193
    with pytest.raises(ValueError):
        CAL.cycle_for_datetime(datetime(2010, 10, 17, 23))


def test_cycles_between_and_latest():
    """cycle ranges"""
    assert CAL.cycles_between(datetime(2010, 1, 1), datetime(2010, 12, 17)) == [1, 2, 3]
    assert not CAL.cycles_between(datetime(2011, 1, 2), datetime(2011, 1, 1))
    now = datetime(2026, 9, 26)
    assert CAL.current_cycle(now) == 195
    assert CAL.latest_cycles(3, now) == [193, 194, 195]
    # latest cycles never go before cycle 1
    assert CycleCalendar(datetime(2026, 9, 1), 30).latest_cycles(5, now) == [1]
