"""cpom.altimetry.projects.csqa.cycles

CSQA data takes: fixed length sub-cycles (30-days by default) of CryoSat-2, numbered from 1
starting at 00:00 UTC on the mission start date (18-Oct-2010 by default).

All datetimes are naive and represent UTC.
"""

from datetime import datetime, timedelta, timezone


class CycleCalendar:
    """Convert between CSQA cycle numbers and dates"""

    def __init__(self, start_date: datetime, length_days: int = 30, data_latency_days: float = 0):
        """
        Args:
            start_date (datetime): start (00:00 UTC) of cycle 1
            length_days (int): cycle length in days
            data_latency_days (float): days after acquisition before input products are
                                       available
        """
        if length_days < 1:
            raise ValueError("cycle length must be at least 1 day")
        self.start_date = start_date
        self.length = timedelta(days=length_days)
        self.data_latency = timedelta(days=data_latency_days)

    def cycle_start(self, cycle: int) -> datetime:
        """start time of a cycle (inclusive)"""
        if cycle < 1:
            raise ValueError(f"cycle numbers start at 1, not {cycle}")
        return self.start_date + (cycle - 1) * self.length

    def cycle_end(self, cycle: int) -> datetime:
        """end time of a cycle (exclusive), ie the start of the next cycle"""
        return self.cycle_start(cycle) + self.length

    def cycle_bounds(self, cycle: int) -> tuple[datetime, datetime]:
        """(start, end) of a cycle. start is inclusive, end is exclusive"""
        return self.cycle_start(cycle), self.cycle_end(cycle)

    def cycle_for_datetime(self, when: datetime) -> int:
        """cycle number containing a time

        Raises:
            ValueError: if the time is before the start of cycle 1
        """
        if when < self.start_date:
            raise ValueError(f"{when} is before the start of cycle 1 ({self.start_date})")
        return int((when - self.start_date) // self.length) + 1

    def cycles_between(self, start: datetime, end: datetime) -> list[int]:
        """cycle numbers overlapping the time range [start, end]"""
        start = max(start, self.start_date)
        if end < start:
            return []
        return list(range(self.cycle_for_datetime(start), self.cycle_for_datetime(end) + 1))

    def current_cycle(self, now: datetime | None = None) -> int:
        """cycle containing the current (or given) time"""
        if now is None:
            now = datetime.now(timezone.utc).replace(tzinfo=None)
        return self.cycle_for_datetime(now)

    def latest_available_cycle(self, now: datetime | None = None) -> int:
        """latest cycle that can have available input data: the cycle containing
        (now - data latency)"""
        if now is None:
            now = datetime.now(timezone.utc).replace(tzinfo=None)
        return self.cycle_for_datetime(max(now - self.data_latency, self.start_date))

    def latest_cycles(self, n_cycles: int, now: datetime | None = None) -> list[int]:
        """the latest n_cycles cycles, ending with the latest cycle that can have data"""
        latest = self.latest_available_cycle(now)
        return list(range(max(1, latest - n_cycles + 1), latest + 1))
