"""pytests of cpom.altimetry.projects.csqa.stats"""

import numpy as np

from cpom.altimetry.projects.csqa.csqa_config import FlagDef
from cpom.altimetry.projects.csqa.stats import flag_stats, float_stats


def test_float_stats():
    """NaN values are excluded from the float statistics"""
    stats = float_stats(np.array([1.0, 2.0, np.nan, 3.0, 10.0], dtype=np.float32))
    assert stats["n_valid"] == 4
    assert stats["mean"] == 4.0
    assert stats["median"] == 2.5
    assert stats["min"] == 1.0 and stats["max"] == 10.0
    assert abs(stats["std"] - np.std([1.0, 2.0, 3.0, 10.0])) < 1e-6


def test_float_stats_empty():
    """statistics of no valid values are None"""
    stats = float_stats(np.array([np.nan, np.nan]))
    assert stats["n_valid"] == 0
    assert stats["mean"] is None and stats["std"] is None


def test_flag_stats():
    """flag percentages are of valid values, and unknown values are counted as other"""
    flags = [FlagDef(1, "LRM", None, "lrm"), FlagDef(2, "SAR", None, "sar")]
    stats = flag_stats(np.array([1, 1, 2, np.nan, 7], dtype=np.float32), flags)
    assert stats["n_valid"] == 4
    assert stats["n_other"] == 1
    assert stats["counts"] == {"lrm": 2, "sar": 1}
    assert stats["pct"] == {"lrm": 50.0, "sar": 25.0}
    assert flag_stats(np.array([]), flags)["pct"] == {"lrm": None, "sar": None}
