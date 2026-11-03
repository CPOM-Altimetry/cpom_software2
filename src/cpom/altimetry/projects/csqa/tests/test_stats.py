"""pytests of cpom.altimetry.projects.csqa.stats"""

import numpy as np

from cpom.altimetry.projects.csqa.csqa_config import FlagDef
from cpom.altimetry.projects.csqa.loader import bit_values
from cpom.altimetry.projects.csqa.stats import bit_flag_stats, flag_stats, float_stats


def test_float_stats():
    """NaN values are excluded from the float statistics"""
    stats = float_stats(np.array([1.0, 2.0, np.nan, 3.0, 10.0], dtype=np.float32))
    assert stats["n_valid"] == 4
    assert stats["mean"] == 4.0
    assert stats["median"] == 2.5
    assert stats["min"] == 1.0 and stats["max"] == 10.0
    assert abs(stats["std"] - np.std([1.0, 2.0, 3.0, 10.0])) < 1e-6
    assert abs(stats["rms"] - np.sqrt(np.mean(np.square([1.0, 2.0, 3.0, 10.0])))) < 1e-6
    # rms of values around 0 is ~ their std, not their (near 0) mean
    assert float_stats(np.array([-2.0, 2.0]))["rms"] == 2.0


def test_float_stats_empty():
    """statistics of no valid values are None"""
    stats = float_stats(np.array([np.nan, np.nan]))
    assert stats["n_valid"] == 0
    assert stats["mean"] is None and stats["std"] is None and stats["rms"] is None


def test_flag_stats():
    """flag percentages are of valid values, and unknown values are counted as other"""
    flags = [FlagDef(1, "LRM", None, "lrm"), FlagDef(2, "SAR", None, "sar")]
    stats = flag_stats(np.array([1, 1, 2, np.nan, 7], dtype=np.float32), flags)
    assert stats["n_valid"] == 4
    assert stats["n_other"] == 1
    assert stats["counts"] == {"lrm": 2, "sar": 1}
    assert stats["pct"] == {"lrm": 50.0, "sar": 25.0}
    assert flag_stats(np.array([]), flags)["pct"] == {"lrm": None, "sar": None}


def test_bit_flag_stats():
    """statistics of a bit of flag words match the flag statistics of its 0/1 values"""
    flags = [FlagDef(0, "Not set", None, "not_set"), FlagDef(1, "Set", None, "set")]
    words = np.array([0, 4, 5, 1, 4, -1, 6], dtype=np.int64)  # -1 = missing
    valid = words >= 0
    for mask in (1, 2, 4, 8):
        expected = flag_stats(bit_values(words, mask), flags)
        assert bit_flag_stats(words, valid, int(valid.sum()), mask, flags) == expected
    # bit 4 is set in 4, 5, 4 and 6
    assert bit_flag_stats(words, valid, 6, 4, flags)["counts"] == {"not_set": 2, "set": 4}
    # all words valid
    words = np.array([4, 0, 4, 4], dtype=np.int64)
    assert bit_flag_stats(words, None, 4, 4, flags)["pct"] == {"not_set": 25.0, "set": 75.0}
