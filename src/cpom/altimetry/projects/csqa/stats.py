"""cpom.altimetry.projects.csqa.stats

Statistics of CSQA parameters for a cycle/area/variant/mode selection.

    float parameters : n_valid, mean, median, std (population), min, max
    flag parameters  : n_valid, and the count and % of each flag value (% of valid records)

Values are NaN where the product variable is missing or set to its fill value.
"""

import numpy as np

from cpom.altimetry.projects.csqa.csqa_config import FlagDef

FLOAT_STATS = ("mean", "median", "std", "min", "max")


def _round(value: float, sig: int = 7) -> float:
    """round to a number of significant digits (for compact json/csv output)"""
    return float(f"{value:.{sig}g}")


def float_stats(vals: np.ndarray) -> dict:
    """Statistics of a float parameter

    Args:
        vals (np.ndarray): values (NaN for missing)

    Returns:
        dict: n_valid, mean, median, std, min, max (None if there are no valid values)
    """
    valid = vals[np.isfinite(vals)]
    stats: dict = {"n_valid": int(valid.size)}
    if valid.size == 0:
        stats.update({name: None for name in FLOAT_STATS})
        return stats
    # accumulate in float64 for accuracy when values are held as float32
    valid64 = valid.astype(np.float64, copy=False)
    stats["mean"] = _round(np.mean(valid64))
    stats["median"] = _round(np.median(valid64))
    stats["std"] = _round(np.std(valid64))
    stats["min"] = _round(np.min(valid64))
    stats["max"] = _round(np.max(valid64))
    return stats


def bit_flag_stats(
    words: np.ndarray, valid: np.ndarray | None, n_valid: int, mask: int, flags: list[FlagDef]
) -> dict:
    """Statistics of one bit of a flag word, in the format of flag_stats() (the flags being
    0 = not set, 1 = set)

    Args:
        words (np.ndarray): flag words (negative for missing)
        valid (np.ndarray|None): words >= 0, or None if every word is valid
        n_valid (int): number of valid words
        mask (int): the bit's mask
        flags (list[FlagDef]): flag definitions of the values 0 (not set) and 1 (set)

    Returns:
        dict: n_valid, n_other, counts {flag key: count}, pct {flag key: % of valid values}
    """
    bit_set = (words & mask) != 0
    n_set = int(np.count_nonzero(bit_set if valid is None else bit_set & valid))
    by_value = {0: n_valid - n_set, 1: n_set}
    counts = {flag.key: by_value.get(flag.value, 0) for flag in flags}
    pct = {
        key: (_round(100.0 * count / n_valid, 6) if n_valid > 0 else None)
        for key, count in counts.items()
    }
    return {"n_valid": n_valid, "n_other": 0, "counts": counts, "pct": pct}


def flag_stats(vals: np.ndarray, flags: list[FlagDef]) -> dict:
    """Statistics of a flag parameter

    Args:
        vals (np.ndarray): flag values (NaN for missing)
        flags (list[FlagDef]): flag definitions

    Returns:
        dict: n_valid, counts {flag key: count}, pct {flag key: % of valid values}.
              Valid values that are not a defined flag value are counted in 'n_other'.
    """
    valid = vals[np.isfinite(vals)]
    n_valid = int(valid.size)
    counts = {flag.key: int(np.count_nonzero(valid == flag.value)) for flag in flags}
    pct = {
        key: (_round(100.0 * count / n_valid, 6) if n_valid > 0 else None)
        for key, count in counts.items()
    }
    return {
        "n_valid": n_valid,
        "n_other": n_valid - sum(counts.values()),
        "counts": counts,
        "pct": pct,
    }
