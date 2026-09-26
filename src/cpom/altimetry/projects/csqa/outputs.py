"""cpom.altimetry.projects.csqa.outputs

Layout of the CSQA output directory (read by the CSQA portal):

    <output_dir>/
        manifest.json                               portal index (build_portal_index.py)
        baseline_<B>/
            cycles/cycle_<NNN>/
                cycle_info.json                     cycle dates, input files, processing info
                stats/<param>.json                  statistics per area/variant/mode
                plots/<param>/<param>[_<variant>][_<mode>]_<area>.webp
            timeseries/<param>.csv                  statistics of all cycles (portal trends)

Files are written atomically (temporary file + rename) so the portal never reads partial files.
"""

import json
import os
import tempfile
from datetime import datetime, timezone


def baseline_dir(output_dir: str, baseline: str) -> str:
    """output directory of a product baseline"""
    return os.path.join(output_dir, f"baseline_{baseline}")


def cycle_dir(output_dir: str, baseline: str, cycle: int) -> str:
    """output directory of a cycle"""
    return os.path.join(baseline_dir(output_dir, baseline), "cycles", f"cycle_{cycle:03d}")


def stats_path(output_dir: str, baseline: str, cycle: int, param_id: str) -> str:
    """statistics file of a parameter for a cycle"""
    return os.path.join(cycle_dir(output_dir, baseline, cycle), "stats", f"{param_id}.json")


def plots_dir(output_dir: str, baseline: str, cycle: int, param_id: str) -> str:
    """plot directory of a parameter for a cycle"""
    return os.path.join(cycle_dir(output_dir, baseline, cycle), "plots", param_id)


def timeseries_path(output_dir: str, baseline: str, param_id: str) -> str:
    """statistics timeseries file of a parameter"""
    return os.path.join(baseline_dir(output_dir, baseline), "timeseries", f"{param_id}.csv")


def utc_now_str() -> str:
    """current UTC time as an ISO string, ie 2026-09-26T12:00:00Z"""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_text_atomic(path: str, text: str):
    """write a text file atomically (readers see the old or new file, never a partial one)"""
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=directory, prefix=".tmp_", suffix=os.path.basename(path))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.chmod(tmp_path, 0o644)
        os.replace(tmp_path, path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def write_json_atomic(path: str, obj):
    """write an object as a json file atomically"""
    write_text_atomic(path, json.dumps(obj, indent=1) + "\n")


def read_json(path: str) -> dict | None:
    """read a json file, returning None if it does not exist or is invalid"""
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
