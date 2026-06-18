"""pytest for cpom.altimetry.projects.clev2er_liiw.uncertainty.aggregate_p2p_diffs"""

import numpy as np

from cpom.altimetry.projects.clev2er_liiw.uncertainty.aggregate_p2p_diffs import (
    aggregate_npz_files,
    find_monthly_files,
)


def _write_monthly(path, **arrays):
    """Write a monthly p2p-diff npz at path (creating parent dirs)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)


def test_find_monthly_files(tmp_path):
    """find_monthly_files returns matching files recursively, sorted, excluding others."""
    _write_monthly(tmp_path / "2020/01/p2p_diffs_antarctica.npz", dh=np.array([1.0]))
    _write_monthly(tmp_path / "2020/02/p2p_diffs_antarctica.npz", dh=np.array([2.0]))
    _write_monthly(tmp_path / "2020/02/other.npz", dh=np.array([9.0]))  # must be ignored

    files = find_monthly_files(tmp_path)

    assert [f.name for f in files] == ["p2p_diffs_antarctica.npz", "p2p_diffs_antarctica.npz"]
    assert "2020/01" in str(files[0]) and "2020/02" in str(files[1])


def test_aggregate_union_nan_fill(tmp_path):
    """Union of keys across files; NaN where a file lacks a key (e.g. LR month no coherence)."""
    _write_monthly(
        tmp_path / "2020/01/p2p_diffs_x.npz",
        dh=np.array([1.0, 2.0]),
        lats=np.array([-80.0, -81.0]),
        lons=np.array([10.0, 11.0]),
        sig0=np.array([0.5, 0.6]),
        coherence=np.array([0.9, 0.8]),
    )
    _write_monthly(
        tmp_path / "2020/02/p2p_diffs_x.npz",
        dh=np.array([3.0]),
        lats=np.array([-82.0]),
        lons=np.array([12.0]),
        sig0=np.array([0.7]),
    )  # no coherence key (e.g. an LR-only month)

    files = find_monthly_files(tmp_path)
    aggregated, counts = aggregate_npz_files(files)

    assert [n for _, n in counts] == [2, 1]
    assert len(aggregated["dh"]) == 3
    assert set(aggregated) >= {"dh", "lats", "lons", "sig0", "coherence"}
    assert np.allclose(aggregated["dh"], [1.0, 2.0, 3.0])
    assert np.allclose(aggregated["sig0"], [0.5, 0.6, 0.7])
    # coherence present for the first file's rows, NaN for the file that lacked it
    assert np.allclose(aggregated["coherence"][:2], [0.9, 0.8])
    assert np.isnan(aggregated["coherence"][2])
