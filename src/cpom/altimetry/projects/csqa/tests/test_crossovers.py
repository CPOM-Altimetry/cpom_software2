"""pytests of cpom.altimetry.projects.csqa.crossovers: the crossover search on synthetic tracks"""

import numpy as np

from cpom.altimetry.projects.csqa.crossovers import Arcs, find_crossovers, track_arcs


def _track(x0, y0, dx, dy, n, h0, dh, t0, ascending, track_id=0):
    """a straight track of n measurements with linearly changing heights"""
    i = np.arange(n, dtype=float)
    nadir_lat = (i if ascending else -i) * 0.001
    return track_arcs(
        x0 + i * dx,
        y0 + i * dy,
        h0 + i * dh,
        t0 + i * 0.05,
        nadir_lat,
        max_arc_length=1000.0,
        track_id=track_id,
    )


def test_single_crossover():
    """an ascending and a descending track crossing once: heights interpolated at the crossing"""
    # ascending along x = y, descending along x = -y + 1000, crossing at (500, 500)
    asc = _track(0.0, 0.0, 300.0 / np.sqrt(2), 300.0 / np.sqrt(2), 10, 100.0, 1.0, 0.0, True)
    desc = _track(
        0.0, 1000.0, 300.0 / np.sqrt(2), -300.0 / np.sqrt(2), 10, 50.0, -2.0, 7200.0, False
    )
    xo = find_crossovers(Arcs.concatenate([asc, desc]), min_time_separation=1800.0)
    assert xo.x.size == 1
    assert np.isclose(xo.x[0], 500.0) and np.isclose(xo.y[0], 500.0)
    # distance along each track to the crossing: 500*sqrt(2) m = 2.357 arcs of 300 m
    steps = 500.0 * np.sqrt(2) / 300.0
    assert np.isclose(xo.difference[0], (100.0 + steps) - (50.0 - 2.0 * steps))
    assert np.isclose(xo.t_desc[0] - xo.t_asc[0], 7200.0)

    # crossings of the same orbit (less than the minimum time apart) are not crossovers
    desc_soon = _track(0.0, 1000.0, 212.0, -212.0, 10, 50.0, 0.0, 600.0, False)
    assert find_crossovers(Arcs.concatenate([asc, desc_soon]), 1800.0).x.size == 0
    # tracks in the same direction do not make crossovers
    asc2 = _track(0.0, 1000.0, 212.0, -212.0, 10, 50.0, 0.0, 7200.0, True)
    assert find_crossovers(Arcs.concatenate([asc, asc2]), 1800.0).x.size == 0


def test_crossing_at_a_measurement():
    """a crossing at a measurement shared by two arcs is only counted once"""
    asc = _track(-1000.0, 0.0, 500.0, 0.0, 5, 0.0, 0.0, 0.0, True)  # through (0, 0)
    desc = _track(0.0, -1000.0, 0.0, 500.0, 5, 1.0, 0.0, 9000.0, False)  # through (0, 0)
    xo = find_crossovers(Arcs.concatenate([asc, desc]))
    assert xo.x.size == 1
    assert np.isclose(xo.difference[0], -1.0)


def test_one_crossover_per_pass_pair():
    """a zig-zagging (POCA) track crossing another track several times gives one crossover:
    the median of the crossings"""
    # ascending track zig-zagging across y = 0 three times, heights 10, 11, 12, 13
    x = np.array([0.0, 50.0, 100.0, 150.0])
    y = np.array([-200.0, 100.0, -100.0, 200.0])
    asc = track_arcs(
        x,
        y,
        np.array([10.0, 11.0, 12.0, 13.0]),
        np.arange(4.0) * 0.05,
        np.arange(4.0),
        1000.0,
        track_id=0,
    )
    desc = _track(-1000.0, 0.0, 200.0, 0.0, 11, 0.0, 0.0, 7200.0, False, track_id=1)
    arcs = Arcs.concatenate([asc, desc])
    # crossings at 2/3, 1/2 and 1/3 along the zig-zag arcs: 10.667, 11.5, 12.333
    all_xo = find_crossovers(arcs, 1800.0)
    assert np.allclose(np.sort(all_xo.difference), [10 + 2 / 3, 11.5, 12 + 1 / 3])
    one = find_crossovers(arcs, 1800.0, one_per_pass_pair=True)
    assert one.difference.size == 1 and np.isclose(one.difference[0], 11.5)
    assert np.isclose(one.y[0], 0.0)

    # an even number of crossings: the mean of the two middle crossings
    two = find_crossovers(
        Arcs.concatenate(
            [
                track_arcs(
                    x[:3],
                    y[:3],
                    np.array([10.0, 11.0, 12.0]),
                    np.arange(3.0) * 0.05,
                    np.arange(3.0),
                    1000.0,
                ),
                desc,
            ]
        ),
        1800.0,
        one_per_pass_pair=True,
    )
    assert two.difference.size == 1 and np.isclose(two.difference[0], (10 + 2 / 3 + 11.5) / 2)

    # different pass pairs keep their own crossovers
    desc2 = _track(-1000.0, 50.0, 200.0, 0.0, 11, 0.0, 0.0, 9000.0, False, track_id=2)
    xo = find_crossovers(Arcs.concatenate([asc, desc, desc2]), 1800.0, one_per_pass_pair=True)
    assert xo.difference.size == 2


def test_pass_ids():
    """the passes of a track (runs of arcs in one direction) have their own ids"""
    lat = np.array([0.0, 1.0, 2.0, 1.5, 1.0, 0.5])
    arcs = track_arcs(
        np.arange(6.0) * 100, np.zeros(6), np.zeros(6), np.arange(6.0), lat, 1000.0, track_id=7
    )
    assert list(arcs.ascending) == [True, True, False, False, False]
    assert list(arcs.pass_id) == [7000, 7000, 7001, 7001, 7001]


def test_track_arcs_gaps():
    """arcs longer than the maximum length (data gaps) are omitted"""
    x = np.array([0.0, 300.0, 600.0, 5000.0, 5300.0])
    arcs = track_arcs(x, np.zeros(5), np.zeros(5), np.arange(5.0), np.arange(5.0), 1000.0)
    assert list(arcs.x1) == [0.0, 300.0, 5000.0]
    assert np.all(arcs.ascending)
    assert find_crossovers(arcs).x.size == 0  # no descending arcs
