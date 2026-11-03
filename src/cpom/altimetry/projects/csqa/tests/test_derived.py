"""pytests of cpom.altimetry.projects.csqa.derived and the loader's pass directions"""

import numpy as np

from cpom.altimetry.projects.csqa.derived import DERIVED_VARIABLES, mispointing_angle
from cpom.altimetry.projects.csqa.loader import pass_directions


def test_mispointing_angle():
    """the mispointing angle is the angle between the rotated boresight and nadir"""
    roll = np.array([0.0, 0.1, 0.0, -0.1, -0.12, 30.0, np.nan])
    pitch = np.array([0.0, 0.0, -0.2, 0.05, -0.06, 40.0, 0.1])
    angle = mispointing_angle(roll, pitch)
    # cos(mispointing) = cos(roll) cos(pitch)
    expected = np.degrees(np.arccos(np.cos(np.radians(roll)) * np.cos(np.radians(pitch))))
    assert np.allclose(angle[:6], expected[:6], atol=1e-6)
    assert angle[0] == 0.0 and np.isclose(angle[1], 0.1) and np.isclose(angle[2], 0.2)
    # ~sqrt(roll^2 + pitch^2) for small angles
    assert np.isclose(angle[4], np.hypot(0.12, 0.06), rtol=1e-6)
    assert np.isnan(angle[6])
    assert DERIVED_VARIABLES["mispointing_angle"][0] == 2


def test_pass_directions():
    """pass directions are from the rate of change of latitude"""
    lats = np.array([80.0, 81.0, 81.5, 81.6, 81.5, 81.0, np.nan, 79.0], dtype=np.float32)
    # 0 at the turning point and for the missing latitude, which is skipped by its neighbours
    assert list(pass_directions(lats)) == [1, 1, 1, 0, -1, -1, 0, -1]
    assert list(pass_directions(np.array([10.0]))) == [0]
