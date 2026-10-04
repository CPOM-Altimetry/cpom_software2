"""cpom.altimetry.projects.csqa.derived

Parameter variants derived from several product variables (ie the antenna mispointing angle
from the roll and pitch angles). A derived variant is configured with the name of a function
in DERIVED_VARIABLES and its input variables:

    derived: mispointing_angle
    inputs: [off_nadir_roll_angle_str_01, off_nadir_pitch_angle_str_01]
"""

from typing import Callable

import numpy as np


def mispointing_angle(roll: np.ndarray, pitch: np.ndarray) -> np.ndarray:
    """Antenna mispointing angle: the angle between the antenna boresight and nadir, for a
    platform rotated by a roll and a pitch angle (yaw does not change the boresight direction)

    The boresight direction cosine with nadir is cos(roll).cos(pitch), so
    sin^2(mispointing) = sin^2(roll) + cos^2(roll).sin^2(pitch), which is numerically stable for
    the small angles of a nadir pointing altimeter (~sqrt(roll^2 + pitch^2)).

    Args:
        roll (np.ndarray): roll angles (degrees)
        pitch (np.ndarray): pitch angles (degrees)

    Returns:
        np.ndarray: mispointing angles (degrees), NaN where an input is NaN
    """
    roll_rad = np.radians(np.asarray(roll, dtype=np.float64))
    pitch_rad = np.radians(np.asarray(pitch, dtype=np.float64))
    sin2 = np.sin(roll_rad) ** 2 + np.cos(roll_rad) ** 2 * np.sin(pitch_rad) ** 2
    return np.degrees(np.arcsin(np.sqrt(np.clip(sin2, 0.0, 1.0))))


# derived variable name -> (number of input variables, function of the inputs' values)
DERIVED_VARIABLES: dict[str, tuple[int, Callable[..., np.ndarray]]] = {
    "mispointing_angle": (2, mispointing_angle),
}
