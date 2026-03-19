"""
cpom.altimetry.datasets.elevation_correction_helper

Default elevation corrections for altimetry datasets.
"""

from datetime import datetime, timedelta

import numpy as np


# ----------------------------
# FDR4ALT Corrections
# ----------------------------
def get_fdr4alt_envisat_elev(
    dataset, nc, input_mask, elevation: np.ndarray, strict_missing
) -> np.ndarray:
    """
    Apply FDR4ALT Envisat elevation correction using doppler slope and doppler corrections.
    """
    # dop_slope_cor_20_ku
    dop_slope_cor_20_ku = dataset.get_variable(
        nc, "expert/range_cor_doppler_slope", replace_fill=True, raise_if_missing=strict_missing
    )[input_mask]
    # dop_cor_20_ku
    dop_cor_20_ku = dataset.get_variable(
        nc, "expert/range_cor_doppler", replace_fill=True, raise_if_missing=strict_missing
    )[input_mask]

    # Apply elevation correction to the expert group
    elevation = elevation - dop_slope_cor_20_ku + dop_cor_20_ku

    return elevation


def get_fdr4alt_ers_elev(
    dataset, nc, input_mask, elevation: np.ndarray, strict_missing
) -> np.ndarray:
    """Apply FDR4ALT ERS elevation correction using doppler correction."""
    # delta_doppler_corr_20hz
    dop_slope_cor_20_ku = dataset.get_variable(
        nc, "expert/range_cor_doppler", replace_fill=True, raise_if_missing=strict_missing
    )[input_mask]
    # Apply elevation correction to the expert group
    elevation = elevation - dop_slope_cor_20_ku

    return elevation


def get_is1_elev(
    dataset,
    nc,
    input_mask,
    elevation: np.ndarray,
    time: np.ndarray,
    standard_epoch: str,
    strict_missing,
) -> np.ndarray:
    """Apply IS1 saturation correction and inter-mission bias corrections to elevation."""
    sat_corr = dataset.get_variable(
        nc,
        "Data_40HZ/Elevation_Corrections/d_satElevCorr",
        replace_fill=False,
        raise_if_missing=strict_missing,
    )[input_mask]

    corrected_elevation = elevation + sat_corr

    laser_bias_corrections = (
        ("2003-09-25", "2004-06-21", -0.017),
        ("2004-10-03", "2008-10-19", +0.011),
    )

    standard_epoch_dt = datetime.fromisoformat(standard_epoch)
    time_dt = np.array(
        [standard_epoch_dt + timedelta(seconds=float(t_sec)) for t_sec in time],
        dtype=object,
    )

    for start_str, end_str, bias in laser_bias_corrections:
        start_dt = datetime.fromisoformat(start_str)
        end_dt = datetime.fromisoformat(end_str)
        correction_mask = (time_dt >= start_dt) & (time_dt <= end_dt)
        if np.any(correction_mask):
            corrected_elevation[correction_mask] = corrected_elevation[correction_mask] + bias

    return corrected_elevation
