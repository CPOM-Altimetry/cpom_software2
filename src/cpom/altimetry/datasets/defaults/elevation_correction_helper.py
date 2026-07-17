"""
cpom.altimetry.datasets.defaults.elevation_correction_helper

Default elevation corrections for altimetry datasets.
"""

import numpy as np


# ----------------------------
# FDR4ALT Corrections
# ----------------------------
def get_fdr4alt_envisat_elev(
    dataset, nc, input_mask, elevation: np.ndarray, strict_missing
) -> tuple[np.ndarray, dict]:
    """
    Applies : elevation = elevation -  expert/range_cor_doppler_slope + expert/range_cor_doppler
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

    return elevation, {
        "dop_slope_cor_20_ku": dop_slope_cor_20_ku,
        "dop_cor_20_ku": dop_cor_20_ku,
    }


def get_fdr4alt_ers_elev(
    dataset, nc, input_mask, elevation: np.ndarray, strict_missing
) -> tuple[np.ndarray, dict]:
    """Applies : elevation = elevation - expert/range_cor_doppler"""
    # delta_doppler_corr_20hz
    dop_cor = dataset.get_variable(
        nc, "expert/range_cor_doppler", replace_fill=True, raise_if_missing=strict_missing
    )[input_mask]
    # Apply elevation correction to the expert group
    elevation = elevation - dop_cor

    return elevation, {"dop_cor": dop_cor}


def get_is1_elev(
    dataset,
    nc,
    input_mask,
    elevation: np.ndarray,
    strict_missing,
) -> tuple[np.ndarray, dict]:
    """Applies : elevation = elevation + Data_40HZ/Elevation_Corrections/d_satElevCorr"""
    sat_corr = dataset.get_variable(
        nc,
        "Data_40HZ/Elevation_Corrections/d_satElevCorr",
        replace_fill=False,
        raise_if_missing=strict_missing,
    )[input_mask]

    corrected_elevation = elevation + sat_corr

    # laser_bias_corrections = (
    #     ("2003-09-25", "2004-06-21", -0.017),
    #     ("2004-10-03", "2008-10-19", +0.011),
    # )

    # standard_epoch_dt = datetime.fromisoformat(standard_epoch)
    # time_dt = np.array(
    #     [standard_epoch_dt + timedelta(seconds=float(t_sec)) for t_sec in time],
    #     dtype=object,
    # )

    # for start_str, end_str, bias in laser_bias_corrections:
    #     start_dt = datetime.fromisoformat(start_str)
    #     end_dt = datetime.fromisoformat(end_str)
    #     correction_mask = (time_dt >= start_dt) & (time_dt <= end_dt)
    #     if np.any(correction_mask):
    #         corrected_elevation[correction_mask] = corrected_elevation[correction_mask] + bias

    return corrected_elevation, {"sat_corr_applied": sat_corr}
