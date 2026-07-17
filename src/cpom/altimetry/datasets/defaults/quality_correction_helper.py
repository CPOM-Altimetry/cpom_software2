"""
cpom.altimetry.datasets.defaults.quality_correction_helper

Default quality masks for altimetry datasets.
"""

import numpy as np

# ----------------------------
# FDR4ALT Corrections
# ----------------------------


def get_fdr4alt_qual_mask(dataset, nc, input_mask, strict_missing) -> tuple[np.ndarray, dict]:
    """Applies : expert/retracking_ice1_qual == 0"""
    retracking_ice1_qual = dataset.get_variable(
        nc,
        "expert/retracking_ice1_qual",
        replace_fill=False,
        raise_if_missing=strict_missing,
    )[input_mask]

    retracking_ice1_qual_mask = retracking_ice1_qual == 0

    return retracking_ice1_qual_mask, {"retracking_ice1_qual": retracking_ice1_qual}


# ----------------------------
# IS2 Corrections
# ----------------------------


def get_is2_qual_mask(dataset, nc, input_mask, strict_missing) -> tuple[np.ndarray, dict]:
    """Applies : land_ice_segments/atl06_quality_summary == 0"""
    atl06_quality_summary = dataset.get_variable(
        nc,
        "land_ice_segments/atl06_quality_summary",
        replace_fill=False,
        raise_if_missing=strict_missing,
    )[input_mask]

    atl06_quality_mask = atl06_quality_summary == 0

    return atl06_quality_mask, {"atl06_quality_summary": atl06_quality_summary}


# ----------------------------
# IS1 Corrections
# ----------------------------


def get_is1_glah12_34_qual_mask(dataset, nc, input_mask, strict_missing) -> tuple[np.ndarray, dict]:
    """Applies : Data_40HZ/Quality/elev_use_flg == 0 and Data_40HZ/Quality/sigma_att_flg == 0
    and Data_40HZ/Quality/sat_corr_flg < 3 and Data_40HZ/Waveform/i_numPk == 1"""
    i_numpk = dataset.get_variable(
        nc, "Data_40HZ/Waveform/i_numPk", replace_fill=False, raise_if_missing=strict_missing
    )[input_mask]
    elev_use_flg = dataset.get_variable(
        nc, "Data_40HZ/Quality/elev_use_flg", replace_fill=False, raise_if_missing=strict_missing
    )[input_mask]
    sigma_att_flg = dataset.get_variable(
        nc, "Data_40HZ/Quality/sigma_att_flg", replace_fill=False, raise_if_missing=strict_missing
    )[input_mask]
    sat_corr_flg = dataset.get_variable(
        nc, "Data_40HZ/Quality/sat_corr_flg", replace_fill=False, raise_if_missing=strict_missing
    )[input_mask]

    is1_quality_mask = (
        (i_numpk == 1)  # N. peaks in returned echo
        & (elev_use_flg == 0)  # Flag to use elevation
        & (sigma_att_flg == 0)  # Attitude quality flag
        & (sat_corr_flg < 3)  # Saturation corr flag
    )

    return is1_quality_mask, {
        "i_numpk": i_numpk,
        "elev_use_flg": elev_use_flg,
        "sigma_att_flg": sigma_att_flg,
        "sat_corr_flg": sat_corr_flg,
    }


def get_is1_qual_mask(dataset, nc, input_mask, strict_missing) -> tuple[np.ndarray, dict]:
    """Applies: Data_40HZ/Quality/elev_use_flg == 0"""
    mask = dataset.get_variable(
        nc, "Data_40HZ/Quality/elev_use_flg", replace_fill=False, raise_if_missing=strict_missing
    )[input_mask]
    return mask, {"elev_use_flg": mask}


# ----------------------------
# CryoTempo EOLIS Corrections
# ----------------------------


def get_cryotempo_eolis_qual_mask(
    dataset, nc, input_mask, strict_missing
) -> tuple[np.ndarray, dict]:
    """Applies : elevation_err <= 7.0"""

    elevation_err = dataset.get_variable(
        nc,
        dataset.get_param_path("elevation_err"),
        replace_fill=False,
        raise_if_missing=strict_missing,
    )[input_mask]

    eolis_quality_mask = elevation_err <= 7.0

    return eolis_quality_mask, {"elevation_err_used": elevation_err}
