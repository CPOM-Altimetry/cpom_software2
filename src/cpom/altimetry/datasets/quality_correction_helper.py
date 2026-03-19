"""
cpom.altimetry.datasets.quality_correction_helper

Default quality masks for altimetry datasets.
"""

import numpy as np

# ----------------------------
# FDR4ALT Corrections
# ----------------------------


def get_fdr4alt_qual_mask(dataset, nc, input_mask, strict_missing) -> np.ndarray:
    """
    Return FDR4ALT quality mask using retracking_ice1_qual flag.
    """
    # Apply quality mask
    retracking_ice1_qual_mask = (
        dataset.get_variable(
            nc,
            "expert/retracking_ice1_qual",
            replace_fill=False,
            raise_if_missing=strict_missing,
        )[input_mask]
        == 0
    )

    return retracking_ice1_qual_mask


# ----------------------------
# IS2 Corrections
# ----------------------------


def get_is2_qual_mask(dataset, nc, input_mask, strict_missing) -> np.ndarray:
    """Return ICESat-2 quality mask using ATL06 quality summary flag."""
    atl06_quality_mask = (
        dataset.get_variable(
            nc,
            "land_ice_segments/atl06_quality_summary",
            replace_fill=False,
            raise_if_missing=strict_missing,
        )[input_mask]
        == 0
    )

    return atl06_quality_mask


# ----------------------------
# IS1 Corrections
# ----------------------------


def get_is1_qual_mask(dataset, nc, input_mask, strict_missing) -> np.ndarray:
    """Return ICESat-1 quality mask using waveform/attitude/saturation flags."""
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

    return (
        (i_numpk == 1)  # N. peaks in returned echo
        & (elev_use_flg == 0)  # Flag to use elevation
        & (sigma_att_flg == 0)  # Attitude quality flag
        & (sat_corr_flg < 3)  # Saturation corr flag
    )
