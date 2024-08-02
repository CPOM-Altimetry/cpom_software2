"""pytests for masks.py: Mask class"""

import numpy as np
import pytest

from cpom.masks.masks import Mask

pytestmark = pytest.mark.requires_external_data


@pytest.mark.parametrize(
    "mask_name,indices_inside,lats,lons,grid_values",
    [
        (
            "greenland_area_xylimits_mask",
            [0],  # indices of points inside
            [75.0, 34],  # lats
            [-38, 16],  # lons
            None,  # grid values in mask to indicate inside,, xylimits have no values
        ),
        (
            "greenland_area_xylimits_mask",
            [],  # indices of points inside
            [72.0],  # lats
            [-10],  # lons
            None,  # grid values in mask to indicate inside, xylimits have no values
        ),
        (
            "greenland_bedmachine_v3_grid_mask",
            [0],  # indices of points inside
            [75.0, 34],  # lats
            [-38, 16],  # lons
            [1, 2],  # grid values in mask to indicate inside
        ),
        ("greenland_bedmachine_v3_grid_mask", [], [72.0], [-10], [1, 2]),
        (
            "antarctica_iceandland_dilated_10km_grid_mask",
            [0, 1],  # indices of points inside
            [-76.82, -70.65, -59.493],  # lats
            [55, -64.057, 98.364],  # lons
            [1],  # grid values in mask to indicate inside
        ),
        (
            "greenland_iceandland_dilated_10km_grid_mask",
            [0],  # indices of points inside
            [78.657, -70.65, -59.493],  # lats
            [-36.33, -64.057, 98.364],  # lons
            [1],  # grid values in mask to indicate inside
        ),
        (
            "greenland_icesheet_2km_grid_mask_mouginot2019",
            [0],  # indices of points inside
            [65.23],  # lats
            [-48.46],  # lons
            [7],  # grid values in mask to indicate inside
        ),
        (
            "greenland_icesheet_2km_grid_mask_mouginot2019",
            [0],  # indices of points inside
            [64.01],  # lats
            [-42.82],  # lons
            [6],  # grid values in mask to indicate inside
        ),
    ],
)
def test_mask_points_inside(  # too-many-arguments, pylint: disable=R0913
    mask_name, indices_inside, lats, lons, grid_values
):
    """test of Mask.points_inside()

    Args:
        mask_name (str): name of Mask
        indices_inside (list[int]): list of indices inside mask, or empty list []
        num_inside (int): number of points inside mask
        lats (_type_): _description_
        lons (_type_): _description_
        grid_values (_type_): _description_
    """
    # Load mask
    thismask = Mask(mask_name)

    # find indices of points inside mask
    true_inside, n_inside = thismask.points_inside(lats, lons, basin_numbers=grid_values)

    assert n_inside == np.count_nonzero(true_inside)

    thismask.clean_up()  # free up shared memory

    expected_number_inside = len(indices_inside)

    # Check number of points inside mask is expected
    assert n_inside == expected_number_inside, (
        f"number of points inside mask should be {expected_number_inside},"
        f"for lats: {lats}, lons: {lons}"
    )

    if expected_number_inside > 0:
        for index_inside in indices_inside:
            assert true_inside[index_inside], f"Index {index_inside} should be inside mask"


@pytest.mark.parametrize(
    "mask_name,lats,lons, expected_surface_type",
    [
        (
            "greenland_bedmachine_v3_grid_mask",
            [75.0, 34, 74],  # lats
            [-38, 16, -58],  # lons
            [
                2,
                0,
                0,
            ],  # expected surface type, grounded ice (2), out of mask (99), ocean(0)
        ),
        (
            "greenland_icesheet_2km_grid_mask_mouginot2019",
            [
                65.23,
                64.01,
                64.49,
                80.59,
                74.38,
                70.59,
                69.57,
                79.67,
                -80.0,
            ],  # lats
            [
                -48.46,
                -42.82,
                -37.08,
                -51.14,
                -52.89,
                -49.61,
                -30.96,
                -30.01,
                75,
            ],  # lons
            [
                7,  # 'SW'
                6,  # 'SE'
                0,  # 'Outside'
                3,  # 'NO'
                5,  # 'NW'
                2,  # 'CW'
                1,  # 'CE'
                4,  # 'NE'
                0,  # 'Outside'
            ],  # expected surface type, grounded ice (2), out of mask (99), ocean(0)
        ),
    ],
)
def test_mask_grid_mask_values(mask_name, lats, lons, expected_surface_type) -> None:
    """test of Mask.grid_mask_values()

    Args:
        mask_name (str): mask name
        lats (np.ndarray): array of latitude N values in degs
        lons (np.ndarray): array of longitude E values in degs
        expected_surface_type (list[int or nan]): list of expected surface type values
    Returns:
        None
    """
    thismask = Mask(mask_name, store_in_shared_memory=False)

    mask_values = thismask.grid_mask_values(lats, lons, unknown_value=0)

    thismask.clean_up()  # free up shared memory

    assert len(mask_values) == len(
        lats
    ), "length of returned mask_values should equal number of lat values"

    for index, expected in enumerate(expected_surface_type):
        if np.isnan(expected):
            assert np.isnan(
                mask_values[index]
            ), f"Surface type at {lats[index]},{lons[index]} should be {expected}"
        else:
            assert expected == mask_values[index], (
                f"Surface type at {lats[index]},{lons[index]} should be"
                f"{expected} but is {mask_values[index]}"
            )


def test_mask_loading():
    """test loading mask file using non-default path"""
    thismask = None
    try:
        thismask = Mask("greenland_bedmachine_v3_grid_mask", mask_path="/tmp/none")
    except FileNotFoundError as exc:
        assert True, f"{exc} raised"
    else:
        assert False, "mask_path is invalid so should fail"
    finally:
        try:
            if thismask is not None:
                thismask.clean_up()
        except IOError:  # pylint: disable=bare-except
            pass
