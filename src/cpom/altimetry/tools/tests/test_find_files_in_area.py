"""pytest for cpom.altimetry.tools.find_files_in_area.py"""

import os
import sys
from unittest import mock

import pytest

from cpom.altimetry.tools.find_files_in_area import main


# Test various command line arg combinations complete successfully to produce a plot
# by checking that the "plot completed ok" string is output at the end.
@pytest.mark.parametrize(
    "test_args, expected_output",
    [
        (
            [
                "find_files_in_area.py",
                "-d",
                (f'{os.environ["CPOM_SOFTWARE_DIR"]}/testdata/cs2/l2/cryotempo/c/landice/greenl'),
                "-a",
                "greenland",
            ],
            "CS_OFFL_SIR_TDP_LI_GREENL_20150131T182746_20150131T183002_07_00423_C001.nc",
        ),
        (
            [
                "find_files_in_area.py",
                "-d",
                (f'{os.environ["CPOM_SOFTWARE_DIR"]}/testdata/cs2/l2/cryotempo/c/landice/greenl'),
                "-a",
                "antarctica",
            ],
            "No files found",
        ),
        (
            [
                "find_files_in_area.py",
                "-d",
                (f'{os.environ["CPOM_SOFTWARE_DIR"]}/testdata/cs2/l2/cryotempo/c/landice/antarc'),
                "-a",
                "antarctica",
            ],
            "CS_OFFL_SIR_TDP_LI_ANTARC_20100718T001450_20100718T001611_01_02650_B001.nc",
        ),
        (
            [
                "find_files_in_area.py",
                "-d",
                (f'{os.environ["CPOM_SOFTWARE_DIR"]}/testdata/cs2/l1b/lrm/vostok'),
                "-a",
                "vostok",
            ],
            "CS_OFFL_SIR_LRM_1B_20200911T023800_20200911T024631_D001.nc",
        ),
        # Add more test cases as needed
    ],
)
def test_find_files_in_area(
    capsys: pytest.CaptureFixture,
    test_args: list[str],
    expected_output: str,
) -> None:
    """
    Test the `plot_map` function from `cpom.altimetry.tools.plot_map`
    with multiple sets of arguments.

    Args:
        capsys (pytest.CaptureFixture): A pytest fixture to capture stdout and stderr.
        directory unique to the test.
        test_args (list[str]): A list of command-line arguments for the test.
        expected_output (str): The expected output to check for in the captured output.

    Raises:
        AssertionError: If the expected output is not found in the captured output.
    """

    with mock.patch("sys.argv", test_args):
        main(sys.argv[1:])
        captured = capsys.readouterr()
        assert expected_output in captured.out.strip()
