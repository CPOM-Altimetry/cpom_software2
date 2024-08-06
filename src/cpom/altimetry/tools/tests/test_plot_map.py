"""pytest for cpom.altimetry.tools.plot_map.py"""
import os
import sys
from unittest import mock

import pytest

from cpom.altimetry.tools.plot_map import main


# Test various command line arg combinations complete successfully to produce a plot
# by checking that the "plot completed ok" string is output at the end.
@pytest.mark.parametrize(
    "test_args, expected_output",
    [
        (
            [
                "plot_map.py",
                "-f",
                (
                    f'{os.environ["CPOM_SOFTWARE_DIR"]}/testdata/cs2/l2/cryotempo/c/landice/greenl/'
                    "CS_OFFL_SIR_TDP_LI_GREENL_20150131T182746_20150131T183002_07_00423_C001.nc"
                ),
                "-a",
                "greenland",
                "-od",
                "dummy_output_dir",
                "--units",
                "m",
            ],
            "plot completed ok",
        ),
        (
            [
                "plot_map.py",
                "-f",
                (
                    f'{os.environ["CPOM_SOFTWARE_DIR"]}/testdata/cs2/l2/cryotempo/c/landice/antarc/'
                    "CS_OFFL_SIR_TDP_LI_ANTARC_20100718T001450_20100718T001611_01_02650_B001.nc"
                ),
                "-a",
                "antarctica",
                "-od",
                "dummy_output_dir",
            ],
            "plot completed ok",
        ),
        # Add more test cases as needed
    ],
)
def test_plot_map(
    capsys: pytest.CaptureFixture,
    tmpdir: pytest.TempPathFactory,
    test_args: list[str],
    expected_output: str,
) -> None:
    """
    Test the `plot_map` function from `cpom.altimetry.tools.plot_map`
    with multiple sets of arguments.

    Args:
        capsys (pytest.CaptureFixture): A pytest fixture to capture stdout and stderr.
        tmpdir (pytest.TempPathFactory): A pytest fixture that provides a temporary
        directory unique to the test.
        test_args (list[str]): A list of command-line arguments for the test.
        expected_output (str): The expected output to check for in the captured output.

    Raises:
        AssertionError: If the expected output is not found in the captured output.
    """
    # Replace the output directory argument with the temporary directory
    test_args[test_args.index("dummy_output_dir")] = str(tmpdir)

    with mock.patch("sys.argv", test_args):
        main(sys.argv[1:])
        captured = capsys.readouterr()
        assert expected_output in captured.out.strip()
