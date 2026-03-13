"""Regression tests for SEC run-chain configuration handling."""

from pathlib import Path
from typing import Any

import pytest

from cpom.altimetry.tools.sec_tools.run_chain import (
    build_args,
    get_algorithms_to_run,
)


def test_build_args_accepts_empty_root_algo_section(tmp_path: Path) -> None:
    """Allow mission overrides to drive an algorithm when the root section is empty."""

    metadata_file = tmp_path / "metadata.json"
    metadata_file.write_text("{}", encoding="utf-8")

    config: dict[str, Any] = {
        "base_out": str(tmp_path / "outputs"),
        "epoch_average": None,
        "cs2": {
            "algorithm_list": ["epoch_average"],
            "base_folder": "ais_cs2",
            "grid_info_json": str(metadata_file),
            "epoch_average": {
                "out_folder_name": "epoch_average",
                "epoch_length": 140,
                "gia_model": "ice5g",
            },
        },
    }

    args = build_args("epoch_average", config, "cs2")

    assert "--epoch_length" in args
    assert "--gia_model" in args
    assert "--out_dir" in args
    assert str(Path(config["base_out"]) / "ais_cs2" / "epoch_average") in args
    assert str(metadata_file) in args


def test_build_args_resolves_in_step_from_mission_override_when_root_is_empty(
    tmp_path: Path,
) -> None:
    """Build in_dir from the mission override of a
    referenced step when the root section is empty."""

    config: dict[str, Any] = {
        "base_out": str(tmp_path / "outputs"),
        "epoch_average": None,
        "calculate_dhdt": {"out_folder_name": "single_mission_dhdt"},
        "cs2": {
            "algorithm_list": ["epoch_average", "calculate_dhdt"],
            "base_folder": "ais_cs2",
            "epoch_average": {
                "out_folder_name": "epoch_average",
                "epoch_length": 140,
            },
            "calculate_dhdt": {
                "in_step": "epoch_average",
            },
        },
    }

    args = build_args("calculate_dhdt", config, "cs2")

    assert "--in_dir" in args
    assert str(Path(config["base_out"]) / "ais_cs2" / "epoch_average") in args
    assert "--out_dir" in args
    assert str(Path(config["base_out"]) / "ais_cs2" / "single_mission_dhdt") in args


def test_get_algorithms_to_run_starts_at_requested_algorithm() -> None:
    """Skip earlier algorithms when a valid starting algorithm is provided."""

    algorithm_list = ["surface_fit", "epoch_average", "calculate_dhdt", "dhdt_plots"]

    assert get_algorithms_to_run(algorithm_list, start_alg="epoch_average") == [
        "epoch_average",
        "calculate_dhdt",
        "dhdt_plots",
    ]


def test_get_algorithms_to_run_stops_at_requested_algorithm() -> None:
    """Skip later algorithms when a valid ending algorithm is provided."""

    algorithm_list = ["surface_fit", "epoch_average", "calculate_dhdt", "dhdt_plots"]

    assert get_algorithms_to_run(algorithm_list, end_alg="calculate_dhdt") == [
        "surface_fit",
        "epoch_average",
        "calculate_dhdt",
    ]


def test_get_algorithms_to_run_can_run_single_algorithm() -> None:
    """Allow start and end to be the same algorithm to run only that step."""

    algorithm_list = ["surface_fit", "epoch_average", "calculate_dhdt", "dhdt_plots"]

    assert get_algorithms_to_run(
        algorithm_list, start_alg="calculate_dhdt", end_alg="calculate_dhdt"
    ) == ["calculate_dhdt"]


def test_get_algorithms_to_run_raises_for_unknown_algorithm() -> None:
    """Reject a start algorithm that is not present in the configured algorithm list."""

    algorithm_list = ["surface_fit", "epoch_average", "calculate_dhdt", "dhdt_plots"]

    with pytest.raises(ValueError, match="Start algorithm 'clip_to_basins' not found"):
        get_algorithms_to_run(algorithm_list, start_alg="clip_to_basins")


def test_get_algorithms_to_run_raises_when_start_is_after_end() -> None:
    """Reject incompatible start/end bounds."""

    algorithm_list = ["surface_fit", "epoch_average", "calculate_dhdt", "dhdt_plots"]

    with pytest.raises(ValueError, match="must come before or equal to end algorithm"):
        get_algorithms_to_run(algorithm_list, start_alg="dhdt_plots", end_alg="epoch_average")
