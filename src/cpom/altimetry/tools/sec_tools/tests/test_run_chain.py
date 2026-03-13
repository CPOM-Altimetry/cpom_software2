"""Regression tests for SEC run-chain configuration handling."""

from pathlib import Path
from typing import Any

import pytest

from cpom.altimetry.tools.sec_tools.run_chain import build_args, get_algorithms_to_run


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


def test_get_algorithms_to_run_starts_at_requested_algorithm() -> None:
    """Skip earlier algorithms when a valid starting algorithm is provided."""

    algorithm_list = ["surface_fit", "epoch_average", "calculate_dhdt", "dhdt_plots"]

    assert get_algorithms_to_run(algorithm_list, "epoch_average") == [
        "epoch_average",
        "calculate_dhdt",
        "dhdt_plots",
    ]


def test_get_algorithms_to_run_raises_for_unknown_algorithm() -> None:
    """Reject a start algorithm that is not present in the configured algorithm list."""

    algorithm_list = ["surface_fit", "epoch_average", "calculate_dhdt", "dhdt_plots"]

    with pytest.raises(ValueError, match="Start algorithm 'clip_to_basins' not found"):
        get_algorithms_to_run(algorithm_list, "clip_to_basins")
