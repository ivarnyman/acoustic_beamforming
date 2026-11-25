import os
import subprocess
import sys
from pathlib import Path

import pytest

from config import SimulationConfig, load_configuration
from simulation import generate_array


@pytest.mark.parametrize("stagger", [0.0, 0.5, 1.0])
def test_row_stagger_scales_with_speaker_pitch(stagger):
    config = SimulationConfig(
        rows=3, columns=2, speaker_diameter_m=0.2, horizontal_gap_m=0.04,
        row_stagger=stagger,
    )
    first, staggered, third = generate_array(config)[::2, 0]
    assert staggered - first == pytest.approx(stagger * 0.24)
    assert third == first


@pytest.mark.parametrize("stagger", [-0.01, 1.01, float("nan"), None, (0.0,), True])
def test_invalid_row_stagger_is_rejected(stagger):
    with pytest.raises(ValueError, match="row_stagger"):
        SimulationConfig(row_stagger=stagger)


def test_legacy_json_meters_convert_to_pitch_fraction(tmp_path):
    path = tmp_path / "old.json"
    path.write_text('{"rows": 3, "speaker_diameter_m": 0.10, "horizontal_gap_m": 0.02, "row_offsets_m": [0.01, 0.07, 0.01]}')
    assert load_configuration(path).row_stagger == pytest.approx(0.5)


@pytest.mark.parametrize("count", [0, -1, 2.5, True])
def test_invalid_z_evaluation_count(count):
    with pytest.raises(ValueError, match="positive integer"):
        SimulationConfig(z_evaluation_points=count)


def test_main_prints_configuration():
    result = subprocess.run(
        [
            sys.executable, "-c",
            "from main import main; from config import SimulationConfig; "
            "main(SimulationConfig(grid_spacing_m=0.01, padding_factor=4))",
        ],
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, "MPLBACKEND": "Agg"},
        capture_output=True,
        text=True,
        check=True,
    )
    assert "exp(-i omega t)" in result.stdout
    assert "normalized relative diaphragm velocity" in result.stdout
    assert f"'frequency_hz': {SimulationConfig().frequency_hz!r}" in result.stdout
    assert f"Peak pressure at z = {SimulationConfig().observation_distance_m:g} m" in result.stdout
