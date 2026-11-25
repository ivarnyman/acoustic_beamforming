import json
from dataclasses import replace

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from compare import beam_metrics, compare_configurations
from config import SimulationConfig, save_configuration


def test_beamwidth_interpolation_and_peak_sidelobe():
    theta = np.arange(-4.0, 5)
    amplitude = [0.3, 0.1, 0.4, 0.6, 1, 0.6, 0.4, 0.1, 0.2]
    result = beam_metrics(theta, amplitude)
    assert result["peak_angle_deg"] == 0
    assert result["beamwidth_deg"] == pytest.approx(2 * (1 - 1 / np.sqrt(2)) / 0.4)
    assert result["sidelobe_level_db"] == pytest.approx(20 * np.log10(0.3))
    assert beam_metrics(theta, np.ones(len(theta)))["beamwidth_deg"] is None
    assert beam_metrics(theta, np.zeros(len(theta)))["peak_angle_deg"] is None


def test_compare_saved_configs_writes_plots_and_keeps_amplitude_differences(tmp_path):
    config = SimulationConfig(rows=1, columns=3, grid_spacing_m=0.02, padding_factor=8)
    loud = tmp_path / "full.json"
    quiet = tmp_path / "half.json"
    save_configuration(loud, config)
    save_configuration(quiet, replace(config, relative_amplitude=0.5))
    output = tmp_path / "plots"
    report = compare_configurations([loud, quiet], output, distance_m=0.18, azimuth_deg=15)
    first, second = report["cases"]
    assert first["distance_m"] == second["distance_m"] == 0.18
    assert first["peak_directivity_level_db"] == first["peak_pressure_level_db"] == 0
    assert second["peak_directivity_level_db"] == pytest.approx(20 * np.log10(0.5))
    assert second["peak_pressure_level_db"] == pytest.approx(20 * np.log10(0.5))
    assert first["beamwidth_deg"] == second["beamwidth_deg"]
    assert json.loads((output / "metrics.json").read_text()) == report
    for name in ("directivity.png", "beam_metrics.png", "pressure_fields.png"):
        assert (output / name).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
