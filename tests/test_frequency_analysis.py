from dataclasses import replace

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from matplotlib import pyplot as plt

import simulation
from config import SimulationConfig
from frequency_analysis import broadband_mean_square_pressure, point_pressure_responses
from grid import XYGrid
from simulation import FrequencyDirectivityResult, FrequencySweepResult, SimulationSession
from visualization import (
    plot_broadband_pressure,
    plot_directivity_heatmap,
    plot_frequency_directivity,
    plot_frequency_pressure,
    plot_point_frequency_responses,
    plot_pressure_planes,
)


def test_discrete_and_peak_tone_weights_combine_magnitudes_incoherently():
    pressure = np.array([[1 + 1j, 3], [-1 - 1j, 4j]])
    frequencies = [200, 100]
    weights = [1, 3]
    expected = [8, 57]
    np.testing.assert_allclose(broadband_mean_square_pressure(pressure, frequencies, weights), expected)
    np.testing.assert_allclose(
        broadband_mean_square_pressure(pressure, frequencies, weights, weighting="tone_peak_squared"),
        np.asarray(expected) / 2,
    )


def test_psd_integration_respects_nonuniform_spacing_and_preserves_weight_alignment():
    pressure = np.array([4, 1, 2j])
    frequencies = [400, 100, 200]
    psd = [2, 1, 3]
    # Sorted sample widths are [50, 150, 100] Hz at [100, 200, 400] Hz.
    actual = broadband_mean_square_pressure(pressure, frequencies, psd, weighting="one_sided_psd")
    assert actual == 50 + 150 * 4 * 3 + 100 * 16 * 2
    explicit = broadband_mean_square_pressure(
        pressure, frequencies, psd, weighting="one_sided_psd", bin_widths_hz=[40, 10, 20],
    )
    assert explicit == 40 * 16 * 2 + 10 + 20 * 4 * 3


def test_one_psd_sample_requires_measured_bandwidth_and_discrete_weights_do_not():
    with pytest.raises(ValueError, match="explicit bin width"):
        broadband_mean_square_pressure([2j], [500], [0.1], weighting="one_sided_psd")
    assert broadband_mean_square_pressure(
        [2j], [500], [0.1], weighting="one_sided_psd", bin_widths_hz=[10],
    ) == 4
    with pytest.raises(ValueError, match="only to one_sided_psd"):
        broadband_mean_square_pressure([2j], [500], [0.1], bin_widths_hz=[10])


@pytest.mark.parametrize(
    ("frequencies", "weights", "message"),
    [([100, 100], [1, 2], "unique"), ([100, 0], [1, 2], "positive"),
     ([100, 200], [1], "one finite"), ([100, 200], [1, -1], "nonnegative")],
)
def test_invalid_broadband_inputs(frequencies, weights, message):
    with pytest.raises(ValueError, match=message):
        broadband_mean_square_pressure([1, 2], frequencies, weights)


def _fake_sweep():
    grid = XYGrid(np.array([-0.1, 0.1]), np.array([-0.15, 0.15]), 0.2, 0.3)
    pressure = np.arange(1, 17).reshape(2, 2, 2, 2) * (1 + 1j)
    return FrequencySweepResult(np.array([300, 100]), np.array([1, 0.5]), grid, pressure)


def test_observation_responses_snap_to_grid_points_and_sort_frequencies():
    sweep = _fake_sweep()
    points = [(0.08, -0.12, 0.54), (-0.05, 0.13, 0.95)]
    frequencies, locations, response = point_pressure_responses(sweep, points)
    np.testing.assert_array_equal(frequencies, [100, 300])
    np.testing.assert_allclose(locations, [(0.1, -0.15, 0.5), (-0.1, 0.15, 1)])
    np.testing.assert_array_equal(response, [
        [sweep.pressure[1, 1, 0, 1], sweep.pressure[1, 0, 1, 0]],
        [sweep.pressure[0, 1, 0, 1], sweep.pressure[0, 0, 1, 0]],
    ])
    with pytest.raises(ValueError, match="within the sampled observation domain"):
        point_pressure_responses(sweep, [(0, 0, 2)])

    figure = plot_point_frequency_responses(sweep, points, reference_pressure=32)
    try:
        level, phase = (ax.lines[0] for ax in figure.axes)
        np.testing.assert_array_equal(level.get_xdata(), frequencies)
        np.testing.assert_allclose(level.get_ydata(), 20 * np.log10(np.abs(response[:, 0]) / 32))
        np.testing.assert_allclose(phase.get_ydata(), np.angle(response[:, 0]))
        assert "Relative pressure" in figure.axes[0].get_ylabel()
        figure.canvas.draw()
    finally:
        plt.close(figure)


def test_selected_frequency_map_and_broadband_map_use_fixed_references():
    sweep = _fake_sweep()
    weaker = replace(sweep, pressure=sweep.pressure / 2)
    figures = [
        plot_frequency_pressure(case, 100, distance_index=1, reference_pressure=32)
        for case in (sweep, weaker)
    ] + [
        plot_broadband_pressure(case, [0.25, 0.75], distance_index=1, reference_pressure=32)
        for case in (sweep, weaker)
    ]
    try:
        original, reduced, broad, broad_reduced = figures
        expected = sweep.pressure[1, 1]
        np.testing.assert_allclose(original.axes[0].images[0].get_array(), np.abs(expected) / 32)
        np.testing.assert_allclose(original.axes[2].images[0].get_array(), np.angle(expected))
        shift = 20 * np.log10(2)
        for first, second in ((original, reduced), (broad, broad_reduced)):
            level_a = first.axes[1].images[0]
            level_b = second.axes[1].images[0]
            assert level_a.get_clim() == level_b.get_clim()
            np.testing.assert_allclose(level_a.get_array() - level_b.get_array(), shift)
        assert "mean-square pressure" in broad.axes[1].get_title()
        for figure in figures:
            figure.canvas.draw()
    finally:
        for figure in figures:
            plt.close(figure)
    with pytest.raises(ValueError, match="sampled bin"):
        plot_frequency_pressure(sweep, 200)


def test_directivity_cuts_and_heatmap_share_reference_and_handle_irregular_frequencies():
    result = FrequencyDirectivityResult(
        np.array([800, 100, 200]), np.array([-90, 0, 90]), 30,
        np.array([[1, 4, 1], [0.5, 1, 0.5], [1, 2, 1]], dtype=complex),
    )
    curves = plot_frequency_directivity(result, [800, 100], reference_directivity=4)
    heatmap = plot_directivity_heatmap(result, reference_directivity=4)
    try:
        assert [line.get_label() for line in curves.axes[0].lines] == ["800 Hz", "100 Hz"]
        mesh = heatmap.axes[0].collections[0]
        heatmap_rows = np.asarray(mesh.get_array()).reshape(3, 3)
        np.testing.assert_allclose(heatmap_rows[2], curves.axes[0].lines[0].get_ydata())
        np.testing.assert_allclose(heatmap_rows[0], curves.axes[0].lines[1].get_ydata())
        edges = mesh.get_coordinates()[:, 0, 1]
        np.testing.assert_allclose(edges[1:-1], [np.sqrt(100 * 200), np.sqrt(200 * 800)])
        assert heatmap.axes[0].get_yscale() == "log"
        curves.canvas.draw()
        heatmap.canvas.draw()
    finally:
        plt.close(curves)
        plt.close(heatmap)


def test_single_frequency_pressure_and_directivity_match_legacy_views(monkeypatch):
    config = SimulationConfig(
        rows=1, columns=2, speaker_diameter_m=0.1, grid_spacing_m=0.01,
        observation_shape=(6, 8), padding_factor=3, frequency_hz=650,
    )
    session = SimulationSession()
    legacy = session.recalculate(config, directivity_azimuth_deg=35)
    pressure = session.recalculate_frequencies(config, [config.frequency_hz])

    def forbid_propagation(*args, **kwargs):
        pytest.fail("a directivity sweep does not need xy propagation")

    monkeypatch.setattr(simulation, "pressure_from_spectrum", forbid_propagation)
    monkeypatch.setattr(simulation, "iter_pressure_planes", forbid_propagation)
    directivity = session.recalculate_frequency_directivity(config, [config.frequency_hz], directivity_azimuth_deg=35)
    assert directivity.azimuth_deg == 35
    np.testing.assert_array_equal(directivity.directivity[0], legacy.directivity)

    original_plot = plot_pressure_planes(legacy.pressure)
    frequency_plot = plot_frequency_pressure(pressure, config.frequency_hz)
    single_heatmap = plot_directivity_heatmap(directivity)
    try:
        for original, added in zip(original_plot.axes[:3], frequency_plot.axes[:3]):
            np.testing.assert_array_equal(original.images[0].get_array(), added.images[0].get_array())
        single_heatmap.canvas.draw()
    finally:
        plt.close(original_plot)
        plt.close(frequency_plot)
        plt.close(single_heatmap)
