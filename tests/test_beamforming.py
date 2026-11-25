from math import tau

import numpy as np
import pytest

from angular_spectrum import far_field_directivity, propagate_pressure
from aperture import rasterize_aperture
from beamforming import focus_weights, normalize_weights, steering_weights
from geometry import generate_speaker_centers
from grid import build_grid


def test_broadside_has_identical_phase_and_peaks_at_zero_theta():
    centers = generate_speaker_centers(3, 5, 0.06, 0.02, 0.02)
    medium = dict(frequency_hz=1800, sound_speed_m_s=340)
    weights = steering_weights(centers, 41, 90, **medium)
    np.testing.assert_array_equal(weights, np.ones(len(centers)))
    grid = build_grid(centers, 0.06, dx=0.01, dy=0.01, padding_factor=10)
    field = rasterize_aperture(grid.source, centers, 0.06, weights=weights)
    theta = np.linspace(-90, 90, 361)
    directivity = far_field_directivity(field, grid, theta, 41, **medium)
    assert theta[np.abs(directivity).argmax()] == 0


def test_azimuth_and_elevation_steering_move_the_rasterized_far_field_peak():
    centers = generate_speaker_centers(5, 7, 0.06, 0.025, 0.025)
    medium = dict(frequency_hz=1800, sound_speed_m_s=340)
    azimuth, elevation = 37, 56
    weights = steering_weights(centers, azimuth, elevation, **medium)
    grid = build_grid(centers, 0.06, dx=0.01, dy=0.01, padding_factor=12)
    field = rasterize_aperture(grid.source, centers, 0.06, weights=weights)
    theta = np.linspace(0, 90, 181)
    phi = np.arange(0, 360, 2)
    directivity = far_field_directivity(field, grid, theta[:, None], phi, **medium)
    i, j = np.unravel_index(np.abs(directivity).argmax(), directivity.shape)
    assert theta[i] == pytest.approx(90 - elevation, abs=2)
    assert phi[j] == pytest.approx(azimuth, abs=3)


def test_near_field_focus_aligns_path_phase_and_concentrates_pressure():
    diameter = 0.065
    centers = generate_speaker_centers(5, 5, diameter, 0.025, 0.025)
    target = (0.11, -0.085, 0.24)
    medium = dict(frequency_hz=1800, sound_speed_m_s=340)
    k = tau * medium["frequency_hz"] / medium["sound_speed_m_s"]
    weights = focus_weights(centers, target, **medium)
    positions = np.column_stack((centers, np.zeros(len(centers))))
    outgoing = np.exp(1j * k * np.linalg.norm(target - positions, axis=1))
    np.testing.assert_allclose(outgoing * weights, np.exp(1j * k * np.linalg.norm(target)))

    grid = build_grid(
        centers, diameter, dx=0.007, dy=0.007, margin=0.02, padding_factor=8,
    )
    focused = rasterize_aperture(grid.source, centers, diameter, weights=weights)
    broadside = rasterize_aperture(grid.source, centers, diameter)
    magnitude = propagate_pressure(focused, grid, target[2], **medium).magnitude[0]
    baseline = propagate_pressure(broadside, grid, target[2], **medium).magnitude[0]
    ix = int(np.abs(grid.observation.x - target[0]).argmin())
    iy = int(np.abs(grid.observation.y - target[1]).argmin())
    peak_y, peak_x = np.unravel_index(magnitude.argmax(), magnitude.shape)
    assert np.hypot(grid.observation.x[peak_x] - target[0],
                    grid.observation.y[peak_y] - target[1]) < 0.025
    assert magnitude[iy, ix] > 2 * baseline[iy, ix]


@pytest.mark.parametrize(("mode", "total"), [("none", None), ("peak", None), ("power", None), ("power", 7.5)])
def test_taper_and_normalization_preserve_individual_phases(mode, total):
    weights = np.array([1 + 1j, -2j, -3, 4j])
    taper = np.array([0, 0.25, 0.5, 1])
    actual = normalize_weights(weights, taper=taper, normalization=mode, total_squared_weight=total)
    tapered = weights * taper
    assert actual[0] == 0
    np.testing.assert_allclose(np.angle(actual[1:]), np.angle(tapered[1:]))
    if mode == "peak":
        assert np.abs(actual).max() == pytest.approx(1)
    elif mode == "power":
        assert np.sum(np.abs(actual)**2) == pytest.approx(len(weights) if total is None else total)
    else:
        np.testing.assert_array_equal(actual, tapered)


def test_invalid_beamforming_inputs():
    with pytest.raises(ValueError, match="elevation"):
        steering_weights([(0, 0)], 0, -1, frequency_hz=1000)
    with pytest.raises(ValueError, match="target_m"):
        focus_weights([(0, 0)], (0, 0, 0), frequency_hz=1000)
    with pytest.raises(ValueError, match="taper"):
        normalize_weights([1, 1], taper=[1, -1])
    with pytest.raises(ValueError, match="all-zero"):
        normalize_weights([0j, 0j], normalization="power")


def test_main_passes_weighted_velocity_to_both_pressure_calculations(monkeypatch):
    import main as app
    from config import SimulationConfig

    fields = {}
    prepare = app.prepare_velocity_spectrum
    propagate = app.iter_pressure_planes
    directivity = app.directivity_from_spectrum

    def capture_source(velocity, grid):
        assert "velocity" not in fields
        fields["velocity"] = velocity
        return prepare(velocity, grid)

    def capture_pressure(spectrum, distances, **kwargs):
        fields["pressure"] = spectrum
        np.testing.assert_allclose(distances, [0, 0.1, 0.2])
        return propagate(spectrum, distances, **kwargs)

    def capture_directivity(spectrum, *args, **kwargs):
        fields["directivity"] = spectrum
        return directivity(spectrum, *args, **kwargs)

    monkeypatch.setattr(app, "prepare_velocity_spectrum", capture_source)
    monkeypatch.setattr(app, "iter_pressure_planes", capture_pressure)
    monkeypatch.setattr(app, "directivity_from_spectrum", capture_directivity)
    for name in ("plot_grid_diagnostic", "plot_aperture", "plot_pressure_maps",
                 "plot_pressure_cross_section", "plot_angular_directivity"):
        monkeypatch.setattr(app, name, lambda *args, **kwargs: None)
    monkeypatch.setattr(app.plt, "show", lambda: None)
    app.main(SimulationConfig(
        rows=1, columns=2, speaker_weights=(1j, -1),
        speaker_taper=(1, 0), weight_normalization="peak", observation_distance_m=0.2,
        z_evaluation_points=3, grid_spacing_m=0.01, padding_factor=2,
        observation_width_x_factor=1.5, observation_width_y_factor=1.5,
    ))

    assert fields["pressure"] is fields["directivity"]
    velocity = fields["velocity"]
    assert velocity.real.max() == velocity.real.min() == 0
    assert velocity.imag.max() == 1
    assert not velocity[:, velocity.shape[1] // 2:].any()


@pytest.mark.parametrize("name", ["uniform", "hann", "hamming", "blackman", "bartlett", "kaiser", "tukey"])
def test_speaker_windows_are_symmetric_and_peak_normalized(name):
    from beamforming import speaker_array_window

    window = speaker_array_window(5, 6, name)
    assert window.shape == (5, 6)
    assert window.max() == pytest.approx(1)
    assert window.min() >= 0
    np.testing.assert_allclose(window, window[::-1])
    np.testing.assert_allclose(window, window[:, ::-1])
    np.testing.assert_array_equal(speaker_array_window(1, 2, name), [[1, 1]])


def test_array_window_applies_before_taper_and_power_normalization():
    from beamforming import speaker_array_window
    from config import SimulationConfig
    from simulation import configured_weights, generate_array

    config = SimulationConfig(
        rows=3, columns=4, array_window="hamming", speaker_weights=(1j,) * 12,
        speaker_taper=(0.5, 1, 1, 1) * 3, weight_normalization="power",
    )
    weights, _ = configured_weights(config, generate_array(config))
    amplitude = speaker_array_window(3, 4, "hamming").ravel() * config.speaker_taper
    expected = 1j * amplitude * np.sqrt(12) / np.linalg.norm(amplitude)
    np.testing.assert_allclose(weights, expected)
