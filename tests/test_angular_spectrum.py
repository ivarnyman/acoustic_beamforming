from math import tau

import numpy as np
import pytest

from angular_spectrum import propagate_pressure
from aperture import rasterize_aperture
from grid import XYGrid, build_grid


def _plane_wave_grid():
    return XYGrid(np.arange(8) * 0.125, np.arange(5) * 0.2, 0.125, 0.2)


def test_oblique_plane_wave_amplitude_phase_and_relative_levels():
    grid = _plane_wave_grid()
    x, y = np.meshgrid(grid.x, grid.y)
    velocity = (1 - 2j) * np.exp(1j * tau * (x - y))
    z = np.array([0.0, 0.19, 0.07])
    frequency, speed, density = 680, 340, 1.2
    omega = tau * frequency
    kz = np.sqrt((omega / speed) ** 2 - 2 * tau**2)
    expected = density * omega / kz * velocity * np.exp(1j * kz * z[:, None, None])

    result = propagate_pressure(
        velocity, grid, z, frequency_hz=frequency,
        sound_speed_m_s=speed, air_density_kg_m3=density,
    )
    np.testing.assert_array_equal(result.distances_m, z)
    np.testing.assert_allclose(result.pressure, expected, atol=1e-11)
    np.testing.assert_allclose(result.magnitude, np.abs(expected))
    np.testing.assert_allclose(np.exp(1j * result.phase_rad), expected / np.abs(expected))
    np.testing.assert_allclose(result.relative_level_db, 0, atol=1e-12)


def test_evanescent_wave_decays_and_can_be_excluded():
    grid = _plane_wave_grid()
    wave = np.broadcast_to(np.exp(1j * tau * 3 * grid.x), grid.shape)
    z = [0, 0.1, 0.3]
    omega = tau * 680
    alpha = tau * np.sqrt(3**2 - 2**2)
    kwargs = dict(frequency_hz=680, sound_speed_m_s=340, air_density_kg_m3=1.2)
    result = propagate_pressure(wave, grid, z, **kwargs)
    expected = 1.2 * omega / (1j * alpha) * np.exp(-alpha * np.array(z))
    np.testing.assert_allclose(result.pressure, expected[:, None, None] * wave, atol=1e-11)
    np.testing.assert_allclose(
        result.relative_level_db[:, 0, 0], -20 * alpha * np.array(z) / np.log(10), atol=1e-12,
    )
    filtered = propagate_pressure(wave, grid, z, include_evanescent=False, **kwargs)
    np.testing.assert_allclose(filtered.pressure, 0, atol=1e-11)


def test_grazing_bin_is_explicitly_omitted():
    grid = _plane_wave_grid()
    # The even x axis has an exact Nyquist mode at |kx| = 4 * tau.
    velocity = np.broadcast_to((-1.0) ** np.arange(len(grid.x)), grid.shape)
    with np.errstate(divide="raise", invalid="raise"):
        result = propagate_pressure(
            velocity, grid, [0, 1], frequency_hz=1360, sound_speed_m_s=340,
        )
    assert np.all(result.pressure == 0)
    assert np.all(result.magnitude == 0)
    assert np.all(np.isnan(result.phase_rad))
    assert np.all(np.isneginf(result.relative_level_db))


def test_centered_disk_is_symmetric_on_observation_crop():
    centers = [(0, 0)]
    grid = build_grid(
        centers, 0.2, dx=0.025, dy=0.025, margin=0.055,
        padding_factor=3, observation_shape=(7, 7),
    )
    velocity = rasterize_aperture(grid.source, centers, 0.2, subsamples=8)
    result = propagate_pressure(velocity, grid, 0.15, frequency_hz=900)
    assert result.grid is grid.observation
    assert result.pressure.shape == (1, 7, 7)
    field = result.pressure[0]
    np.testing.assert_allclose(field, field[::-1], atol=1e-11)
    np.testing.assert_allclose(field, field[:, ::-1], atol=1e-11)
    np.testing.assert_allclose(field, field.T, atol=1e-11)


def test_on_axis_piston_pressure_converges_with_padding_and_resolution():
    centers = [(0, 0)]
    diameter, z = 0.2, 0.12
    frequency, speed, density = 720, 340, 1.2
    k = tau * frequency / speed
    # Integrate the baffled Rayleigh kernel over the circular piston on axis.
    rim_distance = np.hypot(z, diameter / 2)
    exact = density * speed * (np.exp(1j * k * z) - np.exp(1j * k * rim_distance))

    errors = []
    for sample_count, padding in ((5, 2), (21, 5), (41, 20)):
        spacing = diameter / (sample_count - 0.1)  # Odd source lengths sample the axis.
        grid = build_grid(
            centers, diameter, dx=spacing, dy=spacing,
            padding_factor=padding, observation_shape=(1, 1),
        )
        velocity = rasterize_aperture(grid.source, centers, diameter, subsamples=8)
        result = propagate_pressure(
            velocity, grid, z, frequency_hz=frequency,
            sound_speed_m_s=speed, air_density_kg_m3=density,
        )
        errors.append(abs(result.pressure.item() - exact) / abs(exact))
    assert errors[2] < errors[1] < errors[0]
    assert errors[2] < 0.1


@pytest.mark.parametrize("distance", [-1, [], [0, np.nan], [[0]], 1j])
def test_invalid_distances_raise(distance):
    grid = _plane_wave_grid()
    with pytest.raises(ValueError, match="distances_m"):
        propagate_pressure(np.zeros(grid.shape), grid, distance, frequency_hz=680)
