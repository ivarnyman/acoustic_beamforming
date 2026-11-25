from math import tau

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from matplotlib import pyplot as plt

from angular_spectrum import far_field_directivity, propagate_pressure
from grid import build_grid
from visualization import plot_angular_directivity, plot_pressure_planes, plot_pressure_slice


@pytest.mark.parametrize(("target_theta", "phi"), [(0, 0), (27, 0), (31, 63), (-24, 40)])
def test_directivity_peaks_in_prescribed_direction(target_theta, phi):
    grid = build_grid([(0, 0)], 1.0, dx=0.02, dy=0.025, padding_factor=8)
    x, y = np.meshgrid(grid.source.x, grid.source.y)
    frequency, speed = 1020, 340
    k = tau * frequency / speed
    azimuth = np.deg2rad(phi)
    transverse = x * np.cos(azimuth) + y * np.sin(azimuth)
    velocity = np.exp(-20 * (x**2 + y**2)) * np.exp(
        1j * k * np.sin(np.deg2rad(target_theta)) * transverse,
    )
    theta = np.linspace(-90, 90, 721)
    pattern = far_field_directivity(
        velocity, grid, theta, phi, frequency_hz=frequency, sound_speed_m_s=speed,
    )
    assert theta[np.abs(pattern).argmax()] == pytest.approx(target_theta, abs=1.3)
    if target_theta == 0:
        np.testing.assert_allclose(np.abs(pattern), np.abs(pattern[::-1]), atol=1e-12)


def test_far_field_scaling_and_phase_match_direct_fourier_sum():
    grid = build_grid([(0.08, -0.13)], 0.16, dx=0.02, dy=0.025, padding_factor=12)
    x, y = np.meshgrid(grid.source.x, grid.source.y)
    velocity = np.exp(-200 * ((x - 0.08)**2 + (y + 0.13)**2)) * (1 - 1j)
    theta, phi = 49, 27
    density, frequency, speed = 1.2, 900, 340
    k = tau * frequency / speed
    kx = k * np.sin(np.deg2rad(theta)) * np.cos(np.deg2rad(phi))
    ky = k * np.sin(np.deg2rad(theta)) * np.sin(np.deg2rad(phi))
    expected = -1j * density * frequency * grid.source.dx * grid.source.dy * np.sum(
        velocity * np.exp(-1j * (kx * x + ky * y)),
    )
    actual = far_field_directivity(
        velocity, grid, theta, phi, frequency_hz=frequency,
        sound_speed_m_s=speed, air_density_kg_m3=density,
    )
    assert actual == pytest.approx(expected, rel=0.01)


def test_far_field_rejects_directions_outside_spatial_nyquist_limit():
    grid = build_grid([(0, 0)], 1, dx=0.25, dy=0.25).source
    with pytest.raises(ValueError, match="Nyquist"):
        far_field_directivity(np.ones(grid.shape), grid, 90, frequency_hz=1020)


def test_xy_plots_share_color_scales_and_fixed_pressure_reference():
    grid = build_grid([(0, 0)], 1.0, dx=0.25, dy=0.2).source
    result = propagate_pressure(np.ones(grid.shape), grid, [0.1, 0.2], frequency_hz=800)
    reference = 2 * result.magnitude.max()
    fig = plot_pressure_planes(result, plane_indices=(1, 0), reference_pressure=reference)
    try:
        images = [ax.images[0] for ax in fig.axes[:6]]
        for first, second in zip(images[:3], images[3:]):
            assert first.norm is second.norm
        assert fig.axes[0].get_title() == "z = 0.2 m"
        np.testing.assert_allclose(images[0].get_array(), 0.5)
        np.testing.assert_allclose(images[1].get_array(), 20 * np.log10(0.5))
        assert images[2].get_clim() == (-np.pi, np.pi)
        fig.canvas.draw()
    finally:
        plt.close(fig)


@pytest.mark.parametrize("plane", ["xz", "yz"])
def test_slice_extracts_requested_cross_section_at_nonuniform_distances(plane):
    grid = build_grid([(0, 0)], 1.0, dx=0.25, dy=0.2).source
    x, y = np.meshgrid(grid.x, grid.y)
    result = propagate_pressure(np.exp(-7 * (x**2 + 2 * y**2)), grid, [0.4, 0, 0.1], frequency_hz=800)
    index = 1
    coordinate = grid.y[index] if plane == "xz" else grid.x[index]
    fig = plot_pressure_slice(result, plane, coordinate, reference_pressure=1)
    try:
        mesh = fig.axes[0].collections[0]
        np.testing.assert_allclose(mesh.get_coordinates()[:, 0, 1], [0, 0.05, 0.25, 0.55])
        expected = result.magnitude[:, index, :] if plane == "xz" else result.magnitude[:, :, index]
        np.testing.assert_allclose(mesh.get_array().reshape(expected.shape), expected[[1, 2, 0]])
        fig.canvas.draw()
    finally:
        plt.close(fig)


def test_directivity_plot_squares_pressure_for_intensity():
    fig = plot_angular_directivity([-30, 0, 30], [1, 2, 1j], reference_pressure=2)
    try:
        pressure, intensity, levels = fig.axes
        np.testing.assert_allclose(pressure.lines[0].get_ydata(), [0.5, 1, 0.5])
        np.testing.assert_allclose(intensity.lines[0].get_ydata(), [0.25, 1, 0.25])
        np.testing.assert_allclose(levels.lines[0].get_ydata(), 20 * np.log10([0.5, 1, 0.5]))
        np.testing.assert_allclose(levels.lines[1].get_ydata(), 10 * np.log10([0.25, 1, 0.25]))
        fig.canvas.draw()
    finally:
        plt.close(fig)
