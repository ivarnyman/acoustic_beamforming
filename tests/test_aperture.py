from math import pi

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from matplotlib import pyplot as plt

from aperture import RadialProfile, rasterize_aperture
from grid import build_grid
from visualization import plot_aperture


def test_subpixel_sampling_converges_to_disk_area():
    centers = [(0.17, -0.09)]
    grid = build_grid(centers, 1.0, dx=0.13, dy=0.19).source
    coarse = rasterize_aperture(grid, centers, 1.0, subsamples=1).real
    refined = rasterize_aperture(grid, centers, 1.0, subsamples=32).real
    exact_area = pi / 4

    assert np.any((refined > 0) & (refined < 1))
    coarse_error = abs(coarse.sum() * grid.dx * grid.dy - exact_area)
    refined_error = abs(refined.sum() * grid.dx * grid.dy - exact_area)
    assert refined_error < coarse_error / 10
    assert refined_error < 1e-3


@pytest.mark.parametrize("kind", ["uniform", "cosine", "gaussian"])
def test_radial_profiles_are_symmetric_and_peak_normalized(kind):
    profile = RadialProfile(kind)
    centers = [(0, 0)]
    grid = build_grid(centers, 1.0, dx=0.04, dy=0.04, margin=0.04).source
    field = rasterize_aperture(grid, centers, 1.0, profile=profile).real

    assert profile.evaluate([0.0, 1.01]).tolist() == [1.0, 0.0]
    assert 0.99 < field.max() <= 1
    assert field.min() == 0
    np.testing.assert_allclose(field, field[::-1, :], atol=1e-14)
    np.testing.assert_allclose(field, field[:, ::-1], atol=1e-14)
    np.testing.assert_allclose(field, field.T, atol=1e-14)


def test_profile_width_definitions():
    cosine = RadialProfile("cosine", taper_width_fraction=0.4)
    np.testing.assert_allclose(cosine.evaluate([0, 0.6, 0.8, 1]), [1, 1, 0.5, 0])
    gaussian = RadialProfile("gaussian", gaussian_sigma_fraction=0.3)
    np.testing.assert_allclose(gaussian.evaluate([0, 0.3, 1.01]), [1, np.exp(-0.5), 0])


@pytest.mark.parametrize("kind", ["uniform", "cosine", "gaussian"])
def test_equal_energy_normalization_precedes_weight(kind):
    centers = [(0, 0)]
    grid = build_grid(centers, 1.0, dx=0.07, dy=0.09).source
    weight = 1 + 2j
    velocity = rasterize_aperture(
        grid, centers, 1.0, profile=RadialProfile(kind),
        weights=[weight], normalization="equal_energy",
    )
    energy = np.sum(np.abs(velocity) ** 2) * grid.dx * grid.dy
    assert energy == pytest.approx(abs(weight) ** 2 * pi / 4)


def test_complex_weights_sum_when_two_disks_share_a_grid_cell():
    centers = [(-0.5, 0), (0.5, 0)]
    grid = build_grid(centers, 1.0, dx=0.4, dy=0.2).source
    left = rasterize_aperture(grid, centers[:1], 1.0)
    right = rasterize_aperture(grid, centers[1:], 1.0)
    unweighted = rasterize_aperture(grid, centers, 1.0)
    weighted = rasterize_aperture(grid, centers, 1.0, weights=[2j, -1 + 1j])

    assert np.iscomplexobj(weighted)
    assert np.any((left != 0) & (right != 0))
    np.testing.assert_allclose(unweighted, left + right)
    np.testing.assert_allclose(weighted, 2j * left + (-1 + 1j) * right)


@pytest.mark.parametrize(
    ("options", "message"),
    [({"weights": [1, 2]}, "weights"), ({"weights": [complex("nan")]}, "weights"),
     ({"normalization": "unknown"}, "normalization"), ({"subsamples": 0}, "subsamples")],
)
def test_invalid_rasterization_settings(options, message):
    centers = [(0, 0)]
    grid = build_grid(centers, 1.0, dx=0.2, dy=0.2).source
    with pytest.raises(ValueError, match=message):
        rasterize_aperture(grid, centers, 1.0, **options)


def test_aperture_plot_uses_full_grid_and_hides_phase_at_zero_amplitude():
    centers = [(0, 0)]
    grid = build_grid(centers, 1.0, dx=0.1, dy=0.2, margin=0.2).source
    field = rasterize_aperture(grid, centers, 1.0, weights=[-1j])
    profiles = tuple(RadialProfile(kind) for kind in ("uniform", "cosine", "gaussian"))
    figure = plot_aperture(grid, field, 1.0, profiles=profiles)
    try:
        radial, amplitude, phase = figure.axes[:3]
        assert len(radial.lines) == 3
        assert amplitude.images[0].get_extent() == list(grid.extent)
        np.testing.assert_allclose(amplitude.images[0].get_array(), np.abs(field))
        phase_values = phase.images[0].get_array()
        np.testing.assert_array_equal(phase_values.mask, field == 0)
        np.testing.assert_allclose(phase_values.compressed(), -pi / 2)
        figure.canvas.draw()
    finally:
        plt.close(figure)
