import matplotlib
import numpy as np
import pytest
from scipy.fft import next_fast_len

matplotlib.use("Agg")

from matplotlib import pyplot as plt
from matplotlib.patches import Circle

from geometry import generate_speaker_centers
from grid import build_grid
from visualization import plot_grid_diagnostic


def test_spacing_indexing_and_aperture_coverage():
    centers = generate_speaker_centers(2, 3, 0.1, 0.02, 0.03, 0.04)
    grid = build_grid(centers, 0.1, dx=0.013, dy=0.017, margin=(0.023, 0.031))
    source = grid.source

    np.testing.assert_allclose(np.diff(source.x), 0.013)
    np.testing.assert_allclose(np.diff(source.y), 0.017)
    np.testing.assert_allclose(grid.aperture_extent, (-0.19, 0.19, -0.115, 0.115))
    xmin, xmax, ymin, ymax = source.extent
    assert xmin <= -0.213 and xmax >= 0.213
    assert ymin <= -0.146 and ymax >= 0.146
    np.testing.assert_allclose([xmin + xmax, ymin + ymax], [0, 0], atol=1e-15)

    x, y = np.meshgrid(source.x, source.y, indexing="xy")
    assert x.shape == y.shape == source.shape
    assert x[1, 2] == source.x[2] and y[1, 2] == source.y[1]


def test_even_and_odd_frequency_bins_and_fft_order():
    grid = build_grid([(0, 0)], 1, dx=0.25, dy=0.2).computational
    assert grid.shape == (5, 4)
    fx, fy = grid.frequency_axes()
    np.testing.assert_allclose(fx, [0, 1, -2, -1])
    np.testing.assert_allclose(fy, [0, 1, 2, -2, -1])
    shifted_fx, shifted_fy = grid.frequency_axes(shifted=True)
    np.testing.assert_allclose(shifted_fx, [-2, -1, 0, 1])
    np.testing.assert_allclose(shifted_fy, [-2, -1, 0, 1, 2])

    x, y = np.meshgrid(grid.x, grid.y)
    spectrum = np.fft.fft2(np.exp(2j * np.pi * (-x + 2 * y)))
    iy, ix = np.unravel_index(np.abs(spectrum).argmax(), grid.shape)
    assert (fx[ix], fy[iy]) == (-1, 2)
    iy, ix = np.unravel_index(np.abs(np.fft.fftshift(spectrum)).argmax(), grid.shape)
    assert (shifted_fx[ix], shifted_fy[iy]) == (-1, 2)


@pytest.mark.parametrize("fast_fft", [False, True])
def test_padding_preserves_samples_and_aligns_observation_crop(fast_fft):
    kwargs = dict(centers=[(0.125, -0.25)], diameter=1, dx=0.25, dy=0.2)
    unpadded = build_grid(**kwargs)
    grid = build_grid(**kwargs, padding_factor=2.1, fast_fft=fast_fft, observation_shape=(6, 3))
    target = (11, 9)
    expected = tuple(next_fast_len(n) for n in target) if fast_fft else target
    assert grid.computational.shape == expected
    assert grid.observation.shape == (6, 3)
    assert grid.aperture_extent == unpadded.aperture_extent
    assert grid.observation.extent != grid.source.extent
    assert grid.computational.extent != grid.observation.extent

    np.testing.assert_array_equal(grid.source.x, unpadded.source.x)
    np.testing.assert_array_equal(grid.source.y, unpadded.source.y)
    padded = np.zeros(grid.computational.shape)
    padded[grid.source_slice] = 1
    assert np.count_nonzero(padded) == np.prod(grid.source.shape)
    oy, ox = grid.observation_slice
    np.testing.assert_array_equal(grid.observation.x, grid.computational.x[ox])
    np.testing.assert_array_equal(grid.observation.y, grid.computational.y[oy])
    for axis in ("x", "y"):
        vectors = [getattr(domain, axis) for domain in (grid.source, grid.computational, grid.observation)]
        assert vectors[0][len(vectors[0]) // 2] == vectors[1][len(vectors[1]) // 2]
        assert vectors[0][len(vectors[0]) // 2] == vectors[2][len(vectors[2]) // 2]


@pytest.mark.parametrize(
    ("name", "value"),
    [("dx", 0), ("dy", np.inf), ("margin", (-1, 0)), ("padding_factor", 0.5),
     ("observation_shape", (1, 100)), ("observation_shape", 2)],
)
def test_invalid_grid_parameters(name, value):
    kwargs = dict(dx=0.25, dy=0.2)
    kwargs[name] = value
    with pytest.raises(ValueError, match=name):
        build_grid([(0, 0)], 1, **kwargs)


def test_grid_diagnostic_draws_speakers_and_extent_outlines():
    centers = [(0.125, 0)]
    grid = build_grid(centers, 1, dx=0.25, dy=0.2, padding_factor=2)
    ax = plot_grid_diagnostic(grid, centers, 1)
    try:
        assert isinstance(ax.patches[0], Circle)
        assert ax.patches[0].center == centers[0]
        assert len(ax.patches) == 5
        assert [len(lines.get_segments()) for lines in ax.collections] == [4, 5]
        ax.figure.canvas.draw()
    finally:
        plt.close(ax.figure)


def test_observation_widths_in_meters_set_plot_grid_extent():
    from config import SimulationConfig
    from simulation import build_array_grid, generate_array

    config = SimulationConfig(
        rows=1, columns=1, grid_spacing_m=0.02, grid_spacing_y_m=0.03,
        padding_factor=4, observation_width_x_m=0.31, observation_width_y_m=0.21,
    )
    grid = build_array_grid(config, generate_array(config))
    assert grid.observation.shape == (7, 16)
    xmin, xmax, ymin, ymax = grid.observation.extent
    assert xmax - xmin == pytest.approx(0.32)
    assert ymax - ymin == pytest.approx(0.21)


def test_observation_width_must_fit_padded_grid():
    with pytest.raises(ValueError, match="Increase padding"):
        build_grid([(0, 0)], 0.1, dx=0.01, dy=0.01, observation_width_y_m=0.2)


@pytest.mark.parametrize("fast_fft", [False, True])
def test_computational_dimensions_match_full_grid_extent(fast_fft):
    from config import SimulationConfig
    from simulation import build_array_grid, computational_dimensions_m, generate_array

    config = SimulationConfig(
        row_stagger=0.4, grid_spacing_m=0.013, grid_spacing_y_m=0.017,
        grid_margin_m=(0.02, 0.03), padding_factor=3, fast_fft=fast_fft,
        observation_width_x_m=0.25,
    )
    grid = build_array_grid(config, generate_array(config))
    xmin, xmax, ymin, ymax = grid.computational.extent
    assert computational_dimensions_m(config) == pytest.approx((xmax - xmin, ymax - ymin))


def test_observation_width_factors_scale_physical_array_independently():
    from config import SimulationConfig
    from simulation import build_array_grid, generate_array

    config = SimulationConfig(
        rows=2, columns=3, speaker_diameter_m=0.1,
        horizontal_gap_m=0.02, vertical_gap_m=0.04, row_stagger=0.5,
        grid_spacing_m=0.01, padding_factor=2,
        observation_width_x_factor=1.0, observation_width_y_factor=2.0,
    )
    grid = build_array_grid(config, generate_array(config))
    assert grid.observation.shape == (48, 40)
