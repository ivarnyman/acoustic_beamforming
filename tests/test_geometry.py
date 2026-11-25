import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from matplotlib import pyplot as plt
from matplotlib.patches import Circle

from geometry import generate_speaker_centers, validate_speaker_centers
from visualization import plot_speaker_geometry


def test_rectangular_array_count_coordinates_and_spacing():
    centers = generate_speaker_centers(2, 3, 0.1, horizontal_gap=0.02, vertical_gap=0.04)

    assert centers.shape == (6, 2)
    np.testing.assert_allclose(
        centers,
        [(-0.12, -0.07), (0, -0.07), (0.12, -0.07),
         (-0.12, 0.07), (0, 0.07), (0.12, 0.07)],
    )


def test_scalar_offset_shifts_alternating_rows():
    centers = generate_speaker_centers(4, 2, 0.1, row_offsets=0.03)
    row_x = centers.reshape(4, 2, 2)[:, 0, 0]

    np.testing.assert_allclose(row_x - row_x[0], [0, 0.03, 0, 0.03])


def test_explicit_offsets_center_the_bounding_box():
    centers = generate_speaker_centers(3, 2, 0.1, row_offsets=(0.05, -0.02, 0.03))
    row_x = centers.reshape(3, 2, 2)[:, 0, 0]

    np.testing.assert_allclose(row_x - row_x[0], [0, -0.07, -0.02])
    np.testing.assert_allclose(centers.min(axis=0) + centers.max(axis=0), [0, 0], atol=1e-15)


def test_single_speaker_is_centered_even_with_offset():
    np.testing.assert_array_equal(generate_speaker_centers(1, 1, 0.1, row_offsets=(2,)), [[0, 0]])


def test_manual_coordinates_preserve_order_and_position():
    points = [(3, -2), (1, 0), (-4, 5)]
    np.testing.assert_array_equal(validate_speaker_centers(points, 0.1), points)


@pytest.mark.parametrize("centers", [[(0, 0), (0, 0)], [(0, 0), (0.06, 0.06)]])
def test_overlapping_disks_raise(centers):
    with pytest.raises(ValueError, match="speaker disks 0 and 1 overlap"):
        validate_speaker_centers(centers, 0.1)


def test_tangent_disks_are_allowed():
    assert generate_speaker_centers(2, 4, 0.1).shape == (8, 2)
    assert validate_speaker_centers([(0, 0), (3, 4)], 5).shape == (2, 2)


@pytest.mark.parametrize(
    ("parameter", "value"),
    [("n_rows", 0), ("n_cols", 1.5), ("diameter", float("inf")),
     ("horizontal_gap", -0.01), ("vertical_gap", float("nan")), ("row_offsets", [0])],
)
def test_invalid_generation_parameters_raise(parameter, value):
    parameters = dict(n_rows=2, n_cols=2, diameter=0.1)
    parameters[parameter] = value
    with pytest.raises(ValueError, match=parameter):
        generate_speaker_centers(**parameters)


@pytest.mark.parametrize("centers", [[], [1, 2], [(0, float("nan"))], [(0, 1j)]])
def test_invalid_manual_coordinates_raise(centers):
    with pytest.raises(ValueError, match="centers"):
        validate_speaker_centers(centers, 0.1)


@pytest.mark.parametrize("show_indices", [False, True])
def test_geometry_plot(show_indices):
    ax = plot_speaker_geometry([(-0.1, 0), (0.1, 0)], 0.1, show_indices=show_indices)
    try:
        assert len(ax.patches) == 2
        for patch, center in zip(ax.patches, [(-0.1, 0), (0.1, 0)]):
            assert isinstance(patch, Circle)
            assert patch.center == center
            assert patch.radius == 0.05
        assert ax.get_aspect() == 1
        assert (ax.get_xlabel(), ax.get_ylabel()) == ("x (m)", "y (m)")
        assert [label.get_text() for label in ax.texts] == (["0", "1"] if show_indices else [])
    finally:
        plt.close(ax.figure)


def test_layout_plot_colors_types_and_optionally_shows_type_ids():
    ax = plot_speaker_geometry(
        [(-0.2, 0), (0, 0), (0.2, 0)], [0.1, 0.2, 0.1],
        type_ids=("small", "large", "small"), show_indices=True, show_type_ids=True,
    )
    try:
        small, large, other_small = ax.patches
        assert small.get_edgecolor() == other_small.get_edgecolor()
        assert small.get_edgecolor() != large.get_edgecolor()
        assert [patch.radius for patch in ax.patches] == [0.05, 0.1, 0.05]
        assert [label.get_text() for label in ax.texts] == ["0\nsmall", "1\nlarge", "2\nsmall"]
        assert [label.get_text() for label in ax.get_legend().get_texts()] == ["small", "large"]
        assert ax.get_aspect() == 1
        ax.figure.canvas.draw()
    finally:
        plt.close(ax.figure)
