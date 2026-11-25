import numpy as np

from grid import XYGrid
from validation import mode_checks, piston_case, piston_directivity, piston_on_axis


def test_piston_validation_separates_padding_from_spatial_resolution():
    distances = [0.03, 0.12, 0.8]
    short = piston_case(21, 2, 8, distances)
    padded = piston_case(21, 36, 8, distances)
    assert padded["on_axis_relative_error"][1] < 0.08
    assert padded["on_axis_relative_error"][1] < short["on_axis_relative_error"][1] / 10
    assert padded["on_axis_relative_error"][2] > padded["on_axis_relative_error"][0]

    coarse = piston_case(9, 12, 8, [0.12])
    fine = piston_case(41, 12, 8, [0.12])
    assert fine["far_field_max_complex_error_over_broadside"] < 0.004
    assert fine["far_field_max_complex_error_over_broadside"] < coarse["far_field_max_complex_error_over_broadside"]
    assert fine["relative_area_error"] < coarse["relative_area_error"]


def test_boundary_subsampling_improves_area_and_far_field():
    coarse = piston_case(21, 12, 1, [0.12])
    averaged = piston_case(21, 12, 16, [0.12])
    assert averaged["relative_area_error"] < coarse["relative_area_error"] / 10
    assert averaged["far_field_max_complex_error_over_broadside"] < coarse["far_field_max_complex_error_over_broadside"]


def test_on_axis_reference_approaches_far_field_as_distance_increases():
    medium = dict(frequency_hz=720, sound_speed_m_s=340, air_density_kg_m3=1.2)
    z = np.array([0.12, 0.35, 0.8, 10])
    k = 2 * np.pi * medium["frequency_hz"] / medium["sound_speed_m_s"]
    scaled = z * piston_on_axis(z, 0.1, **medium) * np.exp(-1j * k * z)
    far_limit = piston_directivity(0, 0.1, **medium)
    errors = np.abs(scaled / far_limit - 1)
    assert np.all(np.diff(errors) < 0)
    assert errors[-1] < 0.004


def test_grazing_cutoff_and_near_grazing_gain_are_distinct():
    checks = mode_checks()
    assert checks["exact_grazing_peak"] == checks["filtered_peak"] == 0
    assert checks["near_grazing_gain_over_rho_c"] > 1000
    np.testing.assert_allclose(checks["evanescent_ratios"], checks["analytical_ratios"])


def test_frequency_axes_are_cached_by_sampling_and_immutable():
    first = XYGrid(np.arange(8) / 8, np.arange(5) / 5, 1 / 8, 1 / 5)
    shifted_origin = XYGrid(first.x + 0.1, first.y - 0.2, first.dx, first.dy)
    fx, fy = first.frequency_axes()
    assert shifted_origin.frequency_axes()[0] is fx
    assert shifted_origin.frequency_axes()[1] is fy
    assert not fx.flags.writeable and not fy.flags.writeable
    shifted_fx, shifted_fy = first.frequency_axes(shifted=True)
    np.testing.assert_array_equal(shifted_fx, np.fft.fftshift(fx))
    np.testing.assert_array_equal(shifted_fy, np.fft.fftshift(fy))
