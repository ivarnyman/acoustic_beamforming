import json
from dataclasses import replace
from math import pi

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from matplotlib import pyplot as plt

from angular_spectrum import propagate_pressure
from aperture import CallableProfile, FrequencyParameter, RadialProfile, rasterize_aperture
from config import SimulationConfig, load_configuration, save_configuration
from frequency_response import FrequencyResponse
from grid import XYGrid, build_grid
from simulation import SimulationSession
from speakers import SpeakerArray, SpeakerElement, SpeakerType
from visualization import plot_diaphragm_profiles


def test_radial_parameters_interpolate_at_frequency_and_preserve_phase():
    sigma = FrequencyParameter((500, 2000), (0.2, 0.4))
    phase = FrequencyParameter((500, 2000), (0, pi))
    profile = RadialProfile("gaussian", gaussian_sigma_fraction=sigma, rim_phase_rad=phase)
    expected = np.exp(-0.5 * (0.5 / 0.3)**2 + 1j * pi / 8)
    np.testing.assert_allclose(profile.evaluate([0, 0.5, 1.1], 1000), [1, expected, 0])
    assert sigma.evaluate(400) == 0.2
    assert sigma.evaluate(3000) == 0.4
    with pytest.raises(ValueError, match="frequency_hz"):
        profile.evaluate(0.5)


@pytest.mark.parametrize("kind", ["uniform", "cosine", "gaussian"])
def test_legacy_profiles_remain_real_and_independent_of_frequency(kind):
    profile = RadialProfile(kind)
    r = np.linspace(0, 1.2, 100)
    np.testing.assert_array_equal(profile.evaluate(r, 1200), profile.evaluate(r))
    assert np.isrealobj(profile.evaluate(r, 1200))


def test_callable_peak_normalization_keeps_complex_phase_and_disk_sampling():
    def profile_function(r, frequency_hz):
        assert (r <= 1).all()
        assert frequency_hz == 1000
        return 3j

    centers = [(0, 0)]
    grid = build_grid(centers, 1, dx=0.13, dy=0.16, margin=0.2).source
    baseline = rasterize_aperture(grid, centers, 1, subsamples=4)
    field = rasterize_aperture(
        grid, centers, 1, profile=CallableProfile(profile_function), frequency_hz=1000, subsamples=4,
    )
    np.testing.assert_allclose(field, 1j * baseline)
    assert np.any((field.imag > 0) & (field.imag < 1))


def test_subpixel_complex_cancellation_occurs_before_taking_magnitude():
    grid = XYGrid(np.array([0.0]), np.array([0.0]), dx=0.4, dy=0.4)
    profile = CallableProfile(lambda x, y, frequency_hz: np.where(x < 0, 2j, -2j), "xy")
    kwargs = dict(profile=profile, frequency_hz=500, subsamples=2)
    assert rasterize_aperture(grid, [(0, 0)], 1, **kwargs).item() == 0j
    with pytest.raises(ValueError, match="no sampled velocity"):
        rasterize_aperture(grid, [(0, 0)], 1, normalization="equal_energy", **kwargs)


def test_equal_energy_normalization_uses_squared_magnitude_and_keeps_phase():
    centers = [(0, 0)]
    grid = build_grid(centers, 1, dx=0.1, dy=0.12).source
    profile = CallableProfile(lambda x, y, frequency_hz: (1 + x / 2) * np.exp(2j * x), "xy")
    kwargs = dict(profile=profile, frequency_hz=600, weights=[1+2j])
    peak = rasterize_aperture(grid, centers, 1, **kwargs)
    energy = rasterize_aperture(grid, centers, 1, normalization="equal_energy", **kwargs)
    assert np.sum(np.abs(energy)**2) * grid.dx * grid.dy == pytest.approx(5 * pi / 4)
    active = peak != 0
    assert np.abs(peak).max() <= abs(kwargs["weights"][0])
    np.testing.assert_allclose(np.angle(energy[active]), np.angle(peak[active]))


def test_frequency_parameters_round_trip_and_custom_functions_fail_clearly(tmp_path):
    profile = RadialProfile("cosine", taper_width_fraction=FrequencyParameter((500, 1000), (0.2, 0.4)),
                            rim_phase_rad=FrequencyParameter((500, 1000), (0.0, 0.8)))
    speaker_type = SpeakerType("demo", "Demo", 0.1, profile)
    array = SpeakerArray((speaker_type,), (SpeakerElement("demo", (0, 0)),))
    config = SimulationConfig(speaker_array=array)
    path = tmp_path / "profile.json"
    save_configuration(path, config)
    saved = path.read_text()
    assert json.loads(saved)["speaker_array"]["speaker_types"][0]["aperture_profile"]["rim_phase_rad"] == {
        "frequencies_hz": [500, 1000], "values": [0, 0.8],
    }
    assert load_configuration(path) == config

    custom = replace(speaker_type, aperture_profile=CallableProfile(lambda r, f: 1j))
    with pytest.raises(ValueError, match="Python-only"):
        save_configuration(path, replace(config, speaker_array=replace(array, speaker_types=(custom,))))
    assert path.read_text() == saved


def test_mixed_profiles_change_with_frequency_and_keep_drive_response_gain():
    profile = CallableProfile(lambda r, f: 3 * np.exp(1j * pi * f / 2000))
    config = SimulationConfig(
        grid_spacing_m=0.01, padding_factor=4, observation_shape=(7, 9),
        observation_distance_m=0.25,
        speaker_array=SpeakerArray(
            (
                SpeakerType.generic(0.12),
                SpeakerType("complex", "Complex", 0.2, profile,
                            frequency_response=FrequencyResponse((500, 1000), (2j, 2j))),
            ),
            (SpeakerElement("generic", (-0.2, 0)), SpeakerElement("complex", (0.2, 0), weight=0.5)),
        ),
    )
    session = SimulationSession()
    previous = None
    for frequency in (500, 1000):
        config = replace(config, frequency_hz=frequency)
        result = session.recalculate(config)
        expected = rasterize_aperture(
            result.grid.source, [(-0.2, 0), (0.2, 0)], (0.12, 0.2),
            weights=[1, 1j * np.exp(1j * pi * frequency / 2000)],
        )
        np.testing.assert_allclose(result.velocity, expected, atol=1e-14)
        expected_pressure = propagate_pressure(
            expected, result.grid, config.observation_distance_m, frequency_hz=frequency,
            sound_speed_m_s=config.sound_speed_m_s, air_density_kg_m3=config.air_density_kg_m3,
        )
        np.testing.assert_allclose(result.pressure.pressure, expected_pressure.pressure, atol=1e-11)
        if previous is not None:
            assert result.grid is previous.grid
            assert not np.array_equal(result.velocity, previous.velocity)
        previous = result


def test_plot_selected_diaphragm_phase_at_two_frequencies():
    profile = CallableProfile(lambda r, f: np.exp(1j * pi * f / 2000), label="Demo")
    fig = plot_diaphragm_profiles((profile,), 0.1, (500, 1000), samples_across=15)
    try:
        for row, frequency in enumerate((500, 1000)):
            amp_ax, phase_ax = fig.axes[2 * row:2 * row + 2]
            assert f"Demo · {frequency} Hz" in phase_ax.get_title()
            assert amp_ax.images[0].get_array().max() == pytest.approx(1)
            phase = phase_ax.images[0].get_array()
            assert np.ma.getmaskarray(phase).any()
            np.testing.assert_allclose(phase.compressed(), pi * frequency / 2000)
        fig.canvas.draw()
    finally:
        plt.close(fig)
