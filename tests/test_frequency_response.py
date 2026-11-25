import json
from dataclasses import replace

import numpy as np
import pytest

from angular_spectrum import propagate_pressure
from aperture import rasterize_aperture
from config import SimulationConfig, load_configuration, save_configuration
from frequency_response import FrequencyResponse
from simulation import SimulationSession, array_diameters, configured_weights, create_velocity, generate_array
from speakers import SpeakerArray, SpeakerElement, SpeakerGroup, SpeakerType


def test_interpolation_uses_log_magnitude_phase_and_frequency():
    response = FrequencyResponse((100, 10_000), (1, 100j))
    assert response.evaluate(1000) == pytest.approx(10 * np.exp(1j * np.pi / 4))
    np.testing.assert_allclose(
        response.evaluate([[100, 1000], [1000, 10_000]]),
        [[1, 10 * np.exp(1j * np.pi / 4)], [10 * np.exp(1j * np.pi / 4), 100j]],
    )


def test_phase_unwrap_preserves_continuity_and_polarity():
    response = FrequencyResponse(
        (100, 400), np.exp(1j * np.deg2rad([170, -170])),
    )
    assert response.unwrapped_phase_rad == pytest.approx(tuple(np.deg2rad([170, 190])))
    assert response.evaluate(200) == pytest.approx(-1 + 0j, abs=1e-14)
    inverted = FrequencyResponse((100, 400), (-1, -1))
    assert inverted.evaluate(400) == -1 + 0j


@pytest.mark.parametrize("policy", ["zero", "hold", "error"])
def test_out_of_band_policies_and_exact_endpoints(policy):
    response = FrequencyResponse((100, 400), (-2, 3j), out_of_range=policy)
    np.testing.assert_array_equal(response.evaluate([100, 400]), [-2, 3j])
    if policy == "error":
        with pytest.raises(ValueError, match=r"outside.*100, 400"):
            response.evaluate([50, 100, 800])
    else:
        expected = (0, 0) if policy == "zero" else (-2, 3j)
        np.testing.assert_array_equal(response.evaluate([50, 800]), expected)


@pytest.mark.parametrize(
    ("frequencies", "values", "message"),
    [((100, 100), (1, 1), "increasing"), ((200, 100), (1, 1), "increasing"),
     ((0, 100), (1, 1), "positive"), ((100, np.inf), (1, 1), "positive"),
     ((100, 200), (1, 0), "nonzero"), ((100, 200), (1, np.nan), "finite"),
     ((100, 200), (1,), "one complex")],
)
def test_invalid_sample_data(frequencies, values, message):
    with pytest.raises(ValueError, match=message):
        FrequencyResponse(frequencies, values)


def test_flat_and_synthetic_responses_follow_the_time_convention():
    flat = FrequencyResponse.flat()
    np.testing.assert_array_equal(flat.evaluate([0.1, 1024, 20_000]), [1, 1, 1])
    with pytest.raises(ValueError, match="positive"):
        flat.evaluate(0)
    lowpass = FrequencyResponse.synthetic_lowpass((100, 1000, 10_000), 1000)
    assert lowpass.evaluate(1000) == pytest.approx(0.5 + 0.5j)


def test_response_json_round_trip_retains_explicit_unwrapped_turns(tmp_path):
    response = FrequencyResponse.from_magnitude_phase(
        (100, 400), (1, 1), (0, 3 * np.pi), out_of_range="hold",
    )
    generic = SpeakerType.generic(0.1)
    typed = SpeakerType("demo", "Synthetic demo", 0.1, frequency_response=response)
    config = SimulationConfig(
        speaker_array=SpeakerArray(
            (generic, typed),
            (SpeakerElement("demo", (-0.1, 0)), SpeakerElement("generic", (0.1, 0))),
        ),
    )
    path = tmp_path / "response.json"
    save_configuration(path, config)
    types = json.loads(path.read_text())["speaker_array"]["speaker_types"]
    assert "frequency_response" not in types[0]
    assert types[1]["frequency_response"]["out_of_range"] == "hold"
    assert types[1]["frequency_response"]["unwrapped_phase_rad"] == pytest.approx([0, 3 * np.pi])
    reloaded = load_configuration(path)
    assert reloaded == config
    saved_response = reloaded.speaker_array.speaker_types[1].frequency_response
    assert saved_response.evaluate(200) == pytest.approx(-1j, abs=1e-14)


def _simulation_config(response):
    return SimulationConfig(
        rows=1, columns=3, speaker_diameter_m=0.1, horizontal_gap_m=0.02,
        grid_spacing_m=0.0125, padding_factor=3, observation_shape=(9, 11),
        observation_distance_m=0.25, frequency_hz=1000, relative_amplitude=0.8,
        weight_normalization="power", total_squared_weight=2,
        speaker_array=SpeakerArray(
            (SpeakerType.generic(0.1), SpeakerType("demo", "Synthetic demo", 0.1, frequency_response=response)),
            elements=(SpeakerElement("generic", (-0.22, 0), weight=1-1j),),
            groups=(SpeakerGroup("demo", 1, 2, center_m=(0.1, 0), horizontal_gap_m=0.02),),
        ),
    )


def test_response_applied_after_drive_normalization_and_refreshes_cached_aperture():
    response = FrequencyResponse((500, 1000), (0.125, -0.5j))
    config = _simulation_config(response)
    centers = generate_array(config)
    drive, _ = configured_weights(config, centers)
    assert np.sum(np.abs(drive)**2) == pytest.approx(2)
    session = SimulationSession()

    for frequency, gain in ((1000, -0.5j), (500, 0.125)):
        result = session.recalculate(replace(config, frequency_hz=frequency))
        expected_velocity = rasterize_aperture(
            result.grid.source, centers, array_diameters(config),
            weights=config.relative_amplitude * drive * [1, gain, gain],
        )
        np.testing.assert_array_equal(result.velocity, expected_velocity)
        expected_pressure = propagate_pressure(
            expected_velocity, result.grid, config.observation_distance_m, frequency_hz=frequency,
            sound_speed_m_s=config.sound_speed_m_s, air_density_kg_m3=config.air_density_kg_m3,
            include_evanescent=config.include_evanescent,
        )
        np.testing.assert_array_equal(result.pressure.pressure, expected_pressure.pressure)
        batch_velocity, _, _ = create_velocity(replace(config, frequency_hz=frequency), centers, result.grid)
        np.testing.assert_array_equal(batch_velocity, expected_velocity)


def test_zero_response_mutes_type_without_renormalizing_other_speakers():
    muted = _simulation_config(FrequencyResponse((500, 750), (1, 1), out_of_range="zero"))
    session = SimulationSession()
    result = session.recalculate(muted)
    drive, _ = configured_weights(muted, generate_array(muted))
    expected = rasterize_aperture(
        result.grid.source, [(-0.22, 0)], 0.1, weights=[muted.relative_amplitude * drive[0]],
    )
    np.testing.assert_array_equal(result.velocity, expected)
    with pytest.raises(ValueError, match=r"Speaker type 'demo'.*outside"):
        session.recalculate(_simulation_config(FrequencyResponse((500, 750), (1, 1))))


def test_explicit_flat_response_matches_original_simulation():
    config = _simulation_config(FrequencyResponse.flat())
    legacy = replace(config, speaker_array=None)
    groups = SpeakerArray(
        (replace(SpeakerType.generic(0.1), frequency_response=FrequencyResponse.flat()),),
        groups=(SpeakerGroup("generic", 1, 3, horizontal_gap_m=0.02),),
    )
    typed = replace(legacy, speaker_array=groups)
    session = SimulationSession()
    base = session.recalculate(legacy)
    actual = session.recalculate(typed)
    np.testing.assert_array_equal(base.velocity, actual.velocity)
    np.testing.assert_array_equal(base.pressure.pressure, actual.pressure.pressure)
    np.testing.assert_array_equal(base.directivity, actual.directivity)
