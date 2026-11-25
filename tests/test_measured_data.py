from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from matplotlib import pyplot as plt

import simulation
from config import SimulationConfig, load_configuration, save_configuration
from frequency_response import FrequencyResponse
from measured_data import (
    MeasuredElementPattern,
    MeasurementReference,
    PatternOrientation,
    load_element_pattern_csv,
    load_frequency_response_csv,
)
from signal_chain import SignalChain
from simulation import SimulationSession
from speakers import SpeakerArray, SpeakerElement, SpeakerGroup, SpeakerType
from validation import piston_directivity
from visualization import plot_aperture_measured_comparison

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "stage8"
VELOCITY_CSV = EXAMPLES / "synthetic_velocity_response.csv"
PATTERN_CSV = EXAMPLES / "synthetic_piston_pattern.csv"
REFERENCE = MeasurementReference(1, "normalized", "synthetic", "test", "Synthetic test samples")


def _pattern(rings=None, *, frequency_hz=(100, 400), orientation=PatternOrientation()):
    if rings is None:
        rings = np.full((3, 4), 1 + 2j)
    values = [rings * (1 + index) for index in range(len(frequency_hz))]
    return MeasuredElementPattern(
        frequency_hz, (0, 45, 90), (0, 90, 180, 270), values, 10, 343, "relative", REFERENCE, orientation,
    )


def _config(*types, elements=(), groups=(), **settings):
    return SimulationConfig(
        speaker_array=SpeakerArray(types, elements, groups), grid_spacing_m=0.0025, observation_shape=(1, 1),
        padding_factor=16, **settings,
    )


def test_velocity_csv_uses_existing_frequency_response_and_metadata():
    measurement = load_frequency_response_csv(VELOCITY_CSV)
    assert measurement.metadata["reference_input_value"] == 1
    assert measurement.metadata["reference_input_unit"] == "normalized"
    assert measurement.metadata["calibration"] == "synthetic"
    assert measurement.response.evaluate(800) == pytest.approx(0.5 + 0.5j)
    with pytest.raises(ValueError, match="outside the response band"):
        measurement.response.evaluate(200)
    held = load_frequency_response_csv(VELOCITY_CSV, out_of_range="hold").response
    assert held.evaluate(200) == held.evaluate(250)


def test_velocity_csv_db_phase_unwrapping_and_positive_time_conversion(tmp_path):
    source = VELOCITY_CSV.read_text().split("frequency_hz,magnitude,phase")[0]
    source = (source.replace("magnitude_unit=linear", "magnitude_unit=dB")
              .replace("exp(-i omega t)", "exp(+i omega t)").replace("phase_wrapping=unwrapped", "phase_wrapping=wrapped"))
    path = tmp_path / "velocity.csv"
    path.write_text(source + "frequency_hz,magnitude,phase\n400,-20,-170\n100,0,170\n")
    response = load_frequency_response_csv(path).response
    assert response.unwrapped_phase_rad == pytest.approx(np.deg2rad([-170, -190]))
    assert response.evaluate(200) == pytest.approx(-np.sqrt(0.1))


@pytest.mark.parametrize(("filename", "old", "new", "error"), [
    (VELOCITY_CSV, "relative_diaphragm_velocity", "SPL", "relative_diaphragm_velocity"),
    (VELOCITY_CSV, "reference_input_unit=normalized", "reference_input_unit=dBV", "reference_input_unit"),
    (VELOCITY_CSV, "calibration_id=synthetic-lowpass-800Hz", "extra_id=x", "calibration_id"),
    (VELOCITY_CSV, "500,", "250,", "strictly increasing"),
    (PATTERN_CSV, "reference_distance_m=20", "reference_distance_m=0", "finite and positive"),
    (PATTERN_CSV, "frequency_hz,theta_deg,phi_deg", "frequency_hz,elevation_deg,azimuth_deg", "columns"),
    (PATTERN_CSV, "500,0,90,", "500,0,0,", "complete frequency"),
    (PATTERN_CSV, "phase_reference=includes_propagation", "phase_reference=unknown", "phase_reference"),
])
def test_csv_rejects_incompatible_metadata_and_samples(tmp_path, filename, old, new, error):
    path = tmp_path / filename.name
    path.write_text(filename.read_text().replace(old, new, 1))
    loader = load_frequency_response_csv if filename == VELOCITY_CSV else load_element_pattern_csv
    with pytest.raises(ValueError, match=error):
        loader(path)


def test_pattern_interpolates_complex_angular_grid_and_log_frequency_across_seam():
    rings = np.array([[1, 1, 1, 1], [1j, 2j, 3j, 5j], [2j, 4j, 6j, 10j]])
    pattern = _pattern(rings)
    # Halfway in polar angle, across the azimuth seam, and in log frequency.
    actual = pattern.evaluate(67.5, [-45, 315, 675], np.array([[100], [200], [400]]))
    np.testing.assert_allclose(actual, 4.5j * np.array([[1], [1.5], [2]]) * np.ones(3))
    assert pattern.evaluate(-45, 0, 100) == pytest.approx(3j)
    with pytest.raises(ValueError, match="outside the measured pattern band"):
        pattern.evaluate(45, 0, 401)
    with pytest.raises(ValueError, match="outside the measured polar coverage"):
        pattern.evaluate(100, 0, 200)


def test_measurement_and_mounting_orientations_compose():
    rings = np.array([[1, 1, 1, 1], [1j, 2j, 3j, 5j], [2j, 4j, 6j, 10j]])
    pattern = _pattern(rings, frequency_hz=(100,), orientation=PatternOrientation(yaw_deg=90))
    assert pattern.evaluate(90, 90, 100) == pytest.approx(2j)
    mount = PatternOrientation(pitch_deg=90)
    assert pattern.evaluate(90, 0, 100, placement=mount) == pytest.approx(1)
    assert pattern.evaluate(90, 90, 100, placement=mount) == pytest.approx(2j)
    assert pattern.evaluate(0, 0, 100, placement=PatternOrientation(roll_deg=90)) == pytest.approx(2j)
    speaker = SpeakerType("one", "Oriented", 0.1, measured_pattern=pattern)
    config = _config(speaker, groups=(SpeakerGroup("one", 1, 1, orientation=mount),), frequency_hz=100)
    assert SimulationSession().measured_far_field(config, 90, 0) == pytest.approx(1)


def test_imported_piston_pattern_restores_far_field_phase_and_distance(tmp_path):
    pattern = load_element_pattern_csv(PATTERN_CSV)
    theta = np.arange(0, 91, 15)
    expected = piston_directivity(theta, 0.08, frequency_hz=1500, sound_speed_m_s=343, air_density_kg_m3=1.21)
    np.testing.assert_allclose(pattern.evaluate(theta, 25, 1500), expected, atol=1e-12)
    assert pattern.reference_distance_m == 20

    # The same coefficient from a different range with the propagation phase already removed.
    path = tmp_path / "pattern.csv"
    header = PATTERN_CSV.read_text().split("frequency_hz,theta_deg,phi_deg,magnitude,phase")[0]
    header = (header.replace("reference_distance_m=20", "reference_distance_m=7")
              .replace("phase_reference=includes_propagation", "phase_reference=range_removed")
              .replace("exp(-i omega t)", "exp(+i omega t)"))
    path.write_text(header + "frequency_hz,theta_deg,phi_deg,magnitude,phase\n" +
                    "".join(f"1500,{angle},{phi},{abs(value) / 7},90\n"
                            for angle, value in zip(theta, expected) for phi in (0, 180)))
    np.testing.assert_allclose(load_element_pattern_csv(path).evaluate(theta, 25, 1500), expected, atol=1e-12)


def test_measured_array_uses_electrical_weights_and_position_phase_only(monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail("Measured synthesis must not use the aperture/ASM pipeline")

    monkeypatch.setattr(simulation, "prepare_velocity_spectrum", fail)
    monkeypatch.setattr(simulation, "rasterize_aperture", fail)
    pattern = _pattern(frequency_hz=(1000,))
    types = (
        SpeakerType("a", "First", 0.1, measured_pattern=pattern),
        SpeakerType("b", "Second", 0.1, measured_pattern=pattern,
                    frequency_response=FrequencyResponse((1000,), (5j,))),
    )
    spacing = 0.343
    elements = (
        SpeakerElement("a", (0, 0)),
        SpeakerElement("b", (spacing, 0), weight=0.5, delay_s=0.00025, signal_chain=SignalChain(gain=2)),
    )
    config = _config(*types, elements=elements, frequency_hz=1000, relative_amplitude=3)
    session = SimulationSession()
    d = 1 + 2j
    theta = np.array([0, 30, 60])
    expected = 3 * d * (1 + 1j * np.exp(-2j * np.pi * np.sin(np.deg2rad(theta))))
    np.testing.assert_allclose(session.measured_far_field(config, theta, 0), expected, atol=1e-12)

    elements = (elements[0], SpeakerElement("b", (spacing, 0)))
    steered = replace(
        config, speaker_array=replace(config.speaker_array, elements=elements), beamforming_mode="steer",
        steering_elevation_deg=60,
    )
    assert session.measured_far_field(steered, 0, 0) == pytest.approx(0j, abs=1e-12)
    assert session.measured_far_field(steered, 30, 0) == pytest.approx(6 * d)


def test_measured_array_refuses_missing_or_incompatible_patterns():
    speaker = SpeakerType.generic(0.1)
    elements = (SpeakerElement("generic", (0, 0)),)
    config = _config(speaker, elements=elements, frequency_hz=100)
    session = SimulationSession()
    with pytest.raises(ValueError, match="no measured element pattern"):
        session.measured_far_field(config, 0, 0)
    with pytest.raises(ValueError, match="requires an explicit speaker array"):
        session.measured_far_field(SimulationConfig(), 0, 0)

    pattern = _pattern()
    speaker = replace(speaker, measured_pattern=pattern)
    other = SpeakerType("other", "Another", 0.1, measured_pattern=replace(
        pattern, reference=replace(REFERENCE, calibration_id="unrelated"),
    ))
    config = _config(speaker, other, elements=(*elements, SpeakerElement("other", (0.2, 0))), frequency_hz=100)
    with pytest.raises(ValueError, match="share pressure units, reference input, and calibration"):
        session.measured_far_field(config, 0, 0)
    with pytest.raises(ValueError, match="pattern sound speed must match"):
        session.measured_far_field(replace(config, sound_speed_m_s=340), 0, 0)


def test_measured_configuration_round_trip_and_legacy_aperture_behavior(tmp_path):
    frequency_response = load_frequency_response_csv(VELOCITY_CSV)
    pattern = load_element_pattern_csv(PATTERN_CSV)
    speaker = SpeakerType("piston", "Piston", 0.16)
    element = SpeakerElement("piston", (0, 0))
    config = _config(speaker, elements=(element,), frequency_hz=1000)
    assert "measured_pattern" not in config.speaker_array.to_dict()["speaker_types"][0]
    assert "orientation" not in config.speaker_array.to_dict()["elements"][0]
    session = SimulationSession()
    baseline = session.recalculate(config)

    measured_type = replace(speaker, measured_pattern=pattern)
    oriented = replace(element, orientation=PatternOrientation(yaw_deg=19))
    measured = _config(measured_type, elements=(oriented,), frequency_hz=1000)
    actual = session.recalculate(measured)
    np.testing.assert_array_equal(actual.velocity, baseline.velocity)
    np.testing.assert_array_equal(actual.pressure.pressure, baseline.pressure.pressure)
    np.testing.assert_array_equal(actual.directivity, baseline.directivity)

    measured_type = replace(measured_type, frequency_response=frequency_response.response,
                            frequency_response_metadata=frequency_response.metadata)
    config = _config(measured_type, elements=(oriented,), frequency_hz=1000)
    path = tmp_path / "speaker.json"
    save_configuration(path, config)
    loaded = load_configuration(path)
    assert loaded.speaker_array == config.speaker_array
    gain = frequency_response.response.evaluate(1000)
    np.testing.assert_allclose(session.recalculate(loaded).velocity, gain * baseline.velocity)


def test_synthetic_pattern_matches_asm_piston_and_plots_a_shared_reference():
    speaker = SpeakerType("piston", "Piston", 0.16, measured_pattern=load_element_pattern_csv(PATTERN_CSV))
    config = _config(speaker, elements=(SpeakerElement("piston", (0, 0)),))
    session = SimulationSession()
    progress = []
    frequencies = [1500, 1000]
    aperture = session.recalculate_frequency_directivity(config, frequencies, directivity_azimuth_deg=45)
    measured = session.recalculate_measured_directivity(
        config, frequencies, directivity_azimuth_deg=45, progress=lambda done, total: progress.append((done, total)),
    )
    np.testing.assert_array_equal(measured.frequencies_hz, frequencies)
    assert progress == [(1, 2), (2, 2)]
    assert measured.azimuth_deg == 45
    reference = np.abs(measured.directivity).max(axis=1)
    relative_error = np.abs(aperture.directivity - measured.directivity).max(axis=1) / reference
    assert (relative_error < 0.012).all()

    fig = plot_aperture_measured_comparison(
        aperture, measured, 1000, reference_directivity=reference[1], show_phase=True,
    )
    try:
        assert len(fig.axes) == 2
        assert fig.axes[0].get_title() == "shared reference"
        assert fig.axes[0].lines[1].get_ydata().max() == pytest.approx(0)
        fig.canvas.draw()
    finally:
        plt.close(fig)
