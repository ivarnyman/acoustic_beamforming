import json
from dataclasses import replace
from math import pi, tau

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from matplotlib import pyplot as plt

from angular_spectrum import propagate_pressure
from aperture import rasterize_aperture
from config import SimulationConfig, load_configuration, save_configuration
from frequency_response import FrequencyResponse
from signal_chain import Crossover, SignalChain
from simulation import SimulationSession, configured_weights, generate_array
from speakers import SpeakerArray, SpeakerElement, SpeakerGroup, SpeakerType
from visualization import plot_transfer_functions


@pytest.mark.parametrize(
    ("kind", "stopband_points"), [("lowpass", (10_000, 100_000)), ("highpass", (100, 10))],
)
def test_butterworth_cutoff_and_asymptotic_slope(kind, stopband_points):
    crossover = Crossover(kind, 1000, order=2)
    assert abs(crossover.evaluate(1000)) == pytest.approx(np.sqrt(0.5))
    near, far = np.abs(crossover.evaluate(stopband_points))
    assert 20 * np.log10(far / near) == pytest.approx(-40, abs=0.01)
    assert crossover.evaluate(0) == pytest.approx(1 if kind == "lowpass" else 0)


def test_butterworth_bandpass_edges_center_and_both_slopes():
    crossover = Crossover("bandpass", (200, 800), order=2)
    np.testing.assert_allclose(np.abs(crossover.evaluate([200, 400, 800])), [2**-0.5, 1, 2**-0.5])
    for frequencies in ([20, 2], [8000, 80_000]):
        first, second = np.abs(crossover.evaluate(frequencies))
        assert 20 * np.log10(second / first) == pytest.approx(-40, abs=0.25)


def test_first_order_phase_matches_negative_time_exponential():
    assert Crossover("lowpass", 1000, order=1).evaluate(1000) == pytest.approx(0.5 + 0.5j)
    assert Crossover("highpass", 1000, order=1).evaluate(1000) == pytest.approx(0.5 - 0.5j)


@pytest.mark.parametrize(("order", "high_polarity"), [(2, -1), (4, 1), (8, 1)])
def test_linkwitz_riley_pair_at_cutoff_and_coherent_sum(order, high_polarity):
    low = Crossover("lowpass", 1000, order=order, family="linkwitz_riley")
    high = Crossover("highpass", 1000, order=order, family="linkwitz_riley")
    low_at_cutoff = low.evaluate(1000)
    high_at_cutoff = high.evaluate(1000)
    assert abs(low_at_cutoff) == pytest.approx(0.5)
    assert abs(high_at_cutoff) == pytest.approx(0.5)
    assert high_polarity * high_at_cutoff == pytest.approx(low_at_cutoff)
    f = np.geomspace(80, 12_000, 20)
    np.testing.assert_allclose(np.abs(low.evaluate(f) + high_polarity * high.evaluate(f)), 1)
    near, far = np.abs(low.evaluate([10_000, 100_000]))
    assert 20 * np.log10(far / near) == pytest.approx(-20 * order, abs=0.1)


def test_identity_and_delay_gain_polarity_and_phase_are_applied_once():
    np.testing.assert_array_equal(SignalChain().evaluate([0, 500, 2000]), [1, 1, 1])
    delay = 250e-6
    chain = SignalChain(gain=0.75, polarity=-1, phase_offset_rad=pi / 4, delay_s=delay)
    assert chain.evaluate(1000) == pytest.approx(-0.75 * np.exp(3j * pi / 4))
    assert chain.evaluate(1500) / chain.evaluate(1000) == pytest.approx(np.exp(1j * tau * 500 * delay))

    with_alias = SpeakerElement("demo", (0, 0), delay_s=delay, signal_chain=chain)
    assert with_alias.effective_signal_chain.evaluate(1000) == chain.evaluate(1000)
    with pytest.raises(ValueError, match="conflicts"):
        replace(with_alias, delay_s=2 * delay)


@pytest.mark.parametrize(
    "parameters",
    [
        ("lowpass", 0, 2, "butterworth"),
        ("bandpass", (500, 200), 2, "butterworth"),
        ("bandpass", (200, 800), 4, "linkwitz_riley"),
        ("highpass", 500, 3, "linkwitz_riley"),
    ],
)
def test_invalid_crossover_settings(parameters):
    with pytest.raises(ValueError):
        Crossover(*parameters)


def test_group_overrides_are_unique_and_within_the_group():
    with pytest.raises(ValueError, match="within the group"):
        SpeakerGroup("demo", 1, 2, element_signal_overrides=((2, SignalChain()),))
    with pytest.raises(ValueError, match="unique"):
        SpeakerGroup("demo", 1, 2, element_signal_overrides=((0, SignalChain()), (0, SignalChain())))


def _config(**kwargs):
    return SimulationConfig(
        rows=1, columns=3, speaker_diameter_m=0.1, horizontal_gap_m=0.02,
        frequency_hz=1000, grid_spacing_m=0.0125, observation_shape=(7, 9),
        padding_factor=3, observation_distance_m=0.25, **kwargs,
    )


def test_group_and_element_processing_round_trip_and_apply_before_rasterizing(tmp_path):
    speaker_type = SpeakerType(
        "demo", "Demo", 0.1, frequency_response=FrequencyResponse((500, 2000), (2j, 2j)),
    )
    group_chain = SignalChain(gain=0.5, crossover=Crossover("lowpass", 1000, order=1))
    override = SignalChain(crossover=Crossover("highpass", 1000, order=1))
    individual = SpeakerElement("demo", (-0.22, 0), weight=1-1j, delay_s=0.00025,
                                signal_chain=SignalChain(gain=2))
    group = SpeakerGroup(
        "demo", 1, 2, center_m=(0.1, 0), horizontal_gap_m=0.02,
        signal_chain=group_chain, element_signal_overrides=((1, override),),
    )
    config = _config(
        speaker_array=SpeakerArray((speaker_type,), (individual,), (group,)),
        relative_amplitude=0.8, weight_normalization="power",
    )
    path = tmp_path / "chains.json"
    save_configuration(path, config)
    group_data = json.loads(path.read_text())["speaker_array"]["groups"][0]
    assert group_data["signal_chain"]["crossover"]["kind"] == "lowpass"
    assert group_data["element_signal_overrides"][0][0] == 1
    loaded = load_configuration(path)
    assert loaded == config
    assert [item.signal_chain for item in loaded.speaker_array.flattened_elements] == [
        individual.signal_chain, group_chain, override,
    ]

    session = SimulationSession()
    result = session.recalculate(loaded)
    centers = generate_array(loaded)
    drive, _ = configured_weights(loaded, centers)
    electronics = np.array([2j, 0.5 * (0.5 + 0.5j), 0.5 - 0.5j])
    expected = rasterize_aperture(
        result.grid.source, centers, 0.1, weights=0.8 * drive * electronics * 2j,
    )
    np.testing.assert_allclose(result.velocity, expected, atol=1e-14)
    pressure = propagate_pressure(
        expected, result.grid, loaded.observation_distance_m, frequency_hz=loaded.frequency_hz,
        sound_speed_m_s=loaded.sound_speed_m_s, air_density_kg_m3=loaded.air_density_kg_m3,
    )
    np.testing.assert_allclose(result.pressure.pressure, pressure.pressure, atol=1e-12)

    identity_override = replace(group, element_signal_overrides=((1, SignalChain()),))
    edited = replace(loaded, speaker_array=replace(loaded.speaker_array, groups=(identity_override,)))
    updated = session.recalculate(edited)
    assert updated.grid is result.grid
    assert not np.array_equal(updated.velocity, result.velocity)


def test_optional_identity_signal_chain_matches_legacy_simulation():
    legacy = _config(weight_normalization="power")
    typed = replace(legacy, speaker_array=SpeakerArray(
        (SpeakerType.generic(legacy.speaker_diameter_m),),
        groups=(SpeakerGroup("generic", 1, 3, horizontal_gap_m=0.02, signal_chain=SignalChain()),),
    ))
    session = SimulationSession()
    original, with_chain = session.recalculate(legacy), session.recalculate(typed)
    np.testing.assert_array_equal(with_chain.velocity, original.velocity)
    np.testing.assert_array_equal(with_chain.pressure.pressure, original.pressure.pressure)
    np.testing.assert_array_equal(with_chain.directivity, original.directivity)


def test_transfer_plots_show_individual_and_combined_complex_gains():
    f = [500, 1000, 2000]
    chain = SignalChain(gain=0.5, delay_s=0.0001, crossover=Crossover("lowpass", 1000))
    response = FrequencyResponse((500, 2000), (0.5j, 0.5j))
    figure = plot_transfer_functions(chain, f, frequency_response=response)
    try:
        magnitude_ax, phase_ax = figure.axes
        assert "Crossover" in [line.get_label() for line in magnitude_ax.lines]
        expected = chain.evaluate(f) * response.evaluate(f)
        np.testing.assert_allclose(magnitude_ax.lines[-1].get_ydata(), 20 * np.log10(np.abs(expected)))
        np.testing.assert_allclose(phase_ax.lines[-1].get_ydata(), np.angle(expected))
        figure.canvas.draw()
    finally:
        plt.close(figure)
