from concurrent.futures import CancelledError
from dataclasses import replace
from math import pi, tau
from threading import Event
from weakref import ref

import numpy as np
import pytest

import angular_spectrum
import grid
import simulation
from aperture import CallableProfile
from config import SimulationConfig
from frequency_response import FrequencyResponse
from signal_chain import Crossover, SignalChain
from simulation import SimulationSession
from speakers import SpeakerArray, SpeakerGroup, SpeakerType


def _config(**changes):
    defaults = dict(
        rows=1, columns=3, speaker_diameter_m=0.1, horizontal_gap_m=0.03,
        frequency_hz=850, grid_spacing_m=0.01, padding_factor=3,
        observation_shape=(7, 9), observation_distance_m=0.3,
    )
    return SimulationConfig(**{**defaults, **changes})


@pytest.mark.parametrize("include_evanescent", [True, False])
def test_one_frequency_sweep_matches_single_frequency_api(include_evanescent):
    config = _config(include_evanescent=include_evanescent)
    session = SimulationSession()
    legacy = session.recalculate(config)
    sweep = session.recalculate_frequencies(config, [config.frequency_hz])
    assert sweep.pressure.shape == (1, 1, 7, 9)
    assert sweep.grid is legacy.pressure.grid
    np.testing.assert_array_equal(sweep.frequencies_hz, [config.frequency_hz])
    np.testing.assert_array_equal(sweep.distances_m, legacy.pressure.distances_m)
    np.testing.assert_array_equal(sweep.pressure[0], legacy.pressure.pressure)


def test_same_frequency_speaker_contributions_superpose_linearly():
    config = _config(weight_normalization="none")
    session = SimulationSession()
    weights_a = (0.5 + 1j, 0, 0)
    weights_b = (0, -0.2j, 1 - 0.1j)
    weights_ab = tuple(a + b for a, b in zip(weights_a, weights_b))
    frequencies = [1200, 650, 1200]
    distances = [0.4, 0, 0.15]
    a, b, combined = [
        session.recalculate_frequencies(replace(config, speaker_weights=weights), frequencies, distances)
        for weights in (weights_a, weights_b, weights_ab)
    ]
    np.testing.assert_array_equal(combined.frequencies_hz, frequencies)
    np.testing.assert_array_equal(combined.distances_m, distances)
    np.testing.assert_allclose(combined.pressure, a.pressure + b.pressure, atol=1e-12)
    np.testing.assert_array_equal(combined.pressure[0], combined.pressure[2])


def test_sweep_recomputes_steering_response_crossover_delay_and_profile_at_each_frequency(monkeypatch):
    frequencies = [500, 1000, 1500]
    config = _config(
        beamforming_mode="steer", steering_azimuth_deg=0, steering_elevation_deg=65,
    )
    array = SpeakerArray(
        (SpeakerType(
            "demo", "Idealized demo", 0.1,
            aperture_profile=CallableProfile(lambda r, f: np.exp(1j * pi * f / 4000)),
            frequency_response=FrequencyResponse((500, 1500), (0.25j, 1j)),
        ),),
        groups=(SpeakerGroup(
            "demo", 1, 3, horizontal_gap_m=0.03,
            signal_chain=SignalChain(delay_s=0.00025, crossover=Crossover("lowpass", 1000, order=1)),
        ),),
    )
    typed_config = replace(config, speaker_array=array)
    calls = []
    fft2 = angular_spectrum.fft2

    def count_fft(*args, **kwargs):
        calls.append(args[0].shape)
        return fft2(*args, **kwargs)

    monkeypatch.setattr(angular_spectrum, "fft2", count_fft)
    actual = SimulationSession().recalculate_frequencies(typed_config, frequencies)
    assert len(calls) == len(frequencies)  # All speakers share one source FFT per frequency.

    for i, f in enumerate(frequencies):
        base = SimulationSession().recalculate(replace(config, frequency_hz=f))
        magnitude = 0.25 * (f / 500)**(np.log(4) / np.log(3))
        transfer = (1j * magnitude) * np.exp(1j * (tau * 0.00025 * f + pi * f / 4000)) / (1 - 1j * f / 1000)
        np.testing.assert_allclose(actual.pressure[i], base.pressure.pressure * transfer, atol=1e-11)


def test_stream_is_lazy_and_reuses_static_geometry_spectrum_and_frequency_axes(monkeypatch):
    calls = []
    grid._frequency_axes.cache_clear()

    def watch(name):
        original = getattr(simulation, name)

        def tracked(*args, **kwargs):
            calls.append(name)
            return original(*args, **kwargs)

        monkeypatch.setattr(simulation, name, tracked)

    for name in ("generate_array", "build_array_grid", "prepare_velocity_spectrum"):
        watch(name)
    session = SimulationSession()
    frequencies = [600, 700, 800, 900]
    distances = [0, 0.4]
    updates = []
    stream = session.iter_frequency_planes(
        _config(), frequencies, distances, progress=lambda done, total: updates.append((done, total)),
    )
    assert calls == []
    first = next(stream)
    assert updates == [(1, 8)]
    snapshot = first.pressure.copy()
    first_ref = ref(first.pressure)
    second = next(stream)
    np.testing.assert_array_equal(first.pressure, snapshot)
    assert not np.shares_memory(first.pressure, second.pressure)
    del first
    assert first_ref() is None

    remaining = list(stream)
    assert len(remaining) == 6
    assert all(plane.pressure.shape == (7, 9) for plane in remaining)
    assert all(plane.grid is second.grid for plane in remaining)
    assert [(plane.frequency_hz, plane.distance_m) for plane in remaining] == [
        (f, z) for f in frequencies[1:] for z in distances
    ]
    assert updates == [(index, 8) for index in range(1, 9)]
    assert calls == ["generate_array", "build_array_grid", "prepare_velocity_spectrum"]
    assert grid._frequency_axes.cache_info().misses == 1


def test_cancel_skips_remaining_inverse_ffts_and_leaves_the_session_usable(monkeypatch):
    stop = Event()
    calls = []
    ifft2 = angular_spectrum.ifft2

    def count_ifft(*args, **kwargs):
        calls.append(1)
        return ifft2(*args, **kwargs)

    monkeypatch.setattr(angular_spectrum, "ifft2", count_ifft)
    session = SimulationSession()
    config = _config()
    stream = session.iter_frequency_planes(
        config, [800, 1000], [0, 0.4],
        progress=lambda done, total: stop.set(), should_cancel=stop.is_set,
    )
    assert next(stream).pressure.shape == (7, 9)
    with pytest.raises(CancelledError):
        next(stream)
    assert len(calls) == 1

    with pytest.raises(CancelledError):
        session.recalculate_frequencies(config, [800], should_cancel=stop.is_set)
    assert len(calls) == 1
    stop.clear()
    assert session.recalculate_frequencies(config, [800], should_cancel=stop.is_set).pressure.shape == (1, 1, 7, 9)
    assert len(calls) == 2


@pytest.mark.parametrize("frequencies", [[], [0], [-1], [np.inf], [np.nan], [[800]], [1j], True])
def test_invalid_frequency_axes_raise(frequencies):
    with pytest.raises(ValueError, match="frequencies_hz"):
        SimulationSession().recalculate_frequencies(_config(), frequencies)
