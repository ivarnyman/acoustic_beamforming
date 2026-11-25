"""Explicitly idealized array presets for the desktop interface."""

from dataclasses import replace

from config import SimulationConfig
from signal_chain import Crossover, SignalChain
from speakers import SpeakerArray, SpeakerGroup, SpeakerType

PRESETS = ("Generic identical", "Woofer / tweeter", "Three-way")


def preset_configuration(name: str) -> SimulationConfig:
    """Return a standalone configuration with illustrative driver sizes and filters."""
    base = SimulationConfig()
    if name == PRESETS[0]:
        return base
    woofer = SpeakerType.generic(0.18, type_id="woofer")
    tweeter = SpeakerType.generic(0.04, type_id="tweeter")
    if name == PRESETS[1]:
        woofer = replace(woofer, name="Idealized woofer")
        tweeter = replace(tweeter, name="Idealized tweeter")
        types = (woofer, tweeter)
        groups = (
            SpeakerGroup("woofer", 1, 1, (-0.08, 0), signal_chain=SignalChain(
                crossover=Crossover("lowpass", 1800, order=4, family="linkwitz_riley"),
            )),
            SpeakerGroup("tweeter", 1, 1, (0.08, 0), signal_chain=SignalChain(
                crossover=Crossover("highpass", 1800, order=4, family="linkwitz_riley"),
            )),
        )
        frequencies = (400, 800, 1200, 1800, 2800, 4000, 6000)
    elif name == PRESETS[2]:
        types = (
            replace(woofer, name="Idealized woofer"),
            replace(SpeakerType.generic(0.085, type_id="midrange"), name="Idealized midrange"),
            replace(tweeter, name="Idealized tweeter"),
        )
        groups = (
            SpeakerGroup("woofer", 1, 1, (-0.16, 0), signal_chain=SignalChain(
                crossover=Crossover("lowpass", 500),
            )),
            SpeakerGroup("midrange", 1, 1, (0, 0), signal_chain=SignalChain(
                crossover=Crossover("bandpass", (500, 2500)),
            )),
            SpeakerGroup("tweeter", 1, 1, (0.10, 0), signal_chain=SignalChain(
                crossover=Crossover("highpass", 2500),
            )),
        )
        frequencies = (250, 500, 750, 1200, 1800, 2500, 4000, 6000)
    else:
        raise ValueError("choose a listed preset")
    return replace(
        base, speaker_array=SpeakerArray(types, groups=groups), frequencies_hz=frequencies,
        rows=1, columns=len(groups), grid_spacing_m=0.005, observation_distance_m=1,
    )
