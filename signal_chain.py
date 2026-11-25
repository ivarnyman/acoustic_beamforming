"""Optional drive processing before the speaker's diaphragm response."""

from dataclasses import dataclass
from functools import cached_property
from math import isfinite, tau
from numbers import Integral, Real
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.signal import butter, freqs_zpk

FilterKind = Literal["lowpass", "highpass", "bandpass"]
FilterFamily = Literal["butterworth", "linkwitz_riley"]


def _frequencies(frequency_hz: ArrayLike) -> NDArray[np.float64]:
    frequencies = np.asarray(frequency_hz)
    if (
        frequencies.dtype.kind not in "iuf" or not np.isfinite(frequencies).all()
        or (frequencies < 0).any()
    ):
        raise ValueError("frequency_hz must be finite and nonnegative")
    return frequencies.astype(float)


@dataclass(frozen=True)
class Crossover:
    """Analog Butterworth or Linkwitz–Riley transfer in the exp(-i omega t) convention.

    Butterworth band-pass order is the prototype order, giving twice as many
    poles in the realized filter. Linkwitz–Riley order is the total order of
    two cascaded Butterworth sections.
    """

    kind: FilterKind
    cutoff_hz: float | tuple[float, float]
    order: int = 2
    family: FilterFamily = "butterworth"

    def __post_init__(self) -> None:
        if self.kind not in ("lowpass", "highpass", "bandpass"):
            raise ValueError("crossover kind must be lowpass, highpass, or bandpass")
        if self.family not in ("butterworth", "linkwitz_riley"):
            raise ValueError("crossover family must be butterworth or linkwitz_riley")
        if isinstance(self.order, bool) or not isinstance(self.order, Integral) or self.order < 1:
            raise ValueError("crossover order must be a positive integer")
        if self.family == "linkwitz_riley" and (self.order % 2 or self.kind == "bandpass"):
            raise ValueError("Linkwitz–Riley requires an even total order and lowpass or highpass kind")

        cutoff = np.asarray(self.cutoff_hz)
        shape = (2,) if self.kind == "bandpass" else ()
        if (
            cutoff.shape != shape or cutoff.dtype.kind not in "iuf"
            or not np.isfinite(cutoff).all() or (cutoff <= 0).any()
        ):
            raise ValueError("cutoff_hz must be positive, with two frequencies for a bandpass")
        if self.kind == "bandpass":
            low, high = map(float, cutoff)
            if low >= high:
                raise ValueError("bandpass cutoff frequencies must be increasing")
            object.__setattr__(self, "cutoff_hz", (low, high))
        else:
            object.__setattr__(self, "cutoff_hz", float(cutoff))
        object.__setattr__(self, "order", int(self.order))

    @cached_property
    def _section(self):
        cutoff = np.asarray(self.cutoff_hz)
        reference_hz = float(cutoff.flat[0])
        # Normalize the analog frequencies to the first edge for numerical conditioning.
        order = self.order // 2 if self.family == "linkwitz_riley" else self.order
        zpk = butter(order, cutoff / reference_hz, btype=self.kind, analog=True, output="zpk")
        return reference_hz, zpk

    def evaluate(self, frequency_hz: ArrayLike) -> complex | NDArray[np.complex128]:
        frequencies = _frequencies(frequency_hz)
        reference_hz, zpk = self._section
        # SciPy evaluates at s = +i*omega. Negative omega gives exp(-i*omega*t).
        _, response = freqs_zpk(*zpk, worN=-frequencies.ravel() / reference_hz)
        if self.family == "linkwitz_riley":
            response = response**2
        return complex(response.item()) if frequencies.ndim == 0 else response.reshape(frequencies.shape)


@dataclass(frozen=True)
class SignalChain:
    """Optional linear gain, polarity, phase offset, delay, and crossover."""

    gain: float = 1.0
    polarity: Literal[1, -1] = 1
    phase_offset_rad: float = 0.0
    delay_s: float = 0.0
    crossover: Crossover | None = None

    def __post_init__(self) -> None:
        for name in ("gain", "phase_offset_rad", "delay_s"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
                raise ValueError(f"{name} must be finite and real")
            if name != "phase_offset_rad" and value < 0:
                raise ValueError(f"{name} must be nonnegative")
            object.__setattr__(self, name, float(value))
        if isinstance(self.polarity, bool) or not isinstance(self.polarity, Integral) or self.polarity not in (-1, 1):
            raise ValueError("polarity must be +1 or -1")
        object.__setattr__(self, "polarity", int(self.polarity))
        if self.crossover is not None and not isinstance(self.crossover, Crossover):
            raise ValueError("crossover must be a Crossover or None")

    def gain_and_delay(self, frequency_hz: ArrayLike) -> complex | NDArray[np.complex128]:
        frequencies = _frequencies(frequency_hz)
        phase = self.phase_offset_rad + tau * self.delay_s * frequencies
        response = self.gain * self.polarity * np.exp(1j * phase)
        return complex(response.item()) if frequencies.ndim == 0 else response

    def evaluate(self, frequency_hz: ArrayLike) -> complex | NDArray[np.complex128]:
        response = self.gain_and_delay(frequency_hz)
        if self.crossover is not None:
            response = response * self.crossover.evaluate(frequency_hz)
        return response

    @classmethod
    def from_dict(cls, data: dict) -> "SignalChain":
        if not isinstance(data, dict):
            raise ValueError("signal_chain must be a JSON object")
        fields = dict(data)
        if fields.get("crossover") is not None:
            if not isinstance(fields["crossover"], dict):
                raise ValueError("crossover must be a JSON object")
            fields["crossover"] = Crossover(**fields["crossover"])
        return cls(**fields)
