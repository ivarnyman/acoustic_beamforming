"""Complex response from normalized input drive to diaphragm velocity scale."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

OutOfRange = Literal["zero", "hold", "error"]


def _positive_frequencies(values: ArrayLike) -> NDArray[np.float64]:
    frequencies = np.asarray(values)
    if (
        frequencies.dtype.kind not in "iuf" or not np.isfinite(frequencies).all()
        or (frequencies <= 0).any()
    ):
        raise ValueError("frequencies must be finite and positive in Hz")
    return frequencies.astype(float)


@dataclass(frozen=True)
class FrequencyResponse:
    """Tabulated H(f) for the exp(-i omega t) convention.

    Interpolate log magnitude and unwrapped phase along log frequency.
    The default policy raises outside the sampled band.
    """

    frequencies_hz: ArrayLike
    values: ArrayLike
    out_of_range: OutOfRange = "error"
    unwrapped_phase_rad: ArrayLike | None = None

    def __post_init__(self) -> None:
        if self.out_of_range not in ("zero", "hold", "error"):
            raise ValueError("out_of_range must be zero, hold, or error")
        frequencies = _positive_frequencies(self.frequencies_hz)
        if frequencies.ndim != 1 or not frequencies.size or not (np.diff(frequencies) > 0).all():
            raise ValueError("frequency samples must be a nonempty, strictly increasing vector")

        values = np.asarray(self.values)
        if values.shape != frequencies.shape or values.dtype.kind not in "iufc":
            raise ValueError("provide one complex response value per frequency sample")
        values = values.astype(complex)
        magnitude = np.abs(values)
        if not np.isfinite(magnitude).all() or (magnitude <= 0).any():
            raise ValueError("response magnitudes must be finite and nonzero for log interpolation")

        if self.unwrapped_phase_rad is None:
            phase = np.unwrap(np.angle(values))
        else:
            phase = np.asarray(self.unwrapped_phase_rad)
            if (
                phase.shape != values.shape or phase.dtype.kind not in "iuf"
                or not np.isfinite(phase).all()
            ):
                raise ValueError("provide one finite unwrapped phase in radians per frequency sample")
            if not np.allclose(np.exp(1j * phase), np.exp(1j * np.angle(values)), rtol=0, atol=1e-10):
                raise ValueError("unwrapped phase must agree with the complex sample values")

        object.__setattr__(self, "frequencies_hz", tuple(map(float, frequencies)))
        object.__setattr__(self, "values", tuple(map(complex, values)))
        object.__setattr__(self, "unwrapped_phase_rad", tuple(map(float, phase)))

    @classmethod
    def from_magnitude_phase(
        cls,
        frequencies_hz: ArrayLike,
        magnitude: ArrayLike,
        unwrapped_phase_rad: ArrayLike,
        *,
        out_of_range: OutOfRange = "error",
    ) -> "FrequencyResponse":
        """Construct from positive linear magnitude and explicitly unwrapped radians."""
        amplitude = np.asarray(magnitude)
        phase = np.asarray(unwrapped_phase_rad)
        if (
            amplitude.ndim != 1 or amplitude.shape != phase.shape
            or amplitude.dtype.kind not in "iuf" or phase.dtype.kind not in "iuf"
            or not np.isfinite(amplitude).all() or not np.isfinite(phase).all()
            or (amplitude <= 0).any()
        ):
            raise ValueError("magnitude and phase must be matching vectors with positive magnitude and finite phase")
        return cls(frequencies_hz, amplitude * np.exp(1j * phase), out_of_range, phase)

    @classmethod
    def flat(cls) -> "FrequencyResponse":
        """Return the explicitly idealized unity response at all positive frequencies."""
        return cls((1.0,), (1 + 0j,), out_of_range="hold")

    @classmethod
    def synthetic_lowpass(
        cls, frequencies_hz: ArrayLike, corner_hz: float, *, out_of_range: OutOfRange = "error",
    ) -> "FrequencyResponse":
        """Sample an idealized first-order low-pass response for demonstrations."""
        frequencies = _positive_frequencies(frequencies_hz)
        corner = _positive_frequencies(corner_hz)
        if corner.ndim != 0:
            raise ValueError("corner_hz must be a scalar frequency")
        return cls(frequencies, 1 / (1 - 1j * frequencies / corner), out_of_range=out_of_range)

    def evaluate(self, frequency_hz: ArrayLike) -> complex | NDArray[np.complex128]:
        """Evaluate a positive frequency or array of frequencies in Hz."""
        query = _positive_frequencies(frequency_hz)
        samples = np.asarray(self.frequencies_hz)
        values = np.asarray(self.values)
        low, high = samples[0], samples[-1]
        outside = (query < low) | (query > high)
        if self.out_of_range == "error" and outside.any():
            raise ValueError(f"frequency outside the response band [{low:g}, {high:g}] Hz")

        clipped = np.clip(query, low, high)
        log_query, log_samples = np.log(clipped), np.log(samples)
        log_magnitude = np.interp(log_query, log_samples, np.log(np.abs(values)))
        phase = np.interp(log_query, log_samples, self.unwrapped_phase_rad)
        result = np.exp(log_magnitude + 1j * phase)
        indices = np.searchsorted(samples, clipped)
        # Return the stored complex samples exactly, including held band edges.
        result = np.where(clipped == samples[indices], values[indices], result)
        if self.out_of_range == "zero":
            result = np.where(outside, 0j, result)
        return complex(result.item()) if query.ndim == 0 else result

    def to_dict(self) -> dict:
        return {
            "frequencies_hz": list(self.frequencies_hz),
            "values": [[value.real, value.imag] for value in self.values],
            "out_of_range": self.out_of_range,
            "unwrapped_phase_rad": list(self.unwrapped_phase_rad),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "FrequencyResponse":
        if (
            not isinstance(data, dict) or not {"frequencies_hz", "values"} <= data.keys()
            or data.keys() - {"frequencies_hz", "values", "out_of_range", "unwrapped_phase_rad"}
        ):
            raise ValueError("frequency_response must contain frequencies_hz and values")
        pairs = np.asarray(data["values"])
        if pairs.ndim != 2 or pairs.shape[1] != 2 or pairs.dtype.kind not in "iuf" or not np.isfinite(pairs).all():
            raise ValueError("frequency response values must be finite [real, imaginary] pairs")
        fields = dict(data)
        fields["values"] = [complex(real, imag) for real, imag in pairs]
        return cls(**fields)
