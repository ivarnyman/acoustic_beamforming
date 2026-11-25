"""Rasterize prescribed relative normal velocity on the source plane."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from math import isfinite, pi
from numbers import Integral, Real
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from geometry import speaker_radii, validate_speaker_centers
from grid import XYGrid

ProfileKind = Literal["uniform", "cosine", "gaussian"]


def _positive_frequency(frequency_hz: float | None) -> float:
    if (
        isinstance(frequency_hz, bool) or not isinstance(frequency_hz, Real)
        or not isfinite(frequency_hz) or frequency_hz <= 0
    ):
        raise ValueError("frequency_hz must be finite and positive for a frequency-dependent profile")
    return float(frequency_hz)


@dataclass(frozen=True)
class FrequencyParameter:
    """An idealized profile parameter interpolated linearly over log frequency."""

    frequencies_hz: tuple[float, ...]
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        frequencies, values = np.asarray(self.frequencies_hz), np.asarray(self.values)
        if (
            frequencies.ndim != 1 or not frequencies.size or frequencies.dtype.kind not in "iuf"
            or not np.isfinite(frequencies).all() or (frequencies <= 0).any()
            or not (frequencies[1:] > frequencies[:-1]).all()
        ):
            raise ValueError("parameter frequencies must be finite, positive, and strictly increasing")
        if values.shape != frequencies.shape or values.dtype.kind not in "iuf" or not np.isfinite(values).all():
            raise ValueError("provide one finite parameter value per frequency")
        object.__setattr__(self, "frequencies_hz", tuple(map(float, frequencies)))
        object.__setattr__(self, "values", tuple(map(float, values)))

    def evaluate(self, frequency_hz: float | None) -> float:
        """Evaluate at a positive frequency and hold the endpoints outside the band."""
        return float(np.interp(np.log(_positive_frequency(frequency_hz)), np.log(self.frequencies_hz), self.values))


@dataclass(frozen=True)
class RadialProfile:
    """Peak-normalized amplitude parameterized by radius fractions.

    The cosine profile is flat until its outer taper region. The Gaussian
    uses a standard deviation relative to the radius and is clipped at the rim.
    """

    kind: ProfileKind = "uniform"
    taper_width_fraction: float | FrequencyParameter = 0.25
    gaussian_sigma_fraction: float | FrequencyParameter = 0.5
    rim_phase_rad: float | FrequencyParameter = 0.0

    def __post_init__(self) -> None:
        if self.kind not in ("uniform", "cosine", "gaussian"):
            raise ValueError("kind must be uniform, cosine, or gaussian")
        for name, parameter in (
            ("taper_width_fraction", self.taper_width_fraction),
            ("gaussian_sigma_fraction", self.gaussian_sigma_fraction),
            ("rim_phase_rad", self.rim_phase_rad),
        ):
            values = parameter.values if isinstance(parameter, FrequencyParameter) else (parameter,)
            for value in values:
                if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
                    raise ValueError(f"{name} must be finite")
                if name != "rim_phase_rad" and value <= 0:
                    raise ValueError(f"{name} must be positive")
                if name == "taper_width_fraction" and value > 1:
                    raise ValueError("taper_width_fraction must be at most 1")

    @property
    def frequency_dependent(self) -> bool:
        return any(isinstance(value, FrequencyParameter) for value in (
            self.taper_width_fraction, self.gaussian_sigma_fraction, self.rim_phase_rad,
        ))

    def at_frequency(self, frequency_hz: float | None) -> "RadialProfile":
        if not self.frequency_dependent:
            return self
        return replace(self, **{
            name: value.evaluate(frequency_hz)
            for name, value in (
                ("taper_width_fraction", self.taper_width_fraction),
                ("gaussian_sigma_fraction", self.gaussian_sigma_fraction),
                ("rim_phase_rad", self.rim_phase_rad),
            )
            if isinstance(value, FrequencyParameter)
        })

    def evaluate(self, radius_fraction: ArrayLike, frequency_hz: float | None = None) -> NDArray:
        """Evaluate at r/R, with optional quadratic phase from center to rim."""
        profile = self.at_frequency(frequency_hz)
        r = np.asarray(radius_fraction, dtype=float)
        if profile.kind == "cosine":
            width = profile.taper_width_fraction
            t = np.clip((r - (1 - width)) / width, 0.0, 1.0)
            amplitude = 0.5 * (1 + np.cos(pi * t))
        elif profile.kind == "gaussian":
            amplitude = np.exp(-0.5 * (r / profile.gaussian_sigma_fraction) ** 2)
        else:
            amplitude = np.ones_like(r)
        if profile.rim_phase_rad:
            amplitude = amplitude * np.exp(1j * profile.rim_phase_rad * np.clip(r, 0, 1)**2)
        return np.where((r >= 0) & (r <= 1), amplitude, 0.0)


@dataclass(frozen=True)
class CallableProfile:
    """A vectorized complex profile in normalized disk coordinates."""

    function: Callable[..., ArrayLike]
    coordinates: Literal["radial", "xy"] = "radial"
    label: str = "Custom"

    def __post_init__(self) -> None:
        if not callable(self.function):
            raise ValueError("profile function must be callable")
        if self.coordinates not in ("radial", "xy"):
            raise ValueError("profile coordinates must be radial or xy")

    @property
    def frequency_dependent(self) -> bool:
        return True

    def evaluate(
        self, x_fraction: ArrayLike, y_fraction: ArrayLike, frequency_hz: float | None,
    ) -> NDArray[np.complex128]:
        """Call the function at in-disk samples only, with frequency in Hz."""
        frequency = _positive_frequency(frequency_hz)
        x, y = np.broadcast_arrays(x_fraction, y_fraction)
        r = np.hypot(x, y)
        inside = r <= 1
        result = np.zeros(r.shape, dtype=complex)
        if inside.any():
            arguments = (r[inside],) if self.coordinates == "radial" else (x[inside], y[inside])
            values = np.asarray(self.function(*arguments, frequency))
            if (
                values.ndim > 1 or values.size not in (1, arguments[0].size)
                or values.dtype.kind not in "iufc" or not np.isfinite(values).all()
            ):
                raise ValueError("profile callable must return finite values, scalar or one per sample")
            result[inside] = values
        return result


ApertureProfile = RadialProfile | CallableProfile


def rasterize_aperture(
    grid: XYGrid,
    centers: ArrayLike,
    diameter: ArrayLike,
    *,
    profile: ApertureProfile = RadialProfile(),
    weights: ArrayLike | None = None,
    normalization: Literal["peak", "equal_energy"] = "peak",
    subsamples: int = 4,
    profiles: Sequence[ApertureProfile] | None = None,
    frequency_hz: float | None = None,
) -> NDArray[np.complex128]:
    """Return cell-averaged complex normal velocity in (ny, nx) order.

    Each cell uses a subsamples-by-subsamples midpoint rule. Peak normalization
    uses the continuous peak for radial profiles and the quadrature peak for
    callable profiles before averaging. Equal-energy mode
    instead scales each rasterized disk to sum(|v|²) * dx * dy = pi * R² before
    its complex weight is applied.

    `diameter` may be shared or supplied per element. If provided, `profiles`
    supplies a separate radial profile for each element.
    """
    points = validate_speaker_centers(centers, diameter)
    if normalization not in ("peak", "equal_energy"):
        raise ValueError("normalization must be peak or equal_energy")
    if isinstance(subsamples, bool) or not isinstance(subsamples, Integral) or subsamples < 1:
        raise ValueError("subsamples must be a positive integer")
    if weights is None:
        speaker_weights = np.ones(len(points), dtype=complex)
    else:
        speaker_weights = np.asarray(weights, dtype=complex)
        if speaker_weights.shape != (len(points),) or not np.isfinite(speaker_weights).all():
            raise ValueError("weights must contain one finite complex value per speaker")

    if profiles is None:
        profiles = (profile,) * len(points)
    if len(profiles) != len(points) or not all(isinstance(p, (RadialProfile, CallableProfile)) for p in profiles):
        raise ValueError("profiles must contain one aperture profile per speaker")
    radii = speaker_radii(diameter, len(points))
    offsets = (np.arange(subsamples) + 0.5) / subsamples - 0.5
    velocity = np.zeros(grid.shape, dtype=complex)
    for (cx, cy), weight, radius, element_profile in zip(points, speaker_weights, radii, profiles):
        if weight == 0:
            continue
        # Only cells within half a cell of the disk bounding box can contribute.
        x_left = grid.x.searchsorted(cx - radius - grid.dx / 2)
        x_right = grid.x.searchsorted(cx + radius + grid.dx / 2, side="right")
        y_bottom = grid.y.searchsorted(cy - radius - grid.dy / 2)
        y_top = grid.y.searchsorted(cy + radius + grid.dy / 2, side="right")
        region = np.s_[y_bottom:y_top, x_left:x_right]
        x = grid.x[x_left:x_right][None, :] - cx
        y = grid.y[y_bottom:y_top][:, None] - cy
        custom = isinstance(element_profile, CallableProfile)
        if not custom:
            element_profile = element_profile.at_frequency(frequency_hz)
        complex_profile = custom or bool(element_profile.rim_phase_rad)
        amplitude = np.zeros((y.size, x.size), dtype=complex if complex_profile else float)
        peak = 0.0
        for oy in offsets * grid.dy:
            for ox in offsets * grid.dx:
                if custom:
                    samples = element_profile.evaluate((x + ox) / radius, (y + oy) / radius, frequency_hz)
                    if normalization == "peak":
                        peak = max(peak, float(np.abs(samples).max(initial=0)))
                else:
                    samples = element_profile.evaluate(np.hypot(x + ox, y + oy) / radius)
                amplitude += samples
        amplitude /= subsamples**2
        if peak > 0:
            amplitude /= peak

        if normalization == "equal_energy":
            squared = np.abs(amplitude)**2 if complex_profile else amplitude**2
            energy = np.sum(squared) * grid.dx * grid.dy
            if energy == 0:
                raise ValueError("a speaker has no sampled velocity; refine the grid or increase subsamples")
            amplitude *= np.sqrt(pi * radius**2 / energy)
        velocity[region] += weight * amplitude
    return velocity
