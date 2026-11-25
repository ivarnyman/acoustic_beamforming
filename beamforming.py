"""Complex speaker weights for the exp(-i omega t) convention."""

from math import isfinite, tau
from numbers import Integral, Real
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.signal import get_window

WeightNormalization = Literal["none", "peak", "power"]


_WINDOWS = {
    "uniform": "boxcar",
    "hann": "hann",
    "hamming": "hamming",
    "blackman": "blackman",
    "bartlett": "bartlett",
    "kaiser": ("kaiser", 6.0),
    "tukey": ("tukey", 0.5),
}
ARRAY_WINDOWS = tuple(_WINDOWS)


def speaker_array_window(rows: int, columns: int, name: str = "uniform") -> NDArray[np.float64]:
    """Return a separable, peak-normalized amplitude window in speaker-index order."""
    if name not in _WINDOWS:
        raise ValueError(f"array_window must be one of {', '.join(ARRAY_WINDOWS)}")
    if any(isinstance(n, bool) or not isinstance(n, Integral) or n < 1 for n in (rows, columns)):
        raise ValueError("window rows and columns must be positive integers")

    def axis_window(count: int) -> NDArray[np.float64]:
        # An axis of one or two elements has no interior to favor over its edges.
        if count < 3:
            return np.ones(count)
        values = np.maximum(get_window(_WINDOWS[name], count, fftbins=False), 0.0)
        return values / values.max()

    return np.outer(axis_window(rows), axis_window(columns))


def steering_weights(
    centers: ArrayLike,
    azimuth_deg: float,
    elevation_deg: float,
    *,
    frequency_hz: float,
    sound_speed_m_s: float = 343.0,
) -> NDArray[np.complex128]:
    """Return unit-amplitude steering weights for the forward hemisphere.

    Azimuth increases from +x toward +y. Elevation is measured above the xy
    plane, so broadside is 90 degrees. The outgoing path contributes
    exp(-i k s_xy dot r); the weight cancels that phase in the target direction.
    """
    points = _speaker_points(centers)
    k = _wavenumber(frequency_hz, sound_speed_m_s)
    if not isfinite(azimuth_deg) or not 0 <= elevation_deg <= 90:
        raise ValueError("azimuth must be finite and elevation must be within [0, 90] degrees")
    azimuth = np.deg2rad(azimuth_deg)
    # sin of the complementary angle is exactly zero at broadside.
    direction_xy = np.sin(np.deg2rad(90 - elevation_deg)) * np.array(
        [np.cos(azimuth), np.sin(azimuth)],
    )
    return np.exp(1j * k * (points @ direction_xy))


def focus_target(azimuth_deg: float, elevation_deg: float, distance_m: float) -> tuple[float, float, float]:
    """Convert a range and steering direction to a Cartesian focus target."""
    if not isfinite(distance_m) or distance_m <= 0:
        raise ValueError("focus distance must be finite and positive")
    if not isfinite(azimuth_deg) or not 0 < elevation_deg <= 90:
        raise ValueError("focus azimuth must be finite and elevation must be within (0, 90] degrees")
    azimuth = np.deg2rad(azimuth_deg)
    polar = np.deg2rad(90 - elevation_deg)
    horizontal = distance_m * np.sin(polar)
    return (
        float(horizontal * np.cos(azimuth)),
        float(horizontal * np.sin(azimuth)),
        float(distance_m * np.cos(polar)),
    )


def focus_weights(
    centers: ArrayLike,
    target_m: ArrayLike,
    *,
    frequency_hz: float,
    sound_speed_m_s: float = 343.0,
) -> NDArray[np.complex128]:
    """Return unit-amplitude weights for a target (x, y, z) with z > 0.

    Each weight cancels exp(+i k R) from the speaker center to the target,
    leaving a common reference phase associated with the array-plane origin.
    """
    points = _speaker_points(centers)
    k = _wavenumber(frequency_hz, sound_speed_m_s)
    target = np.asarray(target_m)
    if (
        target.shape != (3,) or target.dtype.kind not in "iuf"
        or not np.isfinite(target).all() or target[2] <= 0
    ):
        raise ValueError("target_m must contain finite (x, y, z) coordinates with z > 0")
    delta = target[:2] - points
    ranges = np.hypot(np.hypot(delta[:, 0], delta[:, 1]), target[2])
    reference = np.hypot(np.hypot(target[0], target[1]), target[2])
    return np.exp(-1j * k * (ranges - reference))


def normalize_weights(
    weights: ArrayLike,
    *,
    taper: ArrayLike | None = None,
    normalization: WeightNormalization = "none",
    total_squared_weight: float | None = None,
) -> NDArray[np.complex128]:
    """Apply a nonnegative amplitude taper and normalize the complex weights.

    Peak mode sets max(|w|) to one. Power mode sets sum(|w|²) to the specified
    total, defaulting to the number of speakers (the untapered unity baseline).
    """
    result = np.asarray(weights, dtype=np.complex128)
    if result.ndim != 1 or result.size == 0 or not np.isfinite(result).all():
        raise ValueError("weights must be a nonempty vector of finite complex values")
    if normalization not in ("none", "peak", "power"):
        raise ValueError("normalization must be none, peak, or power")
    if taper is not None:
        amplitudes = np.asarray(taper)
        if (
            amplitudes.shape != result.shape or amplitudes.dtype.kind not in "iuf"
            or not np.isfinite(amplitudes).all() or (amplitudes < 0).any()
        ):
            raise ValueError("taper must contain one finite nonnegative amplitude per speaker")
        result = result * amplitudes

    if total_squared_weight is not None and (
        normalization != "power" or not isfinite(total_squared_weight) or total_squared_weight <= 0
    ):
        raise ValueError("total_squared_weight requires power normalization and a finite positive value")
    if normalization == "none":
        return result

    peak = np.max(np.abs(result))
    if peak == 0:
        raise ValueError("cannot normalize all-zero speaker weights")
    result = result / peak
    if normalization == "power":
        total = result.size if total_squared_weight is None else total_squared_weight
        result *= np.sqrt(total) / np.linalg.norm(result)
    return result


def _speaker_points(centers: ArrayLike) -> NDArray[np.float64]:
    points = np.asarray(centers)
    if (
        points.ndim != 2 or points.shape[1] != 2 or not len(points)
        or points.dtype.kind not in "iuf" or not np.isfinite(points).all()
    ):
        raise ValueError("centers must contain a nonempty array of finite (x, y) pairs")
    return points.astype(float)


def _wavenumber(frequency_hz: float, sound_speed_m_s: float) -> float:
    for name, value in (("frequency_hz", frequency_hz), ("sound_speed_m_s", sound_speed_m_s)):
        if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    return tau * frequency_hz / sound_speed_m_s
