"""Frequency-resolved pressure analysis in the simulator's relative units."""

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from simulation import FrequencySweepResult

SpectralWeighting = Literal["discrete", "tone_peak_squared", "one_sided_psd"]


def frequency_bin_order(frequencies_hz: ArrayLike) -> NDArray[np.intp]:
    """Validate unique positive frequency bins and return an ascending index."""
    frequencies = np.asarray(frequencies_hz)
    if (
        frequencies.ndim != 1 or not frequencies.size or frequencies.dtype.kind not in "iuf"
        or not np.isfinite(frequencies).all() or (frequencies <= 0).any()
    ):
        raise ValueError("frequencies_hz must be a nonempty vector of finite positive frequencies")
    order = np.argsort(frequencies)
    ordered = frequencies[order]
    if (ordered[1:] == ordered[:-1]).any():
        raise ValueError("frequency analysis requires unique frequency bins")
    return order


def broadband_mean_square_pressure(
    pressure: ArrayLike,
    frequencies_hz: ArrayLike,
    spectral_weights: ArrayLike,
    *,
    weighting: SpectralWeighting = "discrete",
    bin_widths_hz: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """Sum incoherent pressure contributions along the first (frequency) axis.

    Discrete weights are mean-square normalized drive per bin. Peak-squared
    tone weights are halved. One-sided PSD weights are integrated in Hz using
    explicit bin widths or trapezoid quadrature over the sampled band.
    """
    frequencies = np.asarray(frequencies_hz)
    order = frequency_bin_order(frequencies)
    fields = np.asarray(pressure)
    if fields.ndim == 0 or fields.shape[0] != len(frequencies) or fields.dtype.kind not in "iufc":
        raise ValueError("pressure must have one real or complex field per frequency on its first axis")
    weights = np.asarray(spectral_weights)
    if (
        weights.shape != frequencies.shape or weights.dtype.kind not in "iuf"
        or not np.isfinite(weights).all() or (weights < 0).any()
    ):
        raise ValueError("spectral_weights must contain one finite nonnegative weight per frequency")
    weights = weights.astype(float)
    if weighting == "one_sided_psd":
        if bin_widths_hz is None:
            if len(frequencies) < 2:
                raise ValueError("one PSD sample requires an explicit bin width in Hz")
            steps = np.diff(frequencies[order].astype(float))
            widths = np.empty_like(weights)
            widths[order] = (np.r_[steps, 0] + np.r_[0, steps]) / 2
        else:
            widths = np.asarray(bin_widths_hz)
            if (
                widths.shape != weights.shape or widths.dtype.kind not in "iuf"
                or not np.isfinite(widths).all() or (widths <= 0).any()
            ):
                raise ValueError("bin_widths_hz must contain one finite positive width per PSD bin")
        weights *= widths
    elif weighting in ("discrete", "tone_peak_squared"):
        if bin_widths_hz is not None:
            raise ValueError("bin_widths_hz apply only to one_sided_psd weights")
        if weighting == "tone_peak_squared":
            weights *= 0.5
    else:
        raise ValueError("weighting must be discrete, tone_peak_squared, or one_sided_psd")

    result = np.zeros(fields.shape[1:], dtype=float)
    scratch = np.empty_like(result)
    for field, weight in zip(fields, weights):
        if not np.isfinite(field).all():
            raise ValueError("pressure fields must be finite")
        if weight:
            np.abs(field, out=scratch)
            np.square(scratch, out=scratch)
            scratch *= weight
            result += scratch
    return result


def point_pressure_responses(
    sweep: FrequencySweepResult, points_m: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.complex128]]:
    """Return sorted frequencies, actual sampled xyz points, and pressure [f, point]."""
    points = np.atleast_2d(np.asarray(points_m))
    if (
        points.ndim != 2 or points.shape[1] != 3 or not len(points)
        or points.dtype.kind not in "iuf" or not np.isfinite(points).all()
    ):
        raise ValueError("points_m must contain finite (x, y, z) observation coordinates")
    x, y, z = sweep.grid.x, sweep.grid.y, sweep.distances_m
    xmin, xmax, ymin, ymax = sweep.grid.extent
    if (
        (points < (xmin, ymin, z.min())).any()
        or (points > (xmax, ymax, z.max())).any()
    ):
        raise ValueError("observation points must lie within the sampled observation domain")
    ix, iy, iz = (np.abs(axis - coordinate[:, None]).argmin(axis=1) for axis, coordinate in zip((x, y, z), points.T))
    locations = np.column_stack((x[ix], y[iy], z[iz]))
    order = frequency_bin_order(sweep.frequencies_hz)
    return sweep.frequencies_hz[order], locations, sweep.pressure[:, iz, iy, ix][order]
