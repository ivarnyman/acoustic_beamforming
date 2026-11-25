"""Worker-side calculation and bounded plot data for the interactive interface."""

from concurrent.futures import CancelledError
from dataclasses import dataclass, replace
from math import isqrt
from threading import Event

import numpy as np

from config import SimulationConfig
from grid import XYGrid
from simulation import FrequencyDirectivityResult, FrequencySweepResult, SimulationOutput, SimulationSession


@dataclass(frozen=True)
class SweepPreview:
    directivity: FrequencyDirectivityResult
    pressure: FrequencySweepResult | None = None
    aperture_grid: XYGrid | None = None
    velocity: np.ndarray | None = None
    peak_pressure: float = 0.0


def _sample_grid(grid: XYGrid, limit: int) -> tuple[XYGrid, tuple[slice, slice]]:
    # Retain a uniformly spaced subset including the center cell on each axis.
    ny, nx = grid.shape
    sy = (ny + limit - 1) // limit
    sx = (nx + limit - 1) // limit
    ys, xs = slice((ny // 2) % sy, None, sy), slice((nx // 2) % sx, None, sx)
    sampled = XYGrid(grid.x[xs].copy(), grid.y[ys].copy(), sx * grid.dx, sy * grid.dy)
    return sampled, (ys, xs)


def calculate_preview(
    session: SimulationSession, config: SimulationConfig, cancel: Event,
) -> SimulationOutput | SweepPreview:
    """Calculate every requested frequency once and retain sampled plot data."""
    if cancel.is_set():
        raise CancelledError
    frequencies = config.frequencies_hz or (config.frequency_hz,)
    if config.propagation_mode == "measured":
        return SweepPreview(session.recalculate_measured_directivity(
            config, frequencies, should_cancel=cancel.is_set,
        ))
    if config.frequencies_hz is None:
        return session.recalculate(config)

    # Keep the two complex preview stacks within a shared 64 MiB budget.
    limit = max(1, min(700, isqrt(64 * 1024**2 // (2 * 16 * len(frequencies)))))
    peak_pressure = 0.0
    for index, frequency in enumerate(frequencies):
        if cancel.is_set():
            raise CancelledError
        output = session.recalculate(replace(config, frequency_hz=frequency))
        if cancel.is_set():
            raise CancelledError
        if index == 0:
            aperture, source_slice = _sample_grid(output.grid.source, limit)
            observation, observation_slice = _sample_grid(output.grid.observation, limit)
            pressures = np.empty((len(frequencies), 1, *observation.shape), dtype=complex)
            velocities = np.empty((len(frequencies), *aperture.shape), dtype=complex)
            theta, azimuth = output.theta_deg, output.azimuth_deg
            coefficients = np.empty((len(frequencies), len(theta)), dtype=complex)
        pressures[index, 0] = output.pressure.pressure[0][observation_slice]
        velocities[index] = output.velocity[source_slice]
        coefficients[index] = output.directivity
        peak_pressure = max(peak_pressure, float(output.pressure.magnitude.max()))
        del output

    frequency_axis = np.asarray(frequencies)
    return SweepPreview(
        FrequencyDirectivityResult(frequency_axis, theta, azimuth, coefficients),
        FrequencySweepResult(frequency_axis, np.array([config.observation_distance_m]), observation, pressures),
        aperture, velocities, peak_pressure,
    )
