"""Angular spectrum pressure for a prescribed-velocity baffled source."""

from collections.abc import Iterator
from dataclasses import dataclass
from math import isfinite, tau
from numbers import Real

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.fft import fft2, ifft2

from grid import SimulationGrid, XYGrid


@dataclass(frozen=True)
class PressureResult:
    """Pressure fields indexed as (distance, y, x), including scalar requests."""

    distances_m: NDArray[np.float64]
    grid: XYGrid
    pressure: NDArray[np.complex128]
    magnitude: NDArray[np.float64]
    phase_rad: NDArray[np.float64]
    relative_level_db: NDArray[np.float64]


@dataclass(frozen=True)
class VelocitySpectrum:
    """Native-order FFT and the grids needed to reuse it across calculations."""

    coefficients: NDArray[np.complex128]
    computational: XYGrid
    observation: XYGrid
    observation_slice: tuple[slice, slice]


def prepare_velocity_spectrum(velocity: ArrayLike, grid: XYGrid | SimulationGrid) -> VelocitySpectrum:
    """Validate, pad, and transform the velocity once."""
    field, computational = _computational_field(velocity, grid)
    if isinstance(grid, SimulationGrid):
        observation = grid.observation
        observation_slice = grid.observation_slice
    else:
        observation = computational
        observation_slice = (slice(None), slice(None))
    # Padded fields are private buffers. Preserve a caller's supplied XYGrid field.
    coefficients = fft2(field, overwrite_x=isinstance(grid, SimulationGrid))
    return VelocitySpectrum(coefficients, computational, observation, observation_slice)


def propagate_pressure(
    velocity: ArrayLike,
    grid: XYGrid | SimulationGrid,
    distances_m: ArrayLike,
    *,
    frequency_hz: float,
    sound_speed_m_s: float = 343.0,
    air_density_kg_m3: float = 1.21,
    include_evanescent: bool = True,
) -> PressureResult:
    """Propagate sampled normal velocity into z >= 0 using exp(-i omega t).

    With a SimulationGrid, embed the source in its computational grid and
    return its observation crop. With an XYGrid, return the full periodic plane.
    Levels use one peak reference across all returned planes. Zero pressure
    has level -inf and phase NaN.

    Omit frequency bins within 32 float64 eps of k in relative transverse
    wavenumber. These bins represent exact grazing or its roundoff neighborhood,
    where the sampled transfer function would divide by zero or amplify noise.
    """
    return pressure_from_spectrum(
        prepare_velocity_spectrum(velocity, grid),
        distances_m,
        frequency_hz=frequency_hz,
        sound_speed_m_s=sound_speed_m_s,
        air_density_kg_m3=air_density_kg_m3,
        include_evanescent=include_evanescent,
    )


def pressure_from_spectrum(
    prepared: VelocitySpectrum,
    distances_m: ArrayLike,
    *,
    frequency_hz: float,
    sound_speed_m_s: float = 343.0,
    air_density_kg_m3: float = 1.21,
    include_evanescent: bool = True,
) -> PressureResult:
    """Propagate an already transformed source using the usual grazing policy."""
    distances = validate_observation_distances(distances_m)
    pressure = np.empty((len(distances), *prepared.observation.shape), dtype=np.complex128)
    for index, plane in enumerate(iter_pressure_planes(
        prepared, distances, frequency_hz=frequency_hz, sound_speed_m_s=sound_speed_m_s,
        air_density_kg_m3=air_density_kg_m3, include_evanescent=include_evanescent,
    )):
        pressure[index] = plane

    magnitude = np.abs(pressure)
    nonzero = magnitude > 0
    phase = np.full(magnitude.shape, np.nan)
    np.arctan2(pressure.imag, pressure.real, out=phase, where=nonzero)
    levels = np.full(magnitude.shape, -np.inf)
    peak = magnitude.max()
    if peak > 0:
        np.log10(magnitude, out=levels, where=nonzero)
        levels -= np.log10(peak)
        levels *= 20
    return PressureResult(distances, prepared.observation, pressure, magnitude, phase, levels)


def validate_observation_distances(distances_m: ArrayLike) -> NDArray[np.float64]:
    """Return one observation-distance axis in meters."""
    distances = np.atleast_1d(np.asarray(distances_m))
    if (
        distances.ndim != 1 or distances.size == 0 or distances.dtype.kind not in "iuf"
        or not np.isfinite(distances).all() or (distances < 0).any()
    ):
        raise ValueError("distances_m must be a nonempty scalar or vector of finite nonnegative distances")
    return distances.astype(float)



def iter_pressure_planes(
    prepared: VelocitySpectrum,
    distances_m: ArrayLike,
    *,
    frequency_hz: float,
    sound_speed_m_s: float = 343.0,
    air_density_kg_m3: float = 1.21,
    include_evanescent: bool = True,
) -> Iterator[NDArray[np.complex128]]:
    """Yield one observation plane at a time. Copy any plane retained after advancing."""
    _validate_medium(frequency_hz, sound_speed_m_s, air_density_kg_m3)
    distances = validate_observation_distances(distances_m)
    computational = prepared.computational

    omega = tau * frequency_hz
    k = omega / sound_speed_m_s
    fx, fy = computational.frequency_axes()
    transverse_ratio = np.hypot((tau / k) * fx[None, :], (tau / k) * fy[:, None])
    grazing = np.abs(transverse_ratio - 1) <= 32 * np.finfo(float).eps
    active = ~grazing
    if not include_evanescent:
        active &= transverse_ratio < 1

    ratio = transverse_ratio[active]
    # Complex sqrt gives +real for propagating modes and +imaginary for evanescent modes.
    kz = ((1 - ratio) * (1 + ratio)).astype(complex)
    del transverse_ratio, grazing, ratio
    np.sqrt(kz, out=kz)
    kz *= k
    base = prepared.coefficients[active]
    base /= kz
    base *= air_density_kg_m3 * omega

    spectrum = np.empty(computational.shape, dtype=np.complex128)
    excluded = not active.all()
    transfer = np.empty_like(kz) if excluded else spectrum.ravel()
    for distance in distances:
        np.multiply(kz, 1j * distance, out=transfer)
        np.exp(transfer, out=transfer)
        transfer *= base
        if excluded:
            spectrum.fill(0)
            spectrum[active] = transfer
        yield ifft2(spectrum, overwrite_x=True)[prepared.observation_slice]



def _computational_field(
    velocity: ArrayLike, grid: XYGrid | SimulationGrid,
) -> tuple[NDArray[np.complex128], XYGrid]:
    source = grid.source if isinstance(grid, SimulationGrid) else grid
    field = np.asarray(velocity, dtype=np.complex128)
    if field.shape != source.shape or not np.isfinite(field).all():
        raise ValueError("velocity must be finite and match the source grid shape")
    if isinstance(grid, SimulationGrid):
        padded = np.zeros(grid.computational.shape, dtype=np.complex128)
        padded[grid.source_slice] = field
        return padded, grid.computational
    return field, grid


def far_field_directivity(
    velocity: ArrayLike,
    grid: XYGrid | SimulationGrid,
    theta_deg: ArrayLike,
    phi_deg: ArrayLike = 0.0,
    *,
    frequency_hz: float,
    sound_speed_m_s: float = 343.0,
    air_density_kg_m3: float = 1.21,
) -> NDArray[np.complex128]:
    """Return D for asymptotic pressure p(R, theta, phi) = D * exp(i*k*R) / R.

    Angles broadcast together. Theta is measured from +z, with negative theta
    sampling the opposite side of the same azimuthal cut. Phi runs from +x
    toward +y. At |theta| = 90 degrees the forward grazing limit is used.
    """
    return directivity_from_spectrum(
        prepare_velocity_spectrum(velocity, grid),
        theta_deg,
        phi_deg,
        frequency_hz=frequency_hz,
        sound_speed_m_s=sound_speed_m_s,
        air_density_kg_m3=air_density_kg_m3,
    )


def directivity_from_spectrum(
    prepared: VelocitySpectrum,
    theta_deg: ArrayLike,
    phi_deg: ArrayLike = 0.0,
    *,
    frequency_hz: float,
    sound_speed_m_s: float = 343.0,
    air_density_kg_m3: float = 1.21,
) -> NDArray[np.complex128]:
    """Evaluate far-field directions from an already transformed source."""
    _validate_medium(frequency_hz, sound_speed_m_s, air_density_kg_m3)
    theta, phi = np.broadcast_arrays(
        np.asarray(theta_deg, dtype=float), np.asarray(phi_deg, dtype=float),
    )
    if not np.isfinite(theta).all() or not np.isfinite(phi).all() or (np.abs(theta) > 90).any():
        raise ValueError("angles must be finite and theta_deg must be within [-90, 90]")

    computational = prepared.computational
    wavelength = sound_speed_m_s / frequency_hz
    transverse = np.sin(np.deg2rad(theta)) / wavelength
    sample_fx = transverse * np.cos(np.deg2rad(phi))
    sample_fy = transverse * np.sin(np.deg2rad(phi))
    nyquist_x = 0.5 / computational.dx
    nyquist_y = 0.5 / computational.dy
    if (np.abs(sample_fx) > nyquist_x).any() or (np.abs(sample_fy) > nyquist_y).any():
        raise ValueError("requested directions exceed the spatial Nyquist limits; refine the grid")

    # Bilinearly sample four native FFT bins, referencing their phases to
    # the central grid sample before interpolation. Modulo indexing wraps
    # both signs of Nyquist without constructing a shifted spectrum.
    ny, nx = computational.shape
    x_bin = sample_fx * (nx * computational.dx)
    y_bin = sample_fy * (ny * computational.dy)
    ix = np.floor(x_bin).astype(np.intp)
    iy = np.floor(y_bin).astype(np.intp)
    tx, ty = x_bin - ix, y_bin - iy
    samples = np.zeros(theta.shape, dtype=np.complex128)
    for offset_y, wy in ((0, 1 - ty), (1, ty)):
        for offset_x, wx in ((0, 1 - tx), (1, tx)):
            i, j = iy + offset_y, ix + offset_x
            center_phase = np.exp(1j * tau * (j * (nx // 2 / nx) + i * (ny // 2 / ny)))
            samples += wx * wy * center_phase * prepared.coefficients[i % ny, j % nx]
    x_ref, y_ref = computational.x[nx // 2], computational.y[ny // 2]
    origin_phase = np.exp(-1j * tau * (sample_fx * x_ref + sample_fy * y_ref))
    velocity_transform = samples * origin_phase * computational.dx * computational.dy

    # The stationary-phase factor kz / (2*pi*i*R) cancels the ASM 1/kz.
    # Equivalently, this is the far-field limit of the baffled Rayleigh integral.
    return -1j * air_density_kg_m3 * frequency_hz * velocity_transform


def _validate_medium(
    frequency_hz: float, sound_speed_m_s: float, air_density_kg_m3: float,
) -> None:
    for name, value in (
        ("frequency_hz", frequency_hz),
        ("sound_speed_m_s", sound_speed_m_s),
        ("air_density_kg_m3", air_density_kg_m3),
    ):
        if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")

