"""Configuration adapters shared by the batch and interactive entry points."""

from collections.abc import Callable, Iterator
from concurrent.futures import CancelledError
from contextlib import closing
from dataclasses import dataclass, replace
from math import isclose, tau

import numpy as np
from numpy.typing import ArrayLike, NDArray

from angular_spectrum import (
    PressureResult,
    VelocitySpectrum,
    directivity_from_spectrum,
    iter_pressure_planes,
    prepare_velocity_spectrum,
    pressure_from_spectrum,
    validate_observation_distances,
)

from aperture import ApertureProfile, RadialProfile, rasterize_aperture
from beamforming import focus_target, focus_weights, normalize_weights, speaker_array_window, steering_weights
from config import SimulationConfig
from geometry import generate_speaker_centers, validate_speaker_centers
from grid import SimulationGrid, XYGrid, build_grid, grid_dimensions, resolve_observation_shape, speaker_bounds
from measured_data import MeasuredElementPattern, direction_vectors


def array_diameters(config: SimulationConfig) -> float | tuple[float, ...]:
    if config.speaker_array is None:
        return config.speaker_diameter_m
    return tuple(speaker_type.diameter_m for speaker_type in config.speaker_array.element_types())


def array_type_ids(config: SimulationConfig) -> tuple[str, ...] | None:
    if config.speaker_array is None:
        return None
    return tuple(element.type_id for element in config.speaker_array.flattened_elements)


def generate_array(config: SimulationConfig) -> np.ndarray:
    if config.speaker_array is not None:
        return validate_speaker_centers(
            [element.position_m for element in config.speaker_array.flattened_elements], array_diameters(config),
        )
    return generate_speaker_centers(
        config.rows, config.columns, config.speaker_diameter_m,
        config.horizontal_gap_m, config.vertical_gap_m,
        config.row_stagger * (config.speaker_diameter_m + config.horizontal_gap_m),
    )


def _grid_spacing(config: SimulationConfig) -> tuple[float, float]:
    dx = config.grid_spacing_m
    return dx, dx if config.grid_spacing_y_m is None else config.grid_spacing_y_m


def _array_grid_dimensions(config: SimulationConfig, centers: np.ndarray, dx: float, dy: float):
    return grid_dimensions(
        speaker_bounds(centers, array_diameters(config)), dx=dx, dy=dy,
        margin=config.grid_margin_m, padding_factor=config.padding_factor, fast_fft=config.fast_fft,
    )


def computational_dimensions_m(
    config: SimulationConfig, *, centers: np.ndarray | None = None,
) -> tuple[float, float]:
    """Return the full padded (x, y) widths in meters without allocating a grid."""
    dx, dy = _grid_spacing(config)
    if centers is None:
        centers = generate_array(config)
    _, (ny, nx) = _array_grid_dimensions(config, centers, dx, dy)
    return float(nx * dx), float(ny * dy)


def _observation_widths(
    config: SimulationConfig, centers: np.ndarray,
) -> tuple[float | None, float | None]:
    xmin, xmax, ymin, ymax = speaker_bounds(centers, array_diameters(config))
    return config.resolve_observation_widths_m(xmax - xmin, ymax - ymin)


def build_array_grid(config: SimulationConfig, centers: np.ndarray) -> SimulationGrid:
    dx, dy = _grid_spacing(config)
    observation_x, observation_y = _observation_widths(config, centers)
    return build_grid(
        centers,
        array_diameters(config),
        dx=dx,
        dy=dy,
        margin=config.grid_margin_m,
        padding_factor=config.padding_factor,
        fast_fft=config.fast_fft,
        observation_shape=config.observation_shape,
        observation_width_x_m=observation_x,
        observation_width_y_m=observation_y,
    )


def create_profile(config: SimulationConfig) -> RadialProfile:
    return RadialProfile(
        kind=config.amplitude_profile,
        taper_width_fraction=config.taper_width_fraction,
        gaussian_sigma_fraction=config.gaussian_sigma_fraction,
    )


def array_profiles(
    config: SimulationConfig, default: RadialProfile,
) -> tuple[ApertureProfile, ...] | None:
    if config.speaker_array is None:
        return None
    return tuple(item.aperture_profile or default for item in config.speaker_array.element_types())


def _processed_drive_weights(config: SimulationConfig, centers: np.ndarray) -> tuple[np.ndarray, float]:
    """Apply element signal chains to the normalized drive."""
    weights, azimuth = configured_weights(config, centers)
    weights = config.relative_amplitude * weights
    if config.speaker_array is not None:
        chain_gains = {}
        for index, element in enumerate(config.speaker_array.flattened_elements):
            chain = element.effective_signal_chain
            if chain is not None:
                if chain not in chain_gains:
                    chain_gains[chain] = chain.evaluate(config.frequency_hz)
                weights[index] *= chain_gains[chain]
    return weights, azimuth


def _velocity_weights(config: SimulationConfig, centers: np.ndarray) -> tuple[np.ndarray, float]:
    """Convert processed drive to diaphragm velocity using each type's response."""
    weights, azimuth = _processed_drive_weights(config, centers)
    if config.speaker_array is not None:
        gains = {}
        for index, speaker_type in enumerate(config.speaker_array.element_types()):
            response = speaker_type.frequency_response
            if response is not None:
                type_id = speaker_type.type_id
                if type_id not in gains:
                    try:
                        gains[type_id] = response.evaluate(config.frequency_hz)
                    except ValueError as exc:
                        raise ValueError(f"Speaker type {type_id!r}: {exc}") from exc
                weights[index] *= gains[type_id]
    return weights, azimuth


def create_velocity(
    config: SimulationConfig, centers: np.ndarray, grid: SimulationGrid,
) -> tuple[np.ndarray, RadialProfile, float]:
    profile = create_profile(config)
    weights, azimuth = _velocity_weights(config, centers)
    velocity = sample_velocity(config, centers, grid, profile, weights)
    return velocity, profile, azimuth


def sample_velocity(
    config: SimulationConfig, centers: np.ndarray, grid: SimulationGrid,
    profile: RadialProfile, weights: np.ndarray,
) -> np.ndarray:
    return rasterize_aperture(
        grid.source, centers, array_diameters(config), profile=profile, weights=weights,
        normalization=config.aperture_normalization, subsamples=config.aperture_subsamples,
        profiles=array_profiles(config, profile), frequency_hz=config.frequency_hz,
    )


def configured_focus_point(config: SimulationConfig) -> tuple[float, float, float]:
    if config.focus_distance_m is None:
        return config.focus_point_m
    return focus_target(
        config.steering_azimuth_deg, config.steering_elevation_deg, config.focus_distance_m,
    )


def configured_array_window(config: SimulationConfig, count: int) -> np.ndarray:
    if config.speaker_array is not None and config.rows * config.columns != count:
        if config.array_window != "uniform":
            raise ValueError("a nonuniform window needs rows × columns to match the explicit element count")
        return np.ones((1, count))
    return speaker_array_window(config.rows, config.columns, config.array_window)


def configured_weights(config: SimulationConfig, centers: np.ndarray) -> tuple[np.ndarray, float]:
    medium = dict(frequency_hz=config.frequency_hz, sound_speed_m_s=config.sound_speed_m_s)
    azimuth = config.steering_azimuth_deg
    if config.beamforming_mode == "steer":
        phases = steering_weights(centers, azimuth, config.steering_elevation_deg, **medium)
    elif config.beamforming_mode == "focus":
        target = configured_focus_point(config)
        phases = focus_weights(centers, target, **medium)
        if config.focus_distance_m is None and (target[0] or target[1]):
            azimuth = float(np.rad2deg(np.arctan2(target[1], target[0])))
    elif config.beamforming_mode == "none":
        phases = np.ones(len(centers), dtype=complex)
    else:
        raise ValueError("beamforming_mode must be none, steer, or focus")

    if config.speaker_array is not None:
        element_weights = np.array([element.weight for element in config.speaker_array.flattened_elements])
        if element_weights.shape != phases.shape:
            raise ValueError("explicit elements must match the number of speaker centers")
        phases *= element_weights
    weights = config.speaker_weights
    if weights is not None:
        weights = np.asarray(weights, dtype=complex)
        if weights.shape != phases.shape:
            raise ValueError("speaker_weights must contain one weight per speaker")
        phases *= weights
    if config.array_window != "uniform":
        window = configured_array_window(config, len(centers))
        if window.size != phases.size:
            raise ValueError("array window shape must match the number of speakers")
        phases *= window.ravel()
    return normalize_weights(
        phases,
        taper=config.speaker_taper,
        normalization=config.weight_normalization,
        total_squared_weight=config.total_squared_weight,
    ), azimuth


@dataclass(frozen=True)
class SimulationOutput:
    grid: SimulationGrid
    velocity: np.ndarray
    pressure: PressureResult
    theta_deg: np.ndarray
    directivity: np.ndarray
    azimuth_deg: float


@dataclass(frozen=True)
class FrequencySweepResult:
    """Complex pressure indexed as (frequency, distance, y, x)."""

    frequencies_hz: NDArray[np.float64]
    distances_m: NDArray[np.float64]
    grid: XYGrid
    pressure: NDArray[np.complex128]


@dataclass(frozen=True)
class FrequencyDirectivityResult:
    """Far-field pressure coefficients indexed as (frequency, polar angle)."""

    frequencies_hz: NDArray[np.float64]
    theta_deg: NDArray[np.float64]
    azimuth_deg: float
    directivity: NDArray[np.complex128]


@dataclass(frozen=True)
class FrequencyPressurePlane:
    """One independently stored xy plane at a given frequency and distance."""

    frequency_index: int
    distance_index: int
    frequency_hz: float
    distance_m: float
    grid: XYGrid
    pressure: NDArray[np.complex128]


def _sweep_frequencies(frequencies_hz: ArrayLike) -> NDArray[np.float64]:
    frequencies = np.atleast_1d(np.asarray(frequencies_hz))
    if (
        frequencies.ndim != 1 or not frequencies.size or frequencies.dtype.kind not in "iuf"
        or not np.isfinite(frequencies).all() or (frequencies <= 0).any()
    ):
        raise ValueError("frequencies_hz must be a nonempty scalar or vector of finite positive frequencies")
    return frequencies.astype(float)


def _sweep_axes(
    config: SimulationConfig, frequencies_hz: ArrayLike, distances_m: ArrayLike | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    return (
        _sweep_frequencies(frequencies_hz),
        validate_observation_distances(config.observation_distance_m if distances_m is None else distances_m),
    )


def _check_cancelled(should_cancel: Callable[[], bool] | None) -> None:
    if should_cancel is not None and should_cancel():
        raise CancelledError("frequency sweep cancelled")


def _measured_patterns(config: SimulationConfig) -> dict[str, MeasuredElementPattern]:
    if config.speaker_array is None:
        raise ValueError("measured far field requires an explicit speaker array")
    used_ids = set(array_type_ids(config))
    patterns = {}
    common_basis = None
    for speaker_type in config.speaker_array.speaker_types:
        if speaker_type.type_id not in used_ids:
            continue
        pattern = speaker_type.measured_pattern
        if pattern is None:
            raise ValueError(f"Speaker type {speaker_type.type_id!r} has no measured element pattern")
        if not isclose(pattern.sound_speed_m_s, config.sound_speed_m_s, rel_tol=1e-9):
            raise ValueError(f"Speaker type {speaker_type.type_id!r} pattern sound speed must match the configuration")
        reference = pattern.reference
        basis = (
            pattern.pressure_unit, reference.reference_input_value, reference.reference_input_unit,
            reference.calibration, reference.calibration_id,
        )
        if common_basis is not None and basis != common_basis:
            raise ValueError("measured element patterns must share pressure units, reference input, and calibration")
        common_basis = basis
        patterns[speaker_type.type_id] = pattern
    return patterns


def _measured_pattern_sum(
    config: SimulationConfig, centers: np.ndarray, patterns: dict[str, MeasuredElementPattern],
    theta_deg: ArrayLike, phi_deg: ArrayLike | None,
) -> tuple[np.ndarray, float]:
    weights, azimuth = _processed_drive_weights(config, centers)
    phi = azimuth if phi_deg is None else phi_deg
    directions = direction_vectors(theta_deg, phi)
    k = tau * config.frequency_hz / config.sound_speed_m_s
    total = np.zeros(directions.shape[:-1], dtype=complex)
    evaluated = {}
    for center, weight, element in zip(centers, weights, config.speaker_array.flattened_elements):
        key = element.type_id, element.orientation
        if key not in evaluated:
            try:
                evaluated[key] = patterns[element.type_id].evaluate(
                    theta_deg, phi, config.frequency_hz, placement=element.orientation,
                )
            except ValueError as exc:
                raise ValueError(f"Speaker type {element.type_id!r}: {exc}") from exc
        # R_element ≈ R - s·r_element for an outgoing exp(+i k R) wave.
        total += weight * evaluated[key] * np.exp(-1j * k * (directions[..., :2] @ center))
    return total, azimuth


class SimulationSession:
    """Reuse the last result of each stage while editing a configuration."""

    def __init__(self) -> None:
        self._cache = {}

    def _cached(self, name, key, compute):
        cached = self._cache.get(name)
        if cached is not None and cached[0] == key:
            return cached[1]
        if cached is not None:
            # Release large stale results before computing their replacements.
            del self._cache[name], cached
        result = compute()
        self._cache[name] = (key, result, object())
        return result

    def _revision(self, name):
        return self._cache[name][2]

    def centers(self, config: SimulationConfig) -> np.ndarray:
        if config.speaker_array is None:
            key = (
                config.rows, config.columns, config.speaker_diameter_m,
                config.horizontal_gap_m, config.vertical_gap_m, config.row_stagger,
            )
        else:
            key = tuple(element.position_m for element in config.speaker_array.flattened_elements), array_diameters(config)
        return self._cached("geometry", key, lambda: generate_array(config))

    def grid(self, config: SimulationConfig) -> SimulationGrid:
        centers = self.centers(config)
        key = (
            self._revision("geometry"), config.grid_spacing_m, config.grid_spacing_y_m,
            config.grid_margin_m, config.padding_factor, config.fast_fft, config.observation_shape,
            config.observation_width_x_m, config.observation_width_y_m,
            config.observation_width_x_factor, config.observation_width_y_factor,
        )
        return self._cached("grid", key, lambda: build_array_grid(config, centers))

    def warnings(self, config: SimulationConfig) -> list[str]:
        """Return sampling and padding heuristics without computing an FFT."""
        centers = self.centers(config)
        dx, dy = _grid_spacing(config)
        (sy, sx), (ny, nx) = _array_grid_dimensions(config, centers, dx, dy)
        resolve_observation_shape(
            (sy, sx), (ny, nx), dx, dy, config.observation_shape,
            *_observation_widths(config, centers),
        )
        profile = create_profile(config)
        spacing = max(dx, dy)
        wavelength = config.sound_speed_m_s / config.frequency_hz
        warnings = []
        diameters = np.broadcast_to(array_diameters(config), (len(centers),))
        if spacing > min(wavelength, diameters.min()) / 6:
            warnings.append("Coarse sampling: aim for six cells across a wavelength and diaphragm.")
        profiles = array_profiles(config, profile) or (profile,) * len(centers)
        for diameter, item in zip(diameters, profiles):
            if not isinstance(item, RadialProfile):
                continue
            item = item.at_frequency(config.frequency_hz)
            width = {
                "cosine": diameter / 2 * item.taper_width_fraction,
                "gaussian": diameter / 2 * item.gaussian_sigma_fraction,
            }.get(item.kind)
            if width is not None and spacing > width / 3:
                warnings.append("Profile width spans fewer than three cells.")
                break
        px = (nx - sx) * dx / 2
        py = (ny - sy) * dy / 2
        if config.padding_factor < 2 or min(px, py) < wavelength / 2:
            warnings.append("Padding may be insufficient: check pressure convergence for wraparound.")
        if nx * ny >= 2**20:
            warnings.append(f"Large FFT ({ny:,} × {nx:,}): recalculation may be slow or memory intensive.")
        return warnings

    def _prepare_source(
        self, config: SimulationConfig,
    ) -> tuple[SimulationGrid, np.ndarray, VelocitySpectrum, float]:
        centers = self.centers(config)
        grid = self.grid(config)
        profile = create_profile(config)
        weights, azimuth = _velocity_weights(config, centers)
        profiles = array_profiles(config, profile)
        profile_frequency = config.frequency_hz if profiles and any(p.frequency_dependent for p in profiles) else None
        velocity_key = (
            self._revision("geometry"), self._revision("grid"), profile, tuple(weights),
            config.aperture_normalization, config.aperture_subsamples, profiles, profile_frequency,
        )
        velocity = self._cached(
            "velocity", velocity_key, lambda: sample_velocity(config, centers, grid, profile, weights),
        )
        prepared = self._cached(
            "spectrum", self._revision("velocity"), lambda: prepare_velocity_spectrum(velocity, grid),
        )
        return grid, velocity, prepared, azimuth

    def recalculate(
        self, config: SimulationConfig, *, directivity_azimuth_deg: float | None = None,
    ) -> SimulationOutput:
        grid, velocity, prepared, azimuth = self._prepare_source(config)
        medium = dict(
            frequency_hz=config.frequency_hz,
            sound_speed_m_s=config.sound_speed_m_s,
            air_density_kg_m3=config.air_density_kg_m3,
        )
        spectral_key = self._revision("spectrum"), *medium.values()
        pressure = self._cached(
            "pressure", (*spectral_key, config.observation_distance_m, config.include_evanescent),
            lambda: pressure_from_spectrum(
                prepared, config.observation_distance_m, **medium, include_evanescent=config.include_evanescent,
            ),
        )
        if directivity_azimuth_deg is not None:
            azimuth = directivity_azimuth_deg
        theta = self._cached("angles", (), lambda: np.linspace(-90, 90, 361))
        directivity = self._cached(
            "directivity", (*spectral_key, azimuth),
            lambda: directivity_from_spectrum(prepared, theta, azimuth, **medium),
        )
        return SimulationOutput(grid, velocity, pressure, theta, directivity, azimuth)


    def recalculate_frequencies(
        self,
        config: SimulationConfig,
        frequencies_hz: ArrayLike,
        distances_m: ArrayLike | None = None,
        *,
        progress: Callable[[int, int], None] | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> FrequencySweepResult:
        """Return a batch of pressure planes in the requested frequency and distance order.

        Progress receives the number of completed and total xy planes.
        Cancellation raises CancelledError between FFTs. Use iter_frequency_planes
        when the complete output array would be too large to keep in memory.
        """
        frequencies, distances = _sweep_axes(config, frequencies_hz, distances_m)
        _check_cancelled(should_cancel)
        grid = self.grid(config).observation
        pressure = np.empty((len(frequencies), len(distances), *grid.shape), dtype=complex)
        with closing(self._iter_sweep_planes(config, frequencies, distances, progress, should_cancel)) as planes:
            for fi, zi, plane in planes:
                pressure[fi, zi] = plane
                del plane
        return FrequencySweepResult(frequencies, distances, grid, pressure)

    def recalculate_frequency_directivity(
        self,
        config: SimulationConfig,
        frequencies_hz: ArrayLike,
        *,
        directivity_azimuth_deg: float | None = None,
        progress: Callable[[int, int], None] | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> FrequencyDirectivityResult:
        """Evaluate the usual signed far-field cut without propagating xy planes."""
        frequencies = _sweep_frequencies(frequencies_hz)
        theta = self._cached("angles", (), lambda: np.linspace(-90, 90, 361))
        coefficients = np.empty((len(frequencies), len(theta)), dtype=complex)
        for index, frequency in enumerate(frequencies):
            _check_cancelled(should_cancel)
            _, _, prepared, azimuth = self._prepare_source(replace(config, frequency_hz=float(frequency)))
            if directivity_azimuth_deg is not None:
                azimuth = directivity_azimuth_deg
            coefficients[index] = directivity_from_spectrum(
                prepared, theta, azimuth, frequency_hz=frequency,
                sound_speed_m_s=config.sound_speed_m_s, air_density_kg_m3=config.air_density_kg_m3,
            )
            del prepared
            if progress is not None:
                progress(index + 1, len(frequencies))
        return FrequencyDirectivityResult(frequencies, theta, float(azimuth), coefficients)

    def measured_far_field(
        self, config: SimulationConfig, theta_deg: ArrayLike, phi_deg: ArrayLike,
    ) -> complex | NDArray[np.complex128]:
        """Synthesize complex far-field coefficients at broadcast-compatible angles."""
        _sweep_frequencies(config.frequency_hz)
        patterns = _measured_patterns(config)
        coefficients, _ = _measured_pattern_sum(config, self.centers(config), patterns, theta_deg, phi_deg)
        return complex(coefficients.item()) if coefficients.ndim == 0 else coefficients

    def recalculate_measured_directivity(
        self,
        config: SimulationConfig,
        frequencies_hz: ArrayLike,
        *,
        directivity_azimuth_deg: float | None = None,
        progress: Callable[[int, int], None] | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> FrequencyDirectivityResult:
        """Synthesize measured-pattern cuts over frequency without running ASM."""
        frequencies = _sweep_frequencies(frequencies_hz)
        _check_cancelled(should_cancel)
        patterns = _measured_patterns(config)
        centers = self.centers(config)
        theta = self._cached("angles", (), lambda: np.linspace(-90, 90, 361))
        coefficients = np.empty((len(frequencies), len(theta)), dtype=complex)
        for index, frequency in enumerate(frequencies):
            _check_cancelled(should_cancel)
            current = replace(config, frequency_hz=float(frequency))
            coefficients[index], azimuth = _measured_pattern_sum(
                current, centers, patterns, theta, directivity_azimuth_deg,
            )
            if progress is not None:
                progress(index + 1, len(frequencies))
        if directivity_azimuth_deg is not None:
            azimuth = directivity_azimuth_deg
        return FrequencyDirectivityResult(frequencies, theta, float(azimuth), coefficients)

    def iter_frequency_planes(
        self,
        config: SimulationConfig,
        frequencies_hz: ArrayLike,
        distances_m: ArrayLike | None = None,
        *,
        progress: Callable[[int, int], None] | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> Iterator[FrequencyPressurePlane]:
        """Stream independently stored xy planes with the same hooks as the batch."""
        frequencies, distances = _sweep_axes(config, frequencies_hz, distances_m)
        _check_cancelled(should_cancel)
        grid = self.grid(config).observation
        with closing(self._iter_sweep_planes(config, frequencies, distances, progress, should_cancel)) as planes:
            for fi, zi, plane in planes:
                yield FrequencyPressurePlane(
                    fi, zi, float(frequencies[fi]), float(distances[zi]), grid, plane.copy(),
                )
                del plane

    def _iter_sweep_planes(
        self,
        config: SimulationConfig,
        frequencies: NDArray[np.float64],
        distances: NDArray[np.float64],
        progress: Callable[[int, int], None] | None,
        should_cancel: Callable[[], bool] | None,
    ) -> Iterator[tuple[int, int, NDArray[np.complex128]]]:
        total = len(frequencies) * len(distances)
        for fi, frequency in enumerate(frequencies):
            _check_cancelled(should_cancel)
            _, _, prepared, _ = self._prepare_source(replace(config, frequency_hz=float(frequency)))
            with closing(iter_pressure_planes(
                prepared, distances, frequency_hz=frequency,
                sound_speed_m_s=config.sound_speed_m_s, air_density_kg_m3=config.air_density_kg_m3,
                include_evanescent=config.include_evanescent,
            )) as planes:
                for zi in range(len(distances)):
                    _check_cancelled(should_cancel)
                    plane = next(planes)
                    if progress is not None:
                        progress(fi * len(distances) + zi + 1, total)
                    yield fi, zi, plane
                    del plane
            del prepared
