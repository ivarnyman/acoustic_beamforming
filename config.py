"""Configuration and conventions shared by the acoustic simulator."""

import json
from dataclasses import asdict, dataclass, replace
from math import isclose, isfinite
from numbers import Integral, Real
from pathlib import Path
from typing import Callable, Literal

from speakers import SpeakerArray

TIME_CONVENTION = "exp(-i omega t)"
DIAPHRAGM_VELOCITY_CONVENTION = "dimensionless, normalized relative diaphragm velocity"


@dataclass(frozen=True)
class SimulationConfig:
    """Default settings for a rectangular speaker array in air."""

    # Diameter of each circular diaphragm in meters.
    speaker_diameter_m: float = 0.25
    # Number of speaker rows along y.
    rows: int = 7
    # Number of speaker columns along x.
    columns: int = 7
    # Horizontal edge-to-edge gap between diaphragms in meters.
    horizontal_gap_m: float = 0.02
    # Vertical edge-to-edge gap before staggering in meters.
    vertical_gap_m: float = 0.02
    # Alternating row shift as a fraction of the speaker diameter plus horizontal gap.
    row_stagger: float = 0.0
    # Source-grid spacing along x in meters.
    grid_spacing_m: float = 0.025
    # Factor used to enlarge each grid dimension for FFT zero-padding.
    padding_factor: int = 16
    # Speed of sound in air in meters per second.
    sound_speed_m_s: float = 343.0
    # Frequency of the single simulated tone in hertz.
    frequency_hz: float = 850.0
    # Air density in kilograms per cubic meter.
    air_density_kg_m3: float = 1.21
    # Distance along z from the diaphragm plane to the observation plane in meters.
    observation_distance_m: float = 5.0
    # Number of evenly spaced z evaluation planes for the pressure slices.
    z_evaluation_points: int = 61
    # Radial profile of the prescribed diaphragm velocity.
    amplitude_profile: Literal["uniform", "cosine", "gaussian"] = "uniform"
    # Common multiplier for all relative diaphragm velocities.
    relative_amplitude: float = 1.0
    # Source-grid spacing along y in meters, inherited from x when None.
    grid_spacing_y_m: float | None = None
    # Margin per side around the aperture in meters, shared or specified as (x, y).
    grid_margin_m: float | tuple[float, float] = 0.02
    # Round padded dimensions up to FFT-friendly lengths when enabled.
    fast_fft: bool = True
    # Observation grid shape (ny, nx), inherited from the source grid when None.
    observation_shape: tuple[int, int] | None = None
    # Plot width along x as a multiple of the physical array width.
    observation_width_x_factor: float = 3
    # Plot width along y as a multiple of the physical array height.
    observation_width_y_factor: float = 3
    # Optional fixed x width in meters, overriding the multiplier.
    observation_width_x_m: float | None = None
    # Optional fixed y width in meters, overriding the multiplier.
    observation_width_y_m: float | None = None
    # Width of the cosine edge taper as a fraction of the diaphragm radius.
    taper_width_fraction: float = 0.25
    # Gaussian standard deviation as a fraction of the diaphragm radius.
    gaussian_sigma_fraction: float = 0.5
    # Normalize each diaphragm profile by its peak or integrated squared velocity.
    aperture_normalization: Literal["peak", "equal_energy"] = "peak"
    # Number of subpixel samples per cell axis for aperture averaging.
    aperture_subsamples: int = 4
    # Include exponentially decaying evanescent components in pressure propagation.
    include_evanescent: bool = True
    # Select unsteered operation, far-field steering, or near-field focusing.
    beamforming_mode: Literal["none", "steer", "focus"] = "none"
    # Steering azimuth in degrees measured from +x toward +y.
    steering_azimuth_deg: float = 0.0
    # Steering elevation in degrees above the array plane, with 90° at broadside.
    steering_elevation_deg: float = 90.0
    # Focus range along the steering direction in meters, or None for a Cartesian target.
    focus_distance_m: float | None = None
    # Cartesian focus target (x, y, z) in meters, used when focus_distance_m is None.
    focus_point_m: tuple[float, float, float] = (0.0, 0.0, observation_distance_m)
    # Complex gain per speaker in row-major order, with unity gains when None.
    speaker_weights: tuple[complex, ...] | None = None
    # Symmetric amplitude window across the speaker rows and columns.
    array_window: str = "uniform"
    # Additional nonnegative amplitude multiplier per speaker, with no taper when None.
    speaker_taper: tuple[float, ...] | None = None
    # Leave weights unchanged or normalize by peak magnitude or total squared magnitude.
    weight_normalization: Literal["none", "peak", "power"] = "none"
    # Target sum of squared weight magnitudes in power mode, defaulting to the speaker count.
    total_squared_weight: float | None = None
    # Optional speaker catalog and explicit planar placements for a custom array.
    speaker_array: SpeakerArray | None = None
    # Optional frequency bins for interactive sweeps, in their evaluation order.
    frequencies_hz: tuple[float, ...] | None = None
    # Interactive solver selection, including the measured-pattern far-field path.
    propagation_mode: Literal["asm", "measured"] = "asm"

    def resolve_observation_widths_m(
        self, array_width_x: float, array_width_y: float,
    ) -> tuple[float | None, float | None]:
        """Scale the physical array bounds for any automatically sized plot axes."""
        if self.observation_shape is not None:
            return self.observation_width_x_m, self.observation_width_y_m
        return (
            self.observation_width_x_factor * array_width_x
            if self.observation_width_x_m is None else self.observation_width_x_m,
            self.observation_width_y_factor * array_width_y
            if self.observation_width_y_m is None else self.observation_width_y_m,
        )

    def __post_init__(self) -> None:
        if self.speaker_array is not None and not isinstance(self.speaker_array, SpeakerArray):
            raise ValueError("speaker_array must be a SpeakerArray or None")
        if self.propagation_mode not in ("asm", "measured"):
            raise ValueError("propagation_mode must be asm or measured")
        if self.frequencies_hz is not None:
            try:
                frequencies = tuple(self.frequencies_hz)
            except TypeError as exc:
                raise ValueError("frequencies_hz must contain unique positive frequencies") from exc
            if (
                not frequencies
                or any(isinstance(f, bool) or not isinstance(f, Real) or not isfinite(f) or f <= 0 for f in frequencies)
                or len(set(frequencies)) != len(frequencies)
            ):
                raise ValueError("frequencies_hz must contain unique positive frequencies")
            object.__setattr__(self, "frequencies_hz", tuple(map(float, frequencies)))
        if self.rows < 1 or self.columns < 1:
            raise ValueError("rows and columns must be positive")
        if (
            isinstance(self.row_stagger, bool)
            or not isinstance(self.row_stagger, Real)
            or not 0 <= self.row_stagger <= 1
        ):
            raise ValueError("row_stagger must be a number from 0 to 1")
        object.__setattr__(self, "row_stagger", float(self.row_stagger))
        count = self.z_evaluation_points
        if isinstance(count, bool) or not isinstance(count, Integral) or count < 1:
            raise ValueError("z_evaluation_points must be a positive integer")

        for name in ("observation_width_x_factor", "observation_width_y_factor"):
            factor = getattr(self, name)
            if isinstance(factor, bool) or not isinstance(factor, Real) or not isfinite(factor) or factor <= 0:
                raise ValueError(f"{name} must be finite and positive")


def save_configuration(path: str | Path, config: SimulationConfig) -> None:
    """Write all settings as JSON, representing complex weights as [real, imag]."""
    data = asdict(replace(config, speaker_array=None))
    data.pop("speaker_array")
    if config.speaker_array is not None:
        data["speaker_array"] = config.speaker_array.to_dict()
    if config.speaker_weights is not None:
        data["speaker_weights"] = [[weight.real, weight.imag] for weight in config.speaker_weights]
    if config.frequencies_hz is None:
        data.pop("frequencies_hz")
    if config.propagation_mode == "asm":
        data.pop("propagation_mode")
    Path(path).expanduser().write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load_configuration(path: str | Path) -> SimulationConfig:
    """Read JSON settings, with omitted fields taking their default values."""
    data = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("configuration must be a JSON object")
    if data.get("speaker_weights") is not None:
        data["speaker_weights"] = tuple(complex(*pair) for pair in data["speaker_weights"])
    if data.get("speaker_array") is not None:
        data["speaker_array"] = SpeakerArray.from_dict(data["speaker_array"])
    if "row_offsets_m" in data:
        offsets = data.pop("row_offsets_m")
        if "row_stagger" not in data:
            data["row_stagger"] = _legacy_row_stagger(offsets, data)
    for key in ("grid_margin_m", "observation_shape", "focus_point_m", "speaker_taper"):
        if isinstance(data.get(key), list):
            data[key] = tuple(data[key])
    return SimulationConfig(**data)


def _legacy_row_stagger(offsets: float | list[float] | None, data: dict) -> float:
    if offsets is None:
        return 0.0
    if isinstance(offsets, list):
        rows = data.get("rows", SimulationConfig.rows)
        if len(offsets) != rows or not offsets:
            raise ValueError("legacy row_offsets_m must contain one offset per row")
        shift = offsets[1] - offsets[0] if rows > 1 else 0.0
        if any(
            not isclose(offset, offsets[index % 2], abs_tol=1e-12)
            for index, offset in enumerate(offsets)
        ):
            raise ValueError("legacy row_offsets_m must alternate to convert to row_stagger")
    else:
        shift = offsets
    pitch = (
        data.get("speaker_diameter_m", SimulationConfig.speaker_diameter_m)
        + data.get("horizontal_gap_m", SimulationConfig.horizontal_gap_m)
    )
    if pitch <= 0 or not 0 <= shift <= pitch:
        raise ValueError("legacy row_offsets_m must be within one horizontal speaker pitch")
    return shift / pitch
