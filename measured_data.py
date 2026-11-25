"""CSV imports for velocity response and far-field element patterns."""

import csv
from dataclasses import asdict, dataclass
from functools import cached_property
from itertools import chain
from math import isfinite, tau
from numbers import Real
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial.transform import Rotation

from frequency_response import FrequencyResponse, OutOfRange

_ANGLE_CONVENTION = "polar_from_z_azimuth_from_x_deg"
_NEGATIVE_TIME = "exp(-i omega t)"
_POSITIVE_TIME = "exp(+i omega t)"


def _positive(value: float) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and isfinite(value) and value > 0


@dataclass(frozen=True)
class MeasurementReference:
    """Drive level and shared amplitude/phase reference for measurement data."""

    reference_input_value: float
    reference_input_unit: Literal["normalized", "V_peak", "V_rms"]
    calibration: Literal["synthetic", "relative", "calibrated"]
    calibration_id: str
    calibration_note: str

    def __post_init__(self) -> None:
        if not _positive(self.reference_input_value):
            raise ValueError("reference_input_value must be finite and positive")
        if self.reference_input_unit not in ("normalized", "V_peak", "V_rms"):
            raise ValueError("reference_input_unit must be normalized, V_peak, or V_rms")
        if self.calibration not in ("synthetic", "relative", "calibrated"):
            raise ValueError("calibration must be synthetic, relative, or calibrated")
        for name in ("calibration_id", "calibration_note"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be nonempty")


@dataclass(frozen=True)
class PatternOrientation:
    """Local-to-parent rotation Rz(yaw) Ry(pitch) Rx(roll), in degrees."""

    yaw_deg: float = 0.0
    pitch_deg: float = 0.0
    roll_deg: float = 0.0

    def __post_init__(self) -> None:
        for angle in (self.yaw_deg, self.pitch_deg, self.roll_deg):
            if not isinstance(angle, Real) or isinstance(angle, bool) or not isfinite(angle):
                raise ValueError("orientation angles must be finite real degrees")

    @cached_property
    def _rotation(self) -> Rotation:
        return Rotation.from_euler("ZYX", (self.yaw_deg, self.pitch_deg, self.roll_deg), degrees=True)

    def to_local(self, directions: NDArray[np.float64]) -> NDArray[np.float64]:
        return self._rotation.apply(directions.reshape(-1, 3), inverse=True).reshape(directions.shape)


def direction_vectors(theta_deg: ArrayLike, phi_deg: ArrayLike) -> NDArray[np.float64]:
    """Convert polar and azimuth angles to unit vectors, including signed cuts."""
    theta, phi = np.broadcast_arrays(np.asarray(theta_deg), np.asarray(phi_deg))
    if (
        theta.dtype.kind not in "iuf" or phi.dtype.kind not in "iuf"
        or not np.isfinite(theta).all() or not np.isfinite(phi).all() or (np.abs(theta) > 180).any()
    ):
        raise ValueError("angles must be finite and theta_deg within [-180, 180]")
    theta, phi = np.deg2rad(theta), np.deg2rad(phi)
    transverse = np.sin(theta)
    return np.stack((transverse * np.cos(phi), transverse * np.sin(phi), np.cos(theta)), axis=-1)


@dataclass(frozen=True)
class MeasuredElementPattern:
    """Far-field coefficients D in p(R) = D exp(i k R)/R, per reference drive."""

    frequencies_hz: ArrayLike
    theta_deg: ArrayLike
    phi_deg: ArrayLike
    coefficients: ArrayLike
    reference_distance_m: float
    sound_speed_m_s: float
    pressure_unit: Literal["relative", "Pa"]
    reference: MeasurementReference
    orientation: PatternOrientation = PatternOrientation()

    def __post_init__(self) -> None:
        axes = []
        for name, minimum_count in (("frequencies_hz", 1), ("theta_deg", 2), ("phi_deg", 2)):
            axis = np.asarray(getattr(self, name))
            if (
                axis.ndim != 1 or len(axis) < minimum_count or axis.dtype.kind not in "iuf"
                or not np.isfinite(axis).all() or not (axis[1:] > axis[:-1]).all()
            ):
                raise ValueError(f"{name} must contain at least {minimum_count} increasing finite samples")
            axes.append(axis)
            object.__setattr__(self, name, tuple(map(float, axis)))
        frequencies, theta, phi = axes
        if frequencies[0] <= 0:
            raise ValueError("frequencies_hz must be positive")
        if theta[0] < 0 or theta[-1] > 180 or phi[0] < 0 or phi[-1] >= 360:
            raise ValueError("sample angles must use theta in [0, 180] and phi in [0, 360) degrees")
        values = np.asarray(self.coefficients)
        if (
            values.shape != tuple(map(len, axes)) or values.dtype.kind not in "iufc"
            or not np.isfinite(values).all()
        ):
            raise ValueError("coefficients must form a finite complex grid indexed [frequency, theta, phi]")
        for pole in np.flatnonzero((theta == 0) | (theta == 180)):
            ring = values[:, pole, :]
            if not np.allclose(ring, ring[:, :1], rtol=1e-8, atol=0):
                raise ValueError("polar-axis coefficients must agree across azimuths")
        if not _positive(self.reference_distance_m) or not _positive(self.sound_speed_m_s):
            raise ValueError("reference_distance_m and sound_speed_m_s must be finite and positive")
        if not isinstance(self.reference, MeasurementReference) or not isinstance(self.orientation, PatternOrientation):
            raise ValueError("pattern requires a measurement reference and orientation")
        if self.pressure_unit not in ("relative", "Pa"):
            raise ValueError("pressure_unit must be relative or Pa")
        if self.pressure_unit == "Pa" and self.reference.calibration == "relative":
            raise ValueError("pressure in Pa requires calibrated or explicitly synthetic data")
        object.__setattr__(self, "coefficients", tuple(
            tuple(tuple(map(complex, row)) for row in sample) for sample in values
        ))

    @cached_property
    def _interpolator(self) -> RegularGridInterpolator:
        phi = (*self.phi_deg, self.phi_deg[0] + 360)
        values = np.asarray(self.coefficients)
        periodic = np.concatenate((values, values[:, :, :1]), axis=2)
        return RegularGridInterpolator(
            (np.log(self.frequencies_hz), self.theta_deg, phi), periodic, method="linear",
        )

    def evaluate(
        self, theta_deg: ArrayLike, phi_deg: ArrayLike, frequency_hz: ArrayLike, *,
        placement: PatternOrientation | None = None,
    ) -> complex | NDArray[np.complex128]:
        """Interpolate complex D at global angles, with periodic azimuth."""
        directions = direction_vectors(theta_deg, phi_deg)
        if placement is not None:
            directions = placement.to_local(directions)
        local = self.orientation.to_local(directions)
        x, y, z = np.moveaxis(local, -1, 0)
        theta = np.rad2deg(np.arctan2(np.hypot(x, y), z))
        phi = np.rad2deg(np.arctan2(y, x))
        phi = (phi - self.phi_deg[0]) % 360 + self.phi_deg[0]
        frequency = np.asarray(frequency_hz)
        if frequency.dtype.kind not in "iuf" or not np.isfinite(frequency).all():
            raise ValueError("pattern frequencies must be finite positive values in Hz")
        if ((frequency < self.frequencies_hz[0]) | (frequency > self.frequencies_hz[-1])).any():
            raise ValueError("requested frequency is outside the measured pattern band")
        low, high = self.theta_deg[0], self.theta_deg[-1]
        if ((theta < low - 1e-10) | (theta > high + 1e-10)).any():
            raise ValueError("direction is outside the measured polar coverage")
        coordinates = np.broadcast_arrays(np.log(frequency), np.clip(theta, low, high), phi)
        samples = self._interpolator(np.stack(coordinates, axis=-1))
        return complex(samples.item()) if coordinates[0].ndim == 0 else samples

    def to_dict(self) -> dict:
        data = asdict(self)
        values = np.asarray(self.coefficients)
        data["coefficients"] = np.stack((values.real, values.imag), axis=-1).tolist()
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "MeasuredElementPattern":
        if not isinstance(data, dict):
            raise ValueError("measured_pattern must be a JSON object")
        fields = dict(data)
        pairs = np.asarray(fields["coefficients"])
        if pairs.ndim != 4 or pairs.shape[-1] != 2 or pairs.dtype.kind not in "iuf":
            raise ValueError("pattern coefficients must contain [real, imaginary] pairs")
        fields["coefficients"] = pairs[..., 0] + 1j * pairs[..., 1]
        fields["reference"] = MeasurementReference(**fields["reference"])
        if "orientation" in fields:
            fields["orientation"] = PatternOrientation(**fields["orientation"])
        return cls(**fields)


@dataclass(frozen=True)
class MeasuredVelocityResponse:
    response: FrequencyResponse
    metadata: dict[str, object]


def _read_csv(path: str | Path, schema: str, columns: tuple[str, ...]) -> tuple[dict, np.ndarray]:
    metadata = {}
    rows = []
    with Path(path).expanduser().open(encoding="utf-8-sig", newline="") as source:
        header = ""
        for line in source:
            text = line.strip()
            if not text:
                continue
            if not text.startswith("#"):
                header = line
                break
            key, separator, value = text[1:].strip().partition("=")
            key, value = key.strip(), value.strip()
            if not separator or not key or not value or key in metadata:
                raise ValueError("CSV metadata must use unique '# key=value' entries")
            metadata[key] = value
        if metadata.get("schema") != schema:
            raise ValueError(f"expected CSV schema {schema}")
        reader = csv.reader(chain([header], source), strict=True)
        if tuple(next(reader)) != columns:
            raise ValueError(f"CSV columns must be {','.join(columns)}")
        for row in reader:
            if not row:
                continue
            if len(row) != len(columns):
                raise ValueError(f"CSV row {reader.line_num} must have {len(columns)} columns")
            try:
                rows.append([float(value) for value in row])
            except ValueError as exc:
                raise ValueError(f"CSV row {reader.line_num} must contain numeric values") from exc
    data = np.asarray(rows, dtype=float)
    if not rows or not np.isfinite(data).all():
        raise ValueError("CSV must contain finite numeric samples")
    return metadata, data


def _required(metadata: dict, key: str) -> str:
    try:
        return metadata[key]
    except KeyError:
        raise ValueError(f"CSV metadata requires {key}") from None


def _reference(metadata: dict) -> MeasurementReference:
    return MeasurementReference(
        float(_required(metadata, "reference_input_value")), _required(metadata, "reference_input_unit"),
        _required(metadata, "calibration"), _required(metadata, "calibration_id"),
        _required(metadata, "calibration_note"),
    )


def _magnitude_phase(metadata: dict, magnitude: np.ndarray, phase: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    magnitude_unit = _required(metadata, "magnitude_unit")
    if magnitude_unit == "dB":
        with np.errstate(over="ignore", under="ignore"):
            magnitude = 10 ** (magnitude / 20)
    elif magnitude_unit != "linear":
        raise ValueError("magnitude_unit must be linear or dB")
    if not np.isfinite(magnitude).all() or (magnitude < 0).any():
        raise ValueError("magnitude must be finite and nonnegative")
    phase_unit = _required(metadata, "phase_unit")
    if phase_unit == "deg":
        phase = np.deg2rad(phase)
    elif phase_unit != "rad":
        raise ValueError("phase_unit must be deg or rad")
    convention = _required(metadata, "time_convention")
    if convention == _POSITIVE_TIME:
        phase = -phase
    elif convention != _NEGATIVE_TIME:
        raise ValueError(f"time_convention must be {_NEGATIVE_TIME} or {_POSITIVE_TIME}")
    return magnitude, phase


def load_frequency_response_csv(
    path: str | Path, *, out_of_range: OutOfRange = "error",
) -> MeasuredVelocityResponse:
    """Import drive-to-relative-velocity data and retain its measurement reference."""
    metadata, data = _read_csv(path, "velocity_response_v1", ("frequency_hz", "magnitude", "phase"))
    if _required(metadata, "quantity") != "relative_diaphragm_velocity":
        raise ValueError("velocity_response_v1 requires quantity=relative_diaphragm_velocity")
    reference = _reference(metadata)
    frequency, magnitude, phase = data[data[:, 0].argsort()].T
    magnitude, phase = _magnitude_phase(metadata, magnitude, phase)
    wrapping = _required(metadata, "phase_wrapping")
    if wrapping == "wrapped":
        phase = np.unwrap(phase)
    elif wrapping != "unwrapped":
        raise ValueError("phase_wrapping must be wrapped or unwrapped")
    response = FrequencyResponse.from_magnitude_phase(frequency, magnitude, phase, out_of_range=out_of_range)
    return MeasuredVelocityResponse(response, {**metadata, **asdict(reference)})


def load_element_pattern_csv(path: str | Path) -> MeasuredElementPattern:
    """Import far-field pressure samples and remove reference-range propagation."""
    columns = ("frequency_hz", "theta_deg", "phi_deg", "magnitude", "phase")
    metadata, data = _read_csv(path, "element_pattern_v1", columns)
    if _required(metadata, "angle_convention") != _ANGLE_CONVENTION:
        raise ValueError(f"angle_convention must be {_ANGLE_CONVENTION}")
    reference = _reference(metadata)
    distance = float(_required(metadata, "reference_distance_m"))
    speed = float(_required(metadata, "sound_speed_m_s"))
    if not _positive(distance) or not _positive(speed):
        raise ValueError("reference_distance_m and sound_speed_m_s must be finite and positive")
    orientation = PatternOrientation(*(float(metadata.get(key, 0)) for key in ("yaw_deg", "pitch_deg", "roll_deg")))
    order = np.lexsort((data[:, 2], data[:, 1], data[:, 0]))
    frequency, theta, phi, magnitude, phase = data[order].T
    axes = tuple(np.unique(axis) for axis in (frequency, theta, phi))
    shape = tuple(map(len, axes))
    if len(data) != np.prod(shape) or len(np.unique(data[:, :3], axis=0)) != len(data):
        raise ValueError("element pattern requires a complete frequency × theta × phi grid with unique rows")
    magnitude, phase = _magnitude_phase(metadata, magnitude, phase)
    phase_reference = _required(metadata, "phase_reference")
    if phase_reference == "includes_propagation":
        phase -= (tau * distance / speed) * frequency
    elif phase_reference != "range_removed":
        raise ValueError("phase_reference must be includes_propagation or range_removed")
    coefficients = distance * magnitude * np.exp(1j * phase)
    return MeasuredElementPattern(
        *axes, coefficients.reshape(shape), distance, speed, _required(metadata, "pressure_unit"),
        reference, orientation,
    )
