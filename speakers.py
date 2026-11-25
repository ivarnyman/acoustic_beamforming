"""Optional speaker type catalog and planar element placement."""

import json
from dataclasses import asdict, dataclass, replace
from functools import cached_property
from math import isfinite
from numbers import Complex, Integral, Real
from typing import Literal

from aperture import ApertureProfile, CallableProfile, FrequencyParameter, RadialProfile
from frequency_response import FrequencyResponse
from geometry import generate_speaker_centers
from measured_data import MeasuredElementPattern, PatternOrientation
from signal_chain import SignalChain


def _finite_real(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, Real) and isfinite(value)


@dataclass(frozen=True)
class SpeakerType:
    """Reusable physical diaphragm properties and descriptive response metadata."""

    type_id: str
    name: str
    diameter_m: float
    aperture_profile: ApertureProfile | None = None
    frequency_response_metadata: dict[str, object] | None = None
    directivity_metadata: dict[str, object] | None = None
    response_model: Literal["prescribed_velocity", "idealized_flat"] = "prescribed_velocity"
    frequency_response: FrequencyResponse | None = None
    measured_pattern: MeasuredElementPattern | None = None

    def __post_init__(self) -> None:
        for field in ("type_id", "name"):
            value = getattr(self, field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field} must be a nonempty string")
        if not _finite_real(self.diameter_m) or self.diameter_m <= 0:
            raise ValueError("diameter_m must be finite and positive")
        if self.aperture_profile is not None and not isinstance(self.aperture_profile, (RadialProfile, CallableProfile)):
            raise ValueError("aperture_profile must be a RadialProfile, CallableProfile, or None")
        if self.response_model not in ("prescribed_velocity", "idealized_flat"):
            raise ValueError("response_model must be prescribed_velocity or idealized_flat")
        if self.measured_pattern is not None and not isinstance(self.measured_pattern, MeasuredElementPattern):
            raise ValueError("measured_pattern must be a MeasuredElementPattern or None")
        response = self.frequency_response
        if response is not None:
            if not isinstance(response, FrequencyResponse):
                raise ValueError("frequency_response must be a FrequencyResponse or None")
            if self.response_model == "idealized_flat" and (
                response.out_of_range != "hold" or any(value != 1 for value in response.values)
                or len(set(response.unwrapped_phase_rad)) != 1
            ):
                raise ValueError("idealized_flat requires unity response at all positive frequencies")
        for field in ("frequency_response_metadata", "directivity_metadata"):
            value = getattr(self, field)
            if value is not None:
                if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
                    raise ValueError(f"{field} must be a JSON object")
                try:
                    metadata = json.loads(json.dumps(value, allow_nan=False))
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"{field} must contain JSON data") from exc
                object.__setattr__(self, field, metadata)

    @classmethod
    def generic(
        cls, diameter_m: float, *, type_id: str = "generic",
        aperture_profile: ApertureProfile | None = None,
    ) -> "SpeakerType":
        """Create a generic diaphragm with explicitly idealized unity response."""
        return cls(
            type_id, "Generic idealized speaker", diameter_m, aperture_profile,
            response_model="idealized_flat",
        )

    def to_dict(self) -> dict:
        if isinstance(self.aperture_profile, CallableProfile):
            raise ValueError("callable aperture profiles are Python-only and cannot be saved to JSON")
        data = asdict(self)
        profile = data["aperture_profile"]
        if profile is not None and profile["rim_phase_rad"] == 0:
            profile.pop("rim_phase_rad")
        for name in ("frequency_response", "measured_pattern"):
            value = getattr(self, name)
            if value is None:
                data.pop(name)
            else:
                data[name] = value.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "SpeakerType":
        if not isinstance(data, dict):
            raise ValueError("speaker type must be a JSON object")
        fields = dict(data)
        if fields.get("aperture_profile") is not None:
            fields["aperture_profile"] = RadialProfile(**{
                name: FrequencyParameter(**value) if isinstance(value, dict) else value
                for name, value in fields["aperture_profile"].items()
            })
        if fields.get("frequency_response") is not None:
            fields["frequency_response"] = FrequencyResponse.from_dict(fields["frequency_response"])
        if fields.get("measured_pattern") is not None:
            fields["measured_pattern"] = MeasuredElementPattern.from_dict(fields["measured_pattern"])
        return cls(**fields)


@dataclass(frozen=True)
class SpeakerElement:
    """A planar element with a phasor weight and optional drive processing."""

    type_id: str
    position_m: tuple[float, float]
    weight: complex = 1 + 0j
    delay_s: float | None = None
    signal_chain: SignalChain | None = None
    orientation: PatternOrientation | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.type_id, str) or not self.type_id.strip():
            raise ValueError("type_id must be a nonempty string")
        object.__setattr__(self, "position_m", _planar_position(self.position_m, "position_m"))
        if (
            not isinstance(self.weight, Complex) or isinstance(self.weight, bool)
            or not isfinite(self.weight.real) or not isfinite(self.weight.imag)
        ):
            raise ValueError("weight must be a finite complex number")
        object.__setattr__(self, "weight", complex(self.weight))
        if self.delay_s is not None and (not _finite_real(self.delay_s) or self.delay_s < 0):
            raise ValueError("delay_s must be finite and nonnegative")
        if self.orientation is not None and not isinstance(self.orientation, PatternOrientation):
            raise ValueError("orientation must be a PatternOrientation or None")

        chain = self.signal_chain
        if chain is not None and not isinstance(chain, SignalChain):
            raise ValueError("signal_chain must be a SignalChain or None")
        if self.delay_s and chain is not None and chain.delay_s not in (0, self.delay_s):
            raise ValueError("element delay_s conflicts with signal_chain.delay_s")

    @cached_property
    def effective_signal_chain(self) -> SignalChain | None:
        """Resolve the legacy delay alias to a single delay in the chain."""
        if not self.delay_s:
            return self.signal_chain
        if self.signal_chain is None:
            return SignalChain(delay_s=self.delay_s)
        return replace(self.signal_chain, delay_s=self.delay_s)


def _planar_position(value: tuple[float, float], name: str) -> tuple[float, float]:
    message = f"{name} must contain finite planar (x, y) coordinates"
    try:
        position = tuple(value)
    except TypeError as exc:
        raise ValueError(message) from exc
    if len(position) != 2 or not all(_finite_real(coordinate) for coordinate in position):
        raise ValueError(message)
    x, y = position
    return float(x), float(y)


@dataclass(frozen=True)
class SpeakerGroup:
    """A rectangular group whose disk bounding box is centered at center_m."""

    type_id: str
    rows: int
    columns: int
    center_m: tuple[float, float] = (0.0, 0.0)
    horizontal_gap_m: float = 0.0
    vertical_gap_m: float = 0.0
    row_offsets_m: float | tuple[float, ...] = 0.0
    signal_chain: SignalChain | None = None
    element_signal_overrides: tuple[tuple[int, SignalChain], ...] = ()
    orientation: PatternOrientation | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.type_id, str) or not self.type_id.strip():
            raise ValueError("type_id must be a nonempty string")
        object.__setattr__(self, "center_m", _planar_position(self.center_m, "center_m"))
        for name, count in (("rows", self.rows), ("columns", self.columns)):
            if isinstance(count, bool) or not isinstance(count, Integral) or count < 1:
                raise ValueError(f"{name} must be a positive integer")
            object.__setattr__(self, name, int(count))
        for name, gap in (("horizontal_gap_m", self.horizontal_gap_m), ("vertical_gap_m", self.vertical_gap_m)):
            if not _finite_real(gap) or gap < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
            object.__setattr__(self, name, float(gap))
        offsets = self.row_offsets_m
        if isinstance(offsets, Real) and _finite_real(offsets):
            object.__setattr__(self, "row_offsets_m", float(offsets))
        elif (
            isinstance(offsets, (list, tuple))
            and len(offsets) == self.rows and all(_finite_real(value) for value in offsets)
        ):
            object.__setattr__(self, "row_offsets_m", tuple(float(value) for value in offsets))
        else:
            raise ValueError("row_offsets_m must be finite and scalar or contain one value per row")

        if self.signal_chain is not None and not isinstance(self.signal_chain, SignalChain):
            raise ValueError("signal_chain must be a SignalChain or None")
        if self.orientation is not None and not isinstance(self.orientation, PatternOrientation):
            raise ValueError("orientation must be a PatternOrientation or None")
        overrides = {}
        for index, chain in self.element_signal_overrides:
            if isinstance(index, bool) or not isinstance(index, Integral) or not 0 <= index < self.rows * self.columns:
                raise ValueError("signal override index must be within the group in row-major order")
            if index in overrides:
                raise ValueError("signal override indices must be unique")
            if not isinstance(chain, SignalChain):
                raise ValueError("each signal override must contain a SignalChain")
            overrides[int(index)] = chain
        object.__setattr__(self, "element_signal_overrides", tuple(sorted(overrides.items())))


@dataclass(frozen=True)
class SpeakerArray:
    """Speaker types, explicit elements, and independently placed layout groups."""

    speaker_types: tuple[SpeakerType, ...]
    elements: tuple[SpeakerElement, ...] = ()
    groups: tuple[SpeakerGroup, ...] = ()

    def __post_init__(self) -> None:
        types, elements, groups = tuple(self.speaker_types), tuple(self.elements), tuple(self.groups)
        if not types or not all(isinstance(item, SpeakerType) for item in types):
            raise ValueError("speaker_types must contain at least one SpeakerType")
        if not all(isinstance(item, SpeakerElement) for item in elements):
            raise ValueError("elements must contain only SpeakerElement objects")
        if not all(isinstance(item, SpeakerGroup) for item in groups):
            raise ValueError("groups must contain only SpeakerGroup objects")
        if not elements and not groups:
            raise ValueError("speaker_array must contain at least one element or group")
        ids = {item.type_id for item in types}
        if len(ids) != len(types):
            raise ValueError("speaker type identifiers must be unique")
        for item in (*elements, *groups):
            if item.type_id not in ids:
                raise ValueError(f"unknown speaker type: {item.type_id!r}")
        object.__setattr__(self, "speaker_types", types)
        object.__setattr__(self, "elements", elements)
        object.__setattr__(self, "groups", groups)

    @cached_property
    def flattened_elements(self) -> tuple[SpeakerElement, ...]:
        """Return explicit elements followed by each group's row-major elements."""
        if not self.groups:
            return self.elements
        diameters = {item.type_id: item.diameter_m for item in self.speaker_types}
        elements = list(self.elements)
        for group in self.groups:
            centers = generate_speaker_centers(
                group.rows, group.columns, diameters[group.type_id],
                group.horizontal_gap_m, group.vertical_gap_m, group.row_offsets_m,
            )
            overrides = dict(group.element_signal_overrides)
            elements.extend(
                SpeakerElement(
                    group.type_id, (x, y), orientation=group.orientation,
                    signal_chain=overrides.get(index, group.signal_chain),
                )
                for index, (x, y) in enumerate(centers + group.center_m)
            )
        return tuple(elements)

    def element_types(self) -> tuple[SpeakerType, ...]:
        by_id = {speaker_type.type_id: speaker_type for speaker_type in self.speaker_types}
        return tuple(by_id[element.type_id] for element in self.flattened_elements)

    def to_dict(self) -> dict:
        data = {
            "speaker_types": tuple(item.to_dict() for item in self.speaker_types),
            "elements": tuple(asdict(item) for item in self.elements),
        }
        if self.groups:
            data["groups"] = tuple(asdict(item) for item in self.groups)
        for group in data.get("groups", ()):
            if group["signal_chain"] is None:
                group.pop("signal_chain")
            if not group["element_signal_overrides"]:
                group.pop("element_signal_overrides")
            if group["orientation"] is None:
                group.pop("orientation")
        for element in data["elements"]:
            if element["signal_chain"] is None:
                element.pop("signal_chain")
            if element["orientation"] is None:
                element.pop("orientation")
            weight = element["weight"]
            element["weight"] = [weight.real, weight.imag]
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "SpeakerArray":
        if (
            not isinstance(data, dict) or "speaker_types" not in data
            or set(data) - {"speaker_types", "elements", "groups"}
        ):
            raise ValueError("speaker_array must contain speaker_types and may contain elements and groups")
        types = tuple(SpeakerType.from_dict(entry) for entry in data["speaker_types"])
        elements = []
        for entry in data.get("elements", ()):
            fields = dict(entry)
            if "weight" in fields:
                pair = fields["weight"]
                if not isinstance(pair, (list, tuple)) or len(pair) != 2 or not all(map(_finite_real, pair)):
                    raise ValueError("element weight must contain a finite [real, imaginary] pair")
                fields["weight"] = complex(*pair)
            if fields.get("signal_chain") is not None:
                fields["signal_chain"] = SignalChain.from_dict(fields["signal_chain"])
            if fields.get("orientation") is not None:
                fields["orientation"] = PatternOrientation(**fields["orientation"])
            elements.append(SpeakerElement(**fields))
        groups = []
        for entry in data.get("groups", ()):
            fields = dict(entry)
            if fields.get("signal_chain") is not None:
                fields["signal_chain"] = SignalChain.from_dict(fields["signal_chain"])
            if fields.get("orientation") is not None:
                fields["orientation"] = PatternOrientation(**fields["orientation"])
            if "element_signal_overrides" in fields:
                fields["element_signal_overrides"] = tuple(
                    (index, SignalChain.from_dict(chain)) for index, chain in fields["element_signal_overrides"]
                )
            groups.append(SpeakerGroup(**fields))
        return cls(types, tuple(elements), tuple(groups))
