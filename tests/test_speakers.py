import json
from math import tau
from dataclasses import replace

import numpy as np
import pytest

from aperture import RadialProfile, rasterize_aperture
from config import SimulationConfig, load_configuration, save_configuration
from simulation import SimulationSession, build_array_grid, generate_array
from speakers import SpeakerArray, SpeakerElement, SpeakerGroup, SpeakerType


def _test_config(**kwargs):
    return SimulationConfig(
        rows=2, columns=3, speaker_diameter_m=0.1, horizontal_gap_m=0.02,
        vertical_gap_m=0.03, row_stagger=0.25, grid_spacing_m=0.0125,
        padding_factor=4, frequency_hz=1200, observation_distance_m=0.4,
        observation_shape=(9, 11), **kwargs,
    )


def _mixed_array():
    return SpeakerArray(
        (
            SpeakerType.generic(0.1),
            SpeakerType(
                "mid", "Example midrange", 0.16, RadialProfile("gaussian", gaussian_sigma_fraction=0.3),
                frequency_response_metadata={"source": "vendor", "file": "mid-response.csv"},
                directivity_metadata={"source": "lab", "angles_deg": [0, 15, 30]},
            ),
        ),
        (
            SpeakerElement("generic", (-0.18, 0), weight=1j),
            SpeakerElement("mid", (0, 0.03), weight=0.8-0.2j, delay_s=0.001),
            SpeakerElement("generic", (0.18, 0), weight=0.5),
        ),
    )


def test_typed_configuration_json_round_trip_and_shared_type(tmp_path):
    config = _test_config(speaker_array=_mixed_array())
    path = tmp_path / "typed.json"
    save_configuration(path, config)
    saved = json.loads(path.read_text())["speaker_array"]
    assert "groups" not in saved
    assert saved["elements"][1]["weight"] == [0.8, -0.2]
    assert saved["elements"][1]["delay_s"] == 0.001
    assert saved["speaker_types"][0]["response_model"] == "idealized_flat"
    reloaded = load_configuration(path)
    assert reloaded == config
    first, middle, last = reloaded.speaker_array.element_types()
    assert first is last is reloaded.speaker_array.speaker_types[0]
    assert middle is reloaded.speaker_array.speaker_types[1]
    assert reloaded.speaker_array.elements[0].weight != reloaded.speaker_array.elements[2].weight


def test_legacy_json_keeps_its_schema_and_uses_generated_speakers(tmp_path):
    path = tmp_path / "legacy.json"
    config = _test_config()
    path.write_text('{"rows": 2, "columns": 3, "speaker_diameter_m": 0.1}')
    assert load_configuration(path).speaker_array is None
    save_configuration(path, config)
    assert "speaker_array" not in json.loads(path.read_text())
    assert load_configuration(path) == config


@pytest.mark.parametrize("normalization", ["peak", "equal_energy"])
def test_generic_type_reproduces_legacy_geometry_velocity_and_pressure(normalization):
    legacy = _test_config(
        amplitude_profile="cosine", aperture_normalization=normalization,
        beamforming_mode="steer", steering_azimuth_deg=30, steering_elevation_deg=70,
        array_window="hamming", weight_normalization="power",
        speaker_weights=(1, 0.4 + 0.2j, 0.5j, 1.2, -0.2j, 0.6),
    )
    centers = generate_array(legacy)
    typed = replace(legacy, speaker_array=SpeakerArray(
        (SpeakerType.generic(legacy.speaker_diameter_m),),
        tuple(SpeakerElement("generic", tuple(center)) for center in centers),
    ))
    np.testing.assert_array_equal(generate_array(typed), centers)
    base_result = SimulationSession().recalculate(legacy)
    typed_result = SimulationSession().recalculate(typed)
    assert typed_result.grid.aperture_extent == base_result.grid.aperture_extent
    np.testing.assert_array_equal(typed_result.velocity, base_result.velocity)
    np.testing.assert_array_equal(typed_result.pressure.pressure, base_result.pressure.pressure)
    np.testing.assert_array_equal(typed_result.directivity, base_result.directivity)


def test_mixed_diaphragms_use_per_type_radii_profiles_and_instance_weights():
    array = _mixed_array()
    config = _test_config(speaker_array=array)
    session = SimulationSession()
    result = session.recalculate(config)
    centers = generate_array(config)
    np.testing.assert_array_equal(centers, [element.position_m for element in array.elements])
    assert result.grid.aperture_extent == pytest.approx((-0.23, 0.23, -0.05, 0.11))
    expected = np.zeros(result.grid.source.shape, dtype=complex)
    for center, element, speaker_type in zip(centers, array.elements, array.element_types()):
        expected += rasterize_aperture(
            result.grid.source, [center], speaker_type.diameter_m,
            profile=speaker_type.aperture_profile or RadialProfile(),
            weights=[element.weight * np.exp(1j * tau * (element.delay_s or 0) * config.frequency_hz)],
        )
    np.testing.assert_allclose(result.velocity, expected, atol=1e-14)
    assert np.max(np.abs(result.pressure.pressure)) > 0

    different_type = replace(array.speaker_types[1], aperture_profile=RadialProfile("uniform"))
    edited = replace(config, speaker_array=replace(array, speaker_types=(array.speaker_types[0], different_type)))
    updated = session.recalculate(edited)
    assert updated.grid is result.grid
    assert not np.array_equal(updated.velocity, result.velocity)


def test_type_references_and_mixed_diameter_overlap_are_validated():
    small = SpeakerType.generic(0.1)
    large = SpeakerType("large", "Large", 0.3)
    with pytest.raises(ValueError, match="identifiers must be unique"):
        SpeakerArray((small, small), (SpeakerElement("generic", (0, 0)),))
    with pytest.raises(ValueError, match="unknown speaker type"):
        SpeakerArray((small,), (SpeakerElement("large", (0, 0)),))

    array = SpeakerArray(
        (small, large),
        (SpeakerElement("generic", (0, 0)), SpeakerElement("large", (0.19, 0))),
    )
    with pytest.raises(ValueError, match="overlap"):
        generate_array(_test_config(speaker_array=array))
    tangent = replace(array, elements=(array.elements[0], SpeakerElement("large", (0.2, 0))))
    config = _test_config(speaker_array=tangent)
    assert build_array_grid(config, generate_array(config)).aperture_extent == pytest.approx(
        (-0.05, 0.35, -0.15, 0.15),
    )


def test_mixed_groups_preserve_positions_order_and_disk_edge_bounds(tmp_path):
    array = SpeakerArray(
        (SpeakerType.generic(0.1, type_id="small"), SpeakerType("large", "Large", 0.2)),
        elements=(SpeakerElement("small", (0.05, 0.45), weight=1j),),
        groups=(
            SpeakerGroup("small", 2, 2, (-0.4, 0.1), 0.02, 0.04, 0.03),
            SpeakerGroup("large", 3, 1, (0.45, -0.05), vertical_gap_m=0.05, row_offsets_m=(0, 0.12, -0.04)),
        ),
    )
    config = _test_config(speaker_array=array)
    path = tmp_path / "groups.json"
    save_configuration(path, config)
    data = json.loads(path.read_text())["speaker_array"]
    assert len(data["groups"]) == 2
    assert len(data["elements"]) == 1
    assert load_configuration(path) == config

    centers = generate_array(config)
    np.testing.assert_allclose(centers, [
        (0.05, 0.45),
        (-0.475, 0.03), (-0.355, 0.03), (-0.445, 0.17), (-0.325, 0.17),
        (0.41, -0.30), (0.53, -0.05), (0.37, 0.20),
    ])
    assert [item.type_id for item in array.flattened_elements] == ["small"] * 5 + ["large"] * 3
    assert array.flattened_elements[0].weight == 1j
    assert all(item.weight == 1 for item in array.flattened_elements[1:])

    grouped = SimulationSession().recalculate(load_configuration(path))
    explicit = replace(config, speaker_array=SpeakerArray(array.speaker_types, array.flattened_elements))
    flattened = SimulationSession().recalculate(explicit)
    assert grouped.grid.aperture_extent == pytest.approx((-0.525, 0.63, -0.4, 0.5))
    np.testing.assert_array_equal(grouped.velocity, flattened.velocity)
    np.testing.assert_array_equal(grouped.pressure.pressure, flattened.pressure.pressure)
    np.testing.assert_array_equal(grouped.directivity, flattened.directivity)


@pytest.mark.parametrize("explicit_small", [False, True])
def test_collisions_across_group_boundaries_use_actual_radii(explicit_small):
    types = (SpeakerType.generic(0.1, type_id="small"), SpeakerType("large", "Large", 0.3))
    small_groups = () if explicit_small else (SpeakerGroup("small", 1, 1),)
    elements = (SpeakerElement("small", (0, 0)),) if explicit_small else ()
    large = SpeakerGroup("large", 1, 1, center_m=(0.19, 0))
    array = SpeakerArray(types, elements, (*small_groups, large))
    with pytest.raises(ValueError, match="speaker disks 0 and 1 overlap"):
        generate_array(_test_config(speaker_array=array))

    tangent = replace(array, groups=(*small_groups, replace(large, center_m=(0.20, 0))))
    centers = generate_array(_test_config(speaker_array=tangent))
    np.testing.assert_allclose(centers, [(0, 0), (0.20, 0)])


def test_group_only_layout_matches_legacy_generated_layout():
    legacy = _test_config()
    group = SpeakerGroup(
        "generic", legacy.rows, legacy.columns,
        horizontal_gap_m=legacy.horizontal_gap_m, vertical_gap_m=legacy.vertical_gap_m,
        row_offsets_m=legacy.row_stagger * (legacy.speaker_diameter_m + legacy.horizontal_gap_m),
    )
    array = SpeakerArray((SpeakerType.generic(legacy.speaker_diameter_m),), groups=(group,))
    np.testing.assert_array_equal(generate_array(replace(legacy, speaker_array=array)), generate_array(legacy))


def test_group_configuration_validates_offsets_and_type_reference():
    with pytest.raises(ValueError, match="one value per row"):
        SpeakerGroup("generic", 2, 1, row_offsets_m=(0.03,))
    with pytest.raises(ValueError, match="unknown speaker type"):
        SpeakerArray((SpeakerType.generic(0.1),), groups=(SpeakerGroup("unknown", 1, 1),))
