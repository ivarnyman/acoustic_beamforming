"""Print the configuration and display the source and pressure diagnostics."""

from dataclasses import fields, replace
from pprint import pprint

import numpy as np
from matplotlib import pyplot as plt

from angular_spectrum import directivity_from_spectrum, iter_pressure_planes, prepare_velocity_spectrum
from config import (
    DIAPHRAGM_VELOCITY_CONVENTION,
    TIME_CONVENTION,
    SimulationConfig,
)
from simulation import (
    array_diameters, array_type_ids, build_array_grid, configured_array_window, configured_focus_point,
    create_velocity, generate_array,
)
from visualization import (
    plot_angular_directivity,
    plot_aperture,
    plot_grid_diagnostic,
    plot_pressure_cross_section,
    plot_pressure_maps,
    plot_speaker_window,
)


def main(config: SimulationConfig | None = None) -> None:
    """Simulate and plot the configured speaker array."""
    if config is None:
        config = SimulationConfig()
    print(f"Time convention: {TIME_CONVENTION}")
    print(f"Diaphragm velocity: {DIAPHRAGM_VELOCITY_CONVENTION}")
    pprint({field.name: getattr(config, field.name) for field in fields(config)}, sort_dicts=False)
    centers = generate_array(config)
    grid = build_array_grid(config, centers)
    plot_grid_diagnostic(
        grid, centers, array_diameters(config), show_indices=True, type_ids=array_type_ids(config),
    )
    velocity, profile, directivity_azimuth = create_velocity(config, centers, grid)
    if config.speaker_array is None:
        plot_aperture(
            grid.source,
            velocity,
            config.speaker_diameter_m,
            profiles=(
                replace(profile, kind="uniform"),
                replace(profile, kind="cosine"),
                replace(profile, kind="gaussian"),
            ),
        )
    else:
        used_ids = {element.type_id for element in config.speaker_array.flattened_elements}
        types = [item for item in config.speaker_array.speaker_types if item.type_id in used_ids]
        plot_aperture(
            grid.source, velocity, [item.diameter_m for item in types],
            profiles=tuple(item.aperture_profile or profile for item in types),
            profile_labels=tuple(item.name for item in types),
            frequency_hz=config.frequency_hz,
        )
    observation_z = config.observation_distance_m
    focus_z = configured_focus_point(config)[2] if config.beamforming_mode == "focus" else observation_z
    end_z = max(observation_z, focus_z)
    sweep = (
        np.linspace(0, end_z, config.z_evaluation_points)
        if config.z_evaluation_points > 1
        else np.array([end_z])
    )
    distances = np.unique(np.r_[sweep, observation_z, focus_z])
    prepared = prepare_velocity_spectrum(velocity, grid)
    planes = iter_pressure_planes(
        prepared,
        distances,
        frequency_hz=config.frequency_hz,
        sound_speed_m_s=config.sound_speed_m_s,
        air_density_kg_m3=config.air_density_kg_m3,
        include_evanescent=config.include_evanescent,
    )
    observation_index = int(np.searchsorted(distances, observation_z))
    focus_index = int(np.searchsorted(distances, focus_z))
    map_indices = tuple(dict.fromkeys((observation_index, focus_index)))
    maps = {}
    observation = grid.observation
    iy = int(np.abs(observation.y).argmin())
    ix = int(np.abs(observation.x).argmin())
    xz = np.empty((len(distances), len(observation.x)), dtype=complex)
    yz = np.empty((len(distances), len(observation.y)), dtype=complex)
    peak_pressure = 0.0
    for index, plane in enumerate(planes):
        xz[index] = plane[iy, :]
        yz[index] = plane[:, ix]
        plane_peak = float(np.abs(plane).max())
        peak_pressure = max(peak_pressure, plane_peak)
        if index in map_indices:
            maps[index] = plane.copy()
        if index == observation_index:
            print(
                f"Peak pressure at z = {observation_z:g} m "
                f"(uncalibrated model units) = {plane_peak:.6g}"
            )
    del plane
    reference = peak_pressure or 1.0
    plot_pressure_maps(
        observation, distances[list(map_indices)], [maps[index] for index in map_indices],
        reference_pressure=reference,
    )
    if len(distances) > 1:
        plot_pressure_cross_section(observation, "xz", iy, distances, xz, reference_pressure=reference)
        plot_pressure_cross_section(observation, "yz", ix, distances, yz, reference_pressure=reference)

    theta = np.linspace(-90, 90, 361)
    directivity = directivity_from_spectrum(
        prepared,
        theta,
        directivity_azimuth,
        frequency_hz=config.frequency_hz,
        sound_speed_m_s=config.sound_speed_m_s,
        air_density_kg_m3=config.air_density_kg_m3,
    )
    plot_angular_directivity(theta, directivity)
    image = plot_speaker_window(
        configured_array_window(config, len(centers)), config.array_window,
    )
    image.figure.colorbar(image, ax=image.axes, label="Amplitude multiplier")
    plt.show()


if __name__ == "__main__":
    main()
