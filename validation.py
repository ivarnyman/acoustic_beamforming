"""Analytical piston checks and reproducible performance measurements."""

import argparse
import json
import tracemalloc
from math import pi, tau
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
from scipy.special import j1

from angular_spectrum import directivity_from_spectrum, prepare_velocity_spectrum, pressure_from_spectrum
from aperture import rasterize_aperture
from grid import XYGrid, build_grid


def piston_on_axis(z_m, radius_m, *, frequency_hz, sound_speed_m_s, air_density_kg_m3, velocity=1):
    """Exact complex pressure on the axis of a uniform baffled piston."""
    z = np.asarray(z_m, dtype=float)
    k = tau * frequency_hz / sound_speed_m_s
    path_difference = radius_m**2 / (np.hypot(z, radius_m) + z)
    return -air_density_kg_m3 * sound_speed_m_s * velocity * np.exp(1j * k * z) * np.expm1(
        1j * k * path_difference,
    )


def piston_directivity(theta_deg, radius_m, *, frequency_hz, sound_speed_m_s, air_density_kg_m3, velocity=1):
    """Exact far-field coefficient D in p = D exp(i k R) / R."""
    ka = tau * frequency_hz * radius_m / sound_speed_m_s
    x = ka * np.sin(np.deg2rad(theta_deg))
    shape = np.divide(2 * j1(x), x, out=np.ones_like(x), where=x != 0)
    return -1j * air_density_kg_m3 * frequency_hz * velocity * pi * radius_m**2 * shape


def piston_case(cells, padding, subsamples, distances, *, include_evanescent=True):
    """Evaluate a 0.2 m piston at 720 Hz on aligned on-axis samples."""
    diameter = 0.2
    spacing = np.nextafter(diameter / cells, np.inf)
    centers = [(0, 0)]
    grid = build_grid(
        centers, diameter, dx=spacing, dy=spacing, padding_factor=padding, observation_shape=(1, 1),
    )
    velocity = rasterize_aperture(grid.source, centers, diameter, subsamples=subsamples)
    prepared = prepare_velocity_spectrum(velocity, grid)
    medium = dict(frequency_hz=720, sound_speed_m_s=340, air_density_kg_m3=1.2)
    result = pressure_from_spectrum(prepared, distances, **medium, include_evanescent=include_evanescent)
    exact = piston_on_axis(distances, diameter / 2, **medium)
    theta = np.linspace(-90, 90, 361)
    # A higher ka exercises a null and the first sidelobes in the same aperture.
    far_medium = {**medium, "frequency_hz": 2800}
    actual_pattern = directivity_from_spectrum(prepared, theta, **far_medium)
    expected_pattern = piston_directivity(theta, diameter / 2, **far_medium)
    area = np.sum(velocity.real) * spacing**2
    return {
        "dx_m": float(spacing),
        "fft_shape": list(grid.computational.shape),
        "padding": padding,
        "subsamples": subsamples,
        "relative_area_error": float(abs(area / (pi * (diameter / 2)**2) - 1)),
        "on_axis_relative_error": (np.abs(result.pressure[:, 0, 0] - exact) / np.abs(exact)).tolist(),
        "far_field_max_complex_error_over_broadside": float(
            np.max(np.abs(actual_pattern - expected_pattern)) / np.max(np.abs(expected_pattern)),
        ),
    }


def mode_checks():
    """Isolate the exact grazing cutoff and exponential evanescent decay."""
    grid = XYGrid(np.arange(8) / 8, np.arange(5) / 5, 1 / 8, 1 / 5)
    wave = np.broadcast_to((-1.0)**np.arange(8), grid.shape)
    prepared = prepare_velocity_spectrum(wave, grid)
    speed = 340
    exact = pressure_from_spectrum(prepared, 0, frequency_hz=4 * speed, sound_speed_m_s=speed)
    nearby = pressure_from_spectrum(
        prepared, 0, frequency_hz=(4 + 1e-6) * speed, sound_speed_m_s=speed,
    )
    medium = dict(frequency_hz=2 * speed, sound_speed_m_s=speed)
    z = [0, 0.1, 0.3]
    decaying = pressure_from_spectrum(prepared, z, **medium).magnitude[:, 0, 0]
    filtered = pressure_from_spectrum(prepared, z, **medium, include_evanescent=False)
    return {
        "exact_grazing_peak": float(exact.magnitude.max()),
        "near_grazing_gain_over_rho_c": float(nearby.magnitude.max() / (1.21 * speed)),
        "evanescent_distances_m": z,
        "evanescent_ratios": (decaying / decaying[0]).tolist(),
        "analytical_ratios": np.exp(-tau * np.sqrt(4**2 - 2**2) * np.array(z)).tolist(),
        "filtered_peak": float(filtered.magnitude.max()),
    }


def _profile(job, repeat):
    job()
    times = []
    for _ in range(repeat):
        start = perf_counter()
        job()
        times.append(perf_counter() - start)
    tracemalloc.start()
    try:
        job()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return {"median_ms": 1000 * median(times), "peak_traced_mib": peak / 2**20}


def performance_report(repeat=7):
    """Measure warmed stage timings and incremental traced allocation peaks."""
    grid = build_grid([(0, 0)], 0.2, dx=0.002, dy=0.002, padding_factor=8, fast_fft=True)
    x, y = np.meshgrid(grid.source.x, grid.source.y)
    velocity = (np.hypot(x, y) < 0.1).astype(complex)
    prepared = prepare_velocity_spectrum(velocity, grid)
    angles = np.linspace(-90, 90, 721)
    return {
        "fft_shape": list(grid.computational.shape),
        "repeat": repeat,
        "rasterize_subsamples_4": _profile(
            lambda: rasterize_aperture(grid.source, [(0, 0)], 0.2), repeat,
        ),
        "prepare_velocity_fft": _profile(lambda: prepare_velocity_spectrum(velocity, grid), repeat),
        "pressure_four_planes": _profile(
            lambda: pressure_from_spectrum(prepared, [0.03, 0.1, 0.3, 1], frequency_hz=1600), repeat,
        ),
        "directivity_721_directions": _profile(
            lambda: directivity_from_spectrum(prepared, angles, frequency_hz=1600), repeat,
        ),
    }


def validation_report():
    distances = [0, 0.03, 0.12, 0.35, 0.8]
    far_distances = np.array([0.12, 0.35, 0.8, 4.0])
    medium = dict(frequency_hz=720, sound_speed_m_s=340, air_density_kg_m3=1.2)
    d0 = piston_directivity(0, 0.1, **medium)
    far_pressure = d0 * np.exp(1j * tau * 720 / 340 * far_distances) / far_distances
    exact = piston_on_axis(far_distances, 0.1, **medium)
    return {
        "diameter_m": 0.2,
        "on_axis_frequency_hz": 720,
        "directivity_frequency_hz": 2800,
        "distances_m": distances,
        "padding_sweep": [piston_case(21, pad, 8, distances) for pad in (2, 6, 12, 24, 36, 40, 56)],
        "spacing_sweep": [piston_case(cells, 12, 8, distances) for cells in (9, 21, 41)],
        "boundary_sweep": [piston_case(21, 12, n, distances) for n in (1, 4, 8, 16)],
        "evanescent_omitted": piston_case(21, 56, 8, distances, include_evanescent=False),
        "axial_far_field_limit": {
            "distance_m": far_distances.tolist(),
            "relative_error": (np.abs(far_pressure - exact) / np.abs(exact)).tolist(),
        },
        "spectral_modes": mode_checks(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Write the JSON report to this path")
    parser.add_argument("--repeat", type=int, default=7, help="Timed repeats per performance stage")
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be positive")
    report = json.dumps({**validation_report(), "performance": performance_report(args.repeat)}, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + "\n", encoding="utf-8")
    else:
        print(report)


if __name__ == "__main__":
    main()
