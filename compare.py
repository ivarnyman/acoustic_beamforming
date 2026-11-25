"""Compare saved array configurations on a common pressure and directivity scale."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from scipy.signal import find_peaks

from config import load_configuration
from simulation import SimulationSession


def beam_metrics(theta_deg, magnitude):
    """Measure the dominant lobe in a signed cut and identify resolved sidelobes."""
    theta = np.asarray(theta_deg, dtype=float)
    amplitude = np.asarray(magnitude, dtype=float)
    if (
        theta.ndim != 1 or len(theta) < 3 or theta.shape != amplitude.shape
        or not np.isfinite(theta).all() or not np.isfinite(amplitude).all()
        or np.any(np.diff(theta) <= 0) or np.any(amplitude < 0)
    ):
        raise ValueError("provide matching finite angle and nonnegative magnitude vectors in increasing angle order")
    metrics = dict(peak_angle_deg=None, beamwidth_deg=None, sidelobe_level_db=None)
    peak = int(amplitude.argmax())
    if amplitude[peak] == 0:
        return metrics
    metrics["peak_angle_deg"] = float(theta[peak])

    ratio = amplitude / amplitude[peak]
    half_power = 1 / np.sqrt(2)
    left = np.flatnonzero(ratio[:peak] < half_power)
    right = np.flatnonzero(ratio[peak + 1:] < half_power) + peak + 1
    if left.size and right.size:
        i, j = left[-1], right[0]
        lower = np.interp(half_power, ratio[i:i + 2], theta[i:i + 2])
        upper = np.interp(half_power, ratio[j - 1:j + 1][::-1], theta[j - 1:j + 1][::-1])
        metrics["beamwidth_deg"] = float(upper - lower)

    minima = find_peaks(-ratio)[0]
    left = minima[minima < peak]
    right = minima[minima > peak]
    # Measure a sidelobe only on sides where the main lobe has a resolved minimum.
    sidelobes = []
    if left.size:
        sidelobes.append(ratio[:left[-1]].max())
    if right.size:
        sidelobes.append(ratio[right[0] + 1:].max())
    if sidelobes and max(sidelobes) > 0:
        metrics["sidelobe_level_db"] = float(20 * np.log10(max(sidelobes)))
    return metrics


def _reference(values, supplied):
    if supplied is None:
        return max(values) or 1.0
    if not np.isfinite(supplied) or supplied <= 0:
        raise ValueError("fixed references must be finite and positive")
    return supplied


def _relative_db(magnitude, reference):
    with np.errstate(divide="ignore"):
        return 20 * (np.log10(magnitude) - np.log10(reference))


def compare_configurations(
    paths, output_dir, *, azimuth_deg=0.0, distance_m=None,
    pressure_reference=None, directivity_reference=None, floor_db=-40.0, ceiling_db=0.0,
):
    """Run saved cases and write PNG comparisons plus a machine-readable summary."""
    paths = [Path(path) for path in paths]
    if len(paths) < 2:
        raise ValueError("at least two configurations are required")
    if not np.isfinite([floor_db, ceiling_db]).all() or floor_db >= ceiling_db:
        raise ValueError("dB display limits must be finite and increasing")

    output_dir = Path(output_dir)
    session = SimulationSession()
    cases, results = [], []
    for path in paths:
        config = load_configuration(path)
        if distance_m is not None:
            config = replace(config, observation_distance_m=distance_m)
        result = session.recalculate(config, directivity_azimuth_deg=azimuth_deg)
        metrics = beam_metrics(result.theta_deg, np.abs(result.directivity))
        cases.append({
            "configuration": str(path),
            "label": path.stem,
            "frequency_hz": config.frequency_hz,
            "distance_m": config.observation_distance_m,
            **metrics,
        })
        results.append(result)

    pressure_ref = _reference((r.pressure.magnitude.max() for r in results), pressure_reference)
    directivity_ref = _reference((np.abs(r.directivity).max() for r in results), directivity_reference)

    for case, result in zip(cases, results):
        for key, magnitude, reference in (
            ("peak_pressure_level_db", result.pressure.magnitude.max(), pressure_ref),
            ("peak_directivity_level_db", np.abs(result.directivity).max(), directivity_ref),
        ):
            case[key] = float(_relative_db(magnitude, reference)) if magnitude else None
    output_dir.mkdir(parents=True, exist_ok=True)
    db_limits = floor_db, ceiling_db
    _plot_patterns(output_dir, cases, results, directivity_ref, db_limits, azimuth_deg)
    _plot_metrics(output_dir, cases)
    _plot_fields(output_dir, cases, results, pressure_ref, db_limits)
    summary = {
        "azimuth_deg": azimuth_deg,
        "pressure_reference_model_units": float(pressure_ref),
        "directivity_reference_model_units_m": float(directivity_ref),
        "relative_level_display_limits_db": list(db_limits),
        "cases": cases,
    }
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return summary


def _plot_patterns(output_dir, cases, results, reference, db_limits, azimuth_deg):
    fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
    try:
        for case, result in zip(cases, results):
            level = _relative_db(np.abs(result.directivity), reference)
            ax.plot(result.theta_deg, np.clip(level, *db_limits), label=case["label"])
        ax.set(
            xlabel="Signed polar angle from broadside (°)", ylabel="Relative pressure level (dB)",
            title=f"Far-field directivity at azimuth {azimuth_deg:g}°", xlim=(-90, 90), ylim=db_limits,
        )
        ax.axhline(0, color="0.65", linewidth=0.8)
        ax.grid(alpha=0.25)
        ax.legend()
        fig.savefig(output_dir / "directivity.png", dpi=150)
    finally:
        plt.close(fig)


def _plot_metrics(output_dir, cases):
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), layout="constrained")
    labels = [case["label"] for case in cases]
    try:
        for ax, key, title, ylabel in (
            (axes[0], "beamwidth_deg", "Half-power beamwidth", "Width (°)"),
            (axes[1], "sidelobe_level_db", "Peak sidelobe", "Level relative to own peak (dB)"),
        ):
            for index, case in enumerate(cases):
                value = case[key]
                if value is None:
                    ax.text(index, 0.03, "Unavailable", ha="center", transform=ax.get_xaxis_transform())
                else:
                    ax.bar(index, value, color=f"C{index % 10}")
            ax.set(xticks=range(len(labels)), xticklabels=labels, title=title, ylabel=ylabel)
            ax.tick_params(axis="x", labelrotation=20)
        fig.savefig(output_dir / "beam_metrics.png", dpi=150)
    finally:
        plt.close(fig)


def _plot_fields(output_dir, cases, results, reference, db_limits):
    fig, axes = plt.subplots(1, len(cases), figsize=(4 * len(cases), 4), squeeze=False, layout="constrained")
    try:
        extents = np.array([result.grid.observation.extent for result in results])
        xlim = extents[:, 0].min(), extents[:, 1].max()
        ylim = extents[:, 2].min(), extents[:, 3].max()
        norm = Normalize(*db_limits, clip=True)
        for ax, case, result, extent in zip(axes[0], cases, results, extents):
            level = _relative_db(result.pressure.magnitude[0], reference)
            ax.imshow(
                np.clip(level, *db_limits), extent=extent, origin="lower",
                interpolation="nearest", cmap="magma", norm=norm,
            )
            ax.set(
                xlabel="x (m)", ylabel="y (m)", xlim=xlim, ylim=ylim,
                title=f"{case['label']} · z = {case['distance_m']:g} m",
            )
        fig.colorbar(ScalarMappable(norm=norm, cmap="magma"), ax=axes[0], label="Relative pressure level (dB)")
        fig.savefig(output_dir / "pressure_fields.png", dpi=150)
    finally:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configurations", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--azimuth-deg", type=float, default=0.0)
    parser.add_argument("--distance-m", type=float)
    parser.add_argument("--pressure-reference", type=float)
    parser.add_argument("--directivity-reference", type=float)
    parser.add_argument("--floor-db", type=float, default=-40.0)
    parser.add_argument("--ceiling-db", type=float, default=0.0)
    args = parser.parse_args()
    try:
        compare_configurations(
            args.configurations, args.output, azimuth_deg=args.azimuth_deg, distance_m=args.distance_m,
            pressure_reference=args.pressure_reference, directivity_reference=args.directivity_reference,
            floor_db=args.floor_db, ceiling_db=args.ceiling_db,
        )
    except (OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
