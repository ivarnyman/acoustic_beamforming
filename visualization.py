"""Matplotlib plots of the array and simulated fields."""

from collections.abc import Sequence
from numbers import Integral
from typing import Literal

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Circle, Rectangle
from numpy.typing import ArrayLike

from angular_spectrum import PressureResult
from aperture import ApertureProfile, CallableProfile, rasterize_aperture
from frequency_analysis import (
    SpectralWeighting, broadband_mean_square_pressure, frequency_bin_order, point_pressure_responses,
)
from frequency_response import FrequencyResponse
from geometry import speaker_radii, validate_speaker_centers
from grid import SimulationGrid, XYGrid, build_grid
from signal_chain import SignalChain
from simulation import FrequencyDirectivityResult, FrequencySweepResult


def plot_speaker_geometry(
    centers: ArrayLike,
    diameter: ArrayLike,
    *,
    show_indices: bool = False,
    type_ids: Sequence[str] | None = None,
    show_type_ids: bool = False,
    ax: Axes | None = None,
) -> Axes:
    """Draw circular layouts with optional per-type colors and element labels."""
    points = validate_speaker_centers(centers, diameter)
    colors = {}
    if type_ids is not None:
        if len(type_ids) != len(points) or not all(isinstance(item, str) and item for item in type_ids):
            raise ValueError("type_ids must contain one nonempty ID per speaker")
        unique_ids = tuple(dict.fromkeys(type_ids))
        if len(unique_ids) <= 20:
            paired_colors = plt.get_cmap("tab20").colors
            palette = (*paired_colors[::2], *paired_colors[1::2])
        else:
            palette = plt.get_cmap("hsv")(np.arange(len(unique_ids)) / len(unique_ids))
        colors = dict(zip(unique_ids, palette))
    if ax is None:
        _, ax = plt.subplots()
    labeled_types = set()
    for index, ((x, y), radius) in enumerate(zip(points, speaker_radii(diameter, len(points)))):
        patch = Circle((x, y), radius=radius, fill=False)
        labels = [str(index)] if show_indices else []
        if type_ids is not None:
            type_id = type_ids[index]
            patch.set_edgecolor(colors[type_id])
            if type_id not in labeled_types:
                patch.set_label(type_id)
                labeled_types.add(type_id)
            if show_type_ids:
                labels.append(type_id)
        ax.add_patch(patch)
        if labels:
            ax.text(x, y, "\n".join(labels), ha="center", va="center")
    ax.set(xlabel="x (m)", ylabel="y (m)", aspect="equal")
    ax.autoscale_view()
    if colors:
        ax.legend()
    return ax


def plot_speaker_window(
    amplitudes: np.ndarray, name: str, *, ax: Axes | None = None,
) -> AxesImage:
    """Plot the selected window on the discrete row and column coordinates."""
    if ax is None:
        _, ax = plt.subplots(layout="constrained")
    image = ax.imshow(
        amplitudes, origin="lower", interpolation="nearest", cmap="viridis", vmin=0, vmax=1,
    )
    ax.set(title=f"Speaker window · {name.capitalize()}", xlabel="Column index", ylabel="Row index")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    return image


def plot_grid_diagnostic(
    grid: SimulationGrid,
    centers: ArrayLike,
    diameter: ArrayLike,
    *,
    show_indices: bool = False,
    type_ids: Sequence[str] | None = None,
    show_type_ids: bool = False,
    ax: Axes | None = None,
) -> Axes:
    """Overlay source sample lines, speaker disks, and domain outlines."""
    ax = plot_speaker_geometry(
        centers, diameter, show_indices=show_indices, type_ids=type_ids,
        show_type_ids=show_type_ids, ax=ax,
    )
    xmin, xmax, ymin, ymax = grid.source.extent
    ax.vlines(grid.source.x, ymin, ymax, colors="0.8", linewidths=0.4, zorder=0)
    ax.hlines(grid.source.y, xmin, xmax, colors="0.8", linewidths=0.4, zorder=0)
    for extent, label, color, style in (
        (grid.computational.extent, "Computational domain", "tab:blue", "-"),
        (grid.source.extent, "Source grid", "tab:gray", "-"),
        (grid.aperture_extent, "Physical aperture bounds", "tab:orange", ":"),
        (grid.observation.extent, "Observation grid", "tab:green", "--"),
    ):
        left, right, bottom, top = extent
        ax.add_patch(Rectangle(
            (left, bottom), right - left, top - bottom,
            fill=False, edgecolor=color, linestyle=style, label=label,
        ))
    ax.autoscale_view()
    ax.legend()
    return ax


def plot_aperture(
    grid: XYGrid,
    velocity: ArrayLike,
    diameter: ArrayLike,
    *,
    profiles: tuple[ApertureProfile, ...],
    profile_labels: tuple[str, ...] | None = None,
    frequency_hz: float | None = None,
) -> Figure:
    """Plot peak-normalized radial profiles and sampled velocity amplitude and phase."""
    field = np.asarray(velocity)
    if field.shape != grid.shape:
        raise ValueError("velocity shape must match the plotted grid")
    radii = speaker_radii(diameter, len(profiles))
    labels = _profile_labels(profiles, profile_labels)
    fig, (radial_ax, amplitude_ax, phase_ax) = plt.subplots(
        1, 3, figsize=(12, 4), layout="constrained",
    )
    r = np.linspace(0.0, 1.0, 201)
    has_callable = any(isinstance(profile, CallableProfile) for profile in profiles)
    for profile, radius, label in zip(profiles, radii, labels):
        if isinstance(profile, CallableProfile):
            values = profile.evaluate(r, 0.0, frequency_hz)
        else:
            values = profile.evaluate(r, frequency_hz)
        radial_ax.plot(r * radius, np.abs(values), label=label)
    if has_callable:
        radial_ax.set(xlabel="Distance along +x (m)", ylabel="Amplitude", title="Profile cuts")
    else:
        radial_ax.set(xlabel="Radius (m)", ylabel="Peak-normalized amplitude", title="Radial profiles")
    radial_ax.legend()
    _plot_velocity_maps(fig, amplitude_ax, phase_ax, grid, field)
    return fig


def _profile_labels(profiles, labels):
    if labels is None:
        return tuple(p.label if isinstance(p, CallableProfile) else p.kind for p in profiles)
    if len(labels) != len(profiles):
        raise ValueError("profile_labels must match profiles")
    return labels


def plot_diaphragm_profiles(
    profiles: Sequence[ApertureProfile],
    diameter: ArrayLike,
    frequencies_hz: Sequence[float],
    *,
    profile_labels: Sequence[str] | None = None,
    normalization: Literal["peak", "equal_energy"] = "peak",
    samples_across: int = 81,
    subsamples: int = 4,
) -> Figure:
    """Compare isolated amplitude and phase maps without drive or response gains."""
    frequencies = np.asarray(frequencies_hz)
    if (
        frequencies.ndim != 1 or not frequencies.size or frequencies.dtype.kind not in "iuf"
        or not np.isfinite(frequencies).all() or (frequencies <= 0).any()
    ):
        raise ValueError("frequencies_hz must be a nonempty sequence of finite positive frequencies")
    if not len(profiles):
        raise ValueError("select at least one aperture profile")
    if isinstance(samples_across, bool) or not isinstance(samples_across, Integral) or samples_across < 3:
        raise ValueError("samples_across must be an integer of at least 3")
    labels = _profile_labels(profiles, profile_labels)
    diameters = 2 * speaker_radii(diameter, len(profiles))
    fields = []
    for profile, width, label in zip(profiles, diameters, labels):
        spacing = width / samples_across
        grid = build_grid([(0, 0)], width, dx=spacing, dy=spacing).source
        for frequency in frequencies:
            field = rasterize_aperture(
                grid, [(0, 0)], width, profile=profile, frequency_hz=frequency,
                normalization=normalization, subsamples=subsamples,
            )
            fields.append((label, frequency, grid, field))

    vmax = max(float(np.abs(field).max()) for _, _, _, field in fields) or 1.0
    fig, axes = plt.subplots(
        len(fields), 2, squeeze=False, figsize=(8, 3.4 * len(fields)), layout="constrained",
    )
    for (amplitude_ax, phase_ax), (label, frequency, grid, field) in zip(axes, fields):
        _plot_velocity_maps(fig, amplitude_ax, phase_ax, grid, field, vmax=vmax)
        title = f"{label} · {frequency:g} Hz"
        amplitude_ax.set_title(f"{title} amplitude")
        phase_ax.set_title(f"{title} phase")
    return fig


def _plot_velocity_maps(fig, amplitude_ax, phase_ax, grid, field, *, vmax=None):
    magnitude = np.abs(field)
    phase = np.ma.masked_where(magnitude == 0, np.angle(field))
    amplitude_image = amplitude_ax.imshow(
        magnitude, extent=grid.extent, origin="lower", interpolation="nearest",
        vmin=0, vmax=vmax, cmap="viridis",
    )
    phase_image = phase_ax.imshow(
        phase, extent=grid.extent, origin="lower", interpolation="nearest",
        vmin=-np.pi, vmax=np.pi, cmap="twilight",
    )
    amplitude_ax.set(title="Normal velocity amplitude", xlabel="x (m)", ylabel="y (m)", aspect="equal")
    phase_ax.set(title="Normal velocity phase", xlabel="x (m)", ylabel="y (m)", aspect="equal")
    fig.colorbar(amplitude_image, ax=amplitude_ax, label="Relative amplitude")
    fig.colorbar(phase_image, ax=phase_ax, label="Phase (rad)")


def plot_transfer_functions(
    chain: SignalChain,
    frequencies_hz: ArrayLike,
    *,
    frequency_response: FrequencyResponse | None = None,
) -> Figure:
    """Plot the drive chain and optional diaphragm response on a common frequency axis."""
    frequencies = np.asarray(frequencies_hz)
    if (
        frequencies.ndim != 1 or not frequencies.size or frequencies.dtype.kind not in "iuf"
        or not np.isfinite(frequencies).all() or (frequencies <= 0).any()
        or not (frequencies[1:] > frequencies[:-1]).all()
    ):
        raise ValueError("plot frequencies must be finite, positive, and strictly increasing")
    combined = chain.evaluate(frequencies)
    curves = []
    if chain.crossover is not None:
        curves.extend((
            ("Gain, polarity, phase and delay", chain.gain_and_delay(frequencies)),
            ("Crossover", chain.crossover.evaluate(frequencies)),
        ))
    curves.append(("Signal chain C(f)", combined))
    if frequency_response is not None:
        response = frequency_response.evaluate(frequencies)
        curves.extend((("Diaphragm H(f)", response), ("Combined C(f) × H(f)", combined * response)))

    fig, (magnitude_ax, phase_ax) = plt.subplots(
        2, 1, sharex=True, figsize=(8, 6), layout="constrained",
    )
    for label, response in curves:
        magnitude = np.abs(response)
        with np.errstate(divide="ignore"):
            level = np.maximum(20 * np.log10(magnitude), -120)
        phase = np.ma.masked_where(magnitude == 0, np.angle(response))
        magnitude_ax.semilogx(frequencies, level, label=label)
        phase_ax.semilogx(frequencies, phase, label=label)
    magnitude_ax.set(ylabel="Magnitude (dB)", title="Drive transfer functions")
    phase_ax.set(xlabel="Frequency (Hz)", ylabel="Wrapped phase (rad)", ylim=(-np.pi, np.pi))
    for ax in (magnitude_ax, phase_ax):
        ax.grid(which="both", alpha=0.25)
    magnitude_ax.legend()
    return fig


def plot_pressure_planes(
    result: PressureResult,
    *,
    plane_indices: Sequence[int] | None = None,
    reference_pressure: float | None = None,
    magnitude_max: float = 1.0,
    level_limits_db: tuple[float, float] = (-40.0, 0.0),
) -> Figure:
    """Plot selected xy planes with a common pressure reference and color scales."""
    indices = tuple(range(len(result.distances_m))) if plane_indices is None else tuple(plane_indices)
    if not indices:
        raise ValueError("select at least one pressure plane")
    reference = _pressure_reference(result.magnitude, reference_pressure)
    return plot_pressure_maps(
        result.grid, result.distances_m[list(indices)], [result.pressure[index] for index in indices],
        reference_pressure=reference, magnitude_max=magnitude_max, level_limits_db=level_limits_db,
    )


def plot_pressure_maps(
    grid: XYGrid, distances_m: Sequence[float], fields: Sequence[np.ndarray], *,
    reference_pressure: float,
    magnitude_max: float = 1.0,
    level_limits_db: tuple[float, float] = (-40.0, 0.0),
) -> Figure:
    """Plot stored xy maps using a shared reference."""
    panels = [
        (field, grid.x, grid.y, "x (m)", "y (m)", f"z = {z:g} m", grid.extent)
        for z, field in zip(distances_m, fields)
    ]
    return _plot_pressure_panels(panels, reference_pressure, magnitude_max, level_limits_db)


def plot_pressure_slice(
    result: PressureResult,
    plane: Literal["xz", "yz"],
    coordinate_m: float = 0.0,
    *,
    reference_pressure: float | None = None,
    magnitude_max: float = 1.0,
    level_limits_db: tuple[float, float] = (-40.0, 0.0),
) -> Figure:
    """Extract a slice from propagated xy planes at the nearest transverse sample."""
    if plane == "xz":
        fixed = result.grid.y
    elif plane == "yz":
        fixed = result.grid.x
    else:
        raise ValueError("plane must be xz or yz")
    fixed_step = result.grid.dy if plane == "xz" else result.grid.dx
    if not fixed[0] - fixed_step / 2 <= coordinate_m <= fixed[-1] + fixed_step / 2:
        raise ValueError("coordinate_m must lie within the observation grid")
    index = int(np.abs(fixed - coordinate_m).argmin())
    # Distances can arrive out of order or repeat. Sort and retain one of each.
    z, order = np.unique(result.distances_m, return_index=True)
    if len(z) < 2:
        raise ValueError("a slice requires at least two distinct observation distances")
    pressure = result.pressure[order, index, :] if plane == "xz" else result.pressure[order, :, index]
    return plot_pressure_cross_section(
        result.grid, plane, index, z, pressure,
        reference_pressure=_pressure_reference(result.magnitude, reference_pressure),
        magnitude_max=magnitude_max, level_limits_db=level_limits_db,
    )


def plot_pressure_cross_section(
    grid: XYGrid, plane: Literal["xz", "yz"], index: int, z: np.ndarray, pressure: np.ndarray,
    *,
    reference_pressure: float,
    magnitude_max: float = 1.0,
    level_limits_db: tuple[float, float] = (-40.0, 0.0),
) -> Figure:
    """Plot a cross-section assembled from propagated pressure planes."""
    if plane == "xz":
        fixed, horizontal, step = grid.y, grid.x, grid.dx
        left, right = grid.extent[:2]
    else:
        fixed, horizontal, step = grid.x, grid.y, grid.dy
        left, right = grid.extent[2:]
    x_edges = np.r_[horizontal - step / 2, right]
    x_edges[0] = left
    z_edges = np.r_[max(0, z[0] - (z[1] - z[0]) / 2),
                    (z[1:] + z[:-1]) / 2, z[-1] + (z[-1] - z[-2]) / 2]
    fixed_axis = "y" if plane == "xz" else "x"
    title = f"{plane} slice at {fixed_axis} = {fixed[index]:g} m"
    panels = [(pressure, x_edges, z_edges, f"{plane[0]} (m)", "z (m)", title, None)]
    return _plot_pressure_panels(panels, reference_pressure, magnitude_max, level_limits_db)


def _pressure_reference(
    magnitude: ArrayLike, reference: float | None, *, name: str = "reference_pressure",
) -> float:
    if reference is None:
        return float(np.max(magnitude)) or 1.0
    if not np.isfinite(reference) or reference <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return reference


def _validate_level_limits(level_limits_db: tuple[float, float]) -> None:
    lower, upper = level_limits_db
    if not np.isfinite([lower, upper]).all() or lower >= upper:
        raise ValueError("level_limits_db must be increasing and finite")


def _plot_pressure_panels(panels, reference, magnitude_max, level_limits_db) -> Figure:
    if not np.isfinite(magnitude_max) or magnitude_max <= 0:
        raise ValueError("magnitude_max must be finite and positive")
    _validate_level_limits(level_limits_db)
    fig, axes = plt.subplots(
        len(panels), 3, squeeze=False, figsize=(12, 3.5 * len(panels)), layout="constrained",
    )
    scales = (
        ("Normalized pressure magnitude", "viridis", Normalize(0, magnitude_max)),
        ("Relative pressure level (dB)", "magma", Normalize(*level_limits_db)),
        ("Pressure phase (rad)", "twilight", Normalize(-np.pi, np.pi)),
    )
    for row, (field, x, y, xlabel, ylabel, title, extent) in zip(axes, panels):
        magnitude = np.abs(field) / reference
        with np.errstate(divide="ignore"):
            level = np.maximum(20 * np.log10(magnitude), level_limits_db[0])
        phase = np.ma.masked_where(magnitude == 0, np.angle(field))
        for ax, values, (_, cmap, norm) in zip(row, (magnitude, level, phase), scales):
            if extent is None:
                ax.pcolormesh(x, y, values, cmap=cmap, norm=norm, shading="flat")
            else:
                ax.imshow(values, extent=extent, origin="lower", interpolation="nearest", cmap=cmap, norm=norm)
            ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    for column, (label, cmap, norm) in enumerate(scales):
        fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=axes[:, column], label=label)
    return fig


def plot_angular_directivity(
    theta_deg: ArrayLike,
    directivity: ArrayLike,
    *,
    reference_pressure: float | None = None,
) -> Figure:
    """Plot a signed forward angular cut from the far-field pressure coefficient.

    A fixed reference_pressure is expressed in the same units as directivity.
    """
    theta = np.asarray(theta_deg, dtype=float)
    magnitude = np.abs(directivity)
    if theta.ndim != 1 or theta.size == 0 or magnitude.shape != theta.shape:
        raise ValueError("theta_deg and directivity must be matching nonempty vectors")
    if not np.isfinite(theta).all() or not np.isfinite(magnitude).all() or (np.abs(theta) > 90).any():
        raise ValueError("directivity must be finite and angles must be within [-90, 90]")
    normalized = magnitude / _pressure_reference(magnitude, reference_pressure)
    intensity = normalized**2
    with np.errstate(divide="ignore"):
        pressure_level = 20 * np.log10(normalized)
        intensity_level = 10 * np.log10(intensity)

    fig, (pressure_ax, intensity_ax, level_ax) = plt.subplots(
        1, 3, figsize=(12, 4), subplot_kw={"projection": "polar"}, layout="constrained",
    )
    theta_rad = np.deg2rad(theta)
    pressure_ax.plot(theta_rad, normalized)
    pressure_ax.set_title("Normalized pressure directivity")
    intensity_ax.plot(theta_rad, intensity)
    intensity_ax.set_title("Normalized far-field intensity")
    level_ax.plot(theta_rad, np.maximum(pressure_level, -40), label="Pressure level (20 log₁₀)")
    level_ax.plot(theta_rad, np.maximum(intensity_level, -40), "--", label="Intensity level (10 log₁₀)")
    level_ax.set(title="Relative levels (dB)", rmin=-40, rmax=max(0.0, float(np.max(pressure_level))))
    level_ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.2))
    for ax in (pressure_ax, intensity_ax, level_ax):
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_thetalim(-np.pi / 2, np.pi / 2)
    return fig


def _selected_frequency_indices(frequencies_hz: np.ndarray, requested_hz: ArrayLike) -> list[int]:
    frequency_bin_order(frequencies_hz)
    requested = np.atleast_1d(np.asarray(requested_hz))
    if requested.ndim != 1 or not requested.size or requested.dtype.kind not in "iuf":
        raise ValueError("select at least one sampled frequency in Hz")
    lookup = {frequency: index for index, frequency in enumerate(frequencies_hz)}
    try:
        return [lookup[frequency] for frequency in requested]
    except KeyError as exc:
        raise ValueError(f"frequency {exc.args[0]:g} Hz must match a sampled bin") from None


def _sweep_pressure_reference(sweep: FrequencySweepResult, reference_pressure: float | None) -> float:
    peak = max(float(np.abs(field).max()) for field in sweep.pressure) if reference_pressure is None else 0.0
    return _pressure_reference(peak, reference_pressure)


def _relative_level(magnitude: np.ndarray, reference: float, factor: int, floor_db: float) -> np.ndarray:
    with np.errstate(divide="ignore"):
        return np.maximum(factor * (np.log10(magnitude) - np.log10(reference)), floor_db)


def plot_frequency_pressure(
    sweep: FrequencySweepResult,
    frequency_hz: float,
    *,
    distance_index: int = 0,
    reference_pressure: float | None = None,
    magnitude_max: float = 1.0,
    level_limits_db: tuple[float, float] = (-40.0, 0.0),
) -> Figure:
    """Plot pressure magnitude, relative level, and phase at a sampled frequency."""
    index, = _selected_frequency_indices(sweep.frequencies_hz, frequency_hz)
    fig = plot_pressure_maps(
        sweep.grid, [sweep.distances_m[distance_index]], [sweep.pressure[index, distance_index]],
        reference_pressure=_sweep_pressure_reference(sweep, reference_pressure),
        magnitude_max=magnitude_max, level_limits_db=level_limits_db,
    )
    fig.suptitle(f"Pressure at {sweep.frequencies_hz[index]:g} Hz")
    return fig


def plot_point_frequency_responses(
    sweep: FrequencySweepResult,
    points_m: ArrayLike,
    *,
    reference_pressure: float | None = None,
    level_limits_db: tuple[float, float] = (-60.0, 0.0),
) -> Figure:
    """Plot levels and phases at the nearest sampled observation points."""
    frequencies, locations, pressure = point_pressure_responses(sweep, points_m)
    _validate_level_limits(level_limits_db)
    reference = _sweep_pressure_reference(sweep, reference_pressure)
    fig, (magnitude_ax, phase_ax) = plt.subplots(
        2, 1, sharex=True, figsize=(8, 6), layout="constrained",
    )
    marker = "o" if len(frequencies) == 1 else None
    for (x, y, z), values in zip(locations, pressure.T):
        magnitude = np.abs(values)
        phase = np.ma.masked_where(magnitude == 0, np.angle(values))
        label = f"({x:g}, {y:g}, {z:g}) m"
        line, = magnitude_ax.semilogx(
            frequencies, _relative_level(magnitude, reference, 20, level_limits_db[0]),
            marker=marker, label=label,
        )
        phase_ax.semilogx(frequencies, phase, marker=marker, color=line.get_color())
    magnitude_ax.set(title="Observation-point responses", ylabel="Relative pressure level (dB)", ylim=level_limits_db)
    phase_ax.set(xlabel="Frequency (Hz)", ylabel="Wrapped phase (rad)", ylim=(-np.pi, np.pi))
    magnitude_ax.legend()
    for ax in (magnitude_ax, phase_ax):
        ax.grid(which="both", alpha=0.25)
    return fig


def plot_broadband_pressure(
    sweep: FrequencySweepResult,
    spectral_weights: ArrayLike,
    *,
    weighting: SpectralWeighting = "discrete",
    bin_widths_hz: ArrayLike | None = None,
    distance_index: int = 0,
    reference_pressure: float | None = None,
    mean_square_max: float = 1.0,
    level_limits_db: tuple[float, float] = (-40.0, 0.0),
) -> Figure:
    """Plot normalized mean-square pressure and relative level at one distance."""
    _validate_level_limits(level_limits_db)
    if not np.isfinite(mean_square_max) or mean_square_max <= 0:
        raise ValueError("mean_square_max must be finite and positive")
    mean_square = broadband_mean_square_pressure(
        sweep.pressure[:, distance_index], sweep.frequencies_hz, spectral_weights,
        weighting=weighting, bin_widths_hz=bin_widths_hz,
    )
    reference = _pressure_reference(np.sqrt(mean_square.max()), reference_pressure)
    normalized = mean_square / reference / reference
    fig, (linear_ax, level_ax) = plt.subplots(1, 2, figsize=(9, 4), layout="constrained")
    linear_image = linear_ax.imshow(
        normalized, extent=sweep.grid.extent, origin="lower", interpolation="nearest",
        vmin=0, vmax=mean_square_max, cmap="viridis",
    )
    level_image = level_ax.imshow(
        _relative_level(normalized, 1.0, 10, level_limits_db[0]), extent=sweep.grid.extent,
        origin="lower", interpolation="nearest", vmin=level_limits_db[0], vmax=level_limits_db[1], cmap="magma",
    )
    for ax in (linear_ax, level_ax):
        ax.set(xlabel="x (m)", ylabel="y (m)", aspect="equal")
    linear_ax.set_title("Normalized mean-square pressure")
    level_ax.set_title("Relative mean-square pressure level")
    fig.colorbar(linear_image, ax=linear_ax, label="Mean-square pressure ratio")
    fig.colorbar(level_image, ax=level_ax, label="Relative level (dB)")
    fig.suptitle(f"Broadband pressure at z = {sweep.distances_m[distance_index]:g} m")
    return fig


def _directivity_magnitude(result: FrequencyDirectivityResult) -> np.ndarray:
    theta = np.asarray(result.theta_deg)
    if (
        theta.ndim != 1 or len(theta) < 2 or theta.dtype.kind not in "iuf"
        or not np.isfinite(theta).all() or (np.abs(theta) > 90).any()
        or not (theta[1:] > theta[:-1]).all()
    ):
        raise ValueError("theta_deg must be an increasing angle vector within [-90, 90]")
    magnitude = np.abs(result.directivity)
    if magnitude.shape != (len(result.frequencies_hz), len(theta)) or not np.isfinite(magnitude).all():
        raise ValueError("directivity must contain finite coefficients for each frequency and angle")
    return magnitude


def plot_frequency_directivity(
    result: FrequencyDirectivityResult,
    frequencies_hz: ArrayLike | None = None,
    *,
    reference_directivity: float | None = None,
    level_limits_db: tuple[float, float] = (-60.0, 0.0),
) -> Figure:
    """Overlay angular cuts at selected frequencies using a shared coefficient reference."""
    order = (
        frequency_bin_order(result.frequencies_hz)
        if frequencies_hz is None else _selected_frequency_indices(result.frequencies_hz, frequencies_hz)
    )
    _validate_level_limits(level_limits_db)
    magnitude = _directivity_magnitude(result)
    reference = _pressure_reference(magnitude, reference_directivity, name="reference_directivity")
    fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
    for index in order:
        level = _relative_level(magnitude[index], reference, 20, level_limits_db[0])
        ax.plot(result.theta_deg, level, label=f"{result.frequencies_hz[index]:g} Hz")
    ax.set(
        xlabel="Signed polar angle (°)", ylabel="Relative directivity level (dB)", ylim=level_limits_db,
        title=f"Pressure directivity · azimuth {result.azimuth_deg:g}°",
    )
    ax.legend()
    ax.grid(alpha=0.25)
    return fig


def plot_aperture_measured_comparison(
    aperture: FrequencyDirectivityResult,
    measured: FrequencyDirectivityResult,
    frequency_hz: float,
    *,
    reference_directivity: float | None = None,
    level_limits_db: tuple[float, float] = (-60.0, 0.0),
    show_phase: bool = False,
) -> Figure:
    """Compare matching aperture and measured cuts, optionally with aligned phase origins."""
    azimuth_difference = (aperture.azimuth_deg - measured.azimuth_deg + 180) % 360 - 180
    if not np.isclose(azimuth_difference, 0, atol=1e-9):
        raise ValueError("aperture and measured cuts must have the same azimuth")
    _validate_level_limits(level_limits_db)
    curves = []
    for result, label in ((aperture, "ASM aperture"), (measured, "Measured element patterns")):
        index, = _selected_frequency_indices(result.frequencies_hz, frequency_hz)
        magnitude = _directivity_magnitude(result)[index]
        reference = _pressure_reference(magnitude, reference_directivity, name="reference_directivity")
        level = _relative_level(magnitude, reference, 20, level_limits_db[0])
        curves.append((result.theta_deg, label, level, np.angle(result.directivity[index], deg=True)))

    fig, axes = plt.subplots(1 + show_phase, 1, sharex=True, figsize=(8, 4 + 2 * show_phase), layout="constrained")
    axes = np.atleast_1d(axes)
    for theta, label, level, phase in curves:
        line, = axes[0].plot(theta, level, label=label)
        if show_phase:
            axes[1].plot(theta, phase, color=line.get_color())
    normalization = "each curve referenced to its own peak" if reference_directivity is None else "shared reference"
    axes[0].set(title=normalization, ylabel="Relative directivity level (dB)", ylim=level_limits_db)
    axes[0].legend()
    if show_phase:
        axes[1].set(ylabel="Phase (°)", ylim=(-180, 180))
    axes[-1].set_xlabel("Signed polar angle (°)")
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.suptitle(f"Directivity at {frequency_hz:g} Hz · azimuth {aperture.azimuth_deg:g}°")
    return fig


def _sample_edges(samples: np.ndarray) -> np.ndarray:
    midpoints = (samples[:-1] + samples[1:]) / 2
    return np.r_[2 * samples[0] - midpoints[0], midpoints, 2 * samples[-1] - midpoints[-1]]


def plot_directivity_heatmap(
    result: FrequencyDirectivityResult,
    *,
    reference_directivity: float | None = None,
    level_limits_db: tuple[float, float] = (-60.0, 0.0),
) -> Figure:
    """Plot relative directivity over polar angle and logarithmic frequency."""
    order = frequency_bin_order(result.frequencies_hz)
    _validate_level_limits(level_limits_db)
    magnitude = _directivity_magnitude(result)
    reference = _pressure_reference(magnitude, reference_directivity, name="reference_directivity")
    level = _relative_level(magnitude[order], reference, 20, level_limits_db[0])
    theta = result.theta_deg
    angles = np.clip(_sample_edges(theta), -90, 90)
    log_frequency = np.log(result.frequencies_hz[order])
    if len(log_frequency) == 1:
        log_edges = log_frequency[0] + np.log([1 / np.sqrt(2), np.sqrt(2)])
    else:
        log_edges = _sample_edges(log_frequency)

    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    mesh = ax.pcolormesh(
        angles, np.exp(log_edges), level, shading="flat",
        cmap="magma", vmin=level_limits_db[0], vmax=level_limits_db[1],
    )
    ax.set(
        xlabel="Signed polar angle (°)", ylabel="Frequency (Hz)", yscale="log",
        title=f"Pressure directivity · azimuth {result.azimuth_deg:g}°",
    )
    fig.colorbar(mesh, ax=ax, label="Relative directivity level (dB)")
    return fig
