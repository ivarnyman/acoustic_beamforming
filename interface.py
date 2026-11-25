"""Matplotlib desktop controls for the acoustic array simulator."""

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from threading import Event

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Button, RadioButtons

from beamforming import ARRAY_WINDOWS
from config import SimulationConfig, load_configuration, save_configuration
from interface_editor import ArrayEditor, InputBox
from interface_run import SweepPreview, calculate_preview
from simulation import (
    SimulationOutput, SimulationSession, array_diameters, array_type_ids, configured_array_window,
    configured_focus_point, configured_weights,
)
from visualization import (
    plot_broadband_pressure, plot_directivity_heatmap, plot_point_frequency_responses,
    plot_speaker_geometry, plot_speaker_window,
)


def _optional_float(text: str) -> float | None:
    return float(text) if text.strip() else None


def _with_focus_controls(config: SimulationConfig) -> SimulationConfig:
    """Initialize the range and angles from a legacy Cartesian target."""
    if config.focus_distance_m is not None:
        return config
    x, y, z = config.focus_point_m
    horizontal = np.hypot(x, y)
    values = {"focus_distance_m": float(np.hypot(horizontal, z))}
    if config.beamforming_mode == "focus":
        values["steering_elevation_deg"] = float(np.rad2deg(np.arctan2(z, horizontal)))
        if horizontal > 0:
            values["steering_azimuth_deg"] = float(np.rad2deg(np.arctan2(y, x)))
    return replace(config, **values)


def _complex_weights(text: str) -> tuple[complex, ...] | None:
    if not text.strip():
        return None
    return tuple(complex(part.replace(" ", "").replace("i", "j")) for part in text.split(","))


def _preview_samples(field: np.ndarray) -> np.ndarray:
    """Limit display work to a raster comparable to the size of the plot."""
    ny, nx = field.shape
    return field[::max(1, (ny + 699) // 700), ::max(1, (nx + 699) // 700)]


class SimulatorInterface:
    """Own the widget state and render results from the simulation session."""

    def __init__(self, config: SimulationConfig | None = None) -> None:
        self._default = _with_focus_controls(config if config is not None else SimulationConfig())
        self.config = self._default
        self.session = SimulationSession()
        self._preview = SimulationSession()
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._pending: tuple[SimulationConfig, Future[SimulationOutput | SweepPreview]] | None = None
        self._pending_cancel: Event | None = None
        self._queued: SimulationConfig | None = None
        self._loading = False
        self._closed = False
        self._last_run = None
        self._last_result = None
        self._editor = None
        self._display_frequency = None
        self._pressure_view = "level"
        self._floor_db = -40.0
        self._pressure_reference = None
        self._directivity_reference = None
        self._shown_layout = None
        self._shown_type_ids = None
        self._shown_window = None
        self._window_colorbar = None
        self.figure = plt.figure(figsize=(15, 10))
        self._timer = self.figure.canvas.new_timer(interval=100)
        self._timer.add_callback(self._finish_recalculation)
        self.figure.canvas.mpl_connect("close_event", self.close)
        panels = self.figure.add_gridspec(
            3, 2, left=0.37, right=0.96, bottom=0.18, top=0.93, wspace=0.40, hspace=0.55,
        )
        self.layout_ax = self.figure.add_subplot(panels[0, 0])
        self.window_ax = self.figure.add_subplot(panels[0, 1])
        self.aperture_ax = self.figure.add_subplot(panels[1, 0])
        self.pressure_ax = self.figure.add_subplot(panels[1, 1])
        self.directivity_ax = self.figure.add_subplot(panels[2, :], projection="polar")
        for ax, title in (
            (self.aperture_ax, "Normal velocity amplitude"),
            (self.pressure_ax, "Relative pressure level"),
            (self.directivity_ax, "Pressure directivity"),
        ):
            ax.set_title(title)
            ax.text(0.5, 0.5, "Select Recalculate", transform=ax.transAxes, ha="center")
        self._colorbars = []
        self._status = self.figure.text(0.37, 0.115, "", va="top", fontsize=9)
        self._warnings = self.figure.text(0.37, 0.088, "", va="top", fontsize=8, color="#93501d")
        self.figure.text(0.37, 0.034, "ASM output is uncalibrated. Display levels are relative.", fontsize=8)

        self.fields = {}
        entries = (
            ("rows", "Rows", int), ("columns", "Columns", int),
            ("speaker_diameter_m", "Diameter (m)", float),
            ("horizontal_gap_m", "Horizontal gap (m)", float),
            ("vertical_gap_m", "Vertical gap (m)", float),
            ("row_stagger", "Row stagger (0–1)", float),
            ("frequency_hz", "Frequency (Hz)", float),
            ("observation_distance_m", "Observation z (m)", float),
            ("focus_distance_m", "Focus distance (m)", float),
            ("observation_width_x_m", "Pressure x width (m)", _optional_float),
            ("observation_width_y_m", "Pressure y width (m)", _optional_float),
            ("taper_width_fraction", "Taper width / radius", float),
            ("steering_azimuth_deg", "Azimuth (°)", float),
            ("steering_elevation_deg", "Elevation (°)", float),
            ("grid_spacing_m", "Grid dx (m)", float),
            ("grid_spacing_y_m", "Grid dy (m)", _optional_float),
            ("padding_factor", "FFT padding", float),
            ("speaker_weights", "Complex weights", _complex_weights),
        )
        for index, (key, label, parse) in enumerate(entries):
            # Put labels to the left of text fields and leave room for long weight vectors.
            y = 0.926 - 0.026 * index
            ax = self.figure.add_axes([0.135, y, 0.17, 0.0225])
            widget = InputBox(
                ax, label + " ", initial=self._field_text(self.config, key), label_pad=0.025,
            )
            widget.label.set_fontsize(9)
            widget.text_disp.set_fontsize(9)
            widget.text_disp.set_clip_on(True)
            widget.on_submit(self._on_edit)
            self.fields[key] = (widget, parse)

        self.figure.text(0.035, 0.469, "Weights in speaker index order, e.g. 1, 0.5+0.5j", fontsize=8)
        self.profile = RadioButtons(
            self.figure.add_axes([0.033, 0.33, 0.086, 0.112]), ("uniform", "cosine", "gaussian"),
            active=("uniform", "cosine", "gaussian").index(self.config.amplitude_profile),
        )
        self.mode = RadioButtons(
            self.figure.add_axes([0.126, 0.33, 0.086, 0.112]), ("none", "steer", "focus"),
            active=("none", "steer", "focus").index(self.config.beamforming_mode),
        )
        self.window = RadioButtons(
            self.figure.add_axes([0.219, 0.33, 0.086, 0.112]), ARRAY_WINDOWS,
            active=ARRAY_WINDOWS.index(self.config.array_window),
        )
        for label in self.window.labels:
            label.set_fontsize(8)
        self.window.ax.set_title("Speaker window", fontsize=9)
        self.window.on_clicked(self._on_edit)
        self.profile.ax.set_title("Aperture profile", fontsize=9)
        self.mode.ax.set_title("Beamforming", fontsize=9)
        self.profile.on_clicked(self._on_edit)
        self.mode.on_clicked(self._on_edit)

        self.path = InputBox(
            self.figure.add_axes([0.080, 0.173, 0.225, 0.028]), "JSON ", initial="array_config.json",
        )
        self.path.text_disp.set_fontsize(9)
        self.path.text_disp.set_clip_on(True)
        self.buttons = {}
        for label, position, callback in (
            ("Recalculate", [0.033, 0.255, 0.168, 0.043], self.recalculate),
            ("Reset", [0.212, 0.255, 0.093, 0.043], self.reset),
            ("Save Configuration", [0.033, 0.113, 0.131, 0.037], self.save),
            ("Load Configuration", [0.174, 0.113, 0.131, 0.037], self.load),
            ("Array & sweep settings", [0.033, 0.048, 0.272, 0.040], self.open_editor),
        ):
            button = Button(self.figure.add_axes(position), label)
            button.label.set_fontsize(8)
            button.on_clicked(callback)
            self.buttons[label] = button

        self._on_edit()

    @staticmethod
    def _field_text(config: SimulationConfig, key: str) -> str:
        value = getattr(config, key)
        if isinstance(value, tuple):
            return ", ".join(str(item) for item in value)
        return "" if value is None else str(value)

    def _set_controls(self, config: SimulationConfig) -> None:
        config = _with_focus_controls(config)
        if config != self.config:
            self._invalidate_pending()
        widgets = (self.profile, self.mode, self.window)
        self._loading = True
        for widget in widgets:
            widget.drawon = False
        try:
            self.config = config
            for key, (widget, _) in self.fields.items():
                widget.set_text(self._field_text(config, key))
            self.profile.set_active(("uniform", "cosine", "gaussian").index(config.amplitude_profile))
            self.mode.set_active(("none", "steer", "focus").index(config.beamforming_mode))
            self.window.set_active(ARRAY_WINDOWS.index(config.array_window))
        finally:
            self._loading = False
            for widget in widgets:
                widget.drawon = True

    def open_editor(self, _event=None) -> ArrayEditor:
        if self._editor is None or not plt.fignum_exists(self._editor.figure.number):
            self._editor = ArrayEditor(self, self._editor)
        plt.show(block=False)
        return self._editor

    def apply_configuration(self, config: SimulationConfig) -> None:
        self._preview.centers(config)
        self._set_controls(config)
        self._on_edit()

    def _read_controls(self) -> SimulationConfig:
        values = {key: parse(widget.text) for key, (widget, parse) in self.fields.items()}
        distance = values["focus_distance_m"]
        if not np.isfinite(distance) or distance <= 0:
            raise ValueError("focus distance must be finite and positive")
        config = replace(
            self.config, **values,
            amplitude_profile=self.profile.value_selected,
            beamforming_mode=self.mode.value_selected, array_window=self.window.value_selected,
        )
        if config.beamforming_mode == "focus":
            config = replace(config, focus_point_m=configured_focus_point(config))
        return config

    def _status_message(self, message: str) -> None:
        self._status.set_text(message)
        self.figure.canvas.draw_idle()

    def _invalidate_pending(self) -> None:
        self._queued = None
        if self._pending_cancel is not None:
            self._pending_cancel.set()

    def _numerical_warnings(self, config: SimulationConfig) -> list[str]:
        if config.propagation_mode == "measured":
            notes = ["Measured synthesis is far field only. Patterns need common calibration and angular coverage."]
            elements = config.speaker_array.element_types() if config.speaker_array else ()
            missing = sorted({item.type_id for item in elements if item.measured_pattern is None})
            if missing:
                notes.append("Missing measured patterns for " + ", ".join(missing))
            return notes
        bins = config.frequencies_hz or (config.frequency_hz,)
        warnings = dict.fromkeys(
            item for f in {min(bins), max(bins)}
            for item in self._preview.warnings(replace(config, frequency_hz=f))
        )
        return list(warnings)

    def _on_edit(self, _event=None) -> None:
        if self._loading:
            return
        try:
            config = self._read_controls()
            centers = self._preview.centers(config)
        except (ValueError, TypeError, OverflowError) as exc:
            self._invalidate_pending()
            self._warnings.set_text("")
            self._status_message(f"Input: {exc}")
            return
        if config != self.config:
            self._invalidate_pending()
            self.config = config
        type_ids = array_type_ids(config)
        if centers is not self._shown_layout or type_ids != self._shown_type_ids:
            self.layout_ax.clear()
            plot_speaker_geometry(
                centers, array_diameters(config), show_indices=True, type_ids=type_ids, ax=self.layout_ax,
            )
            self.layout_ax.set_title("Typed array layout" if type_ids is not None else "Array layout")
            self._shown_layout = centers
            self._shown_type_ids = type_ids
        try:
            window_key = (config.rows, config.columns, config.array_window, len(centers), config.speaker_array is not None)
            if window_key != self._shown_window:
                amplitudes = configured_array_window(config, len(centers))
                self.window_ax.clear()
                image = plot_speaker_window(amplitudes, config.array_window, ax=self.window_ax)
                if self._window_colorbar is None:
                    self._window_colorbar = self.figure.colorbar(
                        image, ax=self.window_ax, label="Amplitude multiplier",
                    )
                else:
                    self._window_colorbar.update_normal(image)
                self._shown_window = window_key
            self._validate_acoustics(config)
            self._warnings.set_text("\n".join(self._numerical_warnings(config)))
        except (ValueError, TypeError, OverflowError, MemoryError) as exc:
            self._invalidate_pending()
            self._warnings.set_text("")
            self._status_message(f"Input: {exc}")
            return
        if self._pending is not None and not self._pending_cancel.is_set():
            self._status_message("Calculating… Plots show the last calculation.")
        elif self._last_run is None:
            self._status_message("Select Recalculate to compute the velocity and pressure fields.")
        elif self._last_run == config:
            self._status_message("Results are current.")
        else:
            self._status_message("Settings changed. Plots show the last calculation.")

    def recalculate(self, _event=None) -> None:
        if self._closed:
            return
        try:
            config = self._read_controls()
            self._validate_acoustics(config)
            centers = self._preview.centers(config)
            self._numerical_warnings(config)
            configured_weights(config, centers)
            if config == self._last_run:
                self._status_message("Results are current.")
                return
        except (ValueError, TypeError, OverflowError, MemoryError) as exc:
            self._invalidate_pending()
            self._status_message(f"Calculation failed: {exc}")
            return
        if config != self.config:
            self._on_edit()
        if self._pending is not None:
            if self._pending_cancel.is_set():
                self._queued = config
                self._status_message("Latest request queued. Plots show the last calculation.")
            return
        self._start_recalculation(config)

    def _start_recalculation(self, config: SimulationConfig) -> None:
        cancel = Event()
        self._pending_cancel = cancel
        self._pending = config, self._executor.submit(calculate_preview, self.session, config, cancel)
        self._status_message("Calculating… Plots show the last calculation.")
        self._timer.start()

    def _finish_recalculation(self) -> None:
        if self._pending is None:
            return
        config, future = self._pending
        if not future.done():
            return
        stale = self._pending_cancel.is_set()
        self._pending = None
        self._pending_cancel = None
        self._timer.stop()
        if stale:
            queued, self._queued = self._queued, None
            if queued is not None and queued == self.config:
                self._start_recalculation(queued)
            else:
                self._on_edit()
            return
        try:
            result = future.result()
            self._render(result)
        except Exception as exc:
            self._status_message(f"Calculation failed: {exc or type(exc).__name__}")
            return
        self._last_run = config
        self._last_result = result
        self._on_edit()

    def close(self, _event=None) -> None:
        self._closed = True
        self._invalidate_pending()
        self._timer.stop()
        self._executor.shutdown(wait=False, cancel_futures=True)
        if self._editor is not None:
            plt.close(self._editor.figure)

    def _render(self, result: SimulationOutput | SweepPreview) -> None:
        for colorbar in self._colorbars:
            colorbar.remove()
        self._colorbars = []
        for ax in (self.aperture_ax, self.pressure_ax, self.directivity_ax):
            ax.clear()

        measured = isinstance(result, SweepPreview) and result.pressure is None
        frequency_label = ""
        if isinstance(result, SweepPreview):
            cuts = result.directivity
            indices = np.flatnonzero(cuts.frequencies_hz == self._display_frequency)
            index = int(indices[0]) if indices.size else 0
            frequency_label = f" · {cuts.frequencies_hz[index]:g} Hz"
            theta, azimuth, coefficients = cuts.theta_deg, cuts.azimuth_deg, cuts.directivity[index]
            if not measured:
                self._render_field_maps(
                    result.velocity[index], result.aperture_grid.extent,
                    result.pressure.pressure[index, 0], result.pressure.grid.extent,
                    result.pressure.distances_m[0], result.peak_pressure, frequency_label,
                )
        else:
            theta, azimuth, coefficients = result.theta_deg, result.azimuth_deg, result.directivity
            self._render_field_maps(
                _preview_samples(result.velocity), result.grid.source.extent,
                _preview_samples(result.pressure.pressure[0]), result.grid.observation.extent,
                result.pressure.distances_m[0], float(result.pressure.magnitude.max()), frequency_label,
            )
        if measured:
            for ax, title in ((self.aperture_ax, "Aperture velocity"), (self.pressure_ax, "Pressure plane")):
                ax.set_title(title)
                ax.text(0.5, 0.5, "Measured patterns provide angular data", transform=ax.transAxes, ha="center")
                ax.set_axis_off()

        magnitude = np.abs(coefficients)
        reference = self._directivity_reference or (
            float(np.abs(result.directivity.directivity).max()) if isinstance(result, SweepPreview)
            else float(magnitude.max())
        ) or 1.0
        normalized = magnitude / reference
        self.directivity_ax.plot(np.deg2rad(theta), normalized)
        title = "Measured far-field directivity" if measured else "Pressure directivity"
        self.directivity_ax.set(
            title=f"{title} · azimuth {azimuth:g}°{frequency_label}", ylim=(0, max(1, normalized.max())),
        )
        self.directivity_ax.set_theta_zero_location("N")
        self.directivity_ax.set_theta_direction(-1)
        self.directivity_ax.set_thetalim(-np.pi / 2, np.pi / 2)
        self.directivity_ax.set_rticks([0.5, 1.0])

    def _render_field_maps(self, velocity, aperture_extent, pressure, extent, z, peak_pressure, frequency_label) -> None:
        self.aperture_ax.set_axis_on()
        self.pressure_ax.set_axis_on()
        velocity_image = self.aperture_ax.imshow(
            np.abs(velocity), extent=aperture_extent, origin="lower",
            cmap="viridis", interpolation="nearest", vmin=0,
        )
        self.aperture_ax.set(title=f"Normal velocity amplitude{frequency_label}", xlabel="x (m)", ylabel="y (m)")
        magnitude = np.abs(pressure)
        reference = self._pressure_reference or peak_pressure or 1.0
        if self._pressure_view == "phase":
            data = np.ma.masked_where(magnitude == 0, np.angle(pressure))
            label, name, cmap, limits = "Phase (rad)", "Pressure phase", "twilight", (-np.pi, np.pi)
        elif self._pressure_view == "magnitude":
            data = magnitude / reference
            label, name, cmap, limits = "Pressure ratio", "Relative pressure magnitude", "magma", (0, 1)
        else:
            with np.errstate(divide="ignore"):
                data = np.maximum(20 * (np.log10(magnitude) - np.log10(reference)), self._floor_db)
            label, name, cmap, limits = "Level (dB)", "Relative pressure", "magma", (self._floor_db, 0)
        pressure_image = self.pressure_ax.imshow(
            data, extent=extent, origin="lower", cmap=cmap, interpolation="nearest", vmin=limits[0], vmax=limits[1],
        )
        self.pressure_ax.set(title=f"{name} at z = {z:g} m{frequency_label}", xlabel="x (m)", ylabel="y (m)")
        self._colorbars = [
            self.figure.colorbar(velocity_image, ax=self.aperture_ax, label="Relative amplitude"),
            self.figure.colorbar(pressure_image, ax=self.pressure_ax, label=label),
        ]

    def set_display(self, *, frequency_hz, pressure_view, floor_db, pressure_reference, directivity_reference) -> None:
        frequency, pressure_ref, cut_ref = (_optional_float(value) for value in (
            frequency_hz, pressure_reference, directivity_reference,
        ))
        if any(value is not None and (not np.isfinite(value) or value <= 0) for value in (frequency, pressure_ref, cut_ref)):
            raise ValueError("Display frequency and references must be positive and finite")
        if not np.isfinite(floor_db) or floor_db >= 0:
            raise ValueError("Level floor must be finite and below 0 dB")
        if pressure_view not in ("level", "magnitude", "phase"):
            raise ValueError("Pressure view must be level, magnitude, or phase")
        requested = self.config.frequencies_hz or (self.config.frequency_hz,)
        displayed = self._last_result
        available = displayed.directivity.frequencies_hz if isinstance(displayed, SweepPreview) else ()
        if frequency is not None and frequency not in requested and frequency not in available:
            raise ValueError("Display frequency must match a configured or calculated frequency")
        self._display_frequency, self._pressure_view, self._floor_db = frequency, pressure_view, floor_db
        self._pressure_reference, self._directivity_reference = pressure_ref, cut_ref
        if displayed is not None:
            self._render(displayed)
            self.figure.canvas.draw_idle()

    def show_sweep_plots(self, *, weights=(), weighting="discrete", bin_widths=None) -> None:
        result = self._last_result
        if not isinstance(result, SweepPreview):
            raise ValueError("Apply a sweep or measured mode, then select Recalculate")
        if weights:
            if result.pressure is None:
                raise ValueError("Mean-square pressure needs an ASM pressure sweep")
            plot_broadband_pressure(
                result.pressure, weights, weighting=weighting, bin_widths_hz=bin_widths,
                reference_pressure=self._pressure_reference, level_limits_db=(self._floor_db, 0),
            )
        elif bin_widths is not None:
            raise ValueError("Enter spectral weights for the specified bin widths")
        plot_directivity_heatmap(
            result.directivity, reference_directivity=self._directivity_reference, level_limits_db=(self._floor_db, 0),
        )
        if result.pressure is not None:
            x, y = result.pressure.grid.x, result.pressure.grid.y
            point = (x[len(x) // 2], y[len(y) // 2], result.pressure.distances_m[0])
            plot_point_frequency_responses(
                result.pressure, point, reference_pressure=self._pressure_reference or result.peak_pressure or 1.0,
                level_limits_db=(self._floor_db, 0),
            )
        plt.show(block=False)

    def reset(self, _event=None) -> None:
        self._set_controls(self._default)
        if self._editor is not None:
            self._editor.sync_configuration(self.config)
        self._on_edit()

    def save(self, _event=None) -> None:
        try:
            config = self._read_controls()
            path = self._configuration_path()
            save_configuration(path, config)
        except (OSError, ValueError, TypeError, OverflowError) as exc:
            self._status_message(f"Save failed: {exc}")
            return
        self._status_message(f"Saved {path}.")

    def load(self, _event=None) -> None:
        try:
            path = self._configuration_path()
            config = load_configuration(path)
            centers = self._preview.centers(config)
            self._validate_acoustics(config)
            self._numerical_warnings(config)
            configured_weights(config, centers)
        except (OSError, ValueError, TypeError, OverflowError, MemoryError, KeyError) as exc:
            self._status_message(f"Load failed: {exc}")
            return
        self._set_controls(config)
        if self._editor is not None:
            self._editor.sync_configuration(self.config)
        self._on_edit()
        self._status_message(f"Loaded {path}. Select Recalculate to update the results.")

    @staticmethod
    def _validate_acoustics(config: SimulationConfig) -> None:
        for name, value in (
            ("frequency", config.frequency_hz),
            ("sound speed", config.sound_speed_m_s),
            ("air density", config.air_density_kg_m3),
        ):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not np.isfinite(config.observation_distance_m) or config.observation_distance_m < 0:
            raise ValueError("observation distance must be finite and nonnegative")

    def _configuration_path(self) -> Path:
        path = self.path.text.strip()
        if not path:
            raise ValueError("enter a JSON path")
        return Path(path).expanduser()


def main() -> SimulatorInterface:
    """Launch the desktop interface and keep the widgets alive."""
    app = SimulatorInterface()
    plt.show()
    return app


if __name__ == "__main__":
    main()
