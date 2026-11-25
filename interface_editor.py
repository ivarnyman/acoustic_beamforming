"""Optional speaker library, layout editor, and sweep controls."""

import csv
import json
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Button, RadioButtons, TextBox

from config import SimulationConfig
from frequency_response import FrequencyResponse
from interface_presets import PRESETS, preset_configuration
from measured_data import load_element_pattern_csv, load_frequency_response_csv
from signal_chain import Crossover, SignalChain
from simulation import generate_array
from speakers import SpeakerArray, SpeakerGroup, SpeakerType

if TYPE_CHECKING:
    from interface import SimulatorInterface


class InputBox(TextBox):
    def set_text(self, value) -> None:
        """Populate a control without drawing or submitting it."""
        self.text_disp.set_text(str(value))
        self.cursor_index = min(self.cursor_index, len(self.text))
        self.cursor.set_visible(False)

    def stop_typing(self) -> None:
        # Only the active box needs to release focus or redraw on outside clicks.
        if self.capturekeystrokes:
            super().stop_typing()


def _numbers(text: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in text.split(",")) if text.strip() else ()


def _frequencies(text: str) -> tuple[float, ...] | None:
    if ":" not in text:
        return _numbers(text) or None
    first, last, count_text = text.split(":")
    start, stop = float(first), float(last)
    count = int(count_text)
    if count < 2 or not np.isfinite([start, stop]).all() or not 0 < start < stop:
        raise ValueError("Enter an increasing sweep as start:stop:count with at least two points")
    return tuple(np.linspace(start, stop, count))


def _joined(values) -> str:
    return ", ".join(f"{value:g}" for value in values)


class ArrayEditor:
    """Manage types and groups without constructing an acoustic grid."""

    def __init__(self, app: "SimulatorInterface", previous: "ArrayEditor | None" = None) -> None:
        self.app = app
        self.figure = plt.figure(figsize=(15, 8))
        self.figure.canvas.manager.set_window_title("Speaker library and sweeps")
        self.fields = {}
        self.buttons = {}
        self._message = self.figure.text(0.025, 0.043, "", fontsize=9)
        self._type_summary = self.figure.text(0.025, 0.356, "", fontsize=9)
        self._group_summary = self.figure.text(0.350, 0.235, "", fontsize=9)
        self._type_index = 0

        self.figure.text(0.025, 0.945, "Speaker type library", fontsize=13)
        self.figure.text(0.350, 0.945, "Layout groups", fontsize=13)
        self.figure.text(0.675, 0.945, "Simulation and display", fontsize=13)
        for index, (key, label, initial) in enumerate((
            ("type_id", "Type ID", ""), ("type_name", "Name", ""), ("diameter", "Diameter (m)", ""),
            ("response", "Velocity H", "flat"), ("response_path", "Corner Hz / CSV", ""),
            ("pattern_path", "Element pattern CSV", ""),
        )):
            self._field(key, label, initial, 0.025, 0.880 - 0.055 * index)
        self.figure.text(0.025, 0.555, "H choices: keep, flat, synthetic_lowpass, csv", fontsize=8)
        self.figure.text(0.025, 0.530, "Pattern path: leave blank to keep it, enter - to clear it.", fontsize=8)
        self._button("Save type", 0.025, 0.462, 0.115, self.save_type)
        self._button("Remove type", 0.155, 0.462, 0.115, self.remove_type)
        self._button("Previous type", 0.025, 0.405, 0.115, lambda: self._step_type(-1))
        self._button("Next type", 0.155, 0.405, 0.115, lambda: self._step_type(1))
        self._field("library_path", "Library JSON", "speaker_types.json", 0.025, 0.265)
        self._button("Import / merge", 0.025, 0.207, 0.115, self.import_library)
        self._button("Export library", 0.155, 0.207, 0.115, self.export_library)

        for index, (key, label, initial) in enumerate((
            ("group_index", "Group number", ""), ("group_type", "Type ID", ""),
            ("group_rows", "Rows", "1"), ("group_columns", "Columns", "1"),
            ("group_center", "Center x, y (m)", "0, 0"),
            ("group_gaps", "Gaps x, y (m)", "0.02, 0.02"),
            ("group_offsets", "Row offsets (m)", "0"),
            ("filter_kind", "Filter kind", "none"), ("filter_family", "Filter family", "butterworth"),
            ("filter_cutoffs", "Cutoff(s) (Hz)", ""), ("filter_order", "Filter order", "2"),
        )):
            self._field(key, label, initial, 0.350, 0.880 - 0.048 * index)
        self.figure.text(0.350, 0.366, "Blank group number adds a group. Existing numbers start at 1.", fontsize=8)
        self.figure.text(0.350, 0.344, "Filter kinds: none, lowpass, highpass, bandpass", fontsize=8)
        self._button("Read group", 0.350, 0.285, 0.094, self.read_group)
        self._button("Save group", 0.454, 0.285, 0.094, self.save_group)
        self._button("Remove group", 0.558, 0.285, 0.094, self.remove_group)

        config = app.config
        self.array_mode = RadioButtons(
            self.figure.add_axes([0.675, 0.822, 0.140, 0.09]), ("Generic", "Speaker-specific"),
            active=int(config.speaker_array is not None),
        )
        self.propagation = RadioButtons(
            self.figure.add_axes([0.830, 0.822, 0.140, 0.09]), ("ASM", "Measured far field"),
            active=int(config.propagation_mode == "measured"),
        )
        self.presets = RadioButtons(self.figure.add_axes([0.675, 0.686, 0.158, 0.111]), PRESETS)
        self._button("Load preset", 0.845, 0.721, 0.125, self.load_preset)
        self._field("frequencies", "Sweep frequencies (Hz)", "", 0.675, 0.632)
        self.figure.text(0.675, 0.605, "Comma-separated or start:stop:count. Blank uses Frequency (Hz).", fontsize=8)
        self._button("Apply run settings", 0.675, 0.548, 0.145, self.apply_run_settings)
        self._button("Recalculate", 0.830, 0.548, 0.140, lambda: app.recalculate())
        for index, (key, label, initial) in enumerate((
            ("display_frequency", "Display frequency (Hz)", ""),
            ("pressure_view", "Pressure view", "level"),
            ("floor_db", "Level floor (dB)", "-40"),
            ("pressure_reference", "Pressure reference", ""),
            ("directivity_reference", "Directivity reference", ""),
            ("weights", "Spectral weights", ""),
            ("weighting", "Weighting scheme", "discrete"),
            ("bin_widths", "PSD bin widths (Hz)", ""),
        )):
            self._field(key, label, initial, 0.675, 0.493 - 0.046 * index)
        self.figure.text(0.675, 0.139, "Pressure view: level, magnitude, phase. Blank references use peaks.", fontsize=8)
        self._button("Apply display", 0.675, 0.083, 0.136, self.apply_display)
        self._button("Sweep plots", 0.830, 0.083, 0.140, self.sweep_plots)

        self.sync_configuration(config)
        if previous is not None:
            self._set_layout(previous.types, previous.groups, previous.elements)
            self._show_type()
            for key in ("display_frequency", "pressure_view", "floor_db", "pressure_reference",
                        "directivity_reference", "weights", "weighting", "bin_widths", "library_path"):
                self._set(key, previous._text(key))

    def _field(self, key, label, initial, x, y) -> None:
        width = 0.153 if x == 0.675 else 0.174
        ax = self.figure.add_axes([x + 0.142, y, width, 0.032])
        box = InputBox(ax, label + " ", initial=initial, label_pad=0.025)
        box.label.set_fontsize(8)
        box.text_disp.set(fontsize=9, clip_on=True)
        self.fields[key] = box

    def _button(self, label, x, y, width, callback) -> None:
        button = Button(self.figure.add_axes([x, y, width, 0.036]), label)
        button.label.set_fontsize(8)

        def run(_event):
            try:
                self._message.set_text("")
                callback()
            except (OSError, ValueError, TypeError, OverflowError, MemoryError, KeyError, IndexError, csv.Error) as exc:
                self._message.set_text(f"Input: {exc}")
            self.figure.canvas.draw_idle()

        button.on_clicked(run)
        self.buttons[label] = button

    def _text(self, key) -> str:
        return self.fields[key].text.strip()

    def _set(self, key, value) -> None:
        self.fields[key].set_text(value)

    def _set_layout(self, types, groups, elements) -> None:
        self.types, self.groups, self.elements = types, groups, elements
        self._group_summary.set_text(f"{len(groups)} groups · {len(elements)} individually placed elements")

    def sync_configuration(self, config: SimulationConfig) -> None:
        array = config.speaker_array
        self._set_layout(
            array.speaker_types if array else (), array.groups if array else (), array.elements if array else (),
        )
        self._type_index = 0
        self._show_type()
        self._set("group_index", "")
        self._set("group_type", self.types[0].type_id if self.types else "")
        self.array_mode.set_active(int(array is not None))
        self.propagation.set_active(int(config.propagation_mode == "measured"))
        frequencies = config.frequencies_hz or ()
        self._set("frequencies", ", ".join(map(str, frequencies)))
        self.figure.canvas.draw_idle()

    def _show_type(self) -> None:
        item = self.types[self._type_index] if self.types else None
        self._set("type_id", item.type_id if item else "")
        self._set("type_name", item.name if item else "")
        self._set("diameter", item.diameter_m if item else "")
        self._set("response", "keep" if item else "flat")
        self._set("response_path", "")
        self._set("pattern_path", "")
        if item:
            model = item.frequency_response_metadata or {}
            response = model.get("schema") or model.get("model") or (
                "tabulated H" if item.frequency_response else item.response_model
            )
            pattern = " · pattern attached" if item.measured_pattern else ""
            self._type_summary.set_text(f"{self._type_index + 1} of {len(self.types)} · {response}{pattern}")
        else:
            self._type_summary.set_text("Library empty. Enter a type or import a library.")

    def _step_type(self, offset: int) -> None:
        if self.types:
            self._type_index = (self._type_index + offset) % len(self.types)
            self._show_type()

    def _update_layout(self, types, groups) -> None:
        config = self.app._read_controls()
        array = SpeakerArray(types, self.elements, groups) if self.elements or groups else None
        if array is not None:
            generate_array(replace(config, speaker_array=array))
        if config.speaker_array is not None:
            if array is None:
                raise ValueError("Switch to Generic before removing the last placement")
            self.app.apply_configuration(replace(config, speaker_array=array))
        self._set_layout(types, groups, self.elements)

    def save_type(self) -> None:
        type_id = self._text("type_id")
        catalog = {item.type_id: item for item in self.types}
        current = catalog.get(type_id) or SpeakerType(type_id, self._text("type_name"), float(self._text("diameter")))
        model = self._text("response").lower()
        changes = {}
        if model == "flat":
            changes = dict(frequency_response=None, frequency_response_metadata=None, response_model="idealized_flat")
        elif model == "synthetic_lowpass":
            corner = float(self._text("response_path"))
            response = FrequencyResponse.synthetic_lowpass(tuple(corner * np.geomspace(0.01, 100, 81)), corner)
            changes = dict(
                frequency_response=response, response_model="prescribed_velocity",
                frequency_response_metadata={"model": "synthetic_lowpass", "corner_hz": corner},
            )
        elif model == "csv":
            measurement = load_frequency_response_csv(self._text("response_path"))
            changes = dict(
                frequency_response=measurement.response, frequency_response_metadata=measurement.metadata,
                response_model="prescribed_velocity",
            )
        elif model != "keep":
            raise ValueError("Choose keep, flat, synthetic_lowpass, or csv for Velocity H")
        path = self._text("pattern_path")
        if path:
            changes["measured_pattern"] = None if path == "-" else load_element_pattern_csv(path)
        catalog[type_id] = replace(
            current, name=self._text("type_name"), diameter_m=float(self._text("diameter")), **changes,
        )
        self._update_layout(tuple(catalog.values()), self.groups)
        self._type_index = tuple(catalog).index(type_id)
        self._show_type()

    def remove_type(self) -> None:
        type_id = self._text("type_id")
        if any(item.type_id == type_id for item in (*self.groups, *self.elements)):
            raise ValueError("Remove or reassign placements using this type first")
        self._update_layout(tuple(item for item in self.types if item.type_id != type_id), self.groups)
        self._type_index = 0
        self._show_type()

    def _group_index(self, *, required=False) -> int | None:
        text = self._text("group_index")
        if not text and not required:
            return None
        if not text or not 1 <= int(text) <= len(self.groups):
            raise ValueError("Enter an existing group number")
        return int(text) - 1

    def read_group(self) -> None:
        index = self._group_index(required=True)
        item = self.groups[index]
        crossover = item.signal_chain.crossover if item.signal_chain else None
        offsets = item.row_offsets_m
        for key, value in (
            ("group_type", item.type_id), ("group_rows", item.rows), ("group_columns", item.columns),
            ("group_center", _joined(item.center_m)), ("group_gaps", _joined((item.horizontal_gap_m, item.vertical_gap_m))),
            ("group_offsets", _joined(offsets) if isinstance(offsets, tuple) else offsets),
            ("filter_kind", crossover.kind if crossover else "none"),
            ("filter_family", crossover.family if crossover else "butterworth"),
            ("filter_cutoffs", _joined(np.atleast_1d(crossover.cutoff_hz)) if crossover else ""),
            ("filter_order", crossover.order if crossover else 2),
        ):
            self._set(key, value)

    def save_group(self) -> None:
        index = self._group_index()
        current = self.groups[index] if index is not None else None
        kind = self._text("filter_kind").lower()
        crossover = None
        if kind != "none":
            cutoffs = _numbers(self._text("filter_cutoffs"))
            crossover = Crossover(
                kind, cutoffs[0] if len(cutoffs) == 1 else cutoffs, int(self._text("filter_order")),
                self._text("filter_family").lower(),
            )
        chain = current.signal_chain if current is not None else None
        if crossover is not None or chain is not None:
            chain = replace(chain or SignalChain(), crossover=crossover)
        offsets = _numbers(self._text("group_offsets")) or (0.0,)
        gap_x, gap_y = _numbers(self._text("group_gaps"))
        data = dict(
            type_id=self._text("group_type"), rows=int(self._text("group_rows")), columns=int(self._text("group_columns")),
            center_m=_numbers(self._text("group_center")), horizontal_gap_m=gap_x, vertical_gap_m=gap_y,
            row_offsets_m=offsets[0] if len(offsets) == 1 else offsets, signal_chain=chain,
        )
        groups = list(self.groups)
        if index is None:
            index = len(groups)
            groups.append(SpeakerGroup(**data))
        else:
            groups[index] = replace(current, **data)
        self._update_layout(self.types, tuple(groups))
        self._set("group_index", index + 1)

    def remove_group(self) -> None:
        index = self._group_index(required=True)
        self._update_layout(self.types, self.groups[:index] + self.groups[index + 1:])
        self._set("group_index", "")

    def _library_path(self) -> Path:
        if not self._text("library_path"):
            raise ValueError("Enter a speaker library JSON path")
        return Path(self._text("library_path")).expanduser()

    def import_library(self) -> None:
        path = self._library_path()
        document = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(document, dict) or set(document) != {"speaker_types"} or not isinstance(document["speaker_types"], list):
            raise ValueError("A library JSON needs a speaker_types list")
        entries = [SpeakerType.from_dict(entry) for entry in document["speaker_types"]]
        if len({item.type_id for item in entries}) != len(entries):
            raise ValueError("Speaker type IDs must be unique within a library")
        catalog = {item.type_id: item for item in (*self.types, *entries)}
        self._update_layout(tuple(catalog.values()), self.groups)
        self._type_index = 0
        self._show_type()
        self._message.set_text(f"Imported {path}.")

    def export_library(self) -> None:
        path = self._library_path()
        content = {"speaker_types": [item.to_dict() for item in self.types]}
        path.write_text(json.dumps(content, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        self._message.set_text(f"Exported {path}.")

    def load_preset(self) -> None:
        config = preset_configuration(self.presets.value_selected)
        self.app.apply_configuration(config)
        self.sync_configuration(config)
        self._message.set_text("Preset loaded. Select Recalculate to run it.")

    def apply_run_settings(self) -> None:
        config = self.app._read_controls()
        types, groups = self.types, self.groups
        array = None
        if self.array_mode.value_selected == "Speaker-specific":
            if not types:
                types = (SpeakerType.generic(config.speaker_diameter_m),)
            if not groups and not self.elements:
                speaker = types[self._type_index]
                groups = (SpeakerGroup(
                    speaker.type_id, config.rows, config.columns,
                    horizontal_gap_m=config.horizontal_gap_m, vertical_gap_m=config.vertical_gap_m,
                    row_offsets_m=config.row_stagger * (speaker.diameter_m + config.horizontal_gap_m),
                ),)
            array = SpeakerArray(types, self.elements, groups)
        measured = self.propagation.value_selected == "Measured far field"
        if measured and array is None:
            raise ValueError("Measured far field requires Speaker-specific mode")
        config = replace(
            config, speaker_array=array, propagation_mode="measured" if measured else "asm",
            frequencies_hz=_frequencies(self._text("frequencies")),
        )
        self.app.apply_configuration(config)
        self._set_layout(types, groups, self.elements)
        self._show_type()
        self._message.set_text("Run settings applied. Select Recalculate to update the plots.")

    def apply_display(self) -> None:
        self.app.set_display(
            frequency_hz=self._text("display_frequency"), pressure_view=self._text("pressure_view"),
            floor_db=float(self._text("floor_db")), pressure_reference=self._text("pressure_reference"),
            directivity_reference=self._text("directivity_reference"),
        )

    def sweep_plots(self) -> None:
        self.app.show_sweep_plots(
            weights=_numbers(self._text("weights")), weighting=self._text("weighting"),
            bin_widths=_numbers(self._text("bin_widths")) or None,
        )
