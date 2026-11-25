from concurrent.futures import CancelledError
from dataclasses import replace
from pathlib import Path
from threading import Event

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from matplotlib import pyplot as plt
from matplotlib.backend_bases import MouseEvent

import simulation
from config import SimulationConfig, load_configuration
from interface import SimulatorInterface
from interface_run import SweepPreview
from simulation import SimulationSession

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "stage8"


@pytest.fixture
def app(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda **kwargs: None)
    before = set(plt.get_fignums())
    interface = SimulatorInterface(SimulationConfig(
        rows=1, columns=1, speaker_diameter_m=0.16, padding_factor=5, grid_spacing_m=0.008,
        observation_distance_m=0.4, observation_shape=(13, 15),
    ))
    yield interface
    interface.close()
    for number in set(plt.get_fignums()) - before:
        plt.close(number)


def _click(button):
    canvas = button.ax.figure.canvas
    x, y = button.ax.transAxes.transform((0.5, 0.5))
    for name in ("button_press_event", "button_release_event"):
        canvas.callbacks.process(name, MouseEvent(name, canvas, x, y, button=1))


def _edit(editor, **fields):
    for key, value in fields.items():
        editor.fields[key].set_text(value)


def _finish(app):
    assert app._pending is not None, app._status.get_text()
    result = app._pending[1].result(timeout=10)
    app._finish_recalculation()
    assert app._last_result is result, app._status.get_text()
    return result


def _run(app):
    _click(app.buttons["Recalculate"])
    return _finish(app)


def test_generic_mode_round_trip_and_illustrative_presets(app):
    assert app.config.speaker_array is None
    assert app.config.frequencies_hz is None
    assert app._editor is None
    assert not app.aperture_ax.images
    initial = _run(app)
    _click(app.buttons["Array & sweep settings"])
    editor = app._editor
    editor.array_mode.set_active(1)
    _click(editor.buttons["Apply run settings"])
    assert app.config.speaker_array.speaker_types[0].response_model == "idealized_flat"
    typed = _run(app)
    np.testing.assert_array_equal(typed.pressure.pressure, initial.pressure.pressure)
    editor.array_mode.set_active(0)
    _click(editor.buttons["Apply run settings"])
    assert app.config.speaker_array is None
    assert len(app.layout_ax.patches) == 1

    for choice, expected in [(1, ("lowpass", "highpass")), (2, ("lowpass", "bandpass", "highpass"))]:
        editor.presets.set_active(choice)
        _click(editor.buttons["Load preset"])
        config = app.config
        assert len(app.layout_ax.patches) == len(expected)
        assert tuple(group.signal_chain.crossover.kind for group in config.speaker_array.groups) == expected
        assert len(config.frequencies_hz) > 2
        assert app._pending is None
    editor.presets.set_active(0)
    _click(editor.buttons["Load preset"])
    assert app.config.speaker_array is None
    assert app.config.frequencies_hz is None


def test_library_groups_complex_responses_and_asm_sweep(app, tmp_path, monkeypatch):
    editor = app.open_editor()
    _edit(editor, type_id="woofer", type_name="Low driver", diameter=0.16, response="csv",
          response_path=EXAMPLES / "synthetic_velocity_response.csv")
    _click(editor.buttons["Save type"])
    _edit(editor, group_type="woofer", group_center="-0.12, 0", filter_kind="lowpass", filter_cutoffs=800)
    _click(editor.buttons["Save group"])
    _edit(editor, type_id="tweeter", type_name="High driver", diameter=0.075, response="synthetic_lowpass",
          response_path=5000)
    _click(editor.buttons["Save type"])
    _edit(editor, group_index="", group_type="tweeter", group_center="0.07, 0", filter_kind="highpass", filter_cutoffs=800)
    _click(editor.buttons["Save group"])
    assert not editor._message.get_text()
    assert len(app.layout_ax.patches) == 1  # Draft groups are inactive in Generic mode.
    editor.array_mode.set_active(1)
    _edit(editor, frequencies="500:1000:2")
    _click(editor.buttons["Apply run settings"])
    assert [patch.radius for patch in app.layout_ax.patches] == pytest.approx([0.08, 0.0375])
    assert app._pending is None

    config = app.config
    assert config.frequencies_hz == (500, 1000)
    result = _run(app)
    assert isinstance(result, SweepPreview)
    expected = SimulationSession().recalculate_frequencies(config, config.frequencies_hz)
    np.testing.assert_allclose(result.pressure.pressure, expected.pressure)
    grid = result.aperture_grid
    iy = np.abs(grid.y).argmin()
    for element, item in zip(config.speaker_array.flattened_elements, config.speaker_array.element_types()):
        ix = np.abs(grid.x - element.position_m[0]).argmin()
        f = np.array(config.frequencies_hz)
        effective_drive = element.signal_chain.evaluate(f) * item.frequency_response.evaluate(f)
        assert np.any(effective_drive.imag != 0)
        np.testing.assert_allclose(result.velocity[:, iy, ix], effective_drive)

    def unexpected_calculation(*args, **kwargs):
        pytest.fail("Changing display settings must reuse the displayed sweep")

    monkeypatch.setattr(simulation, "prepare_velocity_spectrum", unexpected_calculation)
    monkeypatch.setattr(app.session, "recalculate", unexpected_calculation)
    _edit(editor, display_frequency=1000, pressure_view="phase", floor_db=-65, weights="1, 2")
    _click(editor.buttons["Apply display"])
    assert "Pressure phase" in app.pressure_ax.get_title()
    assert "1000 Hz" in app.directivity_ax.get_title()
    assert app._last_result is result
    _click(editor.buttons["Recalculate"])
    assert app._pending is None
    before = len(plt.get_fignums())
    _click(editor.buttons["Sweep plots"])
    assert len(plt.get_fignums()) - before == 3

    configuration = tmp_path / "array.json"
    app.path.set_text(configuration)
    _click(app.buttons["Save Configuration"])
    assert load_configuration(configuration) == config
    library = tmp_path / "speakers.json"
    _edit(editor, library_path=library)
    _click(editor.buttons["Export library"])
    _click(app.buttons["Reset"])
    assert not editor.types
    _click(editor.buttons["Import / merge"])
    assert editor.types == config.speaker_array.speaker_types
    _click(app.buttons["Load Configuration"])
    assert app.config == config
    assert editor.groups == config.speaker_array.groups
    assert app._pending is None


def test_measured_far_field_and_numerical_convergence_through_widgets(app, tmp_path, monkeypatch):
    editor = app.open_editor()
    _edit(editor, type_id="piston", type_name="Synthetic piston", diameter=0.16,
          pattern_path=EXAMPLES / "synthetic_piston_pattern.csv")
    _click(editor.buttons["Save type"])
    _edit(editor, group_type="piston", group_center="0, 0")
    _click(editor.buttons["Save group"])
    editor.array_mode.set_active(1)
    _edit(editor, frequencies="1000, 1500")
    _click(editor.buttons["Apply run settings"])

    app.fields["grid_spacing_m"][0].set_val("0.04")
    app.fields["padding_factor"][0].set_val("3")
    assert "Coarse sampling" in app._warnings.get_text()
    coarse = _run(app).directivity.directivity
    app.fields["grid_spacing_m"][0].set_val("0.0025")
    app.fields["padding_factor"][0].set_val("16")
    fine = _run(app).directivity

    def unexpected_fft(*args, **kwargs):
        pytest.fail("Measured mode must use the far-field pattern sum")

    monkeypatch.setattr(simulation, "prepare_velocity_spectrum", unexpected_fft)
    monkeypatch.setattr(simulation, "build_array_grid", unexpected_fft)
    editor.propagation.set_active(1)
    _click(editor.buttons["Apply run settings"])
    result = _run(app)
    assert result.pressure is None
    assert not app.pressure_ax.images
    assert "Measured far-field" in app.directivity_ax.get_title()
    assert "far field only" in app._warnings.get_text()
    measured = result.directivity.directivity
    # Use the imported angular sample nodes to isolate solver convergence from interpolation.
    nodes = np.remainder(fine.theta_deg, 15) == 0
    peak = np.abs(measured[:, nodes]).max(axis=1)
    coarse_error = np.abs(coarse[:, nodes] - measured[:, nodes]).max(axis=1) / peak
    fine_error = np.abs(fine.directivity[:, nodes] - measured[:, nodes]).max(axis=1) / peak
    assert (fine_error < 0.01).all()
    assert (fine_error < coarse_error / 5).all()

    path = tmp_path / "measured-array.json"
    app.path.set_text(path)
    _click(app.buttons["Save Configuration"])
    assert load_configuration(path).propagation_mode == "measured"
    _click(app.buttons["Reset"])
    _click(app.buttons["Load Configuration"])
    assert editor.propagation.value_selected == "Measured far field"
    assert editor.types[0].measured_pattern is not None


def test_stale_sweep_is_discarded_and_only_latest_request_is_applied(app, monkeypatch):
    baseline = _run(app)
    started, resume = Event(), Event()
    frequencies = []
    calculate = app.session.recalculate

    def wait_on_first_frequency(config):
        frequencies.append(config.frequency_hz)
        if config.frequency_hz == 500:
            started.set()
            assert resume.wait(timeout=10)
        return calculate(config)

    monkeypatch.setattr(app.session, "recalculate", wait_on_first_frequency)
    app.apply_configuration(replace(app.config, frequencies_hz=(500, 600)))
    app.recalculate()
    first = app._pending[1]
    try:
        assert started.wait(timeout=2)
        app.apply_configuration(replace(app.config, frequencies_hz=(700, 800)))
        app.recalculate()
        app.apply_configuration(replace(app.config, frequencies_hz=(900, 1100)))
        app.recalculate()
        assert app._pending[1] is first
    finally:
        resume.set()
    with pytest.raises(CancelledError):
        first.result(timeout=10)
    app._finish_recalculation()
    assert app._last_result is baseline
    result = _finish(app)
    np.testing.assert_array_equal(result.directivity.frequencies_hz, [900, 1100])
    assert frequencies == [500, 900, 1100]
