import json
from dataclasses import replace
from threading import Event

import matplotlib
import pytest

matplotlib.use("Agg")

from matplotlib import pyplot as plt
from matplotlib.backend_bases import KeyEvent, MouseEvent

import simulation
from config import SimulationConfig
from interface import SimulatorInterface, load_configuration, save_configuration


@pytest.fixture
def app():
    interface = SimulatorInterface(SimulationConfig(rows=2, columns=4, speaker_diameter_m=0.10, horizontal_gap_m=0.02, grid_spacing_m=0.005))
    yield interface
    interface.close()
    plt.close(interface.figure)


def _recalculate(app):
    app.recalculate()
    if app._pending is not None:
        app._pending[1].result(timeout=10)
        app._finish_recalculation()


def test_geometry_edits_are_live_and_acoustics_wait_for_recalculate(app, monkeypatch):
    prepare = simulation.prepare_velocity_spectrum
    propagate = simulation.pressure_from_spectrum
    directivity = simulation.directivity_from_spectrum
    calls = []

    def track(name, function):
        def tracked(*args, **kwargs):
            calls.append(name)
            return function(*args, **kwargs)
        return tracked

    monkeypatch.setattr(simulation, "prepare_velocity_spectrum", track("fft", prepare))
    monkeypatch.setattr(simulation, "pressure_from_spectrum", track("pressure", propagate))
    monkeypatch.setattr(simulation, "directivity_from_spectrum", track("directivity", directivity))

    app.fields["rows"][0].set_val("3")
    assert len(app.layout_ax.patches) == 12
    app.fields["row_stagger"][0].set_val("0.25")
    x = [patch.center[0] for patch in app.layout_ax.patches]
    assert x[4] - x[0] == pytest.approx(0.03)
    assert calls == []
    assert len(app.pressure_ax.images) == 0

    _recalculate(app)
    assert calls == ["fft", "pressure", "directivity"]
    assert len(app.aperture_ax.images) == len(app.pressure_ax.images) == 1
    assert len(app.directivity_ax.lines) == 1
    axes_count = len(app.figure.axes)
    _recalculate(app)
    assert calls == ["fft", "pressure", "directivity"]

    app.fields["observation_distance_m"][0].set_val("0.35")
    assert "last calculation" in app._status.get_text()
    _recalculate(app)
    assert calls == ["fft", "pressure", "directivity", "pressure"]
    assert "0.35 m" in app.pressure_ax.get_title()
    assert len(app.figure.axes) == axes_count


def test_json_round_trip_including_complex_weights_and_optional_settings(tmp_path):
    config = SimulationConfig(
        rows=1, columns=2, row_stagger=0.25, speaker_weights=(1 + 2j, -0.5j),
        speaker_taper=(0.4, 1), grid_margin_m=(0.01, 0.02), observation_shape=(4, 5),
        focus_point_m=(0.1, 0.2, 0.3), beamforming_mode="focus",
    )
    path = tmp_path / "array.json"
    save_configuration(path, config)
    assert json.loads(path.read_text())["speaker_weights"] == [[1, 2], [0, -0.5]]
    assert load_configuration(path) == config


def test_save_reset_and_load_restore_controls_without_running_fft(app, tmp_path):
    path = tmp_path / "array.json"
    app.path.set_val(str(path))
    app.fields["rows"][0].set_val("1")
    app.fields["columns"][0].set_val("2")
    app.fields["speaker_weights"][0].set_val("1, 0.5-0.5i")
    app.mode.set_active(1)
    app.profile.set_active(1)
    app.save()
    assert load_configuration(path).speaker_weights == (1, 0.5 - 0.5j)
    app.reset()
    assert len(app.layout_ax.patches) == 8
    assert app.profile.value_selected == "uniform"
    app.load()
    assert len(app.layout_ax.patches) == 2
    assert app.profile.value_selected == "cosine"
    assert app.mode.value_selected == "steer"
    assert not app.pressure_ax.images

    path.write_text('{"rows": 0}', encoding="utf-8")
    app.load()
    assert "Load failed" in app._status.get_text()
    assert app.config.rows == 1


def test_numerical_warnings_can_be_evaluated_without_allocating_grid(monkeypatch):
    def unexpected_grid(*args, **kwargs):
        pytest.fail("preview should calculate sizes without allocating sampled axes")

    monkeypatch.setattr(simulation, "build_array_grid", unexpected_grid)
    session = simulation.SimulationSession()
    default = SimulationConfig()
    coarse = replace(default, grid_spacing_m=0.07, padding_factor=1, observation_shape=(1, 1))
    warnings = session.warnings(coarse)
    assert any("Coarse sampling" in item for item in warnings)
    assert any("Padding" in item for item in warnings)

    large = replace(default, grid_spacing_m=1e-8)
    warnings = session.warnings(large)
    assert any("Large FFT" in item for item in warnings)


def test_switching_textboxes_commits_without_redrawing_for_inactive_inputs(app):
    canvas = app.figure.canvas
    draws = []
    canvas.mpl_connect("draw_event", lambda event: draws.append(event))

    def click(box):
        x, y = box.ax.transAxes.transform((0.5, 0.5))
        for event in ("button_press_event", "button_release_event"):
            canvas.callbacks.process(event, MouseEvent(event, canvas, x, y, button=1))

    def press(key):
        canvas.callbacks.process("key_press_event", KeyEvent("key_press_event", canvas, key))

    rows, columns = (app.fields[name][0] for name in ("rows", "columns"))
    click(rows)
    assert len(draws) <= 2
    press("end")
    press("backspace")
    press("3")
    assert app.config.rows == 2
    click(columns)
    assert app.config.rows == 3
    press("end")
    press("backspace")
    press("5")
    press("enter")
    assert app.config.columns == 5
    assert len(app.layout_ax.patches) == 15


def test_controls_remain_editable_during_recalculation(app, monkeypatch):
    calculate = app.session.recalculate
    started = Event()
    resume = Event()

    def wait_then_calculate(config):
        started.set()
        assert resume.wait(timeout=10)
        return calculate(config)

    monkeypatch.setattr(app.session, "recalculate", wait_then_calculate)
    app.recalculate()
    try:
        assert started.wait(timeout=2)
        pending = app._pending
        app.fields["observation_distance_m"][0].set_val("0.6")
        app.recalculate()
        assert app._pending is pending
        assert app.config.observation_distance_m == 0.6
    finally:
        resume.set()
    pending[1].result(timeout=10)
    app._finish_recalculation()
    assert "last calculation" in app._status.get_text()


def test_pressure_width_controls_resize_evaluated_window(app):
    x_field = app.fields["observation_width_x_m"][0]
    y_field = app.fields["observation_width_y_m"][0]
    x_field.set_val("1.6")
    y_field.set_val("0.8")
    assert app.config.observation_width_x_m == 1.6
    assert app.config.observation_width_y_m == 0.8
    assert not app.pressure_ax.images

    _recalculate(app)
    xmin, xmax, ymin, ymax = app.pressure_ax.images[0].get_extent()
    assert xmax - xmin == pytest.approx(1.6)
    assert ymax - ymin == pytest.approx(0.8)
    assert app.pressure_ax.images[0].get_array().shape == (160, 320)

    x_field.set_val("")
    assert app.config.observation_width_x_m is None
    assert app.config.observation_width_y_m == 0.8
    x_field.set_val("10")
    assert "Increase padding" in app._status.get_text()


def test_window_selection_previews_and_applies_amplitudes(app):
    app.window.set_active(1)  # Hann
    assert app.config.array_window == "hann"
    assert "Hann" in app.window_ax.get_title()
    amplitudes = app.window_ax.images[0].get_array()
    assert amplitudes.shape == (2, 4)
    assert amplitudes[0, 0] == 0
    assert amplitudes[0, 1] == pytest.approx(1)
    assert not app.aperture_ax.images
    _recalculate(app)
    assert app.aperture_ax.images
    assert app.directivity_ax.lines


def test_focus_controls_combine_direction_and_range(app, tmp_path):
    app._set_controls(replace(app.config, beamforming_mode="focus", focus_distance_m=None, focus_point_m=(0.1, -0.2, 0.3)))
    field = app.fields["focus_distance_m"][0]
    assert float(field.text) == pytest.approx((0.1**2 + 0.2**2 + 0.3**2)**0.5)
    assert app._read_controls().focus_point_m == pytest.approx((0.1, -0.2, 0.3))

    app.fields["steering_azimuth_deg"][0].set_val("90")
    app.fields["steering_elevation_deg"][0].set_val("30")
    field.set_val("2")
    target = (0, 3**0.5, 1)
    assert app.config.focus_point_m == pytest.approx(target)
    _recalculate(app)
    assert app._last_run.focus_point_m == pytest.approx(target)

    path = tmp_path / "focus.json"
    app.path.set_val(str(path))
    app.save()
    saved = load_configuration(path)
    assert saved.focus_point_m == pytest.approx(target)
    assert saved.focus_distance_m == 2

    field.set_val("0")
    assert "focus distance must be finite and positive" in app._status.get_text()
