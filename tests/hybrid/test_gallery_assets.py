"""Checks for scenario-derived documentation figures."""

import importlib
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from flowcean.hybrid import build_hybrid_system_dot, simulate
from flowcean.hybrid.hybrid_system import display_label

EXAMPLE_DIR = Path(__file__).resolve().parents[2] / "examples/hybrid_systems"
# The generator is a directly executable example with a sibling scenarios import.
previous_scenarios = sys.modules.pop("scenarios", None)
sys.path.insert(0, str(EXAMPLE_DIR))
try:
    gallery = importlib.import_module("gallery_assets")
finally:
    sys.path.remove(str(EXAMPLE_DIR))
    if previous_scenarios is None:
        sys.modules.pop("scenarios", None)
    else:
        sys.modules["scenarios"] = previous_scenarios


@pytest.mark.parametrize(
    ("slug", "name", "sample_dt"),
    [
        ("thermostat", "Thermostat", 0.02),
        ("bouncing_ball", "Bouncing Ball", 0.005),
        ("wind_turbine", "Wind Turbine", 0.1),
    ],
)
def test_profiles_reuse_scenarios(slug, name, sample_dt):
    profile = gallery.profile_for(slug)
    original = next(s for s in gallery.SCENARIOS if s.name == name)
    assert profile.scenario is original
    assert profile.sample_dt == sample_dt
    assert profile.scenario.t_span == original.t_span
    assert profile.scenario.input_stream is original.input_stream
    assert profile.scenario.factory is original.factory
    with pytest.raises(ValueError, match="Unknown profile"):
        gallery.profile_for("unknown")


def test_mode_mappings_are_complete_explicit_and_theme_invariant():
    for profile in gallery.PROFILES:
        system = profile.scenario.factory()
        names = [display_label(loc) for loc in system.locations]
        assert list(profile.colors) == names
        assert len(profile.colors) == len(set(profile.colors))
        assert all(
            color.startswith("#") and len(color) == 7
            for color in profile.colors.values()
        )
        assert all(
            mode.label and mode.color == profile.colors[mode.name]
            for mode in profile.modes
        )
        for theme in ("light", "dark"):
            dot = gallery.styled_dot(profile, system, theme)
            assert gallery.PALETTES[theme].background in dot
            for color in profile.colors.values():
                assert f'fillcolor="{color}"' in dot


def test_documented_index_matches_configured_scenarios():
    page = EXAMPLE_DIR.parents[1] / "docs/examples/hybrid_systems.md"
    rows = re.findall(
        r"\| \[([^]]+)\]\([^)]+\) \| (\d+) \| (\d+) \| ([^|]+) \|",
        page.read_text(encoding="utf-8"),
    )
    documented = {
        name.casefold(): (int(states), int(modes), signal.strip() != "None")
        for name, states, modes, signal in rows
    }
    expected = {}
    for scenario in gallery.SCENARIOS:
        system = scenario.factory()
        expected[scenario.name.casefold()] = (
            len(system.initial_state),
            len(system.locations),
            scenario.input_stream is not None,
        )
    assert len(rows) == len(expected)
    assert documented == expected


@pytest.mark.parametrize("profile", gallery.PROFILES, ids=lambda p: p.slug)
@pytest.mark.parametrize("theme", ["light", "dark"])
def test_styled_dot_preserves_all_declared_edges_and_start(profile, theme):
    system = profile.scenario.factory()
    original = build_hybrid_system_dot(system, show_direction=True)
    styled = gallery.styled_dot(profile, system, theme)
    assert original == build_hybrid_system_dot(system, show_direction=True)
    for line in original.splitlines():
        if " -> " in line:
            compact = line.replace('label="event: ', 'label="', 1)
            compact = compact.replace(r"\ndirection: ", r"\n", 1)
            if '[label="' in compact:
                compact = compact.replace('label="', 'label="  ', 1)
                compact = compact.replace(r"\n", r"  \n  ")
                compact = compact.replace('"];', '  "];')
            assert styled.count(compact) == original.count(line)
    assert styled.index("  node [") < styled.index("  location_0 [")
    assert styled.index("  edge [") < styled.index("  start -> ")
    assert styled.index("  start [shape=point, color=") < styled.index(
        "  start -> "
    )
    assert 'fontcolor="' + gallery.PALETTES[theme].foreground + '"' in styled
    for index, mode in enumerate(profile.modes):
        assert f"location_{index} [label=" in styled
        assert f'color="{mode.color}", fillcolor="{mode.color}"' in styled
    if profile.slug == "wind_turbine":
        assert "rankdir=TB" in styled
    else:
        assert "rankdir=LR" in styled


def test_ball_reset_uses_library_plot_breaks_and_exact_event_states():
    profile = gallery.profile_for("bouncing_ball")
    system = profile.scenario.factory()
    trace = simulate(
        system,
        t_span=profile.scenario.t_span,
        input_stream=profile.scenario.input_stream,
        sample_dt=profile.sample_dt,
        rtol=1e-7,
        atol=1e-9,
    )
    with plt.rc_context({"svg.fonttype": "none"}):
        fig = plt.figure()
        try:
            gallery._ball_trace(fig, profile, trace, gallery.PALETTES["light"])
            height, velocity = fig.axes
            for ax in (height, velocity):
                data_t = np.asarray(ax.lines[0].get_xdata())
                data_y = np.asarray(ax.lines[0].get_ydata())
                for event in trace.events:
                    exact = np.flatnonzero(data_t == event.time)
                    assert len(exact) == 3
                    assert np.isnan(data_y[exact[1]])
            resets = [event for event in trace.events if event.reset]
            assert len(velocity.lines) == 1 + len(resets)
            for line, event in zip(velocity.lines[1:], resets, strict=True):
                np.testing.assert_array_equal(
                    line.get_xdata(), [event.time, event.time]
                )
                np.testing.assert_array_equal(
                    line.get_ydata(),
                    [event.state_before[1], event.state_after[1]],
                )
                assert line.get_linestyle() == "--"
            assert height.get_legend() is None
            legend = velocity.get_legend()
            assert legend is not None
            assert [text.get_text() for text in legend.get_texts()] == [
                "Bounce reset"
            ]
            assert "x0" not in height.get_legend_handles_labels()[1]
            assert "x1" not in velocity.get_legend_handles_labels()[1]
        finally:
            plt.close(fig)


def test_render_profile_simulates_once_and_renders_four_assets(monkeypatch):
    calls = []
    actual = gallery.simulate

    def recorded(*args, **kwargs):
        calls.append(kwargs)
        return actual(*args, **kwargs)

    monkeypatch.setattr(gallery, "simulate", recorded)
    monkeypatch.setattr(gallery, "render_dot_svg", lambda dot: "<svg/>")
    profile = gallery.profile_for("thermostat")
    assets, result = gallery.render_profile(profile)
    assert set(assets) == {
        f"thermostat-{kind}-{theme}.svg"
        for kind in ("automaton", "trace")
        for theme in ("light", "dark")
    }
    assert "events=5" in result
    assert len(calls) == 1
    assert calls[0] == {
        "t_span": profile.scenario.t_span,
        "input_stream": profile.scenario.input_stream,
        "sample_dt": 0.02,
        "rtol": 1e-7,
        "atol": 1e-9,
    }


def test_default_output_is_independent_of_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert (
        EXAMPLE_DIR.parents[1] / "docs/assets/hybrid_systems"
    ) == gallery.DEFAULT_OUTPUT_DIR
    assert gallery.DEFAULT_OUTPUT_DIR.is_absolute()


def test_missing_graphviz_reports_actionable_error(
    tmp_path, monkeypatch, capsys
):
    def missing_renderer(dot):
        raise RuntimeError("Install Graphviz and make dot available on PATH.")

    monkeypatch.setattr(gallery, "render_dot_svg", missing_renderer)
    with pytest.raises(SystemExit) as error:
        gallery.main(
            ["--profile", "thermostat", "--output-dir", str(tmp_path)]
        )
    assert error.value.code == 1
    assert "Graphviz" in capsys.readouterr().err
    assert not list(tmp_path.iterdir())


def test_check_never_mutates_destination_and_reports_missing_or_different(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(
        gallery,
        "render_profile",
        lambda profile: (
            {f"{profile.slug}-trace-light.svg": "<svg>expected</svg>"},
            "sampled once",
        ),
    )
    args = ["--profile", "thermostat", "--output-dir", str(tmp_path)]
    asset = tmp_path / "thermostat-trace-light.svg"
    assert gallery.main([*args, "--check"]) == 1
    assert "stale asset" in capsys.readouterr().out
    assert not asset.exists()
    asset.write_text("old", encoding="utf-8")
    assert gallery.main([*args, "--check"]) == 1
    assert asset.read_text(encoding="utf-8") == "old"
    asset.write_text("<svg>expected</svg>", encoding="utf-8")
    assert gallery.main([*args, "--check"]) == 0
    assert gallery.main(args) == 0
    assert asset.read_text(encoding="utf-8") == "<svg>expected</svg>"
