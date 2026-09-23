"""DOT export and external rendering of hybrid-system structure."""

from types import SimpleNamespace
from typing import Never
from unittest.mock import Mock

import numpy as np
import pytest

from flowcean.hybrid import _graphviz, build_hybrid_system_dot, render_dot_svg
from flowcean.hybrid.hybrid_system import (
    ContinuousDynamics,
    CrossingDirection,
    EventSurface,
    HybridSystem,
    Location,
    Reset,
    SurfaceEntryPolicy,
    Transition,
    display_label,
)
from flowcean.hybrid.hydra.selector import graph as selector_graph
from flowcean.hybrid.hydra.selector.inspection import (
    SelectorInspection,
    SelectorLeafInspection,
    SelectorNodeInspection,
)


class AnonymousCallback:
    """Callback whose default repr includes an allocation-specific address."""

    def __call__(self, *args: object) -> Never:
        raise AssertionError("graph export must not execute callbacks")


def _system(
    locations: list[Location],
    transitions: list[Transition],
    *,
    initial: Location | None = None,
) -> HybridSystem:
    return HybridSystem(
        locations=locations,
        transitions=transitions,
        initial_location=initial if initial is not None else locations[0],
        initial_state=np.array([0.0]),
    )


def test_exact_dot_preserves_order_nonfirst_initial_isolated_and_parallel_edges() -> (
    None
):
    first = Location(AnonymousCallback(), label="same")
    second = Location(ContinuousDynamics(AnonymousCallback(), label="flow"))
    isolated = Location(AnonymousCallback(), label="same")
    event = EventSurface(AnonymousCallback(), label="arrive")
    transitions = [
        Transition(first, second, event),
        Transition(first, second, event),
        Transition(second, second, event, reset=Reset(AnonymousCallback())),
    ]
    system = _system([first, second, isolated], transitions, initial=second)
    assert build_hybrid_system_dot(system) == "\n".join(
        [
            "digraph HybridSystem {",
            "  rankdir=LR;",
            "  start [shape=point];",
            '  location_0 [label="same"];',
            '  location_1 [label="flow"];',
            '  location_2 [label="same"];',
            "  start -> location_1;",
            '  location_0 -> location_1 [label="event: arrive"];',
            '  location_0 -> location_1 [label="event: arrive"];',
            '  location_1 -> location_1 [label="event: arrive\\nreset: reset_2"];',
            "}",
        ],
    )


def test_anonymous_fallbacks_and_original_display_label_behavior() -> None:
    anonymous = AnonymousCallback()
    location = Location(anonymous)
    event = EventSurface(anonymous)
    reset = Reset(anonymous)
    target = Location(anonymous, label="target")
    system = _system(
        [location, target],
        [Transition(location, target, event, reset)],
    )
    dot = build_hybrid_system_dot(system)
    assert 'location_0 [label="location_0"]' in dot
    assert 'label="event: event_0\\nreset: reset_0"' in dot
    assert "0x" not in dot
    assert display_label(location, fallback="stable") == "stable"
    assert display_label(location) == repr(location)
    assert display_label(event) == repr(event)
    assert display_label(reset) == repr(reset)
    assert display_label(object(), fallback="unknown") == "unknown"


def test_callable_names_and_explicit_label_precedence() -> None:
    def named_callback(*args: object) -> Never:
        raise AssertionError("callback executed")

    location = Location(
        ContinuousDynamics(named_callback, label="dynamics"),
        label="location",
    )
    from_dynamics = Location(ContinuousDynamics(named_callback, label="flow"))
    from_callback = Location(named_callback)
    event = EventSurface(named_callback)
    reset = Reset(named_callback, label="explicit reset")
    system = _system(
        [location, from_dynamics, from_callback],
        [Transition(location, from_callback, event, reset)],
    )
    dot = build_hybrid_system_dot(system)
    assert 'location_0 [label="location"]' in dot
    assert 'location_1 [label="flow"]' in dot
    assert (
        'location_2 [label="test_callable_names_and_explicit_label_precedence.<locals>.named_callback"]'
        in dot
    )
    assert (
        "event: test_callable_names_and_explicit_label_precedence.<locals>.named_callback\\nreset: explicit reset"
        in dot
    )
    assert display_label(location, fallback="unused") == "location"


def test_dot_labels_escape_special_characters_and_crlf() -> None:
    first = Location(
        AnonymousCallback(), label='"quoted"\\slash\n<html>\r\nlast'
    )
    second = Location(AnonymousCallback(), label="other")
    event = EventSurface(AnonymousCallback(), label='e"\\\n<b>')
    reset = Reset(AnonymousCallback(), label="r\r\nnext")
    dot = build_hybrid_system_dot(
        _system([first, second], [Transition(first, second, event, reset)]),
    )
    assert 'location_0 [label="\\"quoted\\"\\\\slash\\n<html>\\nlast"]' in dot
    assert 'label="event: e\\"\\\\\\n<b>\\nreset: r\\nnext"' in dot
    assert "<html>" in dot
    assert "<b>" in dot
    assert "\r" not in dot


@pytest.mark.parametrize("line_ending", ["\n", "\r\n", "\r"])
def test_line_endings_do_not_introduce_graphviz_alignment(
    line_ending: str,
) -> None:
    label = f"first{line_ending}last"
    assert _graphviz._escape_dot_label(label) == r"first\nlast"
    assert _graphviz._escape_dot_label(r"first\rlast") == r"first\\rlast"


@pytest.mark.parametrize("direction", list(CrossingDirection))
@pytest.mark.parametrize("entry_policy", list(SurfaceEntryPolicy))
def test_annotations_and_flags(
    direction: CrossingDirection,
    entry_policy: SurfaceEntryPolicy,
) -> None:
    first = Location(AnonymousCallback())
    second = Location(AnonymousCallback())
    transition = Transition(
        first,
        second,
        EventSurface(AnonymousCallback(), direction=direction),
        entry_policy=entry_policy,
    )
    system = _system([first, second], [transition])
    all_annotations = build_hybrid_system_dot(
        system,
        show_direction=True,
        show_entry_policy=True,
    )
    assert (
        f'location_0 -> location_1 [label="event: event_0\\n'
        f'direction: {direction.name.lower()}\\nentry_policy: {entry_policy.value}"];'
    ) in all_annotations
    assert "reset:" not in all_annotations
    plain = build_hybrid_system_dot(
        system,
        show_event_labels=False,
        show_reset_labels=False,
        show_direction=False,
        show_entry_policy=False,
    )
    assert "  location_0 -> location_1;" in plain
    assert " [label=" not in plain.split("  start -> location_0;", 1)[1]
    only_policy = build_hybrid_system_dot(
        system,
        show_event_labels=False,
        show_entry_policy=True,
    )
    assert f'[label="entry_policy: {entry_policy.value}"]' in only_policy
    only_direction = build_hybrid_system_dot(
        system,
        show_event_labels=False,
        show_direction=True,
    )
    assert f'[label="direction: {direction.name.lower()}"]' in only_direction


def test_reset_label_flag_independent_of_event_label_flag() -> None:
    location = Location(AnonymousCallback())
    transition = Transition(
        location,
        location,
        EventSurface(AnonymousCallback()),
        Reset(AnonymousCallback()),
    )
    system = _system([location], [transition])
    assert (
        'location_0 -> location_0 [label="reset: reset_0"]'
        in build_hybrid_system_dot(
            system,
            show_event_labels=False,
        )
    )
    assert (
        'location_0 -> location_0 [label="event: event_0"]'
        in build_hybrid_system_dot(
            system,
            show_reset_labels=False,
        )
    )


def test_equivalent_reconstructed_systems_produce_byte_equal_dot() -> None:
    def reconstruct() -> HybridSystem:
        first = Location(AnonymousCallback())
        second = Location(AnonymousCallback())
        return _system(
            [first, second],
            [
                Transition(
                    first, second, AnonymousCallback(), AnonymousCallback()
                ),
                Transition(second, first, AnonymousCallback()),
            ],
        )

    assert (
        build_hybrid_system_dot(reconstruct()).encode()
        == build_hybrid_system_dot(
            reconstruct(),
        ).encode()
    )


def test_build_does_not_require_dot_or_run_subprocess(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_graphviz.shutil, "which", lambda _: None)
    run = Mock(side_effect=AssertionError("unexpected subprocess"))
    monkeypatch.setattr(_graphviz.subprocess, "run", run)
    location = Location(AnonymousCallback())
    assert 'location_0 [label="location_0"]' in build_hybrid_system_dot(
        _system([location], []),
    )
    run.assert_not_called()


def test_render_svg_success_uses_utf8_without_shell(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_graphviz.shutil, "which", lambda _: "/usr/bin/dot")
    run = Mock(
        return_value=SimpleNamespace(
            returncode=0, stdout="<svg>λ</svg>", stderr=""
        )
    )
    monkeypatch.setattr(_graphviz.subprocess, "run", run)
    assert render_dot_svg('digraph G { x [label="λ"]; }') == "<svg>λ</svg>"
    run.assert_called_once_with(
        ["/usr/bin/dot", "-Tsvg"],
        input='digraph G { x [label="λ"]; }',
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )


def test_render_svg_missing_executable_and_nonzero_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = Mock()
    monkeypatch.setattr(_graphviz.subprocess, "run", run)
    monkeypatch.setattr(_graphviz.shutil, "which", lambda _: None)
    with pytest.raises(
        RuntimeError, match="Graphviz 'dot' executable not found"
    ):
        render_dot_svg("digraph G {}")
    run.assert_not_called()
    monkeypatch.setattr(_graphviz.shutil, "which", lambda _: "/usr/bin/dot")
    run.return_value = SimpleNamespace(
        returncode=1, stdout="", stderr="bad input\n"
    )
    with pytest.raises(
        RuntimeError, match="Graphviz failed to render SVG: bad input"
    ):
        render_dot_svg("bad DOT")
    run.return_value = SimpleNamespace(returncode=1, stdout="", stderr="")
    with pytest.raises(RuntimeError, match="Graphviz failed to render SVG"):
        render_dot_svg("bad DOT")


def test_selector_shared_helper_regression(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert selector_graph.render_dot_svg is render_dot_svg
    assert selector_graph._escape_dot_label is _graphviz._escape_dot_label
    inspection = SelectorInspection(
        nodes=(
            SelectorNodeInspection(
                node_id=0,
                is_leaf=True,
                sample_count=1,
                impurity=0.0,
                predicted_mode_id=2,
                weighted_class_support={2: 1.0},
            ),
        ),
        leaves=(
            SelectorLeafInspection(
                node_id=0,
                mode_id=2,
                sample_count=1,
                weighted_class_support={2: 1.0},
                flow_summary="line one\nline two",
            ),
        ),
        modes=(),
        feature_columns=(),
        classes=(2,),
        max_depth=0,
        n_leaves=1,
    )
    assert selector_graph.build_selector_dot(inspection) == (
        "digraph Selector {\n  node [shape=box];\n"
        '  node_0 [label="mode=2\\nflow=line one | line two"];\n}'
    )
    monkeypatch.setattr(_graphviz.shutil, "which", lambda _: None)
    with pytest.raises(
        RuntimeError, match="Graphviz 'dot' executable not found"
    ):
        selector_graph.render_dot_svg("digraph Selector {}")
