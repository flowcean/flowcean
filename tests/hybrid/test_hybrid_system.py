"""Tests for hybrid-system construction and parameter handling."""

import numpy as np
import pytest

from flowcean.hybrid import (
    CrossingDirection,
    Event,
    EventSurface,
    Flow,
    HybridSystem,
    Location,
    Reset,
    SurfaceEntryPolicy,
    Transition,
    simulate,
)


def _zero_flow() -> np.ndarray:
    return np.array([0.0])


def test_hybrid_system_normalizes_sequences_and_copies_parameters() -> None:
    """System collections and parameter mappings are captured at creation."""
    location = Location(_zero_flow)
    parameters = {"gain": 2.0}

    system = HybridSystem(
        locations=[location],
        transitions=[],
        initial_location=location,
        initial_state=np.array([1.0]),
        parameters=parameters,
    )
    parameters["gain"] = 10.0

    assert system.locations == (location,)
    assert system.transitions == ()
    assert system.parameters == {"gain": 2.0}


def test_hybrid_system_rejects_duplicate_location_objects() -> None:
    """A location object can occur only once in a system."""
    location = Location(_zero_flow)

    with pytest.raises(ValueError, match="duplicate Location"):
        HybridSystem(
            [location, location],
            [],
            location,
            np.array([0.0]),
        )


def test_hybrid_system_requires_initial_location_in_locations() -> None:
    """The initial location is validated by identity, not equality."""
    included = Location(_zero_flow)
    external = Location(_zero_flow)

    with pytest.raises(ValueError, match="initial_location"):
        HybridSystem([included], [], external, np.array([0.0]))


def test_hybrid_system_requires_transition_endpoints_in_locations() -> None:
    """Transitions cannot refer to locations outside their system."""
    included = Location(_zero_flow)
    external = Location(_zero_flow)
    transition = Transition(included, external, lambda state: state[0])

    with pytest.raises(ValueError, match="transition target"):
        HybridSystem(
            [included],
            [transition],
            included,
            np.array([0.0]),
        )


def test_transition_entry_policy_defaults_and_requires_exact_enum() -> None:
    """Entry policy is explicit, keyword-only, and strongly typed."""
    source = Location(_zero_flow)
    target = Location(_zero_flow)

    transition = Transition(source, target, lambda: 1.0)

    assert transition.entry_policy is SurfaceEntryPolicy.ERROR
    assert [policy.value for policy in SurfaceEntryPolicy] == [
        "error",
        "trigger",
        "continue",
    ]
    for invalid in ("trigger", CrossingDirection.EITHER, 0):
        with pytest.raises(TypeError, match="SurfaceEntryPolicy"):
            Transition(
                source,
                target,
                lambda: 1.0,
                entry_policy=invalid,  # pyright: ignore[reportArgumentType]
            )
    with pytest.raises(TypeError):
        Transition(
            source,
            target,
            lambda: 1.0,
            None,
            SurfaceEntryPolicy.TRIGGER,  # pyright: ignore[reportCallIssue]
        )


def test_self_transition_without_reset_resets_location_time() -> None:
    """Re-entry resets the clock without altering the continuous state."""
    location = Location(lambda location_time: np.array([location_time]))
    transition = Transition(
        location,
        location,
        lambda location_time: location_time - 0.5,
    )
    system = HybridSystem([location], [transition], location, np.array([0.0]))

    trace = simulate(
        system,
        (0.0, 1.25),
    )
    frame = trace.sample([0.0, 0.5, 0.75, 1.0, 1.25], include_derivatives=True)

    assert [event.time for event in trace.events] == pytest.approx([0.5, 1.0])
    assert [
        event.location_time_before for event in trace.events
    ] == pytest.approx([0.5, 0.5])
    np.testing.assert_allclose(
        frame["location_time"], [0.0, 0.0, 0.25, 0.0, 0.25]
    )
    np.testing.assert_allclose(frame["dx0"], [0.0, 0.0, 0.25, 0.0, 0.25])
    np.testing.assert_allclose(
        frame["x0"], [0.0, 0.125, 0.15625, 0.25, 0.28125], atol=1e-6
    )


def test_self_transition_immediate_loop_obeys_max_jumps() -> None:
    location = Location(_zero_flow)
    system = HybridSystem(
        [location],
        [
            Transition(
                location,
                location,
                lambda location_time: location_time,
                entry_policy=SurfaceEntryPolicy.TRIGGER,
            )
        ],
        location,
        np.array([0.0]),
    )

    with pytest.raises(RuntimeError, match="Maximum number of transitions"):
        simulate(system, (0.0, 1.0), max_jumps=3)


def test_event_requires_age_and_has_read_only_detached_snapshots() -> None:
    first = Location(_zero_flow)
    second = Location(_zero_flow)
    before = np.array([0.0])
    after = np.array([1.0])
    transition = Transition(first, second, lambda: 0.0)
    event = Event(
        time=0.5,
        transition=transition,
        state_before=before,
        state_after=after,
        microstep=0,
        location_time_before=0.5,
    )
    before[0] = 10.0
    after[0] = 20.0
    assert event.transition.source is first
    assert event.transition.target is second
    assert event.location_time_before == 0.5
    np.testing.assert_allclose(event.state_before, [0.0])
    np.testing.assert_allclose(event.state_after, [1.0])
    assert not event.state_before.flags["W"]
    assert not event.state_after.flags["W"]

    with pytest.raises(TypeError, match="location_time_before"):
        Event(  # pyright: ignore[reportCallIssue]
            0.5,
            transition,
            np.array([0.0]),
            np.array([1.0]),
            0,
        )


def test_event_retains_selected_transition_identity_with_duplicate_labels() -> (
    None
):
    source = Location(lambda: np.array([1.0]), label="same")
    target = Location(_zero_flow, label="same")
    later_surface = EventSurface(lambda t: t - 1.0, label="same")
    selected_surface = EventSurface(lambda t: t - 0.5, label="same")
    later_reset = Reset(lambda state: state + 100.0, label="same")
    selected_reset = Reset(lambda state: state + 2.0, label="same")
    later = Transition(source, target, later_surface, later_reset)
    selected = Transition(source, target, selected_surface, selected_reset)
    system = HybridSystem(
        [source, target], [later, selected], source, np.array([0.0])
    )

    trajectory = simulate(system, (0.0, 1.25))

    assert len(trajectory.events) == 1
    event = trajectory.events[0]
    assert event.transition is system.transitions[1] is selected
    assert event.transition.event_surface is selected_surface
    assert event.transition.reset is selected_reset
    assert event.transition.source is source
    assert event.transition.target is target
    np.testing.assert_allclose(event.state_before, [0.5])
    np.testing.assert_allclose(event.state_after, [2.5])
    assert not np.shares_memory(event.state_before, event.state_after)
    assert not event.state_before.flags["W"]
    assert not event.state_after.flags["W"]


def test_locations_sharing_flow_have_distinct_visits_and_residence_clocks() -> (
    None
):
    flow = Flow(fn=lambda location_time: np.array([location_time]))
    first = Location(flow=flow, label="same")
    second = Location(flow=flow, label="same")
    surface = EventSurface(lambda location_time: location_time - 0.5)
    system = HybridSystem(
        [first, second],
        [
            Transition(first, second, surface),
            Transition(second, first, surface),
        ],
        first,
        np.array([0.0]),
    )

    trajectory = simulate(system, (2.0, 3.25))
    frame = trajectory.sample(
        [2.0, 2.25, 2.5, 2.75, 3.0, 3.25], include_derivatives=True
    )

    assert first.flow is second.flow is flow
    assert [segment.location for segment in trajectory.segments] == [
        first,
        second,
        first,
    ]
    assert frame["location_id"].to_list() == [0, 0, 1, 1, 0, 0]
    assert frame["location_time"].to_list() == [
        0.0,
        0.25,
        0.0,
        0.25,
        0.0,
        0.25,
    ]
    np.testing.assert_allclose(frame["dx0"], frame["location_time"])
    np.testing.assert_allclose(
        frame["x0"], [0, 0.03125, 0.125, 0.15625, 0.25, 0.28125]
    )
    assert [event.location_time_before for event in trajectory.events] == [
        0.5,
        0.5,
    ]


def test_location_parameters_override_globals_for_callbacks() -> None:
    """Source-location parameters take precedence in every callback."""
    source = Location(
        lambda parameters: np.array([parameters["rate"]]),
        label="source",
        parameters={"rate": 2.0, "threshold": 1.0, "offset": 3.0},
    )
    target = Location(_zero_flow, label="target")
    transition = Transition(
        source,
        target,
        lambda state, parameters: state[0] - parameters["threshold"],
        lambda state, parameters: np.array(
            [state[0] + parameters["offset"]],
        ),
    )
    system = HybridSystem(
        [source, target],
        [transition],
        source,
        np.array([0.0]),
        parameters={"rate": 20.0, "threshold": 10.0, "offset": 30.0},
    )

    trace = simulate(system, (0.0, 1.0))

    assert len(trace.events) == 1
    assert trace.events[0].time == pytest.approx(0.5, abs=1e-7)
    assert trace.events[0].state_after == pytest.approx(np.array([4.0]))
    assert trace.events[0].transition.source is source
    assert trace.events[0].transition.target is target
    frame = trace.sample([0.0, trace.events[0].time, 1.0])
    assert frame["x0"][1] == pytest.approx(4.0)
    assert frame["location_id"].to_list() == [0, 1, 1]
