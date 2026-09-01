"""Behavioral tests for continuous and hybrid simulation."""

import numpy as np
import pytest

from flowcean.hybrid import (
    CrossingDirection,
    EventSurface,
    HybridSystem,
    Location,
    Reset,
    Transition,
    simulate,
)


def test_continuous_flow_matches_exponential_solution() -> None:
    """Dense sampling follows an analytical exponential trajectory."""
    location = Location(
        lambda state, parameters: parameters["rate"] * state,
        label="growth",
    )
    system = HybridSystem(
        [location],
        [],
        location,
        np.array([1.5]),
        parameters={"rate": -0.7},
    )
    times = np.linspace(0.0, 2.0, 9)

    trace = simulate(system, (0.0, 2.0), sample_times=times)

    expected = 1.5 * np.exp(-0.7 * times)
    np.testing.assert_allclose(trace.t, times)
    np.testing.assert_allclose(trace.x[:, 0], expected, rtol=2e-6, atol=1e-8)
    assert trace.location.tolist() == ["growth"] * len(times)
    assert trace.events == ()


@pytest.mark.parametrize(
    ("direction", "initial_state", "velocity"),
    [
        (CrossingDirection.RISING, -0.5, 1.0),
        (CrossingDirection.FALLING, 0.5, -1.0),
    ],
)
def test_event_crossing_direction(
    direction: CrossingDirection,
    initial_state: float,
    velocity: float,
) -> None:
    """Trigger surfaces only in their configured crossing direction."""
    source = Location(lambda: np.array([velocity]), label="source")
    target = Location(lambda: np.array([0.0]), label="target")
    transition = Transition(
        source,
        target,
        EventSurface(
            lambda state: state[0],
            direction=direction,
            label=direction.name.lower(),
        ),
    )
    system = HybridSystem(
        [source, target],
        [transition],
        source,
        np.array([initial_state]),
    )

    trace = simulate(system, (0.0, 1.0), sample_times=[0.0, 0.5, 1.0])

    assert len(trace.events) == 1
    assert trace.events[0].time == pytest.approx(0.5, abs=1e-7)
    assert trace.location.tolist() == ["source", "target", "target"]


def test_opposite_crossing_direction_does_not_trigger() -> None:
    """A rising trajectory does not trigger a falling-only event."""
    source = Location(lambda: np.array([1.0]), label="source")
    target = Location(lambda: np.array([0.0]), label="target")
    transition = Transition(
        source,
        target,
        EventSurface(
            lambda state: state[0],
            direction=CrossingDirection.FALLING,
        ),
    )
    system = HybridSystem(
        [source, target],
        [transition],
        source,
        np.array([-0.5]),
    )

    trace = simulate(system, (0.0, 1.0), sample_times=[0.0, 0.5, 1.0])

    assert trace.events == ()
    assert trace.location.tolist() == ["source", "source", "source"]
    np.testing.assert_allclose(trace.x[:, 0], [-0.5, 0.0, 0.5], atol=1e-7)


def test_transition_reset_is_reflected_in_event_record() -> None:
    """A transition records labels and its post-reset state."""
    source = Location(lambda: np.array([1.0]), label="charging")
    target = Location(lambda: np.array([0.0]), label="idle")
    transition = Transition(
        source,
        target,
        EventSurface(lambda state: state[0] - 1.0, label="full"),
        Reset(lambda state: np.array([state[0] - 0.75]), label="drain"),
    )
    system = HybridSystem(
        [source, target],
        [transition],
        source,
        np.array([0.0]),
    )

    trace = simulate(system, (0.0, 1.5), sample_times=[0.0, 1.0, 1.5])

    event = trace.events[0]
    assert event.time == pytest.approx(1.0, abs=1e-7)
    assert event.source_location == "charging"
    assert event.target_location == "idle"
    assert event.event_surface == "full"
    assert event.reset == "drain"
    np.testing.assert_allclose(event.state, [0.25], atol=1e-7)
    np.testing.assert_allclose(trace.x[:, 0], [0.0, 0.25, 0.25], atol=1e-7)


def _immediate_chain_system() -> HybridSystem:
    first = Location(lambda: np.array([1.0]), label="first")
    second = Location(lambda: np.array([0.0]), label="second")
    third = Location(lambda: np.array([0.0]), label="third")
    return HybridSystem(
        [first, second, third],
        [
            Transition(
                first,
                second,
                EventSurface(lambda state: state[0], label="reach-zero"),
                Reset(lambda: np.array([2.0]), label="to-two"),
            ),
            Transition(
                second,
                third,
                EventSurface(lambda state: state[0] - 2.0, label="at-two"),
                Reset(lambda state: state + 1.0, label="increment"),
            ),
        ],
        first,
        np.array([-0.5]),
    )


def test_immediate_transition_chain_occurs_at_one_time() -> None:
    """A reset onto another surface applies the next transition immediately."""
    trace = simulate(
        _immediate_chain_system(),
        (0.0, 1.0),
        sample_times=[0.0, 0.5, 1.0],
    )

    assert len(trace.events) == 2
    assert [event.time for event in trace.events] == pytest.approx([0.5, 0.5])
    assert [event.target_location for event in trace.events] == [
        "second",
        "third",
    ]
    np.testing.assert_allclose(trace.events[-1].state, [3.0])
    assert trace.location.tolist() == ["first", "third", "third"]
    np.testing.assert_allclose(trace.x[:, 0], [-0.5, 3.0, 3.0], atol=1e-7)


def test_max_jumps_limits_immediate_transition_chains() -> None:
    """Immediate transitions count toward the configured jump limit."""
    with pytest.raises(RuntimeError, match="Maximum number of transitions"):
        simulate(_immediate_chain_system(), (0.0, 1.0), max_jumps=1)


def test_fixed_sample_grid_uses_post_reset_state_at_event() -> None:
    """At an event, report the target location and post-reset state."""
    source = Location(lambda: np.array([1.0]), label="before")
    target = Location(lambda: np.array([2.0]), label="after")
    system = HybridSystem(
        [source, target],
        [
            Transition(
                source,
                target,
                EventSurface(
                    lambda state: state[0] - 0.5,
                    direction=CrossingDirection.RISING,
                ),
                lambda: np.array([10.0]),
            ),
        ],
        source,
        np.array([0.0]),
    )

    trace = simulate(system, (0.0, 1.0), sample_dt=0.25)

    np.testing.assert_allclose(trace.t, [0.0, 0.25, 0.5, 0.75, 1.0])
    np.testing.assert_allclose(
        trace.x[:, 0],
        [0.0, 0.25, 10.0, 10.5, 11.0],
        atol=1e-7,
    )
    assert trace.location.tolist() == [
        "before",
        "before",
        "after",
        "after",
        "after",
    ]


def test_inputs_and_derivatives_are_captured_on_sample_grid() -> None:
    """Captured arrays align with the trace and use effective parameters."""
    location = Location(
        lambda t, parameters, input_stream: np.array(
            [
                parameters["gain"] * input_stream(t)[0],
                input_stream(t)[1],
            ],
        ),
        parameters={"gain": 3.0},
    )
    system = HybridSystem(
        [location],
        [],
        location,
        np.array([0.0, 0.0]),
        parameters={"gain": -1.0},
    )
    times = np.array([0.0, 0.2, 0.7, 1.0])

    trace = simulate(
        system,
        (0.0, 1.0),
        input_stream=lambda time: np.array([1.0 + time, 2.0 - time]),
        capture_derivatives=True,
        sample_times=times,
    )

    expected_inputs = np.column_stack((1.0 + times, 2.0 - times))
    expected_derivatives = np.column_stack(
        (3.0 * (1.0 + times), 2.0 - times),
    )
    expected_states = np.column_stack(
        (3.0 * (times + times**2 / 2.0), 2.0 * times - times**2 / 2.0),
    )
    assert trace.u is not None
    assert trace.dx is not None
    np.testing.assert_allclose(trace.u, expected_inputs)
    np.testing.assert_allclose(trace.dx, expected_derivatives)
    np.testing.assert_allclose(trace.x, expected_states, rtol=2e-6, atol=1e-8)
