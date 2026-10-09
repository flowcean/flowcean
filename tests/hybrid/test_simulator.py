"""Behavioral tests for continuous and hybrid simulation."""

import numpy as np
import pytest

from flowcean.hybrid import (
    AmbiguousTransitionError,
    CrossingDirection,
    EventSurface,
    HybridSystem,
    InvalidEventSurfaceValueError,
    Location,
    Reset,
    SimulationProgressError,
    SurfaceEntryError,
    SurfaceEntryPolicy,
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

    trace = simulate(system, (0.0, 2.0))
    frame = trace.sample(times, include_location_label=True)

    expected = 1.5 * np.exp(-0.7 * times)
    np.testing.assert_allclose(frame["t"], times)
    np.testing.assert_allclose(frame["x0"], expected, rtol=2e-6, atol=1e-8)
    assert frame["location_label"].to_list() == ["growth"] * len(times)
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

    trace = simulate(system, (0.0, 1.0))

    assert len(trace.events) == 1
    assert trace.events[0].time == pytest.approx(0.5, abs=1e-7)
    assert trace.sample(
        [0.0, trace.events[0].time, 1.0], include_location_label=True
    )["location_label"].to_list() == ["source", "target", "target"]


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

    trace = simulate(system, (0.0, 1.0))
    frame = trace.sample([0.0, 0.5, 1.0], include_location_label=True)

    assert trace.events == ()
    assert frame["location_label"].to_list() == ["source", "source", "source"]
    np.testing.assert_allclose(frame["x0"], [-0.5, 0.0, 0.5], atol=1e-7)


def test_entry_evaluation_is_atomic_and_error_precedes_ambiguity() -> None:
    """All surfaces run once before ERROR wins over multiple TRIGGERs."""
    source = Location(lambda: np.array([0.0]), label="source")
    targets = [
        Location(lambda: np.array([0.0]), label=f"target-{index}")
        for index in range(4)
    ]
    calls = [0, 0, 0, 0]

    def surface(index: int) -> float:
        calls[index] += 1
        return 0.0

    transitions = [
        Transition(
            source,
            target,
            EventSurface(
                lambda index=index: surface(index),
                label=f"s{index}",
            ),
            entry_policy=policy,
        )
        for index, (target, policy) in enumerate(
            zip(
                targets,
                [
                    SurfaceEntryPolicy.ERROR,
                    SurfaceEntryPolicy.ERROR,
                    SurfaceEntryPolicy.TRIGGER,
                    SurfaceEntryPolicy.TRIGGER,
                ],
                strict=True,
            ),
        )
    ]
    system = HybridSystem(
        [source, *targets],
        transitions,
        source,
        np.array([0.0]),
    )

    with pytest.raises(SurfaceEntryError, match="s0") as caught:
        simulate(system, (0.0, 1.0))

    assert calls == [1, 1, 1, 1]
    assert "s1" in str(caught.value)
    assert "TRIGGER" in " ".join(caught.value.__notes__)


def test_nan_entry_value_precedes_zero_entry_errors() -> None:
    """NaN is reported before any policy-based zero handling."""
    source = Location(lambda: np.array([0.0]), label="source")
    nan_target = Location(lambda: np.array([0.0]), label="nan-target")
    zero_target = Location(lambda: np.array([0.0]), label="zero-target")
    system = HybridSystem(
        [source, nan_target, zero_target],
        [
            Transition(
                source,
                zero_target,
                EventSurface(lambda: 0.0, label="zero"),
            ),
            Transition(
                source,
                nan_target,
                EventSurface(lambda: np.nan, label="not-a-number"),
            ),
        ],
        source,
        np.array([0.0]),
    )

    with pytest.raises(InvalidEventSurfaceValueError, match="not-a-number"):
        simulate(system, (0.0, 1.0))


def test_multiple_entry_triggers_are_ambiguous() -> None:
    """Two exact-zero TRIGGER surfaces cannot select an ordering."""
    source = Location(lambda: np.array([0.0]), label="source")
    first = Location(lambda: np.array([0.0]), label="first")
    second = Location(lambda: np.array([0.0]), label="second")
    system = HybridSystem(
        [source, first, second],
        [
            Transition(
                source,
                first,
                EventSurface(lambda: 0.0, label="first-trigger"),
                entry_policy=SurfaceEntryPolicy.TRIGGER,
            ),
            Transition(
                source,
                second,
                EventSurface(lambda: -0.0, label="second-trigger"),
                entry_policy=SurfaceEntryPolicy.TRIGGER,
            ),
        ],
        source,
        np.array([0.0]),
    )

    with pytest.raises(
        AmbiguousTransitionError,
        match="first-trigger",
    ) as caught:
        simulate(system, (0.0, 1.0))

    assert "second-trigger" in str(caught.value)


@pytest.mark.parametrize("zero", [0.0, -0.0])
def test_exact_zero_uses_entry_policy(zero: float) -> None:
    """Both signs of exact floating-point zero count as on-surface."""
    source = Location(lambda: np.array([0.0]), label="source")
    target = Location(lambda: np.array([0.0]), label="target")
    system = HybridSystem(
        [source, target],
        [Transition(source, target, lambda: zero)],
        source,
        np.array([0.0]),
    )

    with pytest.raises(SurfaceEntryError):
        simulate(system, (0.0, 0.1))


@pytest.mark.parametrize("value", [1e-300, -1e-300, np.inf, -np.inf])
def test_nonzero_and_infinite_entry_values_are_signed_values(
    value: float,
) -> None:
    """No tolerance collapses tiny values or infinities onto the surface."""
    source = Location(lambda: np.array([0.0]), label="source")
    target = Location(lambda: np.array([0.0]), label="target")
    system = HybridSystem(
        [source, target],
        [Transition(source, target, lambda: value)],
        source,
        np.array([0.0]),
    )

    trace = simulate(system, (0.0, 0.1))

    assert trace.events == ()


def test_continue_allows_departure_in_opposite_direction() -> None:
    """CONTINUE passes an entry-zero surface unchanged to the solver."""
    source = Location(lambda: np.array([1.0]), label="source")
    target = Location(lambda: np.array([0.0]), label="target")
    system = HybridSystem(
        [source, target],
        [
            Transition(
                source,
                target,
                EventSurface(
                    lambda state: state[0],
                    direction=CrossingDirection.FALLING,
                ),
                entry_policy=SurfaceEntryPolicy.CONTINUE,
            ),
        ],
        source,
        np.array([0.0]),
    )

    trace = simulate(system, (0.0, 0.25))

    assert trace.events == ()
    np.testing.assert_allclose(trace.sample([0.0, 0.25])["x0"], [0.0, 0.25])


def test_continue_same_direction_reports_no_progress() -> None:
    """A solver root at the segment start is rejected before recording it."""
    source = Location(lambda: np.array([1.0]), label="source")
    target = Location(lambda: np.array([0.0]), label="target")
    system = HybridSystem(
        [source, target],
        [
            Transition(
                source,
                target,
                EventSurface(
                    lambda state: state[0],
                    direction=CrossingDirection.RISING,
                ),
                entry_policy=SurfaceEntryPolicy.CONTINUE,
            ),
        ],
        source,
        np.array([0.0]),
    )

    with pytest.raises(
        SimulationProgressError,
        match="did not advance",
    ) as caught:
        simulate(system, (0.0, 0.25))

    notes = " ".join(caught.value.__notes__)
    assert "stateful callbacks" in notes
    assert "continuous event surfaces" in notes


def test_nan_during_solver_event_evaluation_is_rejected() -> None:
    """NaN checks also cover evaluations made by continuous integration."""
    source = Location(lambda: np.array([1.0]), label="source")
    target = Location(lambda: np.array([0.0]), label="target")
    system = HybridSystem(
        [source, target],
        [
            Transition(
                source,
                target,
                EventSurface(
                    lambda t: 1.0 if t == 0.0 else np.nan,
                    label="unstable-surface",
                ),
            ),
        ],
        source,
        np.array([0.0]),
    )

    with pytest.raises(
        InvalidEventSurfaceValueError,
        match="continuous integration",
    ):
        simulate(system, (0.0, 1.0))


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

    trace = simulate(system, (0.0, 1.5))

    event = trace.events[0]
    assert event.time == pytest.approx(1.0, abs=1e-7)
    assert event.transition.source is source
    assert event.transition.target is target
    assert event.transition.event_surface.label == "full"
    assert event.transition.reset is not None
    assert event.transition.reset.label == "drain"
    np.testing.assert_allclose(event.state_before, [1.0], atol=1e-7)
    np.testing.assert_allclose(event.state_after, [0.25], atol=1e-7)
    assert event.microstep == 0
    np.testing.assert_allclose(
        trace.sample([0.0, event.time, 1.5])["x0"],
        [0.0, 0.25, 0.25],
        atol=1e-7,
    )


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
                entry_policy=SurfaceEntryPolicy.TRIGGER,
            ),
        ],
        first,
        np.array([-0.5]),
    )


def test_immediate_transition_chain_occurs_at_one_time() -> None:
    """A reset onto another surface applies the next transition immediately."""
    trace = simulate(_immediate_chain_system(), (0.0, 1.0))

    assert len(trace.events) == 2
    assert [event.time for event in trace.events] == pytest.approx([0.5, 0.5])
    assert [event.transition.target.label for event in trace.events] == [
        "second",
        "third",
    ]
    assert [event.microstep for event in trace.events] == [0, 1]
    np.testing.assert_allclose(
        trace.events[0].state_before,
        [0.0],
        atol=1e-12,
    )
    np.testing.assert_allclose(trace.events[0].state_after, [2.0])
    np.testing.assert_allclose(trace.events[1].state_before, [2.0])
    np.testing.assert_allclose(trace.events[1].state_after, [3.0])
    frame = trace.sample(
        [0.0, trace.events[0].time, 1.0], include_location_label=True
    )
    assert frame["location_label"].to_list() == ["first", "third", "third"]
    np.testing.assert_allclose(frame["x0"], [-0.5, 3.0, 3.0], atol=1e-7)


def test_exact_restart_detects_a_root_less_than_epsilon_after_event() -> None:
    """A post-jump segment starts at the event time without skipping ahead."""
    first = Location(lambda: np.array([1.0]), label="first")
    second = Location(lambda: np.array([1.0]), label="second")
    third = Location(lambda: np.array([0.0]), label="third")
    delay = 5e-13
    system = HybridSystem(
        [first, second, third],
        [
            Transition(
                first,
                second,
                EventSurface(lambda t: t - 0.5, label="first-event"),
                Reset(lambda: np.array([0.0])),
            ),
            Transition(
                second,
                third,
                EventSurface(
                    lambda state: state[0] - delay,
                    label="nearby-event",
                ),
            ),
        ],
        first,
        np.array([0.0]),
    )

    trace = simulate(system, (0.0, 1.0), max_step=0.1)

    assert len(trace.events) == 2
    separation = trace.events[1].time - trace.events[0].time
    assert 0.0 < separation < 1e-12
    assert trace.events[1].transition.event_surface.label == "nearby-event"


def test_event_states_are_independent_snapshots() -> None:
    """Event states do not alias reset results or trace storage."""
    source = Location(lambda: np.array([1.0]), label="source")
    target = Location(lambda: np.array([0.0]), label="target")
    reset_result = np.array([4.0])
    system = HybridSystem(
        [source, target],
        [
            Transition(
                source,
                target,
                EventSurface(lambda state: state[0] - 1.0),
                lambda: reset_result,
            ),
        ],
        source,
        np.array([0.0]),
    )

    trace = simulate(system, (0.0, 2.0))
    event = trace.events[0]
    reset_result[0] = 99.0

    np.testing.assert_allclose(event.state_before, [1.0])
    np.testing.assert_allclose(event.state_after, [4.0])
    assert not event.state_before.flags["W"]
    assert not event.state_after.flags["W"]
    with pytest.raises(ValueError, match="read-only"):
        event.state_before[0] = 2.0
    with pytest.raises(ValueError, match="read-only"):
        event.state_after[0] = 5.0
    np.testing.assert_allclose(trace.sample([event.time])["x0"], [4.0])
    assert trace.initial_state[0] == 0.0
    assert not trace.initial_state.flags["W"]


def test_explicit_sampling_is_right_continuous_at_event_boundary() -> None:
    """Sampling an exact event time selects the post-reset target."""
    source = Location(lambda: np.array([1.0]), label="source")
    target = Location(lambda: np.array([2.0]), label="target")
    system = HybridSystem(
        [source, target],
        [
            Transition(
                source,
                target,
                EventSurface(lambda state: state[0] - 0.5),
                lambda: np.array([10.0]),
            ),
        ],
        source,
        np.array([0.0]),
    )

    trace = simulate(system, (0.0, 1.0))
    event_time = trace.events[0].time
    frame = trace.sample([event_time], include_derivatives=True)

    np.testing.assert_allclose(frame["x0"], [10.0])
    assert frame["location_id"].to_list() == [1]
    np.testing.assert_allclose(frame["dx0"], [2.0])


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

    trace = simulate(system, (0.0, 1.0))
    frame = trace.sample(dt=0.25, include_location_label=True)

    np.testing.assert_allclose(frame["t"], [0.0, 0.25, 0.5, 0.75, 1.0])
    np.testing.assert_allclose(
        frame["x0"],
        [0.0, 0.25, 10.0, 10.5, 11.0],
        atol=1e-7,
    )
    assert frame["location_label"].to_list() == [
        "before",
        "before",
        "after",
        "after",
        "after",
    ]


def test_fixed_grid_preserves_off_grid_event_and_requested_times() -> None:
    """Off-grid jumps stay in events without changing the requested grid."""
    source = Location(lambda: np.array([1.0]), label="source")
    target = Location(lambda: np.array([0.0]), label="target")
    system = HybridSystem(
        [source, target],
        [
            Transition(
                source,
                target,
                EventSurface(lambda state: state[0] - 0.5),
                lambda: np.array([3.0]),
            ),
        ],
        source,
        np.array([0.0]),
    )
    requested = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    trace = simulate(system, (0.0, 1.0))
    frame = trace.sample(requested, include_location_label=True)

    np.testing.assert_array_equal(frame["t"], requested)
    assert trace.events[0].time == pytest.approx(0.5)
    assert not np.any(frame["t"].to_numpy() == trace.events[0].time)
    np.testing.assert_allclose(frame["x0"], [0.0, 0.2, 0.4, 3.0, 3.0, 3.0])
    assert frame["location_label"].to_list() == [
        "source",
        "source",
        "source",
        "target",
        "target",
        "target",
    ]


def test_initial_time_transition_is_right_continuous() -> None:
    """A transition on the initial surface replaces the initial trace row."""
    source = Location(lambda: np.array([0.0]), label="source")
    target = Location(lambda: np.array([1.0]), label="target")
    system = HybridSystem(
        [source, target],
        [
            Transition(
                source,
                target,
                EventSurface(lambda state: state[0]),
                lambda: np.array([4.0]),
                entry_policy=SurfaceEntryPolicy.TRIGGER,
            ),
        ],
        source,
        np.array([0.0]),
    )

    trace = simulate(system, (0.0, 1.0))
    frame = trace.sample([0, 0.5, 1.0], include_derivatives=True)

    np.testing.assert_allclose(frame["t"][0], 0.0)
    assert frame["t"].to_list().count(0.0) == 1
    np.testing.assert_allclose(frame["x0"][0], 4.0)
    assert frame["location_id"][0] == 1
    np.testing.assert_allclose(frame["dx0"][0], 1.0)
    event = trace.events[0]
    assert event.time == 0.0
    assert event.microstep == 0
    np.testing.assert_allclose(event.state_before, [0.0])
    np.testing.assert_allclose(event.state_after, [4.0])


def test_initial_trigger_chain_uses_zero_based_microsteps() -> None:
    """An initial entry chain starts with microstep zero at the start time."""
    first = Location(lambda: np.array([0.0]), label="first")
    second = Location(lambda: np.array([0.0]), label="second")
    third = Location(lambda: np.array([0.0]), label="third")
    system = HybridSystem(
        [first, second, third],
        [
            Transition(
                first,
                second,
                lambda state: state[0],
                lambda: np.array([1.0]),
                entry_policy=SurfaceEntryPolicy.TRIGGER,
            ),
            Transition(
                second,
                third,
                lambda state: state[0] - 1.0,
                lambda: np.array([2.0]),
                entry_policy=SurfaceEntryPolicy.TRIGGER,
            ),
        ],
        first,
        np.array([0.0]),
    )

    trace = simulate(system, (0.0, 1.0))
    frame = trace.sample([0.0, 0.5, 1.0], include_location_label=True)

    assert [event.time for event in trace.events] == [0.0, 0.0]
    assert [event.microstep for event in trace.events] == [0, 1]
    assert frame["location_label"].to_list() == ["third", "third", "third"]
    np.testing.assert_allclose(frame["x0"], [2.0, 2.0, 2.0])


def test_final_time_transition_is_included_and_right_continuous() -> None:
    """A jump at the end of t_span supplies the final trace row."""
    source = Location(lambda: np.array([1.0]), label="source")
    target = Location(lambda: np.array([-2.0]), label="target")
    system = HybridSystem(
        [source, target],
        [
            Transition(
                source,
                target,
                EventSurface(lambda t: t - 1.0),
                lambda: np.array([5.0]),
            ),
        ],
        source,
        np.array([0.0]),
    )

    trace = simulate(system, (0.0, 1.0))
    fixed = trace.sample([0.0, 0.5, 1.0], include_derivatives=True)

    assert len(trace.events) == 1
    assert trace.events[0].time == pytest.approx(1.0)
    np.testing.assert_allclose(fixed["x0"][-1], 5.0)
    assert fixed["location_id"][-1] == 1
    np.testing.assert_allclose(fixed["dx0"][-1], -2.0)
    np.testing.assert_array_equal(fixed["t"], [0.0, 0.5, 1.0])
    assert trace.segments[-1].t_span[1] == 1.0


def test_final_time_target_entry_trigger_uses_next_microstep() -> None:
    """Target entry is resolved even when a continuous root is at t_final."""
    first = Location(lambda: np.array([0.0]), label="first")
    second = Location(lambda: np.array([0.0]), label="second")
    third = Location(lambda: np.array([0.0]), label="third")
    system = HybridSystem(
        [first, second, third],
        [
            Transition(
                first,
                second,
                lambda t: t - 1.0,
                lambda: np.array([3.0]),
            ),
            Transition(
                second,
                third,
                lambda state: state[0] - 3.0,
                lambda: np.array([4.0]),
                entry_policy=SurfaceEntryPolicy.TRIGGER,
            ),
        ],
        first,
        np.array([0.0]),
    )

    trace = simulate(system, (0.0, 1.0))
    frame = trace.sample(dt=0.25, include_location_label=True)

    assert [event.time for event in trace.events] == pytest.approx([1.0, 1.0])
    assert [event.microstep for event in trace.events] == [0, 1]
    np.testing.assert_array_equal(frame["t"], [0.0, 0.25, 0.5, 0.75, 1.0])
    np.testing.assert_allclose(frame["x0"][-1], 4.0)
    assert frame["location_label"][-1] == "third"


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
    )
    frame = trace.sample(times, include_inputs=True, include_derivatives=True)

    expected_inputs = np.column_stack((1.0 + times, 2.0 - times))
    expected_derivatives = np.column_stack(
        (3.0 * (1.0 + times), 2.0 - times),
    )
    expected_states = np.column_stack(
        (3.0 * (times + times**2 / 2.0), 2.0 * times - times**2 / 2.0),
    )
    np.testing.assert_allclose(
        frame.select("u0", "u1").to_numpy(), expected_inputs
    )
    np.testing.assert_allclose(
        frame.select("dx0", "dx1").to_numpy(), expected_derivatives
    )
    np.testing.assert_allclose(
        frame.select("x0", "x1").to_numpy(),
        expected_states,
        rtol=2e-6,
        atol=1e-8,
    )


@pytest.mark.parametrize("start", [13.0, 1e8])
def test_initial_age_uses_stable_anchor_and_reaches_callbacks(
    start: float,
) -> None:
    """Nonzero initial ages survive large physical starts without subtraction loss."""
    seen: list[float] = []

    def surface(*, location_time: float) -> float:
        seen.append(location_time)
        return location_time - 0.35

    source = Location(
        lambda *, location_time: np.array([location_time]), label="source"
    )
    target = Location(lambda: np.array([0.0]), label="target")
    system = HybridSystem(
        [source, target],
        [Transition(source, target, surface)],
        source,
        np.array([0.0]),
    )
    trace = simulate(
        system,
        (start, start + 0.5),
        initial_location_time=0.1,
    )
    frame = trace.sample(
        [start, start + 0.125, start + 0.25, start + 0.5],
        include_derivatives=True,
    )

    assert seen[0] == 0.1
    assert trace.events[0].time == pytest.approx(start + 0.25)
    assert trace.events[0].location_time_before == pytest.approx(0.35)
    np.testing.assert_allclose(frame["location_time"], [0.1, 0.225, 0.0, 0.25])
    np.testing.assert_allclose(frame["dx0"], [0.1, 0.225, 0.0, 0.0])
    assert trace.initial_location_time == 0.1


def test_initial_age_is_preserved_without_jump_and_repeated_runs_independent() -> (
    None
):
    location = Location(lambda *, location_time: np.array([location_time]))
    system = HybridSystem([location], [], location, np.array([0.0]))
    first = simulate(system, (9.0, 9.5), initial_location_time=2.0)
    second = simulate(system, (9.0, 9.5))
    np.testing.assert_allclose(
        first.sample(dt=0.25)["location_time"], [2.0, 2.25, 2.5]
    )
    np.testing.assert_allclose(
        second.sample([9.0, 9.5])["location_time"], [0.0, 0.5]
    )
    assert first.events == second.events == ()


@pytest.mark.parametrize("age", [-1.0, np.nan, np.inf, -np.inf])
def test_invalid_initial_age_is_rejected_before_callbacks(age: float) -> None:
    calls: list[float] = []
    location = Location(lambda: calls.append(1.0) or np.array([0.0]))
    system = HybridSystem([location], [], location, np.array([0.0]))
    with pytest.raises(ValueError, match="initial_location_time"):
        simulate(system, (0.0, 1.0), initial_location_time=age)
    assert calls == []


def test_source_age_and_parameters_reset_before_target_entry_and_chain() -> (
    None
):
    """Each same-time jump produces a fresh visit, including a repeated label."""
    seen: list[tuple[str, float, float]] = []

    def observe(role: str, age: float, rate: float) -> None:
        seen.append((role, age, rate))

    first = Location(
        lambda: np.array([0.0]), label="same", parameters={"rate": 1.0}
    )
    second = Location(
        lambda: np.array([0.0]), label="same", parameters={"rate": 2.0}
    )
    third = Location(
        lambda: np.array([0.0]), label="done", parameters={"rate": 3.0}
    )

    def first_surface(
        *, location_time: float, parameters: dict[str, float]
    ) -> float:
        observe("first surface", location_time, parameters["rate"])
        return location_time - 2.0

    def first_reset(
        *,
        location_time: float,
        parameters: dict[str, float],
        state: np.ndarray,
    ) -> np.ndarray:
        observe("first reset", location_time, parameters["rate"])
        return state

    def second_surface(
        *, location_time: float, parameters: dict[str, float]
    ) -> float:
        observe("second surface", location_time, parameters["rate"])
        return location_time

    def second_reset(
        *,
        location_time: float,
        parameters: dict[str, float],
        state: np.ndarray,
    ) -> np.ndarray:
        observe("second reset", location_time, parameters["rate"])
        return state

    system = HybridSystem(
        [first, second, third],
        [
            Transition(
                first,
                second,
                first_surface,
                first_reset,
                entry_policy=SurfaceEntryPolicy.TRIGGER,
            ),
            Transition(
                second,
                third,
                second_surface,
                second_reset,
                entry_policy=SurfaceEntryPolicy.TRIGGER,
            ),
        ],
        first,
        np.array([0.0]),
    )
    trace = simulate(
        system,
        (5.0, 5.5),
        initial_location_time=2.0,
    )
    frame = trace.sample([5.0, 5.0, 5.25, 5.5], include_location_label=True)
    assert seen == [
        ("first surface", 2.0, 1.0),
        ("first reset", 2.0, 1.0),
        ("second surface", 0.0, 2.0),
        ("second reset", 0.0, 2.0),
    ]
    assert [event.microstep for event in trace.events] == [0, 1]
    assert [event.location_time_before for event in trace.events] == [2.0, 0.0]
    np.testing.assert_allclose(frame["location_time"], [0.0, 0.0, 0.25, 0.5])
    assert frame["location_label"].to_list() == ["done"] * 4


@pytest.mark.parametrize(
    "policy",
    [
        SurfaceEntryPolicy.ERROR,
        SurfaceEntryPolicy.TRIGGER,
        SurfaceEntryPolicy.CONTINUE,
    ],
)
def test_exact_zero_initial_timeout_follows_entry_policy(
    policy: SurfaceEntryPolicy,
) -> None:
    source = Location(lambda: np.array([0.0]))
    target = Location(lambda: np.array([0.0]))
    system = HybridSystem(
        [source, target],
        [
            Transition(
                source,
                target,
                EventSurface(
                    lambda location_time: location_time - 0.5,
                    direction=CrossingDirection.RISING,
                ),
                entry_policy=policy,
            )
        ],
        source,
        np.array([0.0]),
    )
    if policy is SurfaceEntryPolicy.ERROR:
        with pytest.raises(SurfaceEntryError):
            simulate(system, (2.0, 2.5), initial_location_time=0.5)
    elif policy is SurfaceEntryPolicy.TRIGGER:
        trace = simulate(
            system,
            (2.0, 2.5),
            initial_location_time=0.5,
        )
        assert trace.events[0].location_time_before == 0.5
        np.testing.assert_allclose(
            trace.sample([2.0, 2.5])["location_time"], [0.0, 0.5]
        )
    else:
        with pytest.raises(SimulationProgressError):
            simulate(system, (2.0, 2.5), initial_location_time=0.5)


def test_overdue_initial_timeout_does_not_fire() -> None:
    source = Location(lambda: np.array([0.0]))
    target = Location(lambda: np.array([0.0]))
    system = HybridSystem(
        [source, target],
        [
            Transition(
                source,
                target,
                EventSurface(
                    lambda location_time: location_time - 0.5,
                    direction=CrossingDirection.RISING,
                ),
            )
        ],
        source,
        np.array([0.0]),
    )
    trace = simulate(system, (2.0, 3.0), initial_location_time=1.0)
    assert trace.events == ()
    np.testing.assert_allclose(
        trace.sample([2.0, 3.0])["location_time"], [1.0, 2.0]
    )


def test_batch_comprehension_retains_independent_initial_states_and_ages() -> (
    None
):
    location = Location(lambda: np.array([0.0]))
    system = HybridSystem([location], [], location, np.array([0.0]))
    traces = [
        simulate(system, (5.0, 6.0), x0=state, initial_location_time=3.0)
        for state in ([1.0], [2.0])
    ]
    assert len(traces) == 2
    for trace, state in zip(traces, [1.0, 2.0], strict=True):
        frame = trace.sample([5.0, 6.0])
        np.testing.assert_allclose(frame["location_time"], [3.0, 4.0])
        np.testing.assert_allclose(frame["x0"], [state, state])
    empty = traces[0].sample([], include_derivatives=True)
    assert empty.height == 0
    assert empty.schema["dx0"].is_numeric()


@pytest.mark.parametrize("mode", ["fixed", "duplicates", "off_grid", "final"])
def test_visit_clock_aligns_at_boundary_for_every_sampling_mode(
    mode: str,
) -> None:
    """Right-continuous visit ages and derivative capture follow the final visit."""
    source = Location(
        lambda *, location_time: np.array([location_time]), label="same"
    )
    target = Location(
        lambda *, location_time: np.array([location_time + 10.0]), label="same"
    )
    system = HybridSystem(
        [source, target],
        [Transition(source, target, lambda *, t: t - 1.0)],
        source,
        np.array([0.0]),
    )
    end = 1.0 if mode == "final" else 1.5
    trace = simulate(system, (0.0, end))
    if mode == "fixed":
        frame = trace.sample(
            dt=0.5, include_derivatives=True, include_location_label=True
        )
    else:
        times = {
            "duplicates": [0.0, 0.5, 1.0, 1.0, 1.25, 1.5],
            "off_grid": [0.0, 0.75, 1.25, 1.5],
            "final": [0.0, 0.5, 1.0],
        }[mode]
        frame = trace.sample(
            times, include_derivatives=True, include_location_label=True
        )
    assert trace.events[0].location_time_before == pytest.approx(1.0)
    sampled_times = frame["t"].to_numpy()
    sampled_ages = frame["location_time"].to_numpy()
    post_jump = sampled_times >= 1.0
    np.testing.assert_allclose(
        sampled_ages[post_jump], sampled_times[post_jump] - 1.0
    )
    np.testing.assert_allclose(
        frame["dx0"].to_numpy()[post_jump], 10.0 + sampled_ages[post_jump]
    )
    np.testing.assert_allclose(
        sampled_ages[~post_jump], sampled_times[~post_jump]
    )
    assert frame["location_label"].to_list() == ["same"] * frame.height
    if mode == "off_grid":
        assert not np.any(sampled_times == trace.events[0].time)
    else:
        assert np.count_nonzero(sampled_times == trace.events[0].time) == (
            2 if mode == "duplicates" else 1
        )
        np.testing.assert_allclose(sampled_ages[sampled_times == 1.0], 0.0)


@pytest.mark.parametrize("role", ["flow", "surface", "reset", "delay"])
@pytest.mark.parametrize(
    "form",
    [
        "positional",
        "varargs",
        "unknown",
        "uninspectable",
        "named",
        "keyword",
        "kwargs",
    ],
)
def test_callback_dispatch_preserves_legacy_forms_and_exposes_age(
    role: str, form: str
) -> None:
    """Every role supports canonical subsets without changing four-arg fallback."""
    calls: list[tuple[object, ...]] = []

    def positional(a: float, b: np.ndarray, c: object, d: object, /) -> float:
        calls.append((a, b, c, d))
        return 0.0

    def varargs(*args: object) -> float:
        calls.append(args)
        return 0.0

    def unknown(a: float, b: np.ndarray, c: object, d: object) -> float:
        calls.append((a, b, c, d))
        return 0.0

    class Uninspectable:
        @property
        def __signature__(self) -> object:
            raise ValueError("signature unavailable")

        def __call__(self, *args: object) -> float:
            calls.append(args)
            return 0.0

    def named(location_time: float) -> float:
        calls.append((location_time,))
        return 0.0

    def keyword(*, location_time: float, state: np.ndarray) -> float:
        calls.append((location_time, state))
        return 0.0

    def kwargs(**values: object) -> float:
        calls.append(tuple(values.keys()))
        return 0.0

    callbacks = {
        "positional": positional,
        "varargs": varargs,
        "unknown": unknown,
        "uninspectable": Uninspectable(),
        "named": named,
        "keyword": keyword,
        "kwargs": kwargs,
    }
    callback = callbacks[form]
    source = Location(callback if role == "flow" else lambda: np.array([0.0]))
    target = Location(lambda: np.array([0.0]))
    system = HybridSystem(
        [source, target],
        [
            Transition(
                source,
                target,
                callback if role == "surface" else lambda: 0.0,
                callback if role == "reset" else lambda state: state,
                entry_policy=SurfaceEntryPolicy.TRIGGER,
                delay=callback if role == "delay" else 0.0,
            )
        ]
        if role != "flow"
        else [],
        source,
        np.array([0.0]),
        parameters={"rate": 1.0},
    )
    trace = simulate(system, (4.0, 4.25), initial_location_time=2.0)
    assert calls
    if form in {"positional", "varargs", "unknown", "uninspectable"}:
        assert len(calls[0]) == 4
        assert calls[0][0] == 4.0
        np.testing.assert_array_equal(calls[0][1], [0.0])
        assert calls[0][2] == {"rate": 1.0}
        assert callable(calls[0][3])
    elif form == "kwargs":
        assert set(calls[0]) == {
            "t",
            "state",
            "parameters",
            "input_stream",
            "location_time",
        }
    else:
        assert calls[0][0] == 2.0
    if role != "flow":
        assert trace.events[0].location_time_before == 2.0
