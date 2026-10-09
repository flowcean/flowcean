"""Switching delays, cancellation, and detection/execution-time context."""

import numpy as np
import pytest

from flowcean.hybrid import (
    AmbiguousTransitionError,
    CrossingDirection,
    EventSurface,
    HybridSystem,
    Location,
    SimulationProgressError,
    SurfaceEntryError,
    SurfaceEntryPolicy,
    Transition,
    build_hybrid_system_dot,
    simulate,
)


def _system(transitions, source, *targets):
    return HybridSystem(
        [source, *targets], transitions, source, np.array([0.0])
    )


@pytest.mark.parametrize(
    "delay", [-1.0, np.inf, -np.inf, np.nan, [1.0], np.array([1.0]), None]
)
@pytest.mark.parametrize("callback", [False, True])
def test_invalid_delay_is_rejected_when_resolved(delay, callback):
    source, target = Location(lambda: 1.0), Location(lambda: 0.0)
    if not callback:
        with pytest.raises(ValueError, match="delay must"):
            Transition(source, target, lambda t: t - 1, delay=delay)
        return

    transition = Transition(
        source, target, lambda t: t - 1, delay=lambda: delay
    )
    system = _system([transition], source, target)
    assert not simulate(system, (0, 0.5)).events
    with pytest.raises(ValueError, match="delay must"):
        simulate(system, (0, 2))


@pytest.mark.parametrize("delay", [0.0, 2.0])
def test_reset_and_samples_use_execution_state_and_source_context(delay):
    source = Location(
        lambda location_time: location_time, parameters={"offset": 10}
    )
    target = Location(lambda: -1.0, parameters={"offset": 100})
    transition = Transition(
        source,
        target,
        lambda t: t - 1,
        lambda t, state, location_time, parameters: (
            state + t + location_time + parameters["offset"]
        ),
        delay=delay,
    )
    trajectory = simulate(_system([transition], source, target), (0, 4))
    (event,) = trajectory.events
    assert event.detection_time == pytest.approx(1)
    assert event.time == pytest.approx(1 + delay)
    assert event.location_time_before == pytest.approx(1 + delay)
    np.testing.assert_allclose(event.state_before, [0.5 * event.time**2])
    np.testing.assert_allclose(
        event.state_after, event.state_before + 2 * event.time + 10
    )
    point = trajectory.evaluate(event.time)
    assert point.location is target
    assert point.location_time == 0
    frame = trajectory.sample([0, event.time, 4], include_derivatives=True)
    np.testing.assert_allclose(frame["dx0"], [0, -1, -1])
    if delay:
        waiting = trajectory.evaluate(2)
        assert waiting.location is source
        assert waiting.location_time == pytest.approx(2)
        np.testing.assert_allclose(waiting.state, [2])


@pytest.mark.parametrize("callback", [False, True])
def test_recrossings_neither_cancel_nor_restart_delay(callback):
    source, target = Location(lambda: 1.0), Location(lambda: 0.0)
    detections = []

    def delay(t):
        detections.append(t)
        return 2.5 + t

    transition = Transition(
        source,
        target,
        lambda t: np.sin(np.pi * (t - 0.5)),
        delay=delay if callback else 3,
    )
    trajectory = simulate(
        _system([transition], source, target), (0, 5), max_step=0.1
    )
    (event,) = trajectory.events
    assert event.detection_time == pytest.approx(0.5)
    assert event.time == pytest.approx(3.5)
    np.testing.assert_allclose(event.state_before, [3.5])
    assert detections == pytest.approx([0.5] if callback else [])


def test_competing_transition_cancels_delay_even_after_return_to_source():
    source, away, target = (
        Location(lambda: 1.0),
        Location(lambda: 1.0),
        Location(lambda: 0.0),
    )
    delayed = Transition(source, target, lambda t: t - 1, delay=3)
    leave = Transition(source, away, lambda t: t - 2)
    return_ = Transition(away, source, lambda t: t - 3)
    trajectory = simulate(
        _system([delayed, leave, return_], source, away, target), (0, 6)
    )
    assert [event.transition for event in trajectory.events] == [
        leave,
        return_,
    ]
    assert trajectory.evaluate(6).location is source


def test_self_transition_cancels_pending_occurrence_and_starts_fresh_visit():
    source, target = Location(lambda: 1.0), Location(lambda: 0.0)
    detections = []

    def delay(t):
        detections.append(t)
        return 2

    delayed = Transition(
        source, target, lambda location_time: location_time - 1, delay=delay
    )
    restart = Transition(
        source, source, lambda state: state[0] - 2, lambda state: state + 10
    )
    trajectory = simulate(_system([delayed, restart], source, target), (0, 6))
    assert [event.transition for event in trajectory.events] == [
        restart,
        delayed,
    ]
    assert trajectory.events[1].detection_time == pytest.approx(3)
    assert trajectory.events[1].time == pytest.approx(5)
    assert detections == pytest.approx([1, 3])


@pytest.mark.parametrize("reverse", [False, True])
def test_simultaneous_detections_with_distinct_deadlines_are_allowed(reverse):
    source, first, second = (
        Location(lambda: 1.0),
        Location(lambda: 0.0),
        Location(lambda: 0.0),
    )
    early = Transition(source, first, lambda t: t - 1, delay=1)
    late = Transition(source, second, lambda t: 1000 * (t - 1), delay=3)
    transitions = [early, late][:: -1 if reverse else 1]
    trajectory = simulate(_system(transitions, source, first, second), (0, 6))
    (event,) = trajectory.events
    assert event.transition is early
    assert event.detection_time == pytest.approx(1)
    assert event.time == pytest.approx(2)


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize(
    ("first_delay", "second_detection", "second_delay"),
    [(2, 1, 2), (2, 2, 1), (2, 3, 0)],
)
def test_equal_scheduled_deadlines_are_ambiguous(
    first_delay, second_detection, second_delay, reverse
):
    source, target = Location(lambda: 1.0), Location(lambda: 0.0)
    first = Transition(source, target, lambda t: t - 1, delay=first_delay)
    second = Transition(
        source,
        target,
        lambda t: t - second_detection,
        delay=lambda: second_delay,
    )
    transitions = [first, second][:: -1 if reverse else 1]
    with pytest.raises(AmbiguousTransitionError):
        simulate(_system(transitions, source, target), (0, 5))


def test_opposite_crossing_direction_does_not_create_ambiguity():
    source, target = Location(lambda: 1.0), Location(lambda: 0.0)
    delayed = Transition(source, target, lambda t: t - 1, delay=2)
    opposite = Transition(
        source,
        target,
        EventSurface(lambda t: t - 3, direction=CrossingDirection.FALLING),
    )
    trajectory = simulate(_system([delayed, opposite], source, target), (0, 4))
    assert trajectory.events[0].transition is delayed


@pytest.mark.parametrize(("end", "event_count"), [(2, 0), (3, 1), (4, 1)])
def test_final_endpoint_and_pending_occurrences(end, event_count):
    source, target = Location(lambda: 1.0), Location(lambda: 0.0)
    delayed = Transition(source, target, lambda t: t - 1, delay=2)
    trajectory = simulate(_system([delayed], source, target), (0, end))
    assert len(trajectory.events) == event_count
    assert trajectory.evaluate(end).location is (
        target if event_count else source
    )
    assert trajectory.evaluate(end).location_time == pytest.approx(
        end - 3 if event_count else end
    )


@pytest.mark.parametrize("callback", [False, True])
def test_trigger_on_entry_starts_delay_and_preserves_initial_age(callback):
    source, target = Location(lambda: 1.0), Location(lambda: 0.0)
    delayed = Transition(
        source,
        target,
        lambda state: state[0],
        delay=(lambda t, location_time: t - location_time + 1)
        if callback
        else 2,
        entry_policy=SurfaceEntryPolicy.TRIGGER,
    )
    trajectory = simulate(
        _system([delayed], source, target), (5, 8), initial_location_time=4
    )
    (event,) = trajectory.events
    assert event.detection_time == 5
    assert event.time == 7
    assert event.location_time_before == 6
    assert trajectory.evaluate(6).location_time == 5


def test_multiple_entry_detections_can_schedule_different_deadlines():
    source, target = Location(lambda: 1.0), Location(lambda: 0.0)
    transitions = [
        Transition(
            source,
            target,
            lambda state: state[0],
            delay=delay,
            entry_policy=SurfaceEntryPolicy.TRIGGER,
        )
        for delay in [lambda: 3, lambda: 1]
    ]
    trajectory = simulate(_system(transitions, source, target), (0, 4))
    assert trajectory.events[0].transition is transitions[1]
    assert trajectory.events[0].detection_time == 0


def test_immediate_entry_jump_supersedes_delayed_detection():
    source, target = Location(lambda: 1.0), Location(lambda: 0.0)
    transitions = [
        Transition(
            source,
            target,
            lambda state: state[0],
            delay=delay,
            entry_policy=SurfaceEntryPolicy.TRIGGER,
        )
        for delay in [lambda: 3, lambda: 0]
    ]
    trajectory = simulate(_system(transitions, source, target), (0, 4))
    (event,) = trajectory.events
    assert event.transition is transitions[1]
    assert event.time == event.detection_time == 0


@pytest.mark.parametrize("delay", [0, 1])
def test_equal_entry_callback_deadlines_are_ambiguous(delay):
    source, target = Location(lambda: 1.0), Location(lambda: 0.0)
    transitions = [
        Transition(
            source,
            target,
            lambda: 0,
            delay=lambda: delay,
            entry_policy=SurfaceEntryPolicy.TRIGGER,
        )
        for _ in range(2)
    ]
    with pytest.raises(AmbiguousTransitionError):
        simulate(_system(transitions, source, target), (0, 2))


def test_error_entry_policy_still_rejects_delayed_surface():
    source = Location(lambda: 1.0)
    delayed = Transition(source, source, lambda state: state[0], delay=1)
    with pytest.raises(SurfaceEntryError):
        simulate(_system([delayed], source), (0, 2))


def test_continue_entry_policy_allows_departure_then_later_detection():
    source, target = Location(lambda: 1.0), Location(lambda: 0.0)
    delayed = Transition(
        source,
        target,
        EventSurface(
            lambda t: t * (2 - t), direction=CrossingDirection.FALLING
        ),
        delay=1,
        entry_policy=SurfaceEntryPolicy.CONTINUE,
    )
    trajectory = simulate(
        _system([delayed], source, target), (0, 4), max_step=0.2
    )
    (event,) = trajectory.events
    assert event.detection_time == pytest.approx(2)
    assert event.time == pytest.approx(3)


def test_initial_age_does_not_infer_pending_occurrences():
    source, target = Location(lambda: 1.0), Location(lambda: 0.0)
    delayed = Transition(
        source, target, lambda location_time: location_time - 1, delay=3
    )
    trajectory = simulate(
        _system([delayed], source, target), (0, 5), initial_location_time=2
    )
    assert not trajectory.events


def test_scheduling_does_not_count_as_jump_and_zero_span_can_schedule():
    source, target = Location(lambda: 1.0), Location(lambda: 0.0)
    delayed = Transition(
        source,
        target,
        lambda state: state[0],
        delay=2,
        entry_policy=SurfaceEntryPolicy.TRIGGER,
    )
    for end in [0, 1]:
        trajectory = simulate(
            _system([delayed], source, target), (0, end), max_jumps=0
        )
        assert not trajectory.events
    with pytest.raises(RuntimeError, match="Maximum number"):
        simulate(_system([delayed], source, target), (0, 3), max_jumps=0)


def test_delayed_event_starts_immediate_chain_with_correct_detection_times():
    source, middle, target = (
        Location(lambda: 1.0),
        Location(lambda: 0.0),
        Location(lambda: 0.0),
    )
    delayed = Transition(
        source, middle, lambda t: t - 1, lambda: np.array([0.0]), delay=2
    )
    immediate = Transition(
        middle,
        target,
        lambda state: state[0],
        entry_policy=SurfaceEntryPolicy.TRIGGER,
    )
    trajectory = simulate(
        _system([delayed, immediate], source, middle, target), (0, 3)
    )
    assert [event.time for event in trajectory.events] == [3, 3]
    assert [event.detection_time for event in trajectory.events] == [1, 3]
    assert [event.microstep for event in trajectory.events] == [0, 1]
    assert trajectory.evaluate(3).location is target


def test_unrepresentable_deadline_reports_progress_error():
    source = Location(lambda: 0.0)
    delayed = Transition(
        source,
        source,
        lambda: 0,
        delay=0.01,
        entry_policy=SurfaceEntryPolicy.TRIGGER,
    )
    with pytest.raises(
        SimulationProgressError, match="representable deadline"
    ):
        simulate(_system([delayed], source), (1e16, 1e16))


def test_graph_annotates_transition_delay_without_evaluating_callbacks():
    source, target = Location(lambda: 1.0), Location(lambda: 0.0)

    def closing_delay():
        pytest.fail("graph export must not evaluate the delay")

    for delay, label in [
        (2, "2"),
        (closing_delay, closing_delay.__qualname__),
    ]:
        delayed = Transition(source, target, lambda t: t - 1, delay=delay)
        assert f"delay: {label}" in build_hybrid_system_dot(
            _system([delayed], source, target), show_event_labels=False
        )


def test_delayed_self_transition_schedules_each_new_visit_and_is_run_local():
    source = Location(lambda: 1.0)
    delayed = Transition(
        source,
        source,
        lambda location_time: location_time,
        delay=1,
        entry_policy=SurfaceEntryPolicy.TRIGGER,
    )
    system = _system([delayed], source)
    for _ in range(2):
        trajectory = simulate(system, (0, 3))
        assert [event.time for event in trajectory.events] == [1, 2, 3]
        assert [event.detection_time for event in trajectory.events] == [
            0,
            1,
            2,
        ]
        assert [event.location_time_before for event in trajectory.events] == [
            1,
            1,
            1,
        ]
        np.testing.assert_allclose(trajectory.evaluate(3).state, [3])


@pytest.mark.parametrize("tied", [False, True])
def test_earlier_deadline_replaces_pending_transition_and_any_conflict(tied):
    source, target = Location(lambda: 1.0), Location(lambda: 0.0)
    later = Transition(source, target, lambda t: t - 1, delay=4)
    earlier = Transition(source, target, lambda t: t - 2, delay=1)
    transitions = [later, earlier]
    if tied:
        transitions.append(
            Transition(source, target, lambda t: t - 1.5, delay=3.5)
        )
    trajectory = simulate(_system(transitions, source, target), (0, 6))
    (event,) = trajectory.events
    assert event.transition is earlier
    assert event.detection_time == pytest.approx(2)
    assert event.time == pytest.approx(3)


@pytest.mark.parametrize("zero", [False, True])
def test_delay_callback_uses_detection_context_and_freezes_deadline(zero):
    source = Location(lambda: 2.0, parameters={"latency": 2})
    target = Location(lambda: 0.0, parameters={"latency": 100})
    contexts = []

    def delay(*, t, state, parameters, input_stream, location_time):
        contexts.append(
            (t, state.copy(), dict(parameters), input_stream(t), location_time)
        )
        if zero:
            return 0.0
        return (
            t
            + state[0]
            + parameters["bias"]
            + parameters["latency"]
            + input_stream(t)[0]
            + location_time
        ) / 8

    transition = Transition(source, target, lambda t: t - 6, delay=delay)
    system = HybridSystem(
        [source, target],
        [transition],
        source,
        np.array([0.0]),
        parameters={"latency": 10, "bias": 1},
    )
    trajectory = simulate(
        system,
        (5, 9),
        initial_location_time=3,
        input_stream=lambda t: np.array([t / 6]),
    )
    ((time, state, parameters, inputs, age),) = contexts
    assert time == pytest.approx(6)
    np.testing.assert_allclose(state, [2])
    assert parameters == {"latency": 2, "bias": 1}
    np.testing.assert_allclose(inputs, [1])
    assert age == pytest.approx(4)
    (event,) = trajectory.events
    assert event.detection_time == pytest.approx(6)
    assert event.time == pytest.approx(6 if zero else 8)
    np.testing.assert_allclose(event.state_before, [2 if zero else 6])
    if not zero:
        point = trajectory.evaluate(7)
        assert point.location is source
        assert point.location_time == pytest.approx(5)
        np.testing.assert_allclose(point.state, [4])
    trajectory.sample([5, 7, 9], include_derivatives=True)
    assert len(contexts) == 1


def test_new_occurrences_reevaluate_delay_and_runs_are_independent():
    source = Location(lambda: 1.0)
    detections = []

    def delay(t):
        detections.append(t)
        return t

    transition = Transition(
        source, source, lambda location_time: location_time - 1, delay=delay
    )
    system = _system([transition], source)
    for _ in range(2):
        detections.clear()
        trajectory = simulate(system, (0, 14))
        assert detections == pytest.approx([1, 3, 7])
        assert [event.time for event in trajectory.events] == pytest.approx(
            [2, 6, 14]
        )


def test_competing_callback_delays_are_not_resampled_after_supersession():
    source, target = Location(lambda: 1.0), Location(lambda: 0.0)
    detections = []

    def delay(t):
        detections.append(t)
        # Recrossing after t=3 would produce an earlier deadline if resampled.
        return 5 if t < 3 else 0

    transitions = [
        Transition(
            source, target, lambda t: np.sin(np.pi * (t - 0.5)), delay=delay
        ),
        Transition(source, target, lambda t: t - 1, delay=lambda: 3),
        Transition(source, target, lambda t: t - 2, delay=delay),
    ]
    trajectory = simulate(
        _system(transitions, source, target), (0, 5), max_step=0.1
    )
    (event,) = trajectory.events
    assert event.transition is transitions[1]
    assert event.time == pytest.approx(4)
    # Both superseded and later candidates retain their first detection.
    assert detections == pytest.approx([0.5, 2])


def test_detection_at_endpoint_leaves_delayed_transition_unexecuted():
    source, target = Location(lambda: 1.0), Location(lambda: 0.0)
    delayed = Transition(source, target, lambda t: t - 1, delay=1)
    trajectory = simulate(_system([delayed], source, target), (0, 1))
    assert not trajectory.events
    assert trajectory.evaluate(1).location is source
