"""Visit-local scheduling and exact-time detection/execution arbitration."""

from itertools import permutations

import numpy as np
import pytest

from flowcean.hybrid import (
    AmbiguousTransitionError,
    CrossingDirection,
    EventSurface,
    HybridSystem,
    Location,
    SurfaceEntryPolicy,
    Transition,
    TransitionSchedulingPolicy,
    simulate,
)


def _system(source, target, transitions):
    return HybridSystem([source, target], transitions, source, np.array([0.0]))


def test_scheduling_policy_defaults_and_requires_exact_enum():
    source, target = Location(lambda: 1.0), Location(lambda: 0.0)
    assert Transition(source, target, lambda: 1).scheduling_policy is (
        TransitionSchedulingPolicy.FIRST_DETECTION
    )
    assert [policy.value for policy in TransitionSchedulingPolicy] == [
        "first_detection",
        "latest_detection",
        "each_detection",
    ]
    for invalid in ["first_detection", SurfaceEntryPolicy.TRIGGER, None, 0]:
        with pytest.raises(TypeError, match="TransitionSchedulingPolicy"):
            Transition(source, target, lambda: 1, scheduling_policy=invalid)
    with pytest.raises(TypeError):
        Transition(
            source,
            target,
            lambda: 1,
            None,
            TransitionSchedulingPolicy.LATEST_DETECTION,  # pyright: ignore[reportCallIssue]
        )


@pytest.mark.parametrize("order", list(permutations(range(3))))
@pytest.mark.parametrize("callback", [False, True])
@pytest.mark.parametrize("entry_detection", [False, True])
def test_all_tied_detections_precede_deadline_independent_of_order(
    order, callback, entry_detection
):
    """An irrelevant B cannot hide C's zero-delay tie with pending A."""
    source, target = Location(lambda: 0.0), Location(lambda: 0.0)
    transitions = [
        Transition(
            source,
            target,
            lambda t: t if entry_detection else t - 0.25,
            entry_policy=SurfaceEntryPolicy.TRIGGER,
            delay=1 if entry_detection else 0.75,
        ),
        Transition(
            source,
            target,
            lambda t: t - 1,
            delay=(lambda: 2) if callback else 2,
        ),
        Transition(source, target, lambda t: t - 1),
    ]
    # Entry scheduling bounds a step at 1; otherwise all three roots can lie
    # inside one accepted step. Both must arbitrate A and C before execution.
    with pytest.raises(AmbiguousTransitionError):
        simulate(
            _system(source, target, [transitions[i] for i in order]), (0, 2)
        )


@pytest.mark.parametrize("policy", list(TransitionSchedulingPolicy))
@pytest.mark.parametrize("second_delay", [0.5, 5])
def test_repeated_detections_can_shorten_or_postpone(policy, second_delay):
    source, target = Location(lambda: 1.0), Location(lambda: 0.0)
    detections = []

    def delay(t):
        detections.append(t)
        return 3 if t < 1.5 else second_delay

    transition = Transition(
        source,
        target,
        lambda t: (t - 1) * (t - 2),
        delay=delay,
        scheduling_policy=policy,
    )
    trace = simulate(
        _system(source, target, [transition]), (0, 8), max_step=0.125
    )
    (event,) = trace.events
    replaced = policy is TransitionSchedulingPolicy.LATEST_DETECTION or (
        policy is TransitionSchedulingPolicy.EACH_DETECTION
        and second_delay < 2
    )
    assert event.detection_time == pytest.approx(2 if replaced else 1)
    assert event.time == pytest.approx(2 + second_delay if replaced else 4)
    assert detections == pytest.approx(
        [1] if policy is TransitionSchedulingPolicy.FIRST_DETECTION else [1, 2]
    )


@pytest.mark.parametrize("end", [2, 3, 4])
def test_latest_at_own_deadline_postpones_before_execution(end):
    source = Location(lambda location_time: location_time)
    target = Location(lambda: 0.0)
    detections = []

    def delay(t):
        detections.append(t)
        return 1 if t < 1.5 else 2

    transition = Transition(
        source,
        target,
        lambda t: (t - 1) * (t - 2),
        delay=delay,
        scheduling_policy=TransitionSchedulingPolicy.LATEST_DETECTION,
    )
    trace = simulate(
        _system(source, target, [transition]), (0, end), max_step=0.125
    )
    assert detections == pytest.approx([1, 2])
    if end < 4:
        assert not trace.events
        assert trace.evaluate(end).location is source
    else:
        (event,) = trace.events
        assert event.time == 4
        assert event.detection_time == 2
    # The obsolete bound at 2 must not introduce a gap or reset the visit age.
    times = np.linspace(0, min(end, 3), 17)
    frame = trace.sample(times)
    np.testing.assert_allclose(frame["x0"], 0.5 * times**2, atol=1e-12)
    np.testing.assert_allclose(frame["location_time"], times)


@pytest.mark.parametrize("tied_initially", [False, True])
@pytest.mark.parametrize("replacement_time", [2, 3])
def test_postponing_earliest_retains_other_candidates_and_recomputes_ties(
    tied_initially,
    replacement_time,
):
    source, target = Location(lambda: 1.0), Location(lambda: 0.0)
    moving = Transition(
        source,
        target,
        lambda t: (t - 1) * (t - replacement_time),
        delay=lambda t: 2 if t < 1.5 else 4,
        scheduling_policy=TransitionSchedulingPolicy.LATEST_DETECTION,
    )
    fixed = Transition(
        source, target, lambda t: t - 1.5, delay=1.5 if tied_initially else 2.5
    )
    trace = simulate(
        _system(source, target, [moving, fixed]), (0, 7), max_step=0.125
    )
    (event,) = trace.events
    assert event.transition is fixed
    assert event.detection_time == pytest.approx(1.5)
    assert event.time == pytest.approx(3 if tied_initially else 4)


def test_latest_replacement_can_create_cross_transition_tie():
    source, target = Location(lambda: 1.0), Location(lambda: 0.0)
    # Initially moving's first deadline is 5, then replacement ties at 4.
    moving = Transition(
        source,
        target,
        lambda t: (t - 1) * (t - 2),
        delay=lambda t: 4 if t < 1.5 else 2,
        scheduling_policy=TransitionSchedulingPolicy.LATEST_DETECTION,
    )
    fixed = Transition(
        source,
        target,
        lambda: 0,
        delay=4,
        entry_policy=SurfaceEntryPolicy.TRIGGER,
    )
    with pytest.raises(AmbiguousTransitionError):
        simulate(
            _system(source, target, [moving, fixed]), (0, 5), max_step=0.125
        )


@pytest.mark.parametrize("competing", [False, True])
def test_each_equal_own_deadlines_coalesce_with_earliest_detection(competing):
    source, target = Location(lambda: 1.0), Location(lambda: 0.0)
    detections = []

    def delay(t):
        detections.append(t)
        return 2 - t

    transition = Transition(
        source,
        target,
        lambda t: (t - 1) * (t - 2),
        delay=delay,
        scheduling_policy=TransitionSchedulingPolicy.EACH_DETECTION,
    )
    transitions = [transition]
    if competing:
        transitions.append(
            Transition(
                source,
                target,
                lambda: 0,
                delay=2,
                entry_policy=SurfaceEntryPolicy.TRIGGER,
            )
        )
        with pytest.raises(AmbiguousTransitionError):
            simulate(
                _system(source, target, transitions), (0, 4), max_step=0.125
            )
    else:
        trace = simulate(
            _system(source, target, transitions), (0, 4), max_step=0.125
        )
        (event,) = trace.events
        assert event.time == 2
        assert event.detection_time == pytest.approx(1)
    assert detections == pytest.approx([1, 2])


@pytest.mark.parametrize("policy", list(TransitionSchedulingPolicy))
def test_triggered_zero_episode_is_consumed_until_a_fresh_visit(policy):
    source = Location(lambda: 1.0)
    detections = []

    def delay(t):
        detections.append(t)
        return 1

    transition = Transition(
        source,
        source,
        lambda: 0,
        delay=delay,
        entry_policy=SurfaceEntryPolicy.TRIGGER,
        scheduling_policy=policy,
    )
    system = HybridSystem([source], [transition], source, np.array([0.0]))
    trace = simulate(system, (0, 2.5), max_step=0.1)
    assert detections == [0, 1, 2]
    assert [event.time for event in trace.events] == [1, 2]
    assert [event.detection_time for event in trace.events] == [0, 1]


def test_latest_only_resamples_direction_qualified_crossings():
    source, target = Location(lambda: 1.0), Location(lambda: 0.0)
    detections = []

    def delay(t):
        detections.append(t)
        return 2

    transition = Transition(
        source,
        target,
        EventSurface(
            lambda t: (t - 1) * (t - 2), direction=CrossingDirection.FALLING
        ),
        delay=delay,
        scheduling_policy=TransitionSchedulingPolicy.LATEST_DETECTION,
    )
    trace = simulate(
        _system(source, target, [transition]), (0, 4), max_step=0.125
    )
    assert detections == pytest.approx([1])
    assert trace.events[0].time == pytest.approx(3)


def test_nearby_distinct_roots_are_not_grouped_and_post_exit_delay_is_not_called():
    source, target = Location(lambda: 0.0), Location(lambda: 0.0)
    first = Transition(source, target, lambda t: t - 0.5)

    def forbidden_delay():
        pytest.fail(
            "a crossing after the source exit must not sample its delay"
        )

    nearby = Transition(
        source, target, lambda t: t - (0.5 + 5e-13), delay=forbidden_delay
    )
    trace = simulate(_system(source, target, [nearby, first]), (0, 1))
    assert trace.events[0].transition is first
    assert trace.events[0].time == 0.5


def test_internal_knot_departure_can_detect_an_unconsumed_root():
    """A wrong-direction arrival must not consume a later qualified departure."""
    source, target = Location(lambda: 0.0), Location(lambda: 0.0)
    postponed = Transition(
        source,
        target,
        lambda t: t * (t - 0.5),
        entry_policy=SurfaceEntryPolicy.TRIGGER,
        delay=lambda t: 1 if t == 0 else 2.5,
        scheduling_policy=TransitionSchedulingPolicy.LATEST_DETECTION,
    )
    departure = Transition(
        source,
        target,
        EventSurface(
            lambda t: (t - 1) ** 2, direction=CrossingDirection.RISING
        ),
    )
    trace = simulate(
        _system(source, target, [postponed, departure]), (0, 4), max_step=0.125
    )
    (event,) = trace.events
    assert event.transition is departure
    assert event.detection_time == 1
    assert event.time == 1
    assert all(
        segment.t_span[0] < segment.t_span[1] for segment in trace.segments
    )
    assert trace.evaluate(1).location is target


def test_consumed_entry_root_rearms_after_departure():
    source, target = Location(lambda: 0.0), Location(lambda: 0.0)
    detections = []

    def delay(t):
        detections.append(t)
        return 2

    transition = Transition(
        source,
        target,
        lambda t: t * (t - 1),
        delay=delay,
        entry_policy=SurfaceEntryPolicy.TRIGGER,
        scheduling_policy=TransitionSchedulingPolicy.LATEST_DETECTION,
    )
    trace = simulate(
        _system(source, target, [transition]), (0, 4), max_step=0.125
    )
    assert detections == pytest.approx([0, 1])
    assert trace.events[0].time == pytest.approx(3)


def test_nonzero_root_residual_does_not_repeat_detection():
    """Approximate irrational roots must not become new step-start brackets."""
    source, target = Location(lambda: 0.0), Location(lambda: 0.0)
    detections = []

    def delay(t):
        detections.append(t)
        return 1

    transition = Transition(
        source,
        target,
        lambda t: t * t - 2,
        delay=delay,
        scheduling_policy=TransitionSchedulingPolicy.LATEST_DETECTION,
    )
    trace = simulate(
        _system(source, target, [transition]), (0, 3), max_step=0.2
    )
    assert detections == pytest.approx([np.sqrt(2)])
    assert trace.events[0].time == pytest.approx(np.sqrt(2) + 1)


def test_repeated_delay_callbacks_use_current_source_context():
    source = Location(lambda: 2.0, parameters={"latency": 1})
    target = Location(lambda: 0.0)
    contexts = []

    def delay(t, state, parameters, input_stream, location_time):
        contexts.append(
            (
                t,
                state[0],
                parameters["latency"],
                input_stream(t)[0],
                location_time,
            )
        )
        return 4

    transition = Transition(
        source,
        target,
        lambda t: (t - 6) * (t - 7),
        delay=delay,
        scheduling_policy=TransitionSchedulingPolicy.LATEST_DETECTION,
    )
    trace = simulate(
        _system(source, target, [transition]),
        (5, 12),
        initial_location_time=3,
        input_stream=lambda t: np.array([t + 1]),
        max_step=0.125,
    )
    np.testing.assert_allclose(contexts, [[6, 2, 1, 7, 4], [7, 4, 1, 8, 5]])
    assert trace.events[0].time == pytest.approx(11)
    assert trace.events[0].location_time_before == pytest.approx(9)
