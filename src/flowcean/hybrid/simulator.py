"""SciPy event-driven hybrid simulation, independent of sampling grids."""

from collections.abc import Callable, Iterable, Sequence
from itertools import groupby
from typing import NamedTuple

import numpy as np
from scipy.integrate import RK45, DenseOutput, OdeSolution
from scipy.optimize import brentq

from ._runtime import (
    _BoundFunction,
    _ResidenceClock,
    _RunBindings,
    ensure_state,
)
from .hybrid_system import (
    HybridSystem,
    InputStream,
    Location,
    SurfaceEntryPolicy,
    Transition,
    TransitionSchedulingPolicy,
    display_label,
)
from .trajectory import ContinuousSegment, Event, HybridTrajectory


class HybridSimulationError(RuntimeError):
    """Base class for hybrid simulation runtime failures."""


class InvalidEventSurfaceValueError(HybridSimulationError):
    """Raised when an event surface returns NaN."""

    def __init__(
        self,
        transitions: Sequence[Transition],
        time: float,
        *,
        context: str,
    ) -> None:
        super().__init__(
            f"Event surfaces returned NaN {context} at t={time!r}: {_transition_descriptions(transitions)}."
        )
        self.add_note(
            "An event surface must return a scalar value other than NaN; exact zero denotes the surface."
        )


class SurfaceEntryError(HybridSimulationError):
    """Raised when an ERROR surface is zero on location entry."""


class AmbiguousTransitionError(HybridSimulationError):
    """Raised for competing entry-time jumps or equal scheduled deadlines."""


class SimulationProgressError(HybridSimulationError):
    """Raised when a solver event does not advance physical time."""


class _EventFn:
    """Bound event surface with runtime validation."""

    def __init__(
        self,
        transition: Transition,
        surface: _BoundFunction[float],
        clock: _ResidenceClock,
    ) -> None:
        self.transition = transition
        self.surface = surface
        self.clock = clock
        self.direction = int(transition.event_surface.direction)

    def __call__(self, t: float, y: np.ndarray) -> float:
        value = self.surface(t, y, self.clock.age(t))
        if np.isnan(value):
            raise InvalidEventSurfaceValueError(
                [self.transition], t, context="during continuous integration"
            )
        return value


class _PendingTransition(NamedTuple):
    transition: Transition
    detection_time: float
    deadline: float


def _schedule(
    pending: dict[Transition, _PendingTransition],
    transition: Transition,
    time: float,
    delay: float,
) -> None:
    candidate = _PendingTransition(transition, time, time + delay)
    if not np.isfinite(candidate.deadline) or (
        delay > 0 and candidate.deadline <= time
    ):
        raise SimulationProgressError(
            "Transition delay cannot advance to a finite representable deadline."
        )
    previous = pending.get(transition)
    policy = transition.scheduling_policy
    if (
        previous is None
        or policy is TransitionSchedulingPolicy.LATEST_DETECTION
        or (
            policy is TransitionSchedulingPolicy.EACH_DETECTION
            and (candidate.deadline, candidate.detection_time)
            < (previous.deadline, previous.detection_time)
        )
    ):
        pending[transition] = candidate


def _earliest(
    pending: dict[Transition, _PendingTransition],
) -> _PendingTransition | None:
    return min(pending.values(), key=lambda item: item.deadline, default=None)


def _check_conflicts(
    pending: dict[Transition, _PendingTransition],
    winner: _PendingTransition,
) -> None:
    tied = [
        item.transition
        for item in pending.values()
        if item.deadline == winner.deadline
    ]
    if len(tied) > 1:
        raise AmbiguousTransitionError(
            f"Multiple transitions are scheduled for t={winner.deadline!r}: "
            f"{_transition_descriptions(tied)}."
        )


class _EntryResult(NamedTuple):
    state: np.ndarray
    location: Location
    events: tuple[Event, ...]
    jumps: int
    clock: _ResidenceClock
    pending: dict[Transition, _PendingTransition]
    consumed: set[Transition]


def simulate(
    system: HybridSystem,
    t_span: tuple[float, float],
    x0: Iterable[float] | None = None,
    location0: Location | None = None,
    *,
    input_stream: InputStream | None = None,
    initial_location_time: float = 0.0,
    max_jumps: int = 256,
    rtol: float = 1e-7,
    atol: float = 1e-9,
    max_step: float | None = None,
) -> HybridTrajectory:
    """Integrate a hybrid execution with dense continuous segments.

    Sampling is a separate operation on the returned trajectory. Callbacks and
    input streams must be pure and deterministic under repeated evaluation.
    Equal endpoints resolve entry transitions without calling the ODE solver.
    Parameters are snapshotted for every location at the start of the run.
    Each transition's scheduling policy selects accepted detections and
    pending occurrences. Delay callbacks run once per accepted detection,
    freezing that occurrence's deadline. A positive delay keeps the source
    flow and residence clock active until execution. Leaving the source visit
    cancels all its occurrences. At exactly equal numerical times, all
    direction-qualified detections precede execution, even at the final
    endpoint; a latest detection can postpone execution beyond the run.
    Distinct transitions with equal earliest deadlines are ambiguous.

    Root detection uses integration-step endpoint signs: multiple roots within
    a step can be missed, and tangencies are step-dependent. Equality means
    exact numerical equality, not mathematical simultaneity or a tolerance.

    Args:
        system: Model to simulate.
        t_span: Finite start and end times, with end at or after start.
        x0: Initial continuous state, overriding the model's initial state.
        location0: Initial location, overriding the model's initial location.
        input_stream: Function returning a one-dimensional input vector at a
            requested physical time.
        initial_location_time: Finite, nonnegative age of the initial visit.
            Each subsequent transition begins a new visit at age zero.
            Each run starts with no pending transition.
        max_jumps: Maximum number of transitions, including same-time chains.
        rtol: Relative tolerance for SciPy integration.
        atol: Absolute tolerance for SciPy integration.
        max_step: Maximum integration step; None uses SciPy's default.

    Returns:
        The trajectory, retaining its supplied initial condition and ordered
        continuous segments and events.

    Raises:
        HybridSimulationError: The transition limit is exceeded or a subclass
            reports an invalid surface, ambiguous entry, or progress failure.
        ValueError: Time bounds, initial conditions, or callback outputs are
            invalid.
    """
    start, end = (float(value) for value in t_span)
    if not np.isfinite(start) or not np.isfinite(end):
        raise ValueError("t_span endpoints must be finite.")
    if end < start:
        raise ValueError("t_span must not be reversed.")
    if not np.isfinite(initial_location_time) or initial_location_time < 0:
        raise ValueError(
            "initial_location_time must be finite and nonnegative."
        )
    initial_location = (
        system.initial_location if location0 is None else location0
    )
    if not isinstance(initial_location, Location):
        raise TypeError("location0 must be a Location.")
    if initial_location not in system.locations:
        raise ValueError("location0 must be included in system.locations.")
    initial_state = ensure_state(system.initial_state if x0 is None else x0)
    bindings = _RunBindings(system, input_stream)
    execution: list[ContinuousSegment | Event] = []
    clock = _ResidenceClock(start, float(initial_location_time))
    entry = _settle_location_entries(
        system,
        initial_location,
        initial_state.copy(),
        start,
        bindings,
        first_microstep=0,
        clock=clock,
        jumps=0,
        max_jumps=max_jumps,
    )
    state, location, events, jumps, clock, pending, consumed = entry
    execution.extend(events)
    current = start
    while current < end:
        segment, winner = _integrate_visit(
            system,
            location,
            state,
            (current, end),
            bindings,
            clock,
            pending,
            consumed,
            rtol=rtol,
            atol=atol,
            max_step=max_step,
        )
        execution.append(segment)
        if winner is None:
            break
        transition = winner.transition
        event_time = winner.deadline
        event_state = segment.evaluate(event_time).state
        jumps = _increment_jumps(jumps, max_jumps)
        state, event = _apply_transition(
            transition,
            event_time,
            event_state,
            bindings,
            microstep=0,
            location_time=clock.age(event_time),
            detection_time=winner.detection_time,
        )
        execution.append(event)

        entry = _settle_location_entries(
            system,
            transition.target,
            state,
            event_time,
            bindings,
            first_microstep=1,
            clock=_ResidenceClock(event_time, 0.0),
            jumps=jumps,
            max_jumps=max_jumps,
        )
        state, location, events, jumps, clock, pending, consumed = entry
        execution.extend(events)
        current = event_time
    return HybridTrajectory(
        system,
        (start, end),
        initial_state,
        initial_location,
        float(initial_location_time),
        tuple(execution),
        bindings,
    )


def _wrap_flow(
    bound_flow: _BoundFunction[np.ndarray],
    clock: _ResidenceClock,
) -> Callable[[float, np.ndarray], np.ndarray]:
    """Supply a visit's residence clock to a bound flow for SciPy integration."""

    def flow(time: float, state: np.ndarray) -> np.ndarray:
        return bound_flow(time, state, clock.age(time))

    return flow


def _step_roots(
    surfaces: Sequence[_EventFn],
    values: dict[Transition, float],
    consumed: set[Transition],
    interval: tuple[float, float],
    state: np.ndarray,
    dense: DenseOutput,
) -> list[tuple[float, Transition]]:
    """Locate all direction-qualified roots of one accepted integration step.

    Use actual endpoint values, not a value reevaluated at an approximate
    root. A consumed endpoint zero remains latched until a nonzero endpoint;
    this suppresses duplicate detections on departure or a zero plateau.
    """
    left, right = interval
    roots = []
    for surface in surfaces:
        transition = surface.transition
        before = values[transition]
        after = surface(right, state)
        rising = before <= 0 <= after
        falling = before >= 0 >= after
        crossing = (rising and surface.direction >= 0) or (
            falling and surface.direction <= 0
        )
        if crossing and not (before == 0 and transition in consumed):
            # Dense-output endpoint rounding can differ from the accepted
            # state. Preserve the actual endpoint values defining the bracket.
            def bracket_value(
                time: float,
                surface: _EventFn = surface,
                before: float = before,
                after: float = after,
            ) -> float:
                if time == left:
                    return before
                if time == right:
                    return after
                return surface(time, dense(time))

            root = brentq(
                bracket_value,
                left,
                right,
                xtol=4 * np.finfo(float).eps,
                rtol=4 * np.finfo(float).eps,
            )
            roots.append((root, transition))
            if after == 0:
                consumed.add(transition)
        if after != 0:
            consumed.discard(transition)
        values[transition] = after
    return sorted(roots, key=lambda item: item[0])


def _integrate_visit(
    system: HybridSystem,
    location: Location,
    state: np.ndarray,
    t_span: tuple[float, float],
    bindings: _RunBindings,
    clock: _ResidenceClock,
    pending: dict[Transition, _PendingTransition],
    consumed: set[Transition],
    *,
    rtol: float,
    atol: float,
    max_step: float | None,
) -> tuple[ContinuousSegment, _PendingTransition | None]:
    """Integrate a visit, arbitrating every same-time root before execution.

    Detections do not change the flow, so retain accepted steps rather than
    restarting at approximate roots. A changed deadline can clip the current
    interpolant or tighten the next solver's bound. Postponement can leave an
    obsolete bound; resume exactly there, retaining endpoint signs and clock.
    """
    start, end = t_span
    current = start
    surfaces = [
        _EventFn(transition, bindings.surfaces[transition], clock)
        for transition in system.transitions_from(location)
    ]
    values = {
        surface.transition: surface(start, state) for surface in surfaces
    }
    knots = [start]
    interpolants: list[DenseOutput] = []
    solver = None
    winner = None
    while current < end:
        earliest = _earliest(pending)
        bound = min(end, earliest.deadline) if earliest else end
        if (
            solver is None
            or solver.status == "finished"
            or solver.t_bound > bound
        ):
            solver = RK45(
                _wrap_flow(bindings.flows[location], clock),
                current,
                state.copy(),
                bound,
                rtol=rtol,
                atol=atol,
                max_step=np.inf if max_step is None else max_step,
            )
        message = solver.step()
        if solver.status == "failed":
            raise HybridSimulationError(f"ODE integration failed: {message}")
        dense = solver.dense_output()
        right = float(solver.t)
        # FIRST remembers even a currently losing candidate for the full visit.
        active = [
            surface
            for surface in surfaces
            if surface.transition not in pending
            or surface.transition.scheduling_policy
            is not TransitionSchedulingPolicy.FIRST_DETECTION
        ]
        roots = _step_roots(
            active, values, consumed, (current, right), solver.y, dense
        )
        for time, group in groupby(roots, key=lambda item: item[0]):
            earliest = _earliest(pending)
            if earliest is not None and earliest.deadline < time:
                break
            if time <= start:
                error = SimulationProgressError(
                    f"An event did not advance physical time (segment start={start!r}, event time={time!r})."
                )
                error.add_note(
                    "This can result from stateful callbacks, discontinuous event surfaces, "
                    "or insufficient floating-point time resolution. Use deterministic "
                    "callbacks and continuous event surfaces."
                )
                raise error
            event_state = dense(time)
            for _, transition in group:
                delay = bindings.delay(
                    transition, time, event_state, clock.age(time)
                )
                _schedule(pending, transition, time, delay)
        earliest = _earliest(pending)
        if earliest is not None and earliest.deadline <= right:
            _check_conflicts(pending, earliest)
            winner = earliest
            right = earliest.deadline
        if right > current:
            knots.append(right)
            interpolants.append(dense)
        if winner is not None:
            break
        current = right
        state = solver.y
    times = np.array(knots)
    return (
        ContinuousSegment(
            location,
            (start, knots[-1]),
            OdeSolution(times, interpolants),
            clock,
            times,
        ),
        winner,
    )


def _apply_transition(
    transition: Transition,
    time: float,
    state: np.ndarray,
    bindings: _RunBindings,
    *,
    microstep: int,
    location_time: float,
    detection_time: float,
) -> tuple[np.ndarray, Event]:
    before = ensure_state(state)
    if transition.reset is None:
        after = before.copy()
    else:
        after = bindings.resets[transition](time, before.copy(), location_time)
    return after, Event(
        time,
        transition,
        before,
        after,
        microstep,
        location_time,
        detection_time,
    )


def _settle_location_entries(
    system: HybridSystem,
    location: Location,
    state: np.ndarray,
    time: float,
    bindings: _RunBindings,
    *,
    first_microstep: int,
    clock: _ResidenceClock,
    jumps: int,
    max_jumps: int,
) -> _EntryResult:
    """Resolve entry policies and apply immediate jumps until entry settles.

    Evaluate all outgoing surfaces first so NaN, ERROR, and ambiguity checks
    precede any reset.
    """
    events: list[Event] = []
    microstep = first_microstep
    while True:
        transitions = system.transitions_from(location)
        values = [
            bindings.surfaces[transition](time, state, clock.age(time))
            for transition in transitions
        ]
        invalid = [
            transition
            for transition, value in zip(transitions, values, strict=True)
            if np.isnan(value)
        ]
        if invalid:
            raise InvalidEventSurfaceValueError(
                invalid,
                time,
                context=f"while entering {display_label(location)!r}",
            )
        errors = [
            transition
            for transition, value in zip(transitions, values, strict=True)
            if value == 0.0
            and transition.entry_policy is SurfaceEntryPolicy.ERROR
        ]
        if errors:
            error = SurfaceEntryError(
                f"Event surfaces are zero while entering {display_label(location)!r} "
                f"at t={time!r}: {_transition_descriptions(errors)}.",
            )
            error.add_note(
                "Choose TRIGGER to detect the transition on entry or CONTINUE to begin continuous integration from the surface."
            )
            raise error
        triggers = [
            transition
            for transition, value in zip(transitions, values, strict=True)
            if value == 0.0
            and transition.entry_policy is SurfaceEntryPolicy.TRIGGER
        ]
        delays = [
            (
                transition,
                bindings.delay(transition, time, state, clock.age(time)),
            )
            for transition in triggers
        ]
        immediate = [transition for transition, delay in delays if delay == 0]
        if len(immediate) > 1:
            error = AmbiguousTransitionError(
                f"Multiple transitions request an entry-time jump from {display_label(location)!r} "
                f"at t={time!r}: {_transition_descriptions(immediate)}.",
            )
            error.add_note(
                "Make at most one outgoing zero-delay TRIGGER surface zero on entry."
            )
            raise error
        if not immediate:
            pending: dict[Transition, _PendingTransition] = {}
            for transition, delay in delays:
                _schedule(pending, transition, time, delay)
            return _EntryResult(
                state,
                location,
                tuple(events),
                jumps,
                clock,
                pending,
                set(triggers),
            )
        transition = immediate[0]
        jumps = _increment_jumps(jumps, max_jumps)
        state, event = _apply_transition(
            transition,
            time,
            state,
            bindings,
            microstep=microstep,
            location_time=clock.age(time),
            detection_time=time,
        )
        events.append(event)
        location = transition.target
        clock = _ResidenceClock(time, 0.0)
        microstep += 1


def _transition_descriptions(transitions: Sequence[Transition]) -> str:
    return ", ".join(
        f"{display_label(transition.source)} -> {display_label(transition.target)} [{display_label(transition.event_surface)}]"
        for transition in transitions
    )


def _increment_jumps(jumps: int, max_jumps: int) -> int:
    jumps += 1
    if jumps > max_jumps:
        raise HybridSimulationError("Maximum number of transitions exceeded.")
    return jumps
