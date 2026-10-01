"""SciPy event-driven hybrid simulation, independent of sampling grids."""

from collections.abc import Callable, Iterable, Sequence
from typing import NamedTuple

import numpy as np
from scipy.integrate import solve_ivp

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
    """Terminal event wrapper retaining SciPy's crossing-direction semantics."""

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
        self.terminal = True

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
    conflicts: tuple[Transition, ...] = ()

    @property
    def deadline(self) -> float:
        return self.detection_time + self.transition.delay


def _schedule(
    pending: _PendingTransition | None, transition: Transition, time: float
) -> _PendingTransition:
    candidate = _PendingTransition(transition, time)
    if not np.isfinite(candidate.deadline) or (
        transition.delay > 0 and candidate.deadline <= time
    ):
        raise SimulationProgressError(
            "Transition delay cannot advance to a finite representable deadline."
        )
    if pending is None or candidate.deadline < pending.deadline:
        return candidate
    if candidate.deadline == pending.deadline:
        return pending._replace(conflicts=(*pending.conflicts, transition))
    return pending


class _EntryResult(NamedTuple):
    state: np.ndarray
    location: Location
    events: tuple[Event, ...]
    jumps: int
    clock: _ResidenceClock
    pending: _PendingTransition | None


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
    A positive delay keeps the source flow and residence clock active until
    execution. Leaving the source visit cancels its pending transition.
    Deadlines at the final endpoint execute.

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
    state, location, events, jumps, clock, pending = entry
    execution.extend(events)
    current = start
    allow_same_time_detection = False
    while current < end:
        # The earliest deadline ends the source visit and cancels later ones.
        # Skip surfaces whose delay would finish too late even if detected now.
        transitions = [
            transition
            for transition in system.transitions_from(location)
            if pending is None
            or (
                transition is not pending.transition
                and transition not in pending.conflicts
                and current + transition.delay <= pending.deadline
            )
        ]
        event_fns = [
            _EventFn(transition, bindings.surfaces[transition], clock)
            for transition in transitions
        ]

        result = solve_ivp(
            _wrap_flow(bindings.flows[location], clock),
            (current, min(end, pending.deadline) if pending else end),
            state.copy(),
            events=event_fns or None,
            rtol=rtol,
            atol=atol,
            dense_output=True,
            max_step=np.inf if max_step is None else max_step,
        )
        if not result.success:
            raise HybridSimulationError(
                f"ODE integration failed: {result.message}"
            )
        has_event = result.t_events and any(
            len(times) for times in result.t_events
        )
        selected = (
            _first_event(result.t_events, result.y_events)
            if has_event
            else None
        )
        if selected is not None:
            _, event_time, _ = selected
            if event_time <= current and not allow_same_time_detection:
                error = SimulationProgressError(
                    f"An event did not advance physical time (segment start={current!r}, event time={event_time!r}).",
                )
                error.add_note(
                    "This can result from stateful callbacks, discontinuous event surfaces, "
                    "or insufficient floating-point time resolution. Use deterministic "
                    "callbacks and continuous event surfaces.",
                )
                raise error
        segment_end = float(result.t[-1])
        if segment_end > current:
            execution.append(
                ContinuousSegment(
                    location,
                    (current, segment_end),
                    result.sol,
                    clock,
                    result.t,
                )
            )

        if selected is not None:
            index, event_time, _ = selected
            pending = _schedule(pending, transitions[index], event_time)
        if pending is None or (selected is None and pending.deadline > end):
            break
        current = segment_end
        state = result.y[:, -1].copy()
        if pending.deadline > current:
            # Until execution, allow solver restarts without time progress.
            # Each detected candidate is excluded as pending/conflicting or
            # pruned for finishing too late, so it cannot block the restart.
            allow_same_time_detection = True
            continue

        if pending.conflicts:
            raise AmbiguousTransitionError(
                f"Multiple transitions are scheduled for t={current!r}: "
                f"{_transition_descriptions((pending.transition, *pending.conflicts))}."
            )
        transition = pending.transition
        event_time, event_state = current, state
        jumps = _increment_jumps(jumps, max_jumps)
        state, event = _apply_transition(
            transition,
            event_time,
            event_state,
            bindings,
            microstep=0,
            location_time=clock.age(event_time),
            detection_time=pending.detection_time,
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
        state, location, events, jumps, clock, pending = entry
        execution.extend(events)
        current = event_time
        allow_same_time_detection = False
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


def _first_event(
    t_events: Sequence[np.ndarray],
    y_events: Sequence[np.ndarray],
) -> tuple[int, float, np.ndarray]:
    """Select the earliest reported event; ties retain SciPy's ordering."""
    earliest_time = float("inf")
    earliest_index = -1
    earliest_state = np.zeros(0, dtype=float)
    for index, (times, states) in enumerate(
        zip(t_events, y_events, strict=False)
    ):
        if len(times) == 0:
            continue
        time = float(times[0])
        if time < earliest_time:
            earliest_index, earliest_time, earliest_state = (
                index,
                time,
                states[0],
            )
    if earliest_index < 0:
        raise RuntimeError("Event requested but none were detected.")
    return earliest_index, earliest_time, earliest_state


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
        immediate = [
            transition for transition in triggers if transition.delay == 0
        ]
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
            pending = None
            for transition in triggers:
                pending = _schedule(pending, transition, time)
            return _EntryResult(
                state, location, tuple(events), jumps, clock, pending
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
