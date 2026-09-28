"""SciPy event-driven hybrid simulation, independent of sampling grids."""

from collections.abc import Callable, Iterable, Sequence
from typing import NamedTuple

import numpy as np
from scipy.integrate import solve_ivp

from ._runtime import (
    _coerce_vector,
    _ResidenceClock,
    _RunContext,
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


class SurfaceEntryError(HybridSimulationError):
    """Raised when an ERROR surface is zero on location entry."""


class AmbiguousTransitionError(HybridSimulationError):
    """Raised when multiple TRIGGER surfaces are zero on location entry."""


class SimulationProgressError(HybridSimulationError):
    """Raised when a solver event does not advance physical time."""


class _EventFn:
    """Terminal event wrapper retaining SciPy's crossing-direction semantics."""

    def __init__(
        self,
        transition: Transition,
        context: _RunContext,
        clock: _ResidenceClock,
    ) -> None:
        self.transition = transition
        self.context = context
        self.clock = clock
        self.direction = int(transition.event_surface.direction)
        self.terminal = True

    def __call__(self, t: float, y: np.ndarray) -> float:
        value = float(
            self.context.call(
                self.transition.event_surface.fn,
                self.transition.source,
                t,
                y,
                self.clock.age(t),
            )
        )
        if np.isnan(value):
            raise _invalid_surface_value_error(
                [self.transition], t, context="during continuous integration"
            )
        return value


class _EntryResult(NamedTuple):
    state: np.ndarray
    location: Location
    events: tuple[Event, ...]
    jumps: int
    clock: _ResidenceClock


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
    context = _RunContext(system, input_stream)
    execution: list[ContinuousSegment | Event] = []
    clock = _ResidenceClock(start, float(initial_location_time))
    entry = _settle_location_entries(
        system,
        initial_location,
        initial_state.copy(),
        start,
        context,
        first_microstep=0,
        clock=clock,
        jumps=0,
        max_jumps=max_jumps,
    )
    state, location, events, jumps, clock = entry
    execution.extend(events)
    current = start
    while current < end:
        transitions = system.transitions_from(location)
        event_fns = [
            _EventFn(transition, context, clock) for transition in transitions
        ]

        result = solve_ivp(
            _wrap_flow(context, location, clock),
            (current, end),
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
            if event_time <= current:
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
        if selected is None:
            break
        index, event_time, event_state = selected
        transition = transitions[index]
        jumps = _increment_jumps(jumps, max_jumps)
        state, event = _apply_transition(
            transition,
            event_time,
            event_state,
            context,
            microstep=0,
            location_time=clock.age(event_time),
        )
        execution.append(event)
        entry = _settle_location_entries(
            system,
            transition.target,
            state,
            event_time,
            context,
            first_microstep=1,
            clock=_ResidenceClock(event_time, 0.0),
            jumps=jumps,
            max_jumps=max_jumps,
        )
        state, location, events, jumps, clock = entry
        execution.extend(events)
        current = event_time
    return HybridTrajectory(
        system,
        (start, end),
        initial_state,
        initial_location,
        float(initial_location_time),
        tuple(execution),
        context,
    )


def _wrap_flow(
    context: _RunContext,
    location: Location,
    clock: _ResidenceClock,
) -> Callable[[float, np.ndarray], np.ndarray]:
    def flow(time: float, state: np.ndarray) -> np.ndarray:
        return context.derivative(location, time, state, clock.age(time))

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
    context: _RunContext,
    *,
    microstep: int,
    location_time: float,
) -> tuple[np.ndarray, Event]:
    before = ensure_state(state)
    if transition.reset is None:
        after = before.copy()
    else:
        after = _coerce_vector(
            context.call(
                transition.reset.fn,
                transition.source,
                time,
                before.copy(),
                location_time,
            ),
            state_dim=before.size,
            name="Reset",
        )
    return after, Event(
        time,
        transition,
        before,
        after,
        microstep,
        location_time,
    )


def _settle_location_entries(
    system: HybridSystem,
    location: Location,
    state: np.ndarray,
    time: float,
    context: _RunContext,
    *,
    first_microstep: int,
    clock: _ResidenceClock,
    jumps: int,
    max_jumps: int,
) -> _EntryResult:
    """Evaluate each entry atomically: NaN, ERROR, ambiguity, then reset."""
    events: list[Event] = []
    microstep = first_microstep
    while True:
        transitions = system.transitions_from(location)
        values = [
            float(
                context.call(
                    transition.event_surface.fn,
                    location,
                    time,
                    state,
                    clock.age(time),
                )
            )
            for transition in transitions
        ]
        invalid = [
            transition
            for transition, value in zip(transitions, values, strict=True)
            if np.isnan(value)
        ]
        if invalid:
            raise _invalid_surface_value_error(
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
                "Choose TRIGGER for an immediate jump or CONTINUE to begin continuous integration from the surface."
            )
            raise error
        triggers = [
            transition
            for transition, value in zip(transitions, values, strict=True)
            if value == 0.0
            and transition.entry_policy is SurfaceEntryPolicy.TRIGGER
        ]
        if len(triggers) > 1:
            error = AmbiguousTransitionError(
                f"Multiple transitions request an entry-time jump from {display_label(location)!r} "
                f"at t={time!r}: {_transition_descriptions(triggers)}.",
            )
            error.add_note(
                "Make at most one outgoing TRIGGER surface zero on entry."
            )
            raise error
        if not triggers:
            return _EntryResult(state, location, tuple(events), jumps, clock)
        transition = triggers[0]
        jumps = _increment_jumps(jumps, max_jumps)
        state, event = _apply_transition(
            transition,
            time,
            state,
            context,
            microstep=microstep,
            location_time=clock.age(time),
        )
        events.append(event)
        location = transition.target
        clock = _ResidenceClock(time, 0.0)
        microstep += 1


def _invalid_surface_value_error(
    transitions: Sequence[Transition],
    time: float,
    *,
    context: str,
) -> InvalidEventSurfaceValueError:
    error = InvalidEventSurfaceValueError(
        f"Event surfaces returned NaN {context} at t={time!r}: {_transition_descriptions(transitions)}.",
    )
    error.add_note(
        "An event surface must return a scalar value other than NaN; exact zero denotes the surface."
    )
    return error


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
