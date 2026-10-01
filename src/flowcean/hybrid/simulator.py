"""SciPy event-driven hybrid simulation, independent of sampling grids."""

from collections.abc import Callable, Iterable, Sequence
from typing import NamedTuple, cast

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
    """Raised when multiple transitions request execution at the same time."""


class SimulationProgressError(HybridSimulationError):
    """Raised when a solver event does not advance physical time."""


class _EventFn:
    """Event wrapper retaining SciPy's crossing-direction semantics."""

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


class _EntryResult(NamedTuple):
    state: np.ndarray
    location: Location
    events: tuple[Event, ...]
    jumps: int
    clock: _ResidenceClock
    pending: dict[Transition, float]


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
    Positive-delay detections keep the source flow and residence clock active.
    Leaving the source visit cancels all its pending transitions. Deadlines at
    the final endpoint execute; later deadlines are discarded from the result.
    Execution times indistinguishable at root-finding precision are treated as
    simultaneous; multiple execution requests raise AmbiguousTransitionError.

    Args:
        system: Model to simulate.
        t_span: Finite start and end times, with end at or after start.
        x0: Initial continuous state, overriding the model's initial state.
        location0: Initial location, overriding the model's initial location.
        input_stream: Function returning a one-dimensional input vector at a
            requested physical time.
        initial_location_time: Finite, nonnegative age of the initial visit.
            Each subsequent transition begins a new visit at age zero. No
            pending transitions are inferred from the initial residence age.
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
    while current < end:
        transitions = [
            transition
            for transition in system.transitions_from(location)
            if transition not in pending
        ]
        deadline = min(
            (
                detected + transition.delay
                for transition, detected in pending.items()
            ),
            default=float("inf"),
        )
        segment, detected = _integrate(
            location,
            state,
            current,
            min(end, deadline),
            transitions,
            bindings,
            clock,
            rtol=rtol,
            atol=atol,
            max_step=max_step,
        )
        execution.append(segment)
        current = segment.t_span[1]
        state = segment.evaluate(current).state.copy()
        for transition in detected:
            _schedule(pending, transition, current)
        due = [
            transition
            for transition, detection_time in pending.items()
            if detection_time + transition.delay <= current
        ]
        if not due:
            continue
        # Check every scheduled occurrence when execution is due. Tolerance
        # groups competing deadlines, but never advances a lone deadline.
        simultaneous = [
            transition
            for transition, detection_time in pending.items()
            if detection_time + transition.delay <= end
            and _same_time(detection_time + transition.delay, current)
        ]
        _check_ambiguity(simultaneous, current)
        transition = due[0]
        detection_time = pending[transition]
        event_time = current
        event_state = state
        jumps = _increment_jumps(jumps, max_jumps)
        state, event = _apply_transition(
            transition,
            event_time,
            event_state,
            bindings,
            microstep=0,
            location_time=clock.age(event_time),
            detection_time=detection_time,
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


# Match the root solver's absolute/relative time resolution, not ODE state
# tolerances. Indistinguishable execution times must not acquire list priority.
_ROOT_TOLERANCE = 4 * np.finfo(float).eps


def _same_time(first: float, second: float) -> bool:
    return abs(first - second) <= _ROOT_TOLERANCE * (
        1 + max(abs(first), abs(second))
    )


def _schedule(
    pending: dict[Transition, float], transition: Transition, time: float
) -> None:
    deadline = time + transition.delay
    if not np.isfinite(deadline) or (
        transition.delay > 0 and deadline <= time
    ):
        raise SimulationProgressError(
            "Transition delay cannot advance to a finite representable deadline."
        )
    pending[transition] = time


def _check_ambiguity(transitions: Sequence[Transition], time: float) -> None:
    if len(transitions) > 1:
        raise AmbiguousTransitionError(
            f"Multiple transitions request execution at t={time!r}: {_transition_descriptions(transitions)}."
        )


def _step_crossings(
    functions: Sequence[_EventFn],
    before: Sequence[float],
    after: Sequence[float],
    start: float,
    end: float,
    solution: DenseOutput,
) -> list[tuple[float, Transition]]:
    roots: list[tuple[float, Transition]] = []
    for function, left, right in zip(functions, before, after, strict=True):
        rising = left <= 0 <= right and function.direction >= 0
        falling = left >= 0 >= right and function.direction <= 0
        if rising or falling:
            root = brentq(
                lambda time, fn=function: fn(time, solution(time)),
                start,
                end,
                xtol=_ROOT_TOLERANCE,
                rtol=_ROOT_TOLERANCE,
            )
            roots.append((cast(float, root), function.transition))
    return roots


def _integrate(
    location: Location,
    state: np.ndarray,
    start: float,
    end: float,
    transitions: Sequence[Transition],
    bindings: _RunBindings,
    clock: _ResidenceClock,
    *,
    rtol: float,
    atol: float,
    max_step: float | None,
) -> tuple[ContinuousSegment, list[Transition]]:
    """Integrate until the first crossing or deadline, retaining tied roots.

    Use SciPy's public RK45 stepping and dense output APIs so every surface
    in the terminating step is examined. solve_ivp's terminal events truncate
    the root list and can hide simultaneous execution requests.
    """
    solver = RK45(
        _wrap_flow(bindings.flows[location], clock),
        start,
        state,
        end,
        rtol=rtol,
        atol=atol,
        max_step=np.inf if max_step is None else max_step,
    )
    functions = [
        _EventFn(transition, bindings.surfaces[transition], clock)
        for transition in transitions
    ]
    values = [function(start, state) for function in functions]
    knots = [start]
    interpolants: list[DenseOutput] = []
    detected: list[Transition] = []
    while solver.status == "running":
        previous = float(solver.t)
        message = solver.step()
        if solver.status == "failed":
            raise HybridSimulationError(f"ODE integration failed: {message}")
        solution = solver.dense_output()
        time = float(solver.t)
        next_values = [function(time, solver.y) for function in functions]
        roots = _step_crossings(
            functions, values, next_values, previous, time, solution
        )
        if roots:
            time = min(root for root, _ in roots)
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
            if _same_time(time, end):
                time = end
            detected = [
                transition
                for root, transition in roots
                if _same_time(root, time)
            ]
        # A root at the previous step boundary needs no additional interpolant.
        if time > knots[-1]:
            knots.append(time)
            interpolants.append(solution)
        if roots:
            break
        values = next_values
    return ContinuousSegment(
        location,
        (start, knots[-1]),
        OdeSolution(knots, interpolants),
        clock,
        np.array(knots),
    ), detected


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
            pending: dict[Transition, float] = {}
            for transition in triggers:
                _schedule(pending, transition, time)
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
