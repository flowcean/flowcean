"""Hybrid system simulation utilities."""

from collections.abc import Callable, Iterable, Sequence
from functools import cache
from inspect import Parameter, signature
from typing import Any, NamedTuple

import numpy as np
from scipy.integrate import solve_ivp

from .hybrid_system import (
    Event,
    HybridSystem,
    Input,
    InputStream,
    Location,
    Parameters,
    State,
    SurfaceEntryPolicy,
    Trace,
    Transition,
    display_label,
)


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


def ensure_state(state: Iterable[float]) -> State:
    """Validate and coerce a state vector into a 1D array."""
    array = np.asarray(state, dtype=float)
    if array.ndim != 1:
        message = "State must be a 1D array."
        raise ValueError(message)
    return array


class _EventFn:
    """Wrap an event-surface function with metadata for SciPy."""

    def __init__(
        self,
        transition: Transition,
        parameters: Parameters,
        input_stream: InputStream,
    ) -> None:
        self._transition = transition
        self._parameters = parameters
        self._input_stream = input_stream
        self.direction = int(transition.event.direction)
        self.terminal = True

    def __call__(self, t: float, y: np.ndarray) -> float:
        value = float(
            _call_hybrid_callback(
                self._transition.event.fn,
                t,
                y,
                self._parameters,
                self._input_stream,
            ),
        )
        if np.isnan(value):
            raise _invalid_surface_value_error(
                [self._transition],
                t,
                context="during continuous integration",
            )
        return value


class _RolloutResult(NamedTuple):
    """Sampled result for dense rollout."""

    t: np.ndarray
    eval_t: np.ndarray
    x: np.ndarray
    location: np.ndarray


class _Boundary(NamedTuple):
    """Final quiescent state and location at a physical event time."""

    time: float
    state: np.ndarray
    location: Location


class _EntryResult(NamedTuple):
    """State after resolving a chain of location-entry transitions."""

    state: np.ndarray
    location: Location
    events: tuple[Event, ...]
    jumps: int


def simulate(  # noqa: C901, PLR0912, PLR0915
    system: HybridSystem,
    t_span: tuple[float, float],
    x0: Iterable[float] | None = None,
    location0: Location | None = None,
    *,
    input_stream: InputStream | None = None,
    capture_inputs: bool | None = None,
    capture_derivatives: bool = False,
    max_jumps: int = 256,
    rtol: float = 1e-7,
    atol: float = 1e-9,
    max_step: float | None = None,
    dense_output: bool = False,
    sample_times: Iterable[float] | None = None,
    sample_dt: float | None = None,
) -> Trace:
    """Simulate a hybrid system and return a trace.

    Args:
        system: Hybrid system to simulate.
        t_span: Start and end time for integration.
        x0: Optional initial state override.
        location0: Optional initial location override.
        input_stream: Optional input stream accessor for callbacks.
        capture_inputs: Input capture mode. If ``None``, capture iff an
            input stream is provided.
        capture_derivatives: Whether to re-evaluate ``Location.dynamics.flow``
            on the returned trace grid and store the sampled derivatives in
            ``Trace.dx``. This assumes pure flow callbacks under repeated
            evaluation. Scalar derivative returns are accepted only for
            single-state systems.
        max_jumps: Maximum number of transitions allowed.
        rtol: Relative tolerance for the solver.
        atol: Absolute tolerance for the solver.
        max_step: Optional maximum step size.
        dense_output: Whether to build a continuous solution per segment.
        sample_times: Monotone time grid to sample from the dense solution.
        sample_dt: Fixed sampling interval to generate a time grid.

    Returns:
        Trace: The simulation trace with location labels and events.
    """
    start_location = (
        system.initial_location if location0 is None else location0
    )
    if not isinstance(start_location, Location):
        message = "location0 must be a Location."
        raise TypeError(message)
    location_ids = {id(location) for location in system.locations}
    if id(start_location) not in location_ids:
        message = "location0 must be included in system.locations."
        raise ValueError(message)

    state = ensure_state(x0 if x0 is not None else system.initial_state)
    location = start_location
    effective_input_stream = input_stream or _missing_input_stream
    should_capture = _resolve_capture_inputs(
        capture_inputs=capture_inputs,
        input_stream=input_stream,
    )

    t_segments: list[np.ndarray] = []
    x_segments: list[np.ndarray] = []
    location_segments: list[np.ndarray] = []
    sol_segments: list[Callable[[np.ndarray], np.ndarray] | None] = []
    events: list[Event] = []
    boundaries: list[_Boundary] = []

    t_current = float(t_span[0])
    t_final = float(t_span[1])
    jumps = 0

    sample_grid = _prepare_sample_times(t_span, sample_times, sample_dt)
    needs_dense = dense_output or sample_grid is not None

    initial_entry = _settle_location_entries(
        system,
        location,
        state,
        t_current,
        effective_input_stream,
        first_microstep=0,
        jumps=jumps,
        max_jumps=max_jumps,
    )
    state = initial_entry.state
    location = initial_entry.location
    jumps = initial_entry.jumps
    events.extend(initial_entry.events)
    if initial_entry.events:
        boundaries.append(_Boundary(t_current, state.copy(), location))

    while t_current < t_final:
        transitions = system.transitions_from(location)
        event_fns = _build_event_functions(
            transitions,
            system.parameters,
            location.parameters,
            effective_input_stream,
        )
        segment_start = t_current

        solve_kwargs = {
            "fun": _wrap_flow(
                location,
                system.parameters,
                effective_input_stream,
            ),
            "t_span": (segment_start, t_final),
            "y0": state,
            "events": event_fns or None,
            "rtol": rtol,
            "atol": atol,
            "dense_output": needs_dense,
        }
        if max_step is not None:
            solve_kwargs["max_step"] = max_step

        result = solve_ivp(**solve_kwargs)
        if not result.success:
            message = f"ODE integration failed: {result.message}"
            raise HybridSimulationError(message)

        t_segments.append(result.t)
        x_segments.append(result.y.T)
        location_segments.append(
            np.full(result.t.shape, location, dtype=object),
        )
        sol_segments.append(result.sol)

        if not result.t_events or all(
            len(event_list) == 0 for event_list in result.t_events
        ):
            break

        triggered_index, event_time, event_state = _first_event(
            result.t_events,
            result.y_events,
        )
        if event_time <= segment_start:
            progress_context = (
                f"segment start={segment_start!r}, event time={event_time!r}"
            )
            message = (
                f"An event did not advance physical time ({progress_context})."
            )
            error = SimulationProgressError(message)
            error.add_note(
                "This can result from stateful callbacks, discontinuous event "
                "surfaces, or insufficient floating-point time resolution. "
                "Use deterministic callbacks and continuous event surfaces.",
            )
            raise error

        transition = transitions[triggered_index]
        jumps = _increment_jumps(jumps, max_jumps)
        state, event = _apply_transition(
            transition,
            event_time,
            event_state,
            system.parameters,
            effective_input_stream,
            microstep=0,
        )
        events.append(event)
        location = transition.target

        target_entry = _settle_location_entries(
            system,
            location,
            state,
            event_time,
            effective_input_stream,
            first_microstep=1,
            jumps=jumps,
            max_jumps=max_jumps,
        )
        state = target_entry.state
        location = target_entry.location
        jumps = target_entry.jumps
        events.extend(target_entry.events)
        boundaries.append(_Boundary(event_time, state.copy(), location))
        t_current = event_time

    if sample_grid is None:
        t_all = _concat_segments(t_segments)
        x_all = _concat_segments(x_segments)
        location_objects = _concat_segments(location_segments)
        _apply_boundaries(
            t_all,
            x_all,
            location_objects,
            boundaries,
        )
        unique_times = _unique_time_mask(t_all)
        t_all = t_all[unique_times]
        x_all = x_all[unique_times]
        location_objects = location_objects[unique_times]
        location_all = _location_labels(location_objects)
        u_all = None
        dx_all = None
        if should_capture:
            if input_stream is None:
                message = "Internal error: expected input_stream for capture."
                raise RuntimeError(message)
            u_all = _capture_inputs(t_all, input_stream)
        if capture_derivatives:
            dx_all = _capture_derivatives(
                system=system,
                times=t_all,
                states=x_all,
                locations=location_objects,
                input_stream=effective_input_stream,
            )
        return Trace(
            t=t_all,
            x=x_all,
            location=location_all,
            events=tuple(events),
            u=u_all,
            dx=dx_all,
        )

    rolled = _rollout_segments(
        sample_grid,
        t_segments,
        x_segments,
        location_segments,
        sol_segments,
        boundaries,
    )
    u_all = None
    dx_all = None
    if should_capture:
        if input_stream is None:
            message = "Internal error: expected input_stream for capture."
            raise RuntimeError(message)
        u_all = _capture_inputs(rolled.t, input_stream)
    if capture_derivatives:
        dx_all = _capture_derivatives(
            system=system,
            times=rolled.eval_t,
            states=rolled.x,
            locations=rolled.location,
            input_stream=effective_input_stream,
        )
    location_all = _location_labels(rolled.location)
    return Trace(
        t=rolled.t,
        x=rolled.x,
        location=location_all,
        events=tuple(events),
        u=u_all,
        dx=dx_all,
    )


def generate_traces(
    system: HybridSystem,
    t_span: tuple[float, float],
    initial_states: Iterable[Iterable[float]],
    *,
    input_stream: InputStream | None = None,
    capture_inputs: bool | None = None,
    capture_derivatives: bool = False,
    max_jumps: int = 256,
    rtol: float = 1e-7,
    atol: float = 1e-9,
    max_step: float | None = None,
    dense_output: bool = False,
    sample_times: Iterable[float] | None = None,
    sample_dt: float | None = None,
) -> list[Trace]:
    """Simulate a batch of traces for a set of initial states.

    The input stream and capture semantics match :func:`simulate`, including
    the requirement that ``capture_derivatives=True`` assumes pure flow
    callbacks under repeated evaluation on the returned trace grid. Scalar
    derivative returns are accepted only for single-state systems.
    """
    return [
        simulate(
            system,
            t_span,
            x0=state,
            input_stream=input_stream,
            capture_inputs=capture_inputs,
            capture_derivatives=capture_derivatives,
            max_jumps=max_jumps,
            rtol=rtol,
            atol=atol,
            max_step=max_step,
            dense_output=dense_output,
            sample_times=sample_times,
            sample_dt=sample_dt,
        )
        for state in initial_states
    ]


def _wrap_flow(
    location: Location,
    system_parameters: Parameters,
    input_stream: InputStream,
) -> Callable[[float, np.ndarray], np.ndarray]:
    """Bind location dynamics and system parameters for SciPy."""

    def flow(t: float, y: np.ndarray) -> np.ndarray:
        dynamics = location.dynamics
        parameters = {**system_parameters, **location.parameters}
        return _coerce_derivative(
            _call_hybrid_callback(
                dynamics.flow,
                t,
                y,
                parameters,
                input_stream,
            ),
            state_dim=y.shape[0],
        )

    return flow


def _build_event_functions(
    transitions: Sequence[Transition],
    system_parameters: Parameters,
    location_parameters: Parameters,
    input_stream: InputStream,
) -> list[_EventFn]:
    """Create SciPy-compatible event functions for transitions."""
    event_functions: list[_EventFn] = []
    for transition in transitions:
        parameters = {**system_parameters, **location_parameters}
        event_functions.append(
            _EventFn(
                transition,
                parameters,
                input_stream,
            ),
        )
    return event_functions


def _call_hybrid_callback(
    callback: Callable[..., Any],
    t: float,
    state: np.ndarray,
    parameters: Parameters,
    input_stream: InputStream,
) -> Any:
    canonical_arguments = {
        "t": t,
        "state": state,
        "parameters": parameters,
        "input_stream": input_stream,
    }
    callback_parameters = _callback_parameters(callback)
    if callback_parameters is None:
        return callback(t, state, parameters, input_stream)

    selected_arguments: dict[str, object] = {}
    has_var_keyword = False
    for parameter in callback_parameters:
        if parameter.kind in {
            Parameter.POSITIONAL_ONLY,
            Parameter.VAR_POSITIONAL,
        }:
            return callback(t, state, parameters, input_stream)
        if parameter.kind == Parameter.VAR_KEYWORD:
            has_var_keyword = True
            continue
        if parameter.name in canonical_arguments:
            selected_arguments[parameter.name] = canonical_arguments[
                parameter.name
            ]
        elif parameter.default is Parameter.empty:
            return callback(t, state, parameters, input_stream)

    if has_var_keyword:
        return callback(**canonical_arguments)
    return callback(**selected_arguments)


def _callback_parameters(
    callback: Callable[..., Any],
) -> tuple[Parameter, ...] | None:
    try:
        return _cached_callback_parameters(callback)
    except TypeError:
        return _inspect_callback_parameters(callback)


@cache
def _cached_callback_parameters(
    callback: Callable[..., Any],
) -> tuple[Parameter, ...] | None:
    return _inspect_callback_parameters(callback)


def _inspect_callback_parameters(
    callback: Callable[..., Any],
) -> tuple[Parameter, ...] | None:
    try:
        return tuple(signature(callback).parameters.values())
    except (TypeError, ValueError):
        return None


def _first_event(
    t_events: Sequence[np.ndarray],
    y_events: Sequence[np.ndarray],
) -> tuple[int, float, np.ndarray]:
    """Select the earliest triggered event across all event surfaces."""
    earliest_time = float("inf")
    earliest_index = -1
    earliest_state = np.zeros(0, dtype=float)
    for index, (times, states) in enumerate(
        zip(t_events, y_events, strict=False),
    ):
        if len(times) == 0:
            continue
        time = float(times[0])
        if time < earliest_time:
            earliest_time = time
            earliest_index = index
            earliest_state = states[0]

    if earliest_index < 0:
        message = "Event requested but none were detected."
        raise RuntimeError(message)

    return earliest_index, earliest_time, earliest_state


def _apply_transition(
    transition: Transition,
    event_time: float,
    event_state: np.ndarray,
    system_parameters: Parameters,
    input_stream: InputStream,
    *,
    microstep: int,
) -> tuple[np.ndarray, Event]:
    state_before = ensure_state(event_state).copy()
    parameters = {**system_parameters, **transition.source.parameters}
    if transition.reset is None:
        new_state = state_before.copy()
        reset_label = None
    else:
        new_state = _coerce_reset(
            _call_hybrid_callback(
                transition.reset.fn,
                event_time,
                state_before.copy(),
                parameters,
                input_stream,
            ),
            state_dim=state_before.shape[0],
        )
        reset_label = display_label(transition.reset)

    return new_state, Event(
        time=event_time,
        source_location=display_label(transition.source),
        target_location=display_label(transition.target),
        event_surface=display_label(transition.event),
        reset=reset_label,
        state_before=state_before.copy(),
        state_after=new_state.copy(),
        microstep=microstep,
    )


def _settle_location_entries(
    system: HybridSystem,
    location: Location,
    state: np.ndarray,
    time: float,
    input_stream: InputStream,
    *,
    first_microstep: int,
    jumps: int,
    max_jumps: int,
) -> _EntryResult:
    """Resolve explicit entry-trigger transitions until a location settles."""
    entry_events: list[Event] = []
    microstep = first_microstep

    while True:
        transitions = system.transitions_from(location)
        values = _evaluate_entry_surfaces(
            transitions,
            location,
            time,
            state,
            system.parameters,
            input_stream,
        )
        zero_error = [
            transition
            for transition, value in zip(transitions, values, strict=True)
            if value == 0.0
            and transition.entry_policy is SurfaceEntryPolicy.ERROR
        ]
        if zero_error:
            descriptions = _transition_descriptions(zero_error)
            error = SurfaceEntryError(
                "Event surfaces are zero while entering "
                f"{display_label(location)!r} at t={time!r}: {descriptions}.",
            )
            error.add_note(
                "Choose an explicit entry_policy for each implicated "
                "transition: TRIGGER for an immediate jump or CONTINUE to "
                "begin continuous integration from the surface.",
            )
            raise error

        zero_trigger = [
            transition
            for transition, value in zip(transitions, values, strict=True)
            if value == 0.0
            and transition.entry_policy is SurfaceEntryPolicy.TRIGGER
        ]
        if len(zero_trigger) > 1:
            descriptions = _transition_descriptions(zero_trigger)
            error = AmbiguousTransitionError(
                "Multiple transitions request an entry-time jump from "
                f"{display_label(location)!r} at t={time!r}: {descriptions}.",
            )
            error.add_note(
                "Make at most one outgoing TRIGGER surface zero on entry, "
                "or change the model so the entry state selects one target.",
            )
            raise error
        if not zero_trigger:
            return _EntryResult(
                state=state,
                location=location,
                events=tuple(entry_events),
                jumps=jumps,
            )

        transition = zero_trigger[0]
        jumps = _increment_jumps(jumps, max_jumps)
        state, event = _apply_transition(
            transition,
            time,
            state,
            system.parameters,
            input_stream,
            microstep=microstep,
        )
        entry_events.append(event)
        location = transition.target
        microstep += 1


def _evaluate_entry_surfaces(
    transitions: Sequence[Transition],
    location: Location,
    time: float,
    state: np.ndarray,
    system_parameters: Parameters,
    input_stream: InputStream,
) -> list[float]:
    """Evaluate every outgoing event surface once before making a decision."""
    parameters = {**system_parameters, **location.parameters}
    values = [
        float(
            _call_hybrid_callback(
                transition.event.fn,
                time,
                state,
                parameters,
                input_stream,
            ),
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
    return values


def _invalid_surface_value_error(
    transitions: Sequence[Transition],
    time: float,
    *,
    context: str,
) -> InvalidEventSurfaceValueError:
    descriptions = _transition_descriptions(transitions)
    error = InvalidEventSurfaceValueError(
        f"Event surfaces returned NaN {context} at t={time!r}: "
        f"{descriptions}.",
    )
    error.add_note(
        "An event surface must return a scalar value other than NaN; exact "
        "zero denotes the surface and signed nonzero values denote its sides.",
    )
    return error


def _transition_descriptions(transitions: Sequence[Transition]) -> str:
    return ", ".join(
        f"{display_label(transition.source)} -> "
        f"{display_label(transition.target)} "
        f"[{display_label(transition.event)}]"
        for transition in transitions
    )


def _increment_jumps(jumps: int, max_jumps: int) -> int:
    jumps += 1
    if jumps > max_jumps:
        message = "Maximum number of transitions exceeded."
        raise HybridSimulationError(message)
    return jumps


def _concat_segments(segments: Sequence[np.ndarray]) -> np.ndarray:
    """Concatenate solver segments while avoiding duplicate boundary points."""
    if not segments:
        return np.array([], dtype=float)
    if len(segments) == 1:
        return segments[0]
    return np.concatenate(
        [segments[0], *[segment[1:] for segment in segments[1:]]],
    )


def _unique_time_mask(times: np.ndarray) -> np.ndarray:
    """Select one row for each time from a monotone adaptive trace."""
    mask = np.ones(times.shape, dtype=bool)
    mask[1:] = np.diff(times) != 0.0
    return mask


def _apply_boundaries(
    times: np.ndarray,
    states: np.ndarray,
    locations: np.ndarray,
    boundaries: Sequence[_Boundary],
) -> None:
    """Replace event-time rows with their final quiescent values."""
    for boundary in boundaries:
        matches = times == boundary.time
        states[matches] = boundary.state
        locations[matches] = boundary.location


def _location_labels(locations: np.ndarray) -> np.ndarray:
    return np.array(
        [display_label(location) for location in locations],
        dtype=object,
    )


def _prepare_sample_times(
    t_span: tuple[float, float],
    sample_times: Iterable[float] | None,
    sample_dt: float | None,
) -> np.ndarray | None:
    if sample_times is None and sample_dt is None:
        return None
    if sample_times is not None and sample_dt is not None:
        message = "Provide either sample_times or sample_dt, not both."
        raise ValueError(message)
    if sample_times is not None:
        times = np.asarray(list(sample_times), dtype=float)
        if times.size and not np.all(np.isfinite(times)):
            message = "sample_times must be finite."
            raise ValueError(message)
    else:
        if sample_dt is None or not np.isfinite(sample_dt):
            message = "sample_dt must be finite."
            raise ValueError(message)
        if sample_dt <= 0:
            message = "sample_dt must be positive."
            raise ValueError(message)
        times = np.arange(t_span[0], t_span[1], sample_dt, dtype=float)
        endpoint_atol = float(
            np.finfo(float).eps * max(1.0, abs(float(t_span[1]))),
        )
        if times.size and np.isclose(
            times[-1],
            t_span[1],
            rtol=0.0,
            atol=endpoint_atol,
        ):
            times[-1] = float(t_span[1])
        else:
            times = np.append(times, float(t_span[1]))
    if times.size and np.any(np.diff(times) < 0):
        message = "sample_times must be sorted in ascending order."
        raise ValueError(message)
    if times.size and (
        float(times[0]) < t_span[0] or float(times[-1]) > t_span[1]
    ):
        message = "sample_times must lie within t_span."
        raise ValueError(message)
    return times


def _capture_inputs(
    times: np.ndarray,
    input_stream: InputStream,
) -> np.ndarray:
    values: list[np.ndarray] = []
    input_dim: int | None = None
    for time in times:
        value = _coerce_input(input_stream(float(time)))
        if input_dim is None:
            input_dim = value.shape[0]
        elif value.shape[0] != input_dim:
            message = "Input stream dimension changed during simulation."
            raise ValueError(message)
        values.append(value)

    if not values:
        return np.empty((0, 0), dtype=float)

    return np.vstack(values)


def _capture_derivatives(
    *,
    system: HybridSystem,
    times: np.ndarray,
    states: np.ndarray,
    locations: np.ndarray,
    input_stream: InputStream,
) -> np.ndarray:
    derivatives: list[np.ndarray] = []
    matrix_ndim = 2
    state_dim = states.shape[1] if states.ndim == matrix_ndim else 0
    for time, state, location in zip(
        times,
        states,
        locations,
        strict=True,
    ):
        if not isinstance(location, Location):
            message = "Derivative capture requires Location objects."
            raise TypeError(message)
        dynamics = location.dynamics
        derivative = _coerce_derivative(
            _call_hybrid_callback(
                dynamics.flow,
                float(time),
                state,
                {**system.parameters, **location.parameters},
                input_stream,
            ),
            state_dim=state_dim,
        )
        derivatives.append(derivative)

    if not derivatives:
        return np.empty((0, state_dim), dtype=float)

    return np.vstack(derivatives)


def _coerce_reset(candidate: object, *, state_dim: int) -> np.ndarray:
    reset_state = np.asarray(candidate, dtype=float)
    if reset_state.ndim == 0 and state_dim == 1:
        return reset_state.reshape(1)
    if reset_state.ndim != 1:
        message = "Reset must return a 1D state matching the state dimension."
        raise ValueError(message)
    if reset_state.shape[0] != state_dim:
        message = "Reset state must match the state dimension."
        raise ValueError(message)
    return reset_state.copy()


def _coerce_derivative(candidate: object, *, state_dim: int) -> np.ndarray:
    derivative = np.asarray(candidate, dtype=float)
    if derivative.ndim == 0 and state_dim == 1:
        return derivative.reshape(1)
    if derivative.ndim != 1:
        message = (
            "Flow must return a 1D derivative matching the state dimension."
        )
        raise ValueError(message)
    if derivative.shape[0] != state_dim:
        message = "Flow derivative must match the state dimension."
        raise ValueError(message)
    return derivative


def _coerce_input(candidate: object) -> np.ndarray:
    try:
        values = np.asarray(candidate, dtype=float)
    except (TypeError, ValueError) as error:
        message = "Input stream must return numeric values."
        raise ValueError(message) from error

    if values.ndim != 1:
        message = "Input stream must return a 1D array."
        raise ValueError(message)

    return values


def _missing_input_stream(time: float) -> Input:
    message = (
        "input_stream is required for this system; callback accessed input "
        f"at t={time}."
    )
    raise ValueError(message)


def _resolve_capture_inputs(
    *,
    capture_inputs: bool | None,
    input_stream: InputStream | None,
) -> bool:
    if capture_inputs is None:
        return input_stream is not None
    if capture_inputs:
        if input_stream is None:
            message = "capture_inputs=True requires an input_stream."
            raise ValueError(message)
        return True
    return False


def _rollout_segments(
    sample_times: np.ndarray,
    t_segments: Sequence[np.ndarray],
    x_segments: Sequence[np.ndarray],
    location_segments: Sequence[np.ndarray],
    sol_segments: Sequence[Callable[[np.ndarray], np.ndarray] | None],
    boundaries: Sequence[_Boundary],
) -> _RolloutResult:
    if sample_times.size == 0:
        return _RolloutResult(
            t=sample_times,
            eval_t=sample_times,
            x=np.empty((0, 0), dtype=float),
            location=np.empty((0,), dtype=object),
        )

    sampled_t: list[np.ndarray] = []
    sampled_eval_t: list[np.ndarray] = []
    sampled_x: list[np.ndarray] = []
    sampled_location: list[np.ndarray] = []

    last_segment_index = len(t_segments) - 1
    for index, (t_seg, x_seg, location_seg, sol) in enumerate(
        zip(
            t_segments,
            x_segments,
            location_segments,
            sol_segments,
            strict=False,
        ),
    ):
        t_start = float(t_seg[0])
        t_end = float(t_seg[-1])
        if index < last_segment_index:
            mask = (sample_times >= t_start) & (sample_times < t_end)
        else:
            mask = (sample_times >= t_start) & (sample_times <= t_end)
        if not np.any(mask):
            continue
        times = sample_times[mask]
        eval_times = times.copy()
        exact_boundary_mask = np.zeros(times.shape, dtype=bool)
        if index > 0:
            exact_boundary_mask = times == t_start
        sampled_t.append(times)
        sampled_eval_t.append(eval_times)
        if sol is not None:
            values = sol(eval_times).T
        else:
            values = _interpolate_segment(t_seg, x_seg, eval_times)
        if np.any(exact_boundary_mask):
            values[exact_boundary_mask] = x_seg[0]
        sampled_x.append(values)
        sampled_location.append(
            np.full(times.shape, location_seg[0], dtype=object),
        )

    if not sampled_t:
        return _RolloutResult(
            t=np.array([], dtype=float),
            eval_t=np.array([], dtype=float),
            x=np.empty((0, 0), dtype=float),
            location=np.empty((0,), dtype=object),
        )

    result = _RolloutResult(
        t=np.concatenate(sampled_t),
        eval_t=np.concatenate(sampled_eval_t),
        x=np.concatenate(sampled_x),
        location=np.concatenate(sampled_location),
    )
    _apply_boundaries(
        result.t,
        result.x,
        result.location,
        boundaries,
    )
    return result


def _interpolate_segment(
    t_segment: np.ndarray,
    x_segment: np.ndarray,
    sample_times: np.ndarray,
) -> np.ndarray:
    if t_segment.size == 0:
        return np.empty((0, x_segment.shape[1]), dtype=float)
    return np.vstack(
        [
            np.interp(sample_times, t_segment, x_segment[:, dim])
            for dim in range(x_segment.shape[1])
        ],
    ).T
