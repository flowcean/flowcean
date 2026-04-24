"""Hybrid system simulation utilities."""

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import NamedTuple

import numpy as np
from scipy.integrate import solve_ivp

from .hybrid_system import (
    Event,
    GuardFn,
    HybridSystem,
    Input,
    InputStream,
    Location,
    State,
    Trace,
    Transition,
)


def ensure_state(state: Iterable[float]) -> State:
    """Validate and coerce a state vector into a 1D array."""
    array = np.asarray(state, dtype=float)
    if array.ndim != 1:
        message = "State must be a 1D array."
        raise ValueError(message)
    return array


class _EventFn:
    """Wrap a guard function with event metadata for SciPy."""

    def __init__(
        self,
        guard: GuardFn,
        params: Mapping[str, float],
        direction: int,
        input_stream: InputStream,
    ) -> None:
        self._guard = guard
        self._params = params
        self._input_stream = input_stream
        self.direction = direction
        self.terminal = True

    def __call__(self, t: float, y: np.ndarray) -> float:
        return self._guard(t, y, self._params, self._input_stream)


class _RolloutResult(NamedTuple):
    """Sampled result for dense rollout."""

    t: np.ndarray
    eval_t: np.ndarray
    x: np.ndarray
    location: np.ndarray


def simulate(  # noqa: C901, PLR0912, PLR0915
    system: HybridSystem,
    t_span: tuple[float, float],
    x0: Iterable[float] | None = None,
    location0: str | None = None,
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
    start_location = location0 or system.initial_location
    if start_location not in system.locations:
        message = f"Unknown initial location: {start_location}"
        raise ValueError(message)

    state = ensure_state(x0 if x0 is not None else system.initial_state)
    location = system.locations[start_location]
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

    t_current = float(t_span[0])
    t_final = float(t_span[1])
    jumps = 0

    time_epsilon = 1e-12
    sample_grid = _prepare_sample_times(t_span, sample_times, sample_dt)
    needs_dense = dense_output or sample_grid is not None

    while t_current < t_final:
        transitions = system.transitions_from(location.name)
        event_fns = _build_event_functions(
            transitions,
            system.params,
            location.dynamics.params,
            effective_input_stream,
        )

        solve_kwargs = {
            "fun": _wrap_flow(location, system.params, effective_input_stream),
            "t_span": (t_current, t_final),
            "y0": state,
            "events": event_fns or None,
            "rtol": rtol,
            "atol": atol,
            "dense_output": needs_dense,
        }
        if max_step is not None:
            solve_kwargs["max_step"] = max_step

        result = solve_ivp(**solve_kwargs)

        t_segments.append(result.t)
        x_segments.append(result.y.T)
        location_segments.append(
            np.full(result.t.shape, location.name, dtype=object),
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
        if event_time - t_current <= time_epsilon:
            break
        transition = transitions[triggered_index]
        jumps += 1
        if jumps > max_jumps:
            message = "Maximum number of transitions exceeded."
            raise RuntimeError(message)

        if transition.reset is None:
            new_state = event_state
            reset_name = None
        else:
            new_state = transition.reset.fn(
                event_time,
                event_state,
                transition.reset.params,
                effective_input_stream,
            )
            reset_name = transition.reset.name

        events.append(
            Event(
                time=event_time,
                source_location=transition.source_location,
                target_location=transition.target_location,
                guard=transition.guard.name,
                reset=reset_name,
                state=new_state,
            ),
        )

        if transition.target_location not in system.locations:
            message = f"Unknown target location: {transition.target_location}"
            raise ValueError(message)

        location = system.locations[transition.target_location]
        t_current = min(t_final, event_time + time_epsilon)
        state = new_state

    if sample_grid is None:
        t_all = _concat_segments(t_segments)
        x_all = _concat_segments(x_segments)
        location_all = _concat_segments(location_segments)
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
                locations=location_all,
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
    return Trace(
        t=rolled.t,
        x=rolled.x,
        location=rolled.location,
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
    system_params: Mapping[str, float],
    input_stream: InputStream,
) -> Callable[[float, np.ndarray], np.ndarray]:
    """Bind location dynamics and system parameters for SciPy."""

    def flow(t: float, y: np.ndarray) -> np.ndarray:
        dynamics = location.dynamics
        return _coerce_derivative(
            dynamics.flow(
                t,
                y,
                {**system_params, **dynamics.params},
                input_stream,
            ),
            state_dim=y.shape[0],
        )

    return flow


def _build_event_functions(
    transitions: Sequence[Transition],
    system_params: Mapping[str, float],
    location_params: Mapping[str, float],
    input_stream: InputStream,
) -> list[_EventFn]:
    """Create SciPy-compatible event functions for transitions."""
    event_functions: list[_EventFn] = []
    for transition in transitions:
        guard = transition.guard

        params = {**system_params, **location_params}
        event_functions.append(
            _EventFn(
                guard.fn,
                params,
                guard.direction,
                input_stream,
            ),
        )
    return event_functions


def _first_event(
    t_events: Sequence[np.ndarray],
    y_events: Sequence[np.ndarray],
) -> tuple[int, float, np.ndarray]:
    """Select the earliest triggered event across all guards."""
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


def _concat_segments(segments: Sequence[np.ndarray]) -> np.ndarray:
    """Concatenate solver segments while avoiding duplicate boundary points."""
    if not segments:
        return np.array([], dtype=float)
    if len(segments) == 1:
        return segments[0]
    return np.concatenate(
        [segments[0], *[segment[1:] for segment in segments[1:]]],
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
    else:
        if sample_dt is None or sample_dt <= 0:
            message = "sample_dt must be positive."
            raise ValueError(message)
        times = np.arange(
            t_span[0],
            t_span[1] + 0.5 * sample_dt,
            sample_dt,
            dtype=float,
        )
    if times.size and np.any(np.diff(times) < 0):
        message = "sample_times must be sorted in ascending order."
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
    for time, state, location_name in zip(
        times,
        states,
        locations,
        strict=True,
    ):
        location = system.locations[str(location_name)]
        dynamics = location.dynamics
        derivative = _coerce_derivative(
            dynamics.flow(
                float(time),
                state,
                {**system.params, **dynamics.params},
                input_stream,
            ),
            state_dim=state_dim,
        )
        derivatives.append(derivative)

    if not derivatives:
        return np.empty((0, state_dim), dtype=float)

    return np.vstack(derivatives)


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
        t_start = (
            float(t_segments[index - 1][-1]) if index > 0 else float(t_seg[0])
        )
        t_end = float(t_seg[-1])
        if index < last_segment_index:
            mask = (sample_times >= t_start) & (sample_times < t_end)
        else:
            mask = (sample_times >= t_start) & (sample_times <= t_end)
        if not np.any(mask):
            continue
        times = sample_times[mask]
        segment_start = float(t_seg[0])
        eval_times = times.copy()
        reported_times = times.copy()
        exact_boundary_mask = np.zeros(times.shape, dtype=bool)
        if index > 0:
            exact_boundary_mask = times == t_start
            epsilon_mask = (times > t_start) & (times < segment_start)
            eval_times[epsilon_mask] = segment_start
            reported_times[epsilon_mask] = segment_start
        sampled_t.append(reported_times)
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

    return _RolloutResult(
        t=np.concatenate(sampled_t),
        eval_t=np.concatenate(sampled_eval_t),
        x=np.concatenate(sampled_x),
        location=np.concatenate(sampled_location),
    )


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
