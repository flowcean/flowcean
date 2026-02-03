"""Hybrid system simulation utilities."""

from collections.abc import Callable, Iterable, Mapping, Sequence

import numpy as np
from scipy.integrate import solve_ivp

from .hybrid_system import (
    Event,
    GuardFn,
    HybridSystem,
    Mode,
    Trace,
    Transition,
    ensure_state,
)


class _EventFn:
    """Wrap a guard function with event metadata for SciPy."""

    def __init__(
        self,
        guard: GuardFn,
        params: Mapping[str, float],
        direction: int,
    ) -> None:
        self._guard = guard
        self._params = params
        self.direction = direction
        self.terminal = True

    def __call__(self, t: float, y: np.ndarray) -> float:
        return self._guard(t, y, self._params)


def simulate(  # noqa: C901, PLR0915
    system: HybridSystem,
    t_span: tuple[float, float],
    x0: Iterable[float] | None = None,
    mode0: str | None = None,
    *,
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
        mode0: Optional initial mode override.
        max_jumps: Maximum number of transitions allowed.
        rtol: Relative tolerance for the solver.
        atol: Absolute tolerance for the solver.
        max_step: Optional maximum step size.
        dense_output: Whether to build a continuous solution per segment.
        sample_times: Monotone time grid to sample from the dense solution.
        sample_dt: Fixed sampling interval to generate a time grid.

    Returns:
        Trace: The simulation trace with mode labels and events.
    """
    start_mode = mode0 or system.initial_mode
    if start_mode not in system.modes:
        message = f"Unknown initial mode: {start_mode}"
        raise ValueError(message)

    state = ensure_state(x0 if x0 is not None else system.initial_state)
    mode = system.modes[start_mode]

    t_segments: list[np.ndarray] = []
    x_segments: list[np.ndarray] = []
    mode_segments: list[np.ndarray] = []
    sol_segments: list[Callable[[np.ndarray], np.ndarray] | None] = []
    events: list[Event] = []

    t_current = float(t_span[0])
    t_final = float(t_span[1])
    jumps = 0

    time_epsilon = 1e-12
    sample_grid = _prepare_sample_times(t_span, sample_times, sample_dt)
    needs_dense = dense_output or sample_grid is not None

    while t_current < t_final:
        transitions = system.transitions_from(mode.name)
        event_fns = _build_event_functions(
            transitions,
            system.params,
            mode.params,
        )

        solve_kwargs = {
            "fun": _wrap_flow(mode, system.params),
            "t_span": (t_current, t_final),
            "y0": state,
            "events": event_fns if event_fns else None,
            "rtol": rtol,
            "atol": atol,
            "dense_output": needs_dense,
        }
        if max_step is not None:
            solve_kwargs["max_step"] = max_step

        result = solve_ivp(**solve_kwargs)

        t_segments.append(result.t)
        x_segments.append(result.y.T)
        mode_segments.append(np.full(result.t.shape, mode.name, dtype=object))
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
            )
            reset_name = transition.reset.name

        events.append(
            Event(
                time=event_time,
                source_mode=transition.source,
                target_mode=transition.target,
                guard=transition.guard.name,
                reset=reset_name,
                state=new_state,
            ),
        )

        if transition.target not in system.modes:
            message = f"Unknown target mode: {transition.target}"
            raise ValueError(message)

        mode = system.modes[transition.target]
        t_current = min(t_final, event_time + time_epsilon)
        state = new_state

        if transition.guard.terminal:
            continue

    if sample_grid is None:
        t_all = _concat_segments(t_segments)
        x_all = _concat_segments(x_segments)
        mode_all = _concat_segments(mode_segments)
        return Trace(t=t_all, x=x_all, mode=mode_all, events=tuple(events))

    rolled = _rollout_segments(
        sample_grid,
        t_segments,
        x_segments,
        mode_segments,
        sol_segments,
    )
    return Trace(
        t=rolled[0],
        x=rolled[1],
        mode=rolled[2],
        events=tuple(events),
    )


def generate_traces(
    system: HybridSystem,
    t_span: tuple[float, float],
    initial_states: Iterable[Iterable[float]],
    *,
    max_jumps: int = 256,
    rtol: float = 1e-7,
    atol: float = 1e-9,
    max_step: float | None = None,
    dense_output: bool = False,
    sample_times: Iterable[float] | None = None,
    sample_dt: float | None = None,
) -> list[Trace]:
    """Simulate a batch of traces for a set of initial states."""
    return [
        simulate(
            system,
            t_span,
            x0=state,
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
    mode: Mode,
    system_params: Mapping[str, float],
) -> Callable[[float, np.ndarray], np.ndarray]:
    """Bind mode and system parameters into a solver-compatible function."""

    def flow(t: float, y: np.ndarray) -> np.ndarray:
        return mode.flow(t, y, {**system_params, **mode.params})

    return flow


def _build_event_functions(
    transitions: Sequence[Transition],
    system_params: Mapping[str, float],
    mode_params: Mapping[str, float],
) -> list[_EventFn]:
    """Create SciPy-compatible event functions for transitions."""
    event_functions: list[_EventFn] = []
    for transition in transitions:
        guard = transition.guard

        params = {**system_params, **mode_params}
        event_functions.append(_EventFn(guard.fn, params, guard.direction))
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


def _rollout_segments(
    sample_times: np.ndarray,
    t_segments: Sequence[np.ndarray],
    x_segments: Sequence[np.ndarray],
    mode_segments: Sequence[np.ndarray],
    sol_segments: Sequence[Callable[[np.ndarray], np.ndarray] | None],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if sample_times.size == 0:
        return (
            sample_times,
            np.empty((0, 0), dtype=float),
            np.empty((0,), dtype=object),
        )

    sampled_t: list[np.ndarray] = []
    sampled_x: list[np.ndarray] = []
    sampled_mode: list[np.ndarray] = []

    for t_seg, x_seg, mode_seg, sol in zip(
        t_segments,
        x_segments,
        mode_segments,
        sol_segments,
        strict=False,
    ):
        t_start = float(t_seg[0])
        t_end = float(t_seg[-1])
        mask = (sample_times >= t_start) & (sample_times <= t_end)
        if not np.any(mask):
            continue
        times = sample_times[mask]
        sampled_t.append(times)
        if sol is not None:
            sampled_x.append(sol(times).T)
        else:
            sampled_x.append(_interpolate_segment(t_seg, x_seg, times))
        sampled_mode.append(np.full(times.shape, mode_seg[0], dtype=object))

    if not sampled_t:
        return (
            np.array([], dtype=float),
            np.empty((0, 0), dtype=float),
            np.empty((0,), dtype=object),
        )

    return (
        np.concatenate(sampled_t),
        np.concatenate(sampled_x),
        np.concatenate(sampled_mode),
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
