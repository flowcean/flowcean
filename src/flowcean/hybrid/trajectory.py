"""Hybrid executions and explicit tabular sampling."""

from bisect import bisect_left
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

import numpy as np
import polars as pl

from ._runtime import _coerce_input, _readonly, _ResidenceClock, _RunBindings
from .hybrid_system import (
    HybridSystem,
    Location,
    Parameters,
    State,
    Transition,
    display_label,
)


@dataclass(frozen=True, eq=False)
class TrajectoryPoint:
    """State, location, and residence age at one point in an execution.

    The state is a detached, read-only array; location is the active model object.
    """

    state: State
    location: Location
    location_time: float

    def __post_init__(self) -> None:
        state = _readonly(self.state)
        if state.ndim != 1:
            raise ValueError("State must be a 1D array.")
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "location_time", float(self.location_time))


@dataclass(frozen=True, eq=False)
class Event:
    """A recorded transition with states immediately before and after it.

    Pre- and post-reset state snapshots are detached and read-only.
    """

    time: float
    transition: Transition
    state_before: State
    state_after: State
    microstep: int
    location_time_before: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "state_before", _readonly(self.state_before))
        object.__setattr__(self, "state_after", _readonly(self.state_after))


@dataclass(frozen=True, eq=False)
class ContinuousSegment:
    """Continuous evolution in one location over a time interval.

    The solver's dense solution and plotting knots are private. The residence
    clock stays anchored throughout this visit, independent of sampling grids.
    """

    location: Location
    t_span: tuple[float, float]
    _solution: Callable[[np.ndarray], np.ndarray] = field(repr=False)
    _clock: _ResidenceClock = field(repr=False)
    _knots: np.ndarray = field(repr=False)

    def __post_init__(self) -> None:
        if self.t_span[1] <= self.t_span[0]:
            raise ValueError(
                "Continuous segments must have positive duration."
            )
        object.__setattr__(self, "_knots", _readonly(self._knots))

    def evaluate(self, time: float) -> TrajectoryPoint:
        """Evaluate within the inclusive interval without invoking callbacks.

        The start is after any preceding jump chain; the end is before any
        following jump. Times must be finite; extrapolation is not allowed.
        """
        _validate_time(time, self.t_span)
        return TrajectoryPoint(
            self._solution(np.array([time]))[:, 0],
            self.location,
            self._clock.age(time),
        )

    def location_time(self, time: float) -> float:
        """Return residence age without evaluating the continuous state."""
        _validate_time(time, self.t_span)
        return self._clock.age(time)


@dataclass(frozen=True, eq=False)
class HybridTrajectory:
    """A run's ordered execution, associated model, and parameter snapshots.

    Numerical snapshots are read-only. Model identity is retained, but later
    model parameter edits do not affect this run. Callbacks and input streams
    must remain deterministic and pure when explicitly resampled.
    """

    system: HybridSystem
    t_span: tuple[float, float]
    initial_state: State
    initial_location: Location
    initial_location_time: float
    execution: tuple[ContinuousSegment | Event, ...]
    _bindings: _RunBindings = field(repr=False)
    _segments: tuple[ContinuousSegment, ...] = field(init=False, repr=False)
    _events: tuple[Event, ...] = field(init=False, repr=False)
    _segment_ends: tuple[float, ...] = field(init=False, repr=False)
    _last_events: Mapping[float, Event] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "initial_state", _readonly(self.initial_state)
        )
        object.__setattr__(self, "execution", tuple(self.execution))
        # These indexes reference execution records, never separate boundary states.
        segments = tuple(
            item
            for item in self.execution
            if isinstance(item, ContinuousSegment)
        )
        events = tuple(
            item for item in self.execution if isinstance(item, Event)
        )
        object.__setattr__(self, "_segments", segments)
        object.__setattr__(self, "_events", events)
        object.__setattr__(
            self,
            "_segment_ends",
            tuple(segment.t_span[1] for segment in segments),
        )
        object.__setattr__(
            self,
            "_last_events",
            MappingProxyType({event.time: event for event in events}),
        )

    @property
    def segments(self) -> tuple[ContinuousSegment, ...]:
        """Continuous pieces in execution order."""
        return self._segments

    @property
    def events(self) -> tuple[Event, ...]:
        """Individual transitions, including every same-time microstep."""
        return self._events

    @property
    def parameters(self) -> Mapping[Location, Parameters]:
        """Read-only effective parameter snapshots for all declared locations."""
        return self._bindings.parameters

    def evaluate(self, time: float) -> TrajectoryPoint:
        """Evaluate a finite time in the inclusive run interval, without callbacks.

        At an exact event time, return the final state and location after the
        entire jump chain, with zero residence age. This includes initial and
        final events. Nearby times are not snapped to event boundaries.
        """
        _validate_time(time, self.t_span)
        event = self._last_events.get(time)
        if event is not None:
            return TrajectoryPoint(
                event.state_after, event.transition.target, 0.0
            )
        if self._segments:
            segment = self._segments[bisect_left(self._segment_ends, time)]
            return segment.evaluate(time)
        return TrajectoryPoint(
            self.initial_state,
            self.initial_location,
            self.initial_location_time,
        )

    def sample(
        self,
        times: Iterable[float] | None = None,
        *,
        dt: float | None = None,
        include_state: bool = True,
        include_location_id: bool = True,
        include_location_label: bool = False,
        include_location_time: bool = True,
        include_inputs: bool = False,
        include_derivatives: bool = False,
    ) -> pl.DataFrame:
        """Sample exactly one explicit grid, with right-continuous event values.

        Default sampling invokes no callbacks. Inputs and flow derivatives are
        evaluated only when explicitly requested, at the requested times.
        These callbacks and input streams must remain pure and deterministic.
        """
        grid = _sample_grid(self.t_span, times, dt)
        if include_inputs and self._bindings.input_stream is None:
            raise ValueError("include_inputs=True requires an input_stream.")
        if include_inputs and not grid.size:
            raise ValueError(
                "Cannot sample inputs on an empty grid: undeclared input width."
            )
        points = [self.evaluate(float(time)) for time in grid]
        states = np.empty((len(points), self.initial_state.size), dtype=float)
        for index, point in enumerate(points):
            states[index] = point.state
        data = {"t": pl.Series("t", grid, dtype=pl.Float64)}
        if include_state:
            data.update(
                {
                    f"x{i}": pl.Series(f"x{i}", states[:, i])
                    for i in range(states.shape[1])
                }
            )
        if include_location_id:
            ids = {
                location: i for i, location in enumerate(self.system.locations)
            }
            data["location_id"] = pl.Series(
                "location_id",
                [ids[point.location] for point in points],
                dtype=pl.Int64,
            )
        if include_location_label:
            data["location_label"] = pl.Series(
                "location_label",
                [display_label(point.location) for point in points],
                dtype=pl.String,
            )
        if include_location_time:
            data["location_time"] = pl.Series(
                "location_time",
                [point.location_time for point in points],
                dtype=pl.Float64,
            )
        if include_inputs:
            stream = self._bindings.effective_input_stream
            inputs = [_coerce_input(stream(float(time))) for time in grid]
            if any(value.size != inputs[0].size for value in inputs):
                raise ValueError(
                    "Input stream dimension changed across requested rows."
                )
            values = np.vstack(inputs)
            data.update(
                {
                    f"u{i}": pl.Series(f"u{i}", values[:, i])
                    for i in range(values.shape[1])
                }
            )
        if include_derivatives:
            derivatives = np.empty_like(states)
            for index, (time, point) in enumerate(
                zip(grid, points, strict=True)
            ):
                derivatives[index] = self._bindings.flows[point.location](
                    float(time), point.state.copy(), point.location_time
                )
            data.update(
                {
                    f"dx{i}": pl.Series(f"dx{i}", derivatives[:, i])
                    for i in range(states.shape[1])
                }
            )
        return pl.DataFrame(data)


def _validate_time(time: float, t_span: tuple[float, float]) -> None:
    if not np.isfinite(time) or not t_span[0] <= time <= t_span[1]:
        raise ValueError("time must be finite and lie within t_span.")


def _sample_grid(
    t_span: tuple[float, float],
    times: Iterable[float] | None,
    dt: float | None,
) -> np.ndarray:
    """Validate explicit times or build a step grid including both endpoints."""
    if (times is None) == (dt is None):
        raise ValueError("Exactly one of times or dt is required.")
    start, end = t_span
    if times is not None:
        grid = np.array(
            times if isinstance(times, np.ndarray) else list(times),
            dtype=float,
            copy=True,
        )
        if grid.ndim != 1:
            raise ValueError("times must be a 1D grid.")
        if not np.all(np.isfinite(grid)):
            raise ValueError("times must be finite.")
        if np.any(np.diff(grid) < 0):
            raise ValueError("times must be non-descending.")
        if np.any(grid < start) or np.any(grid > end):
            raise ValueError("times must lie within t_span.")
        return grid
    if dt is None or not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive.")
    if start == end:
        return np.array([start])
    if start + dt <= start:
        raise ValueError("dt does not advance floating-point time.")
    grid = np.arange(start, end, dt, dtype=float)
    grid = np.append(grid[grid < end], end)
    if np.any(np.diff(grid) <= 0):
        raise ValueError("dt does not advance floating-point time.")
    return grid
