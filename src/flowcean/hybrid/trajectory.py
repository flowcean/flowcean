"""Continuous hybrid executions and explicit tabular sampling."""

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field

import numpy as np
import polars as pl

from ._runtime import _coerce_input, _readonly, _ResidenceClock, _RunContext
from .hybrid_system import (
    HybridSystem,
    Location,
    Parameters,
    State,
    display_label,
)


@dataclass(frozen=True, eq=False)
class Event:
    """One microstep with detached, read-only pre- and post-reset snapshots."""

    time: float
    source_location: Location
    target_location: Location
    event_surface: str
    reset: str | None
    state_before: State
    state_after: State
    microstep: int
    location_time_before: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "state_before", _readonly(self.state_before))
        object.__setattr__(self, "state_after", _readonly(self.state_after))


@dataclass(frozen=True, eq=False)
class ContinuousSegment:
    """Positive-duration evolution, post-chain at start and pre-jump at end.

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

    def location_time(self, time: float) -> float:
        """Return residence age at a time in this segment."""
        if not self.t_span[0] <= time <= self.t_span[1]:
            raise ValueError("time must lie within the segment.")
        return self._clock.age(time)


@dataclass(frozen=True, eq=False)
class HybridTrajectory:
    """A run's ordered execution, associated model, and frozen run context.

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
    _context: _RunContext = field(repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "initial_state", _readonly(self.initial_state)
        )
        object.__setattr__(self, "execution", tuple(self.execution))

    @property
    def segments(self) -> tuple[ContinuousSegment, ...]:
        """Continuous pieces in execution order."""
        return tuple(
            item
            for item in self.execution
            if isinstance(item, ContinuousSegment)
        )

    @property
    def events(self) -> tuple[Event, ...]:
        """Individual transitions, including every same-time microstep."""
        return tuple(
            item for item in self.execution if isinstance(item, Event)
        )

    @property
    def parameters(self) -> Mapping[Location, Parameters]:
        """Read-only effective parameter snapshots for all declared locations."""
        return self._context.parameters

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
        """
        grid = _sample_grid(self.t_span, times, dt)
        if include_inputs and self._context.input_stream is None:
            raise ValueError("include_inputs=True requires an input_stream.")
        if include_inputs and not grid.size:
            raise ValueError(
                "Cannot sample inputs on an empty grid: undeclared input width."
            )
        states, locations, ages = self._evaluate(grid)
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
                "location_id", [ids[loc] for loc in locations], dtype=pl.Int64
            )
        if include_location_label:
            data["location_label"] = pl.Series(
                "location_label",
                [display_label(loc) for loc in locations],
                dtype=pl.String,
            )
        if include_location_time:
            data["location_time"] = pl.Series("location_time", ages)
        if include_inputs:
            stream = self._context.effective_input_stream
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
            for i, (time, state, location, age) in enumerate(
                zip(grid, states, locations, ages, strict=True)
            ):
                derivatives[i] = self._context.derivative(
                    location, float(time), state.copy(), float(age)
                )
            data.update(
                {
                    f"dx{i}": pl.Series(f"dx{i}", derivatives[:, i])
                    for i in range(states.shape[1])
                }
            )
        return pl.DataFrame(data)

    def _evaluate(
        self, times: np.ndarray
    ) -> tuple[np.ndarray, list[Location], np.ndarray]:
        # Derived lookup only: execution is the single authoritative record.
        last_events = {event.time: event for event in self.events}
        segments = self.segments
        ends = np.array([segment.t_span[1] for segment in segments])
        states = np.empty((len(times), self.initial_state.size), dtype=float)
        locations: list[Location] = []
        ages = np.empty(len(times), dtype=float)
        for i, time in enumerate(times):
            event = last_events.get(float(time))
            if event is not None:
                states[i] = event.state_after
                locations.append(event.target_location)
                ages[i] = 0.0
            elif segments:
                segment = segments[
                    int(np.searchsorted(ends, time, side="left"))
                ]
                states[i] = segment._solution(np.array([time]))[:, 0]
                locations.append(segment.location)
                ages[i] = segment._clock.age(float(time))
            else:
                states[i] = self.initial_state
                locations.append(self.initial_location)
                ages[i] = self.initial_location_time
        return states, locations, ages


def _sample_grid(
    t_span: tuple[float, float],
    times: Iterable[float] | None,
    dt: float | None,
) -> np.ndarray:
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
