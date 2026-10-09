"""Run-local callback bindings, parameter snapshots, and numerical coercion."""

from collections.abc import Callable, Iterable, Mapping
from inspect import Parameter, signature
from types import MappingProxyType
from typing import NamedTuple

import numpy as np

from .hybrid_system import (
    HybridSystem,
    Input,
    InputStream,
    Location,
    Parameters,
    State,
    Transition,
    _validate_delay,
)

type _BoundFunction[T] = Callable[[float, State, float], T]


class _ResidenceClock(NamedTuple):
    """Stable reference for one uninterrupted location visit."""

    anchor: float
    age_at_anchor: float

    def age(self, time: float) -> float:
        return self.age_at_anchor + (time - self.anchor)


def ensure_state(state: Iterable[float]) -> np.ndarray:
    """Copy a one-dimensional state, including one-shot iterables."""
    array = np.array(
        state if isinstance(state, np.ndarray) else list(state),
        dtype=float,
        copy=True,
    )
    if array.ndim != 1:
        raise ValueError("State must be a 1D array.")
    return array


def _readonly(values: np.ndarray) -> np.ndarray:
    """Detach numerical data into immutable backing storage."""
    array = np.asarray(values, dtype=float)
    return np.frombuffer(array.tobytes(), dtype=float).reshape(array.shape)


def _coerce_vector(
    candidate: object, *, state_dim: int, name: str
) -> np.ndarray:
    """Copy a state-sized vector, also accepting scalars for 1D systems."""
    values = np.asarray(
        list(candidate)
        if isinstance(candidate, Iterable)
        and not isinstance(candidate, np.ndarray)
        else candidate,
        dtype=float,
    )
    if values.ndim == 0 and state_dim == 1:
        values = values.reshape(1)
    if values.ndim != 1 or values.size != state_dim:
        raise ValueError(
            f"{name} must return a 1D vector matching the state dimension."
        )
    return values.copy()


def _coerce_input(candidate: object) -> np.ndarray:
    try:
        values = np.asarray(candidate, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("Input stream must return numeric values.") from error
    if values.ndim != 1:
        raise ValueError("Input stream must return a 1D array.")
    return values.copy()


def _missing_input_stream(time: float) -> Input:
    raise ValueError(
        f"input_stream is required; callback accessed input at t={time}."
    )


class _Callback[T]:
    """Inspect once, then dispatch canonical subsets or four positional inputs."""

    def __init__(self, callback: Callable[..., T]) -> None:
        self.callback = callback
        canonical = (
            "t",
            "state",
            "parameters",
            "input_stream",
            "location_time",
        )
        self.names: tuple[str, ...] | None = None
        try:
            parameters = tuple(signature(callback).parameters.values())
        except (TypeError, ValueError):
            return
        if any(
            p.kind in {Parameter.POSITIONAL_ONLY, Parameter.VAR_POSITIONAL}
            or (
                p.kind != Parameter.VAR_KEYWORD
                and p.name not in canonical
                and p.default is Parameter.empty
            )
            for p in parameters
        ):
            return
        self.names = (
            tuple(canonical)
            if any(p.kind == Parameter.VAR_KEYWORD for p in parameters)
            else tuple(p.name for p in parameters if p.name in canonical)
        )

    def __call__(
        self,
        t: float,
        state: np.ndarray,
        parameters: Parameters,
        input_stream: InputStream,
        location_time: float,
    ) -> T:
        if self.names is None:
            return self.callback(t, state, parameters, input_stream)
        values = {
            "t": t,
            "state": state,
            "parameters": parameters,
            "input_stream": input_stream,
            "location_time": location_time,
        }
        return self.callback(**{name: values[name] for name in self.names})


def _bind_callback[T](
    callback: Callable[..., T],
    parameters: Parameters,
    input_stream: InputStream,
) -> _BoundFunction[T]:
    """Capture run inputs and adapt a callback to time, state, and residence age."""
    adapted = _Callback(callback)

    def bound(time: float, state: State, location_time: float) -> T:
        return adapted(time, state, parameters, input_stream, location_time)

    return bound


def _vector_function(
    callback: _BoundFunction[object], *, name: str
) -> _BoundFunction[State]:
    """Validate and detach the output of a bound flow or reset."""

    def vector(time: float, state: State, location_time: float) -> State:
        return _coerce_vector(
            callback(time, state, location_time),
            state_dim=state.size,
            name=name,
        )

    return vector


def _surface_function(
    callback: _BoundFunction[float],
) -> _BoundFunction[float]:
    """Coerce surface outputs to scalars; the simulator handles NaN values."""

    def surface(time: float, state: State, location_time: float) -> float:
        return float(callback(time, state, location_time))

    return surface


class _RunBindings:
    """Captured parameters, input stream, and typed functions for one run.

    Every location's effective parameters are snapshotted before binding any
    callback. Model objects key the bindings; callback objects need not be
    hashable. Input streams and other external resources are not copied.
    """

    def __init__(
        self, system: HybridSystem, input_stream: InputStream | None
    ) -> None:
        self.parameters: Mapping[Location, Parameters] = MappingProxyType(
            {
                location: MappingProxyType(
                    {**system.parameters, **location.parameters}
                )
                for location in system.locations
            }
        )
        self.input_stream = input_stream
        self.effective_input_stream = (
            _missing_input_stream if input_stream is None else input_stream
        )
        self.flows: Mapping[Location, _BoundFunction[State]] = (
            MappingProxyType(
                {
                    location: _vector_function(
                        _bind_callback(
                            location.flow.fn,
                            self.parameters[location],
                            self.effective_input_stream,
                        ),
                        name="Flow",
                    )
                    for location in system.locations
                }
            )
        )
        self.surfaces: Mapping[Transition, _BoundFunction[float]] = (
            MappingProxyType(
                {
                    transition: _surface_function(
                        _bind_callback(
                            transition.event_surface.fn,
                            self.parameters[transition.source],
                            self.effective_input_stream,
                        )
                    )
                    for transition in system.transitions
                }
            )
        )
        self.delays: Mapping[Transition, _BoundFunction[float]] = (
            MappingProxyType(
                {
                    transition: _bind_callback(
                        transition.delay,
                        self.parameters[transition.source],
                        self.effective_input_stream,
                    )
                    for transition in system.transitions
                    if callable(transition.delay)
                }
            )
        )
        self.resets: Mapping[Transition, _BoundFunction[State]] = (
            MappingProxyType(
                {
                    transition: _vector_function(
                        _bind_callback(
                            transition.reset.fn,
                            self.parameters[transition.source],
                            self.effective_input_stream,
                        ),
                        name="Reset",
                    )
                    for transition in system.transitions
                    if transition.reset is not None
                }
            )
        )

    def delay(
        self,
        transition: Transition,
        time: float,
        state: State,
        location_time: float,
    ) -> float:
        """Resolve a delay using the source context at detection."""
        if callable(transition.delay):
            return _validate_delay(
                self.delays[transition](time, state, location_time)
            )
        return transition.delay
