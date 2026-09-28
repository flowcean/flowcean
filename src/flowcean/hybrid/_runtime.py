"""Run-local callback bindings, parameter snapshots, and numerical coercion."""

from collections.abc import Callable, Iterable, Mapping
from inspect import Parameter, signature
from types import MappingProxyType
from typing import Any, NamedTuple

import numpy as np

from .hybrid_system import (
    HybridSystem,
    Input,
    InputStream,
    Location,
    Parameters,
)


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


class _Callback:
    """Inspect once, then dispatch canonical subsets or four positional inputs."""

    def __init__(self, callback: Callable[..., Any]) -> None:
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
    ) -> Any:
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


class _RunContext:
    """Bindings retained by a trajectory under pure callback semantics.

    Callbacks and input streams must be deterministic under repeated evaluation.
    Their external resources are not copied. All effective parameter maps are
    snapshotted before the first callback, including those of unvisited locations.
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
        callbacks = [location.dynamics.flow for location in system.locations]
        callbacks.extend(
            transition.event.fn for transition in system.transitions
        )
        callbacks.extend(
            transition.reset.fn
            for transition in system.transitions
            if transition.reset is not None
        )
        self.callbacks = {
            key: _Callback(callback)
            for key, callback in {
                id(callback): callback for callback in callbacks
            }.items()
        }

    def call(
        self,
        callback: Callable[..., Any],
        location: Location,
        time: float,
        state: np.ndarray,
        age: float,
    ) -> Any:
        return self.callbacks[id(callback)](
            time,
            state,
            self.parameters[location],
            self.effective_input_stream,
            age,
        )

    def derivative(
        self,
        location: Location,
        time: float,
        state: np.ndarray,
        age: float,
    ) -> np.ndarray:
        return _coerce_vector(
            self.call(location.dynamics.flow, location, time, state, age),
            state_dim=state.size,
            name="Flow",
        )
