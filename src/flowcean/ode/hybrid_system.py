"""Hybrid system core types."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Protocol, overload

import numpy as np

State = np.ndarray
Input = np.ndarray
InputStream = Callable[[float], Input]
Parameters = Mapping[str, float]
Derivative = State | float


class FlowFunction(Protocol):
    """Continuous dynamics callback."""

    def __call__(
        self,
        t: float,
        state: State,
        parameters: Parameters,
        input_stream: InputStream,
        /,
    ) -> Derivative: ...


class EventSurfaceFunction(Protocol):
    """Scalar event-surface callback."""

    def __call__(
        self,
        t: float,
        state: State,
        parameters: Parameters,
        input_stream: InputStream,
        /,
    ) -> float: ...


class ResetFunction(Protocol):
    """State reset callback."""

    def __call__(
        self,
        t: float,
        state: State,
        parameters: Parameters,
        input_stream: InputStream,
        /,
    ) -> State: ...


class CrossingDirection(IntEnum):
    """Direction in which an event surface must cross zero."""

    FALLING = -1
    EITHER = 0
    RISING = 1


@dataclass(frozen=True, eq=False)
class ContinuousDynamics:
    """Reusable continuous dynamics definition.

    Args:
        flow: Dynamics function returning the state derivative.
            Scalar derivative returns are accepted only for single-state
            systems, both during solver evaluation and when derivatives are
            captured on the returned trace grid.
        label: Optional display label.
    """

    flow: Callable[..., Derivative]
    label: str | None = None

    def __post_init__(self) -> None:
        if not callable(self.flow):
            message = "flow must be callable."
            raise TypeError(message)


@dataclass(frozen=True, eq=False, init=False)
class Location:
    """Discrete hybrid-automaton location.

    Args:
        dynamics: Continuous dynamics or bare flow callback active here.
        label: Optional display label.
        parameters: Location-local parameter map.
    """

    dynamics: ContinuousDynamics
    label: str | None
    parameters: Parameters

    @overload
    def __init__(
        self,
        dynamics: ContinuousDynamics,
        *,
        label: str | None = None,
        parameters: Parameters | None = None,
    ) -> None: ...

    @overload
    def __init__(
        self,
        dynamics: Callable[..., Derivative],
        *,
        label: str | None = None,
        parameters: Parameters | None = None,
    ) -> None: ...

    def __init__(
        self,
        dynamics: ContinuousDynamics | Callable[..., Derivative],
        *,
        label: str | None = None,
        parameters: Parameters | None = None,
    ) -> None:
        if isinstance(dynamics, ContinuousDynamics):
            continuous_dynamics = dynamics
        elif callable(dynamics):
            continuous_dynamics = ContinuousDynamics(dynamics)
        else:
            message = (
                "Location requires ContinuousDynamics or a flow callback."
            )
            raise TypeError(message)
        if parameters is not None and not isinstance(parameters, Mapping):
            message = "parameters must be a mapping."
            raise TypeError(message)
        object.__setattr__(self, "dynamics", continuous_dynamics)
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "parameters", dict(parameters or {}))


@dataclass(frozen=True, eq=False)
class EventSurface:
    """Scalar event surface defining a simulated transition event.

    Flowcean transitions fire when ``fn`` reaches zero in ``direction``.
    This is event-surface semantics, not Boolean guard-region semantics.

    Args:
        fn: Root function; transitions when it crosses zero.
        direction: Crossing direction. Defaults to either direction.
        label: Optional display label.
    """

    fn: Callable[..., float]
    direction: CrossingDirection = CrossingDirection.EITHER
    label: str | None = None

    def __post_init__(self) -> None:
        if not callable(self.fn):
            message = "fn must be callable."
            raise TypeError(message)
        if not isinstance(self.direction, CrossingDirection):
            message = "direction must be a CrossingDirection."
            raise TypeError(message)


@dataclass(frozen=True, eq=False)
class Reset:
    """State reset applied on a transition.

    Args:
        fn: Reset function applied at the event time.
        label: Optional display label.
    """

    fn: Callable[..., State]
    label: str | None = None

    def __post_init__(self) -> None:
        if not callable(self.fn):
            message = "fn must be callable."
            raise TypeError(message)


@dataclass(frozen=True, eq=False, init=False)
class Transition:
    """Discrete event-triggered transition between locations.

    ``event`` is a scalar zero-crossing surface.

    Args:
        source: Source location.
        target: Target location.
        event: Event surface that triggers the transition.
        reset: Optional reset applied upon transition.
    """

    source: Location
    target: Location
    event: EventSurface
    reset: Reset | None = None

    def __init__(
        self,
        source: Location,
        target: Location,
        event: EventSurface | Callable[..., float],
        reset: Reset | Callable[..., State] | None = None,
    ) -> None:
        if not isinstance(source, Location):
            message = "source must be a Location."
            raise TypeError(message)
        if not isinstance(target, Location):
            message = "target must be a Location."
            raise TypeError(message)
        if isinstance(event, EventSurface):
            event_surface = event
        elif callable(event):
            event_surface = EventSurface(event)
        else:
            message = "event must be an EventSurface or callable."
            raise TypeError(message)
        if isinstance(reset, Reset) or reset is None:
            transition_reset = reset
        elif callable(reset):
            transition_reset = Reset(reset)
        else:
            message = "reset must be a Reset, callable, or None."
            raise TypeError(message)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "event", event_surface)
        object.__setattr__(self, "reset", transition_reset)


@dataclass(frozen=True, eq=False)
class HybridSystem:
    """Hybrid system with locations and transitions.

    Args:
        locations: Location objects in this system.
        transitions: Transition list defining event surfaces and resets.
        initial_location: Starting location object.
        initial_state: Initial state vector.
        parameters: Global parameter map passed to callbacks.
    """

    locations: Sequence[Location]
    transitions: Sequence[Transition]
    initial_location: Location
    initial_state: State
    parameters: Parameters = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.locations, Mapping):
            message = "locations must be a sequence of Location objects."
            raise TypeError(message)
        locations = tuple(self.locations)
        transitions = tuple(self.transitions)
        if any(not isinstance(location, Location) for location in locations):
            message = "locations must contain only Location objects."
            raise TypeError(message)
        if any(
            not isinstance(transition, Transition)
            for transition in transitions
        ):
            message = "transitions must contain only Transition objects."
            raise TypeError(message)
        if not isinstance(self.initial_location, Location):
            message = "initial_location must be a Location."
            raise TypeError(message)
        if self.parameters is not None and not isinstance(
            self.parameters,
            Mapping,
        ):
            message = "parameters must be a mapping."
            raise TypeError(message)

        location_ids = {id(location) for location in locations}
        if len(location_ids) != len(locations):
            message = "duplicate Location objects are not allowed."
            raise ValueError(message)
        if id(self.initial_location) not in location_ids:
            message = "initial_location must be present in locations."
            raise ValueError(message)
        _validate_transition_locations(transitions, location_ids)

        object.__setattr__(self, "locations", locations)
        object.__setattr__(self, "transitions", transitions)
        object.__setattr__(self, "parameters", dict(self.parameters or {}))

    def transitions_from(self, location: Location) -> list[Transition]:
        """Return transitions leaving the given location."""
        return [
            transition
            for transition in self.transitions
            if transition.source is location
        ]


def display_label(obj: object) -> str:
    """Return the human-readable label for a hybrid-system object."""
    if isinstance(obj, Location):
        return (
            obj.label
            or obj.dynamics.label
            or _callback_label(obj.dynamics.flow)
            or repr(obj)
        )
    if isinstance(obj, ContinuousDynamics):
        return obj.label or _callback_label(obj.flow) or repr(obj)
    if isinstance(obj, EventSurface):
        return obj.label or _callback_label(obj.fn) or repr(obj)
    if isinstance(obj, Reset):
        return obj.label or _callback_label(obj.fn) or repr(obj)
    return repr(obj)


def _validate_transition_locations(
    transitions: Sequence[Transition],
    location_ids: set[int],
) -> None:
    for transition in transitions:
        if id(transition.source) not in location_ids:
            message = "transition source must be present in locations."
            raise ValueError(message)
        if id(transition.target) not in location_ids:
            message = "transition target must be present in locations."
            raise ValueError(message)


def _callback_label(callback: object) -> str | None:
    qualified = getattr(callback, "__qualname__", None)
    if isinstance(qualified, str) and qualified:
        return qualified
    name = getattr(callback, "__name__", None)
    if isinstance(name, str) and name:
        return name
    return None


@dataclass(frozen=True)
class Event:
    """Transition event information for a trace."""

    time: float
    source_location: str
    target_location: str
    event_surface: str
    reset: str | None
    state: State


@dataclass(frozen=True)
class Trace:
    """Simulation trace with time, state, and location labels."""

    t: np.ndarray
    x: np.ndarray
    location: np.ndarray
    events: Sequence[Event]
    u: np.ndarray | None = None
    dx: np.ndarray | None = None

    def as_dict(self) -> dict[str, object]:
        """Return a dictionary view of the trace."""
        return {
            "t": self.t,
            "x": self.x,
            "location": self.location,
            "events": self.events,
            "u": self.u,
            "dx": self.dx,
        }
