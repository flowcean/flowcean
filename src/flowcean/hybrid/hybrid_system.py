"""Hybrid system core types."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import IntEnum, StrEnum
from typing import Protocol, overload

import numpy as np

State = np.ndarray
Input = np.ndarray
InputStream = Callable[[float], Input]
Parameters = Mapping[str, float]
Derivative = State | float


class FlowFunction(Protocol):
    """Continuous-state derivative callback.

    Flow, event-surface, and reset callbacks share these inputs: physical time
    (``t``), the continuous ``state`` vector, effective ``parameters``, an
    ``input_stream`` returning a vector for a requested time, and the current
    visit's elapsed ``location_time``.

    Model callbacks may declare any subset using these names, including
    keyword-only arguments. Named callbacks with ``**kwargs`` receive all five
    inputs. Four-positional callbacks receive ``(t, state, parameters,
    input_stream)``; positional-only arguments, ``*args``, and noncanonical
    required names select that form.

    Callbacks and input streams must be pure and deterministic: integration
    and derivative sampling can evaluate the same inputs repeatedly.
    """

    def __call__(
        self,
        *,
        t: float,
        state: State,
        parameters: Parameters,
        input_stream: InputStream,
        location_time: float,
    ) -> Derivative: ...


class EventSurfaceFunction(Protocol):
    """Scalar event-surface callback.

    See [FlowFunction][flowcean.hybrid.FlowFunction] for shared inputs and
    supported callback signatures.
    """

    def __call__(
        self,
        *,
        t: float,
        state: State,
        parameters: Parameters,
        input_stream: InputStream,
        location_time: float,
    ) -> float: ...


class ResetFunction(Protocol):
    """State reset callback.

    See [FlowFunction][flowcean.hybrid.FlowFunction] for shared inputs and
    supported callback signatures.
    """

    def __call__(
        self,
        *,
        t: float,
        state: State,
        parameters: Parameters,
        input_stream: InputStream,
        location_time: float,
    ) -> State: ...


class CrossingDirection(IntEnum):
    """Direction in which an event surface must cross zero."""

    FALLING = -1
    EITHER = 0
    RISING = 1


class SurfaceEntryPolicy(StrEnum):
    """Behavior when a transition surface is exactly zero on location entry.

    Entry includes initialization and arrival after a transition. ``ERROR``
    raises [SurfaceEntryError][flowcean.hybrid.SurfaceEntryError]; ``TRIGGER``
    applies the transition immediately at the same physical time; ``CONTINUE``
    begins continuous integration with the surface at zero. For ``CONTINUE``,
    choose a flow that departs in the direction opposite to the accepted
    crossing so integration can advance.

    All outgoing surfaces are evaluated before resolving entry. NaN values
    are rejected first, then zero ``ERROR`` surfaces. Exactly one zero
    ``TRIGGER`` surface performs a jump; multiple such surfaces raise
    [AmbiguousTransitionError][flowcean.hybrid.AmbiguousTransitionError].
    Entry handling repeats at the target after an immediate transition.
    """

    ERROR = "error"
    TRIGGER = "trigger"
    CONTINUE = "continue"


@dataclass(frozen=True, eq=False)
class Flow:
    """Reusable continuous-state derivative law.

    Args:
        fn: Function returning the state derivative. See
            [FlowFunction][flowcean.hybrid.FlowFunction] for callback inputs.
            Scalar derivative returns are accepted only for single-state
            systems, both during solver evaluation and when derivatives are
            explicitly sampled from a trajectory.
        label: Optional display label.
    """

    fn: Callable[..., Derivative]
    label: str | None = None

    def __post_init__(self) -> None:
        if not callable(self.fn):
            message = "fn must be callable."
            raise TypeError(message)


@dataclass(frozen=True, eq=False, init=False)
class Location:
    """Discrete hybrid-automaton location.

    Args:
        flow: Flow definition or bare derivative callback active here.
        label: Optional display label.
        parameters: Location-local parameters overriding system parameters
            with the same names. Effective maps are snapshotted for each run.
    """

    flow: Flow
    label: str | None
    parameters: Parameters

    @overload
    def __init__(
        self,
        flow: Flow,
        *,
        label: str | None = None,
        parameters: Parameters | None = None,
    ) -> None: ...

    @overload
    def __init__(
        self,
        flow: Callable[..., Derivative],
        *,
        label: str | None = None,
        parameters: Parameters | None = None,
    ) -> None: ...

    def __init__(
        self,
        flow: Flow | Callable[..., Derivative],
        *,
        label: str | None = None,
        parameters: Parameters | None = None,
    ) -> None:
        if isinstance(flow, Flow):
            location_flow = flow
        elif callable(flow):
            location_flow = Flow(flow)
        else:
            message = "Location requires Flow or a flow callback."
            raise TypeError(message)
        if parameters is not None and not isinstance(parameters, Mapping):
            message = "parameters must be a mapping."
            raise TypeError(message)
        object.__setattr__(self, "flow", location_flow)
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "parameters", dict(parameters or {}))


@dataclass(frozen=True, eq=False)
class EventSurface:
    """Scalar event surface defining a simulated transition event.

    Supply a continuous scalar function whose zero crossings identify the
    switching boundary. ``RISING`` accepts negative-to-positive crossings,
    ``FALLING`` positive-to-negative crossings, and ``EITHER`` both. Exact zero
    on entry follows the transition's
    [SurfaceEntryPolicy][flowcean.hybrid.SurfaceEntryPolicy]. NaN values raise
    [InvalidEventSurfaceValueError][flowcean.hybrid.InvalidEventSurfaceValueError];
    nonzero values, including infinities, retain their sign in entry checks.
    Continuous crossing detection uses SciPy's event solver.

    Args:
        fn: Root function. See [FlowFunction][flowcean.hybrid.FlowFunction]
            for callback inputs.
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

    The callback receives the source location's effective parameters and
    residence time. Its result must match the continuous-state dimension;
    a scalar is accepted for a single-state system.

    Args:
        fn: Reset function applied at the event time. See
            [FlowFunction][flowcean.hybrid.FlowFunction] for callback inputs.
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

    ``event_surface`` is a scalar zero-crossing surface.

    Args:
        source: Source location.
        target: Target location.
        event_surface: Event surface that triggers the transition.
        reset: Optional reset applied upon transition.
        entry_policy: Behavior when the event surface is exactly zero upon
            entry to the source location.
    """

    source: Location
    target: Location
    event_surface: EventSurface
    reset: Reset | None = None
    entry_policy: SurfaceEntryPolicy = SurfaceEntryPolicy.ERROR

    def __init__(
        self,
        source: Location,
        target: Location,
        event_surface: EventSurface | Callable[..., float],
        reset: Reset | Callable[..., State] | None = None,
        *,
        entry_policy: SurfaceEntryPolicy = SurfaceEntryPolicy.ERROR,
    ) -> None:
        if not isinstance(source, Location):
            message = "source must be a Location."
            raise TypeError(message)
        if not isinstance(target, Location):
            message = "target must be a Location."
            raise TypeError(message)
        if isinstance(event_surface, EventSurface):
            surface = event_surface
        elif callable(event_surface):
            surface = EventSurface(event_surface)
        else:
            message = "event_surface must be an EventSurface or callable."
            raise TypeError(message)
        if isinstance(reset, Reset) or reset is None:
            transition_reset = reset
        elif callable(reset):
            transition_reset = Reset(reset)
        else:
            message = "reset must be a Reset, callable, or None."
            raise TypeError(message)
        if type(entry_policy) is not SurfaceEntryPolicy:
            message = "entry_policy must be a SurfaceEntryPolicy."
            raise TypeError(message)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "event_surface", surface)
        object.__setattr__(self, "reset", transition_reset)
        object.__setattr__(self, "entry_policy", entry_policy)


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


def display_label(obj: object, *, fallback: str | None = None) -> str:
    """Return the human-readable label, using fallback before repr if set."""

    def last_resort() -> str:
        return fallback if fallback is not None else repr(obj)

    if isinstance(obj, Location):
        return (
            obj.label
            or obj.flow.label
            or _callback_label(obj.flow.fn)
            or last_resort()
        )
    if isinstance(obj, Flow):
        return obj.label or _callback_label(obj.fn) or last_resort()
    if isinstance(obj, EventSurface):
        return obj.label or _callback_label(obj.fn) or last_resort()
    if isinstance(obj, Reset):
        return obj.label or _callback_label(obj.fn) or last_resort()
    return last_resort()


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
