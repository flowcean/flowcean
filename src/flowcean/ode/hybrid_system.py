"""Hybrid system core types."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

State = np.ndarray
Input = np.ndarray
InputStream = Callable[[float], Input]
Derivative = State | float
FlowFn = Callable[[float, State, Mapping[str, float], InputStream], Derivative]
GuardFn = Callable[[float, State, Mapping[str, float], InputStream], float]
ResetFn = Callable[[float, State, Mapping[str, float], InputStream], State]


@dataclass(frozen=True)
class ContinuousDynamics:
    """Reusable continuous dynamics definition.

    Args:
        name: Dynamics identifier.
        flow: Dynamics function returning the state derivative.
            Scalar derivative returns are accepted only for single-state
            systems, both during solver evaluation and when derivatives are
            captured on the returned trace grid.
        params: Optional parameter map merged with system parameters.
    """

    name: str
    flow: FlowFn
    params: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class Location:
    """Discrete hybrid-automaton location.

    Args:
        name: Location identifier.
        dynamics: Continuous dynamics active while this location is active.
    """

    name: str
    dynamics: ContinuousDynamics


@dataclass(frozen=True)
class Guard:
    """Guard function defining a transition event.

    Args:
        name: Guard identifier.
        fn: Root function; transitions when it crosses zero.
        direction: Crossing direction (-1, 0, +1).
    """

    name: str
    fn: GuardFn
    direction: int = 0


@dataclass(frozen=True)
class Reset:
    """State reset applied on a transition.

    Args:
        name: Reset identifier.
        fn: Reset function applied at the event time.
        params: Optional parameter map for the reset.
    """

    name: str
    fn: ResetFn
    params: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class Transition:
    """Discrete transition between locations.

    Args:
        source_location: Source location name.
        target_location: Target location name.
        guard: Guard that triggers the transition.
        reset: Optional reset applied upon transition.
    """

    source_location: str
    target_location: str
    guard: Guard
    reset: Reset | None = None


@dataclass(frozen=True)
class HybridSystem:
    """Hybrid system with locations and transitions.

    Args:
        locations: Location registry by name.
        transitions: Transition list defining guards and resets.
        initial_location: Name of the starting location.
        initial_state: Initial state vector.
        params: Global parameter map passed to dynamics and guards.
    """

    locations: Mapping[str, Location]
    transitions: Sequence[Transition]
    initial_location: str
    initial_state: State
    params: Mapping[str, float] = field(default_factory=dict)

    def transitions_from(self, location: str) -> list[Transition]:
        """Return transitions leaving the given location."""
        return [
            transition
            for transition in self.transitions
            if transition.source_location == location
        ]


@dataclass(frozen=True)
class Event:
    """Transition event information for a trace."""

    time: float
    source_location: str
    target_location: str
    guard: str
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
