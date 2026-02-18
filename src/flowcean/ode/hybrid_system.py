"""Hybrid system core types."""

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

State = np.ndarray
Input = np.ndarray
InputStream = Callable[[float], Input]
FlowFn = Callable[[float, State, Mapping[str, float], InputStream], State]
GuardFn = Callable[[float, State, Mapping[str, float], InputStream], float]
ResetFn = Callable[[float, State, Mapping[str, float], InputStream], State]


@dataclass(frozen=True)
class Mode:
    """Continuous dynamics definition for a single mode.

    Args:
        name: Mode identifier.
        flow: Dynamics function returning the state derivative.
        params: Optional parameter map merged with system parameters.
    """

    name: str
    flow: FlowFn
    params: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class Guard:
    """Guard function defining a transition event.

    Args:
        name: Guard identifier.
        fn: Root function; transitions when it crosses zero.
        direction: Crossing direction (-1, 0, +1).
        terminal: Whether the solver should stop at the event.
    """

    name: str
    fn: GuardFn
    direction: int = 0
    terminal: bool = True


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
    """Discrete transition between modes.

    Args:
        source: Source mode name.
        target: Target mode name.
        guard: Guard that triggers the transition.
        reset: Optional reset applied upon transition.
    """

    source: str
    target: str
    guard: Guard
    reset: Reset | None = None


@dataclass(frozen=True)
class HybridSystem:
    """Hybrid system with modes and transitions.

    Args:
        modes: Mode registry by name.
        transitions: Transition list defining guards and resets.
        initial_mode: Name of the starting mode.
        initial_state: Initial state vector.
        params: Global parameter map passed to dynamics and guards.
    """

    modes: Mapping[str, Mode]
    transitions: Sequence[Transition]
    initial_mode: str
    initial_state: State
    params: Mapping[str, float] = field(default_factory=dict)

    def transitions_from(self, mode: str) -> list[Transition]:
        """Return transitions leaving the given mode."""
        return [
            transition
            for transition in self.transitions
            if transition.source == mode
        ]


@dataclass(frozen=True)
class Event:
    """Transition event information for a trace."""

    time: float
    source_mode: str
    target_mode: str
    guard: str
    reset: str | None
    state: State


@dataclass(frozen=True)
class Trace:
    """Simulation trace with time, state, and mode labels."""

    t: np.ndarray
    x: np.ndarray
    mode: np.ndarray
    events: Sequence[Event]
    u: np.ndarray | None = None

    def as_dict(self) -> dict[str, object]:
        """Return a dictionary view of the trace."""
        return {
            "t": self.t,
            "x": self.x,
            "mode": self.mode,
            "events": self.events,
            "u": self.u,
        }
