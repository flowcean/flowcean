"""Hybrid-system modeling, automaton diagrams, simulation, and traces."""

from . import benchmarks, hydra
from .graph import build_hybrid_system_dot, render_dot_svg
from .hybrid_system import (
    CrossingDirection,
    EventSurface,
    EventSurfaceFunction,
    Flow,
    FlowFunction,
    HybridSystem,
    Input,
    InputStream,
    Location,
    Parameters,
    Reset,
    ResetFunction,
    SurfaceEntryPolicy,
    Transition,
)
from .plotting import plot_locations, plot_phase, plot_trajectory
from .simulator import (
    AmbiguousTransitionError,
    HybridSimulationError,
    InvalidEventSurfaceValueError,
    SimulationProgressError,
    SurfaceEntryError,
    simulate,
)
from .trajectory import ContinuousSegment, Event, HybridTrajectory

__all__ = (
    "AmbiguousTransitionError",
    "ContinuousSegment",
    "CrossingDirection",
    "Event",
    "EventSurface",
    "EventSurfaceFunction",
    "Flow",
    "FlowFunction",
    "HybridSimulationError",
    "HybridSystem",
    "HybridTrajectory",
    "Input",
    "InputStream",
    "InvalidEventSurfaceValueError",
    "Location",
    "Parameters",
    "Reset",
    "ResetFunction",
    "SimulationProgressError",
    "SurfaceEntryError",
    "SurfaceEntryPolicy",
    "Transition",
    "benchmarks",
    "build_hybrid_system_dot",
    "hydra",
    "plot_locations",
    "plot_phase",
    "plot_trajectory",
    "render_dot_svg",
    "simulate",
)
