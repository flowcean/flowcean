"""Hybrid-system modeling, automaton diagrams, simulation, and traces."""

from . import benchmarks, hydra
from .graph import build_hybrid_system_dot, render_dot_svg
from .hybrid_system import (
    ContinuousDynamics,
    CrossingDirection,
    EventSurface,
    EventSurfaceFunction,
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
from .plotting import plot_locations, plot_phase, plot_trace
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
    "ContinuousDynamics",
    "ContinuousSegment",
    "CrossingDirection",
    "Event",
    "EventSurface",
    "EventSurfaceFunction",
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
    "plot_trace",
    "render_dot_svg",
    "simulate",
)
