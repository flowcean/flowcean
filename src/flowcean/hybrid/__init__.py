"""Hybrid-system simulation and identification."""

from . import benchmarks
from .benchmarks import BenchmarkSpec, all_specs, registry
from .benchmarks.bouncing_ball import bouncing_ball
from .benchmarks.hybrid_oscillator import hybrid_oscillator
from .benchmarks.impact_oscillator import (
    impact_input_stream,
    impact_oscillator,
)
from .benchmarks.mode_cycle import mode_cycle
from .benchmarks.pid_controlled_plant import pid_controlled_plant
from .benchmarks.piecewise_affine import piecewise_affine
from .benchmarks.relay_integrator import relay_integrator
from .benchmarks.switched_linear import switched_linear
from .benchmarks.tank_valves import tank_valves
from .benchmarks.thermostat import thermostat, thermostat_target_stream
from .benchmarks.time_forced_switch import time_forced_switch
from .benchmarks.time_varying_event_surface import (
    time_varying_event_surface,
    time_varying_input_stream,
)
from .hybrid_system import (
    ContinuousDynamics,
    CrossingDirection,
    Event,
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
    Trace,
    Transition,
)
from .hydra.callbacks import LogCallback, PlotCallback
from .hydra.learner import HyDRALearner
from .hydra.model import HyDRAModel
from .hydra.schema import HyDRATraceSchema
from .hydra.selector.config import SelectorFeatureConfig
from .hydra.selector.evaluation import (
    SelectorEvaluationReport,
    evaluate_selector_autoregressive,
    evaluate_selector_oracle,
)
from .hydra.selector.inspection import (
    SelectorInspection,
    SelectorLeafInspection,
    SelectorModeInspection,
    SelectorNodeInspection,
)
from .hydra.selector.learner import HybridDecisionTreeLearner
from .hydra.selector.model import (
    HybridDecisionTreeModel,
    ModePredictionResult,
)
from .hydra.selector.runtime import StatefulHybridDecisionTreeSelector
from .hydra.simulation import StateTraceComparison, compare_state_traces
from .io import (
    save_traces_csv,
    save_traces_parquet,
    trace_to_polars,
    traces_to_polars,
)
from .plotting import plot_phase, plot_trace
from .simulator import (
    AmbiguousTransitionError,
    HybridSimulationError,
    InvalidEventSurfaceValueError,
    SimulationProgressError,
    SurfaceEntryError,
    generate_traces,
    simulate,
)

__all__ = (
    "AmbiguousTransitionError",
    "BenchmarkSpec",
    "ContinuousDynamics",
    "CrossingDirection",
    "Event",
    "EventSurface",
    "EventSurfaceFunction",
    "FlowFunction",
    "HyDRALearner",
    "HyDRAModel",
    "HyDRATraceSchema",
    "HybridDecisionTreeLearner",
    "HybridDecisionTreeModel",
    "HybridSimulationError",
    "HybridSystem",
    "Input",
    "InputStream",
    "InvalidEventSurfaceValueError",
    "Location",
    "LogCallback",
    "ModePredictionResult",
    "Parameters",
    "PlotCallback",
    "Reset",
    "ResetFunction",
    "SelectorEvaluationReport",
    "SelectorFeatureConfig",
    "SelectorInspection",
    "SelectorLeafInspection",
    "SelectorModeInspection",
    "SelectorNodeInspection",
    "SimulationProgressError",
    "StateTraceComparison",
    "StatefulHybridDecisionTreeSelector",
    "SurfaceEntryError",
    "SurfaceEntryPolicy",
    "Trace",
    "Transition",
    "all_specs",
    "benchmarks",
    "bouncing_ball",
    "compare_state_traces",
    "evaluate_selector_autoregressive",
    "evaluate_selector_oracle",
    "generate_traces",
    "hybrid_oscillator",
    "impact_input_stream",
    "impact_oscillator",
    "mode_cycle",
    "pid_controlled_plant",
    "piecewise_affine",
    "plot_phase",
    "plot_trace",
    "registry",
    "relay_integrator",
    "save_traces_csv",
    "save_traces_parquet",
    "simulate",
    "switched_linear",
    "tank_valves",
    "thermostat",
    "thermostat_target_stream",
    "time_forced_switch",
    "time_varying_event_surface",
    "time_varying_input_stream",
    "trace_to_polars",
    "traces_to_polars",
)
