"""Hybrid-system simulation and identification."""

from . import benchmarks
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
    Trace,
    Transition,
)
from .hydra.callbacks import LogCallback, PlotCallback
from .hydra.learner import HyDRALearner
from .hydra.model import HyDRAModel
from .hydra.schema import HyDRATraceSchema
from .hydra.selector.config import SelectorFeatureConfig
from .hydra.selector.learner import HybridDecisionTreeLearner
from .hydra.selector.model import (
    HybridDecisionTreeModel,
    ModePredictionResult,
)
from .hydra.simulation import StateTraceComparison, compare_state_traces
from .io import (
    save_traces_csv,
    save_traces_parquet,
    trace_to_polars,
    traces_to_polars,
)
from .plotting import plot_phase, plot_trace
from .simulator import generate_traces, simulate

__all__ = (
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
    "HybridSystem",
    "Input",
    "InputStream",
    "Location",
    "LogCallback",
    "ModePredictionResult",
    "Parameters",
    "PlotCallback",
    "Reset",
    "ResetFunction",
    "SelectorFeatureConfig",
    "StateTraceComparison",
    "Trace",
    "Transition",
    "benchmarks",
    "compare_state_traces",
    "generate_traces",
    "plot_phase",
    "plot_trace",
    "save_traces_csv",
    "save_traces_parquet",
    "simulate",
    "trace_to_polars",
    "traces_to_polars",
)
