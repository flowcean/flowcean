"""Tests for the canonical hybrid-system public namespaces."""

import importlib
import importlib.util

from flowcean import hybrid
from flowcean.hybrid import benchmarks, hydra
from flowcean.hybrid.benchmarks import bouncing_ball
from flowcean.hybrid.hybrid_system import HybridSystem, Location, Trace
from flowcean.hybrid.hydra import HyDRALearner, HyDRAModel, selector
from flowcean.hybrid.simulator import generate_traces, simulate

EXPECTED_HYBRID_EXPORTS = (
    "AmbiguousTransitionError",
    "ContinuousDynamics",
    "CrossingDirection",
    "Event",
    "EventSurface",
    "EventSurfaceFunction",
    "FlowFunction",
    "HybridSimulationError",
    "HybridSystem",
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
    "Trace",
    "Transition",
    "benchmarks",
    "generate_traces",
    "hydra",
    "plot_phase",
    "plot_trace",
    "save_traces_csv",
    "save_traces_parquet",
    "simulate",
    "trace_to_polars",
    "traces_to_polars",
)


def test_hybrid_facade_exports_modeling_and_simulation_api() -> None:
    """The hybrid facade retains the modeling and simulation API."""
    assert hybrid.HybridSystem is HybridSystem
    assert hybrid.Location is Location
    assert hybrid.Trace is Trace
    assert hybrid.simulate is simulate
    assert hybrid.generate_traces is generate_traces
    assert hybrid.__all__ == EXPECTED_HYBRID_EXPORTS


def test_hybrid_facade_exposes_exact_nested_module_handles() -> None:
    """Benchmarks and identification remain explicit nested APIs."""
    assert hybrid.benchmarks is benchmarks
    assert hybrid.hydra is hydra
    assert hydra.selector is selector
    assert hybrid.benchmarks is importlib.import_module(
        "flowcean.hybrid.benchmarks",
    )
    assert hybrid.hydra is importlib.import_module("flowcean.hybrid.hydra")
    assert hydra.selector is importlib.import_module(
        "flowcean.hybrid.hydra.selector",
    )


def test_canonical_nested_facades_export_their_symbols() -> None:
    """Benchmarks and HyDRA types are exported only from nested facades."""
    assert benchmarks.bouncing_ball is bouncing_ball
    assert hydra.HyDRALearner is HyDRALearner
    assert hydra.HyDRAModel is HyDRAModel
    for name in selector.__all__:
        assert getattr(hydra, name) is getattr(selector, name)


def test_hybrid_facade_does_not_flatten_benchmarks_or_hydra() -> None:
    """Nested public symbols do not leak into the hybrid facade."""
    removed = (
        *benchmarks.__all__,
        *(name for name in hydra.__all__ if name != "selector"),
    )

    for name in removed:
        assert name not in hybrid.__all__
        assert not hasattr(hybrid, name)


def test_removed_namespaces_have_no_module_specs() -> None:
    """The clean namespace break leaves no compatibility packages."""
    assert importlib.util.find_spec("flowcean.ode") is None
    assert importlib.util.find_spec("flowcean.hydra") is None
