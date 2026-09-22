"""Tests for the canonical hybrid-system public namespace."""

import importlib.util

from flowcean import hybrid
from flowcean.hybrid import benchmarks, hydra
from flowcean.hybrid.hybrid_system import HybridSystem, Location, Trace
from flowcean.hybrid.hydra import selector
from flowcean.hybrid.hydra.learner import HyDRALearner
from flowcean.hybrid.hydra.model import HyDRAModel
from flowcean.hybrid.hydra.schema import HyDRATraceSchema
from flowcean.hybrid.simulator import generate_traces, simulate


def test_hybrid_facade_exports_simulation_api() -> None:
    """The canonical facade exports implementations from its leaf modules."""
    assert hybrid.HybridSystem is HybridSystem
    assert hybrid.Location is Location
    assert hybrid.Trace is Trace
    assert hybrid.simulate is simulate
    assert hybrid.generate_traces is generate_traces


def test_hybrid_facade_exports_identification_api() -> None:
    """The canonical facade exports the HyDRA API from its nested package."""
    assert hybrid.HyDRALearner is HyDRALearner
    assert hybrid.HyDRAModel is HyDRAModel
    assert hybrid.HyDRATraceSchema is HyDRATraceSchema


def test_hybrid_facade_exports_benchmark_api() -> None:
    """Benchmark factories are available from the canonical facade."""
    for name in benchmarks.__all__:
        assert getattr(hybrid, name) is getattr(benchmarks, name)


def test_hybrid_facades_export_selector_api() -> None:
    """Selector types and helpers are available from both hybrid facades."""
    for name in selector.__all__:
        implementation = getattr(selector, name)
        assert getattr(hydra, name) is implementation
        assert getattr(hybrid, name) is implementation


def test_removed_namespaces_have_no_module_specs() -> None:
    """The clean namespace break leaves no compatibility packages."""
    assert importlib.util.find_spec("flowcean.ode") is None
    assert importlib.util.find_spec("flowcean.hydra") is None
