"""Tests for the canonical hybrid-system public namespaces."""

import importlib
import subprocess
import sys

from flowcean import hybrid
from flowcean.hybrid import benchmarks, dainarx, famos, hydra, mmsr
from flowcean.hybrid.benchmarks import bouncing_ball, duffing_oscillator
from flowcean.hybrid.graph import build_hybrid_system_dot, render_dot_svg
from flowcean.hybrid.hybrid_system import HybridSystem, Location
from flowcean.hybrid.hydra import (
    HyDRACallback,
    HyDRAIdentificationError,
    HyDRALearner,
    HyDRAModel,
    LearnedFlow,
    LearnedFlows,
    PlotCallback,
    TraceSegment,
    selector,
)
from flowcean.hybrid.plotting import plot_locations, plot_state_space
from flowcean.hybrid.simulator import simulate
from flowcean.hybrid.trajectory import HybridTrajectory, TrajectoryPoint


def test_hybrid_facade_exports_modeling_and_simulation_api() -> None:
    assert hybrid.HybridSystem is HybridSystem
    assert hybrid.Location is Location
    assert hybrid.HybridTrajectory is HybridTrajectory
    assert hybrid.TrajectoryPoint is TrajectoryPoint
    assert hybrid.simulate is simulate
    assert hybrid.build_hybrid_system_dot is build_hybrid_system_dot
    assert hybrid.render_dot_svg is render_dot_svg
    assert hybrid.plot_locations is plot_locations
    assert hybrid.plot_state_space is plot_state_space
    assert {"TrajectoryPoint", "plot_state_space", "simulate"} <= set(
        hybrid.__all__
    )


def test_hybrid_facade_exposes_nested_module_handles() -> None:
    assert hybrid.benchmarks is benchmarks
    assert hybrid.hydra is hydra
    assert hybrid.dainarx is dainarx
    assert hybrid.famos is famos
    assert hybrid.mmsr is mmsr
    assert "mmsr" in hybrid.__all__
    assert hydra.selector is selector
    assert hybrid.benchmarks is importlib.import_module(
        "flowcean.hybrid.benchmarks",
    )
    assert hybrid.hydra is importlib.import_module("flowcean.hybrid.hydra")
    assert hybrid.dainarx is importlib.import_module("flowcean.hybrid.dainarx")
    assert hybrid.famos is importlib.import_module("flowcean.hybrid.famos")
    assert hybrid.mmsr is importlib.import_module("flowcean.hybrid.mmsr")
    assert hydra.selector is importlib.import_module(
        "flowcean.hybrid.hydra.selector",
    )


def test_hybrid_import_does_not_load_optional_symbolic_backend() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import flowcean.hybrid; "
            "assert not {'pysr', 'juliacall', 'juliapkg'} & sys.modules.keys()",
        ],
        check=True,
    )


def test_nested_facades_own_benchmark_and_identification_symbols() -> None:
    assert benchmarks.bouncing_ball is bouncing_ball
    assert benchmarks.duffing_oscillator is duffing_oscillator
    assert "duffing_oscillator" in benchmarks.__all__
    assert hydra.HyDRACallback is HyDRACallback
    assert hydra.PlotCallback is PlotCallback
    assert {"HyDRACallback", "PlotCallback"} <= set(hydra.__all__)
    assert hydra.HyDRALearner is HyDRALearner
    assert hydra.HyDRAIdentificationError is HyDRAIdentificationError
    assert hydra.HyDRAModel is HyDRAModel
    assert hydra.LearnedFlow is LearnedFlow
    assert hydra.LearnedFlows is LearnedFlows
    assert hydra.TraceSegment is TraceSegment
    assert {
        "HyDRAIdentificationError",
        "LearnedFlow",
        "LearnedFlows",
        "TraceSegment",
    } <= set(hydra.__all__)
    for name in selector.__all__:
        assert getattr(hydra, name) is getattr(selector, name)
    for name in (
        "bouncing_ball",
        "duffing_oscillator",
        "HyDRALearner",
        "HyDRAModel",
        "FamosModel",
    ):
        assert name not in hybrid.__all__
        assert not hasattr(hybrid, name)
