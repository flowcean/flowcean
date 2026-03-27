"""Hybrid system benchmark suite."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from flowcean.ode import HybridSystem, InputStream

from .bouncing_ball import bouncing_ball
from .hybrid_oscillator import hybrid_oscillator
from .impact_oscillator import impact_oscillator
from .mode_cycle import mode_cycle
from .pid_controlled_plant import pid_controlled_plant
from .piecewise_affine import piecewise_affine
from .relay_integrator import relay_integrator
from .switched_linear import switched_linear
from .tank_valves import tank_valves
from .thermostat import thermostat
from .time_forced_switch import time_forced_switch
from .time_varying_guard import time_varying_guard


@dataclass(frozen=True)
class BenchmarkSpec:
    """Metadata and factory for a benchmark system."""

    name: str
    factory: Callable[[], HybridSystem]
    tags: tuple[str, ...]
    description: str
    t_span: tuple[float, float]
    input_stream: InputStream | None = None


def _thermostat_input_stream(t: float) -> np.ndarray:
    target = 22.0 + 0.8 * np.sin(0.7 * t)
    return np.array([target], dtype=float)


def _time_guard_input_stream(t: float) -> np.ndarray:
    value = 0.5 * np.sin(t) + 0.15 * np.sin(2.3 * t)
    return np.array([value], dtype=float)


def _impact_input_stream(t: float) -> np.ndarray:
    value = 0.5 * np.sin(1.5 * t) + 0.2 * np.sin(0.2 * t)
    return np.array([value], dtype=float)


def registry() -> dict[str, BenchmarkSpec]:
    """Return the benchmark registry keyed by name."""
    specs = [
        BenchmarkSpec(
            name="Bouncing Ball",
            factory=bouncing_ball,
            tags=("reset", "impact", "nonlinear"),
            description="Ballistic motion with velocity reset on impact.",
            t_span=(0.0, 10.0),
        ),
        BenchmarkSpec(
            name="Thermostat",
            factory=thermostat,
            tags=("hysteresis", "threshold", "switching"),
            description="Two-mode thermostat with temperature thresholds.",
            t_span=(0.0, 10.0),
            input_stream=_thermostat_input_stream,
        ),
        BenchmarkSpec(
            name="Hybrid Oscillator",
            factory=hybrid_oscillator,
            tags=("oscillator", "piecewise", "damping"),
            description="Oscillator with side-dependent damping.",
            t_span=(0.0, 15.0),
        ),
        BenchmarkSpec(
            name="Switched Linear",
            factory=switched_linear,
            tags=("linear", "threshold", "switching"),
            description="Switching linear dynamics by state threshold.",
            t_span=(0.0, 10.0),
        ),
        BenchmarkSpec(
            name="Relay Integrator",
            factory=relay_integrator,
            tags=("relay", "hysteresis", "control"),
            description="Relay-controlled integrator with hysteresis.",
            t_span=(0.0, 20.0),
        ),
        BenchmarkSpec(
            name="Time-Varying Guard",
            factory=time_varying_guard,
            tags=("time", "guard", "switching"),
            description="Time-varying guard surface induces switching.",
            t_span=(0.0, 20.0),
            input_stream=_time_guard_input_stream,
        ),
        BenchmarkSpec(
            name="Time-Forced Switch",
            factory=time_forced_switch,
            tags=("time", "periodic", "switching"),
            description="Periodic time-driven mode switching.",
            t_span=(0.0, 5.0),
        ),
        BenchmarkSpec(
            name="Piecewise Affine",
            factory=piecewise_affine,
            tags=("affine", "multidim", "threshold"),
            description="Piecewise affine dynamics with linear guard.",
            t_span=(0.0, 20.0),
        ),
        BenchmarkSpec(
            name="Impact Oscillator",
            factory=impact_oscillator,
            tags=("impact", "time", "reset"),
            description="Oscillator with periodic forcing and impacts.",
            t_span=(0.0, 20.0),
            input_stream=_impact_input_stream,
        ),
        BenchmarkSpec(
            name="PID-Controlled Plant",
            factory=pid_controlled_plant,
            tags=("control", "pid", "saturation"),
            description="PID-controlled plant with actuator saturation.",
            t_span=(0.0, 20.0),
        ),
        BenchmarkSpec(
            name="Tank Valves",
            factory=tank_valves,
            tags=("flow", "valves", "nonlinear"),
            description="Two-tank system with valve-controlled flow.",
            t_span=(0.0, 5.0),
        ),
        BenchmarkSpec(
            name="Mode Cycle",
            factory=lambda: mode_cycle(modes=6, dimension=3, dwell_time=0.4),
            tags=("scalable", "time", "multimode"),
            description="Scalable cycle of linear modes with clock resets.",
            t_span=(0.0, 10.0),
        ),
    ]

    return {spec.name: spec for spec in specs}


def all_specs() -> Sequence[BenchmarkSpec]:
    """Return all benchmarks in deterministic order."""
    return list(registry().values())


__all__ = [
    "BenchmarkSpec",
    "all_specs",
    "bouncing_ball",
    "hybrid_oscillator",
    "impact_oscillator",
    "mode_cycle",
    "pid_controlled_plant",
    "piecewise_affine",
    "registry",
    "relay_integrator",
    "switched_linear",
    "tank_valves",
    "thermostat",
    "time_forced_switch",
    "time_varying_guard",
]
