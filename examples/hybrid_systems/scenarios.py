"""Simulation settings for the hybrid systems example gallery."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from flowcean.hybrid import HybridSystem, InputStream
from flowcean.hybrid.benchmarks import (
    bouncing_ball,
    buck_converter,
    hybrid_oscillator,
    impact_oscillator,
    mode_cycle,
    pid_controlled_plant,
    piecewise_affine,
    relay_integrator,
    switched_linear,
    tank_valves,
    thermostat,
    time_forced_switch,
    time_varying_event_surface,
    wind_turbine,
)


@dataclass(frozen=True)
class Scenario:
    """One gallery model with its simulation settings and display metadata."""

    name: str
    factory: Callable[[], HybridSystem]
    tags: tuple[str, ...]
    description: str
    t_span: tuple[float, float]
    input_stream: InputStream | None = None


def thermostat_target_temperature(t: float) -> np.ndarray:
    return np.array([22.0 + 0.8 * np.sin(0.7 * t)])


def impact_force(t: float) -> np.ndarray:
    return np.array([0.5 * np.sin(1.5 * t) + 0.2 * np.sin(0.2 * t)])


def event_surface_threshold(t: float) -> np.ndarray:
    return np.array([0.5 * np.sin(t) + 0.15 * np.sin(2.3 * t)])


def pid_reference_and_rate(t: float) -> np.ndarray:
    return np.array([2.0 * np.sin(t), 2.0 * np.cos(t)])


def wind_speed(t: float) -> np.ndarray:
    return np.array([11.0 - 4.0 * np.cos(2.0 * np.pi * t / 120.0)])


WIND_TURBINE = Scenario(
    name="Wind Turbine",
    factory=wind_turbine,
    tags=("wind", "nonlinear", "control", "hysteresis", "input"),
    description=(
        "Running turbine with five torque-control regimes and pitch control."
    ),
    t_span=(0.0, 120.0),
    input_stream=wind_speed,
)

SCENARIOS = (
    Scenario(
        "Bouncing Ball",
        bouncing_ball,
        ("reset", "impact", "nonlinear"),
        "Ballistic motion with velocity reset on impact.",
        (0.0, 3.0),
    ),
    Scenario(
        "Thermostat",
        thermostat,
        ("hysteresis", "threshold", "switching"),
        "Two-location thermostat with temperature thresholds.",
        (0.0, 10.0),
        thermostat_target_temperature,
    ),
    Scenario(
        "Hybrid Oscillator",
        hybrid_oscillator,
        ("oscillator", "piecewise", "damping"),
        "Oscillator with side-dependent damping.",
        (0.0, 15.0),
    ),
    Scenario(
        "Switched Linear",
        switched_linear,
        ("linear", "threshold", "switching"),
        "Switching linear dynamics by state threshold.",
        (0.0, 10.0),
    ),
    Scenario(
        "Relay Integrator",
        relay_integrator,
        ("relay", "hysteresis", "control"),
        "Relay-controlled integrator with hysteresis.",
        (0.0, 20.0),
    ),
    Scenario(
        "Time-Varying Event Surface",
        time_varying_event_surface,
        ("time", "event-surface", "switching"),
        "Time-varying event surface induces switching.",
        (0.0, 20.0),
        event_surface_threshold,
    ),
    Scenario(
        "Time-Forced Switch",
        time_forced_switch,
        ("time", "periodic", "switching"),
        "Periodic time-driven location switching.",
        (0.0, 5.0),
    ),
    Scenario(
        "Piecewise Affine",
        piecewise_affine,
        ("affine", "multidim", "threshold"),
        "Piecewise affine dynamics with a linear event surface.",
        (0.0, 20.0),
    ),
    Scenario(
        "Impact Oscillator",
        impact_oscillator,
        ("impact", "time", "reset"),
        "Oscillator with periodic forcing and impacts.",
        (0.0, 20.0),
        impact_force,
    ),
    Scenario(
        "PID-Controlled Plant",
        pid_controlled_plant,
        ("control", "pid", "saturation"),
        "PID-controlled plant with actuator saturation.",
        (0.0, 20.0),
        pid_reference_and_rate,
    ),
    Scenario(
        "Tank Valves",
        tank_valves,
        ("flow", "valves", "nonlinear"),
        "Two-tank system with valve-controlled flow.",
        (0.0, 300.0),
    ),
    Scenario(
        "Location Cycle",
        lambda: mode_cycle(modes=6, dimension=3, dwell_time=0.4),
        ("scalable", "time", "multimode"),
        "Scalable cycle of linear locations with clock resets.",
        (0.0, 10.0),
    ),
    Scenario(
        "Buck Converter",
        buck_converter,
        ("power-electronics", "hysteresis", "affine", "diode"),
        "Hysteretic buck converter with ideal-diode current blocking.",
        (0.0, 0.02),
    ),
    WIND_TURBINE,
)
