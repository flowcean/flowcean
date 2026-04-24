import importlib
import sys
from pathlib import Path

import numpy as np

from flowcean.ode import simulate

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

impact_oscillator = importlib.import_module(
    "examples.hybrid_systems.benchmarks.impact_oscillator",
).impact_oscillator
time_varying_guard = importlib.import_module(
    "examples.hybrid_systems.benchmarks.time_varying_guard",
).time_varying_guard


def no_input(_t: float) -> np.ndarray:
    return np.array([], dtype=float)


def test_impact_oscillator_forcing_parameters_affect_flow() -> None:
    low = impact_oscillator(forcing=0.0, forcing_freq=1.0)
    high = impact_oscillator(forcing=2.0, forcing_freq=1.0)
    slow = impact_oscillator(forcing=2.0, forcing_freq=1.0)
    fast = impact_oscillator(forcing=2.0, forcing_freq=2.0)
    state = np.array([0.5, 0.0], dtype=float)

    low_dx = low.locations["oscillate"].dynamics.flow(
        0.25,
        state,
        low.params,
        no_input,
    )
    high_dx = high.locations["oscillate"].dynamics.flow(
        0.25,
        state,
        high.params,
        no_input,
    )
    slow_dx = slow.locations["oscillate"].dynamics.flow(
        0.5,
        state,
        slow.params,
        no_input,
    )
    fast_dx = fast.locations["oscillate"].dynamics.flow(
        0.5,
        state,
        fast.params,
        no_input,
    )

    assert high_dx[1] != low_dx[1]
    assert fast_dx[1] != slow_dx[1]


def test_time_varying_guard_parameters_affect_guard_surface() -> None:
    low = time_varying_guard(amplitude=0.0, frequency=1.0)
    high = time_varying_guard(amplitude=2.0, frequency=1.0)
    slow = time_varying_guard(amplitude=2.0, frequency=1.0)
    fast = time_varying_guard(amplitude=2.0, frequency=2.0)
    state = np.array([0.0, 0.0], dtype=float)

    low_value = low.transitions[0].guard.fn(
        0.25,
        state,
        low.params,
        no_input,
    )
    high_value = high.transitions[0].guard.fn(
        0.25,
        state,
        high.params,
        no_input,
    )
    slow_value = slow.transitions[0].guard.fn(
        0.5,
        state,
        slow.params,
        no_input,
    )
    fast_value = fast.transitions[0].guard.fn(
        0.5,
        state,
        fast.params,
        no_input,
    )

    assert high_value != low_value
    assert fast_value != slow_value


def test_benchmarks_simulate_without_external_input_stream() -> None:
    for system in [impact_oscillator(), time_varying_guard()]:
        trace = simulate(system, t_span=(0.0, 0.1), sample_times=[0.0, 0.1])

        assert trace.t.tolist() == [0.0, 0.1]
