"""Canonical one-tank system and deterministic simulation data."""

import numpy as np
import polars as pl

from flowcean.hybrid import (
    ContinuousDynamics,
    HybridSystem,
    InputStream,
    Location,
    Parameters,
    simulate,
    trace_to_polars,
)


def one_tank_system() -> HybridSystem:
    """Build the one-tank system as a single-location hybrid system."""

    def tank_flow(
        t: float,
        state: np.ndarray,
        parameters: Parameters,
        _input_stream: InputStream,
    ) -> np.ndarray:
        pump_voltage = max(0.0, np.sin(2.0 * np.pi * t / 10.0))
        level_rate = (
            parameters["inflow_rate"] * pump_voltage
            - parameters["outflow_rate"] * np.sqrt(state[0])
        ) / parameters["area"]
        return np.array([level_rate], dtype=float)

    tank = Location(
        ContinuousDynamics(tank_flow, label="tank_flow"),
        label="tank",
    )
    return HybridSystem(
        locations=[tank],
        transitions=[],
        initial_location=tank,
        initial_state=np.array([1.0], dtype=float),
        parameters={
            "area": 5.0,
            "outflow_rate": 0.5,
            "inflow_rate": 2.0,
        },
    )


def simulate_one_tank() -> pl.DataFrame:
    """Simulate a fixed one-tank trace and return its time and level data."""
    trace = simulate(
        one_tank_system(),
        t_span=(0.0, 25.0),
        sample_dt=0.1,
    )
    return trace_to_polars(trace, state_names=("h",)).select("t", "h")
