from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from flowcean.hybrid import (
    CrossingDirection,
    EventSurface,
    Flow,
    HybridSystem,
    InputStream,
    Location,
    Parameters,
    Transition,
    simulate,
)
from flowcean.utils import initialize_random

DATA_PATH = Path(__file__).parent / "data" / "thermostat_trace.csv"


def target_input(time: float) -> np.ndarray:
    target = 20.0 + 1.0 * np.sin(time / 4.0) + 0.5 * np.sin(time / 9.0)
    return np.array([target], dtype=float)


def thermostat_system() -> HybridSystem:
    def cooling_flow(
        _t: float,
        state: np.ndarray,
        _params: Parameters,
        input_stream: InputStream,
    ) -> np.ndarray:
        target = input_stream(_t)[0]
        ambient = target - 4.0
        return np.array([0.18 * (ambient - state[0])], dtype=float)

    def heating_flow(
        _t: float,
        state: np.ndarray,
        _params: Parameters,
        input_stream: InputStream,
    ) -> np.ndarray:
        target = input_stream(_t)[0]
        heat_source = target + 4.0
        return np.array([0.35 * (heat_source - state[0])], dtype=float)

    def too_cold(
        _t: float,
        state: np.ndarray,
        _params: Parameters,
        input_stream: InputStream,
    ) -> float:
        return state[0] - input_stream(_t)[0] + 0.35

    def warm_enough(
        _t: float,
        state: np.ndarray,
        _params: Parameters,
        input_stream: InputStream,
    ) -> float:
        return state[0] - input_stream(_t)[0] - 0.35

    cooling = Location(
        Flow(cooling_flow, label="cooling_flow"),
        label="cooling",
    )
    heating = Location(
        Flow(heating_flow, label="heating_flow"),
        label="heating",
    )

    return HybridSystem(
        locations=[cooling, heating],
        transitions=[
            Transition(
                source=cooling,
                target=heating,
                event_surface=EventSurface(
                    too_cold,
                    direction=CrossingDirection.FALLING,
                    label="too_cold",
                ),
            ),
            Transition(
                source=heating,
                target=cooling,
                event_surface=EventSurface(
                    warm_enough,
                    direction=CrossingDirection.RISING,
                    label="warm_enough",
                ),
            ),
        ],
        initial_location=cooling,
        initial_state=np.array([20.5], dtype=float),
    )


def generate_trace_frame() -> pl.DataFrame:
    initialize_random(42)
    trajectory = simulate(
        thermostat_system(),
        t_span=(0.0, 40.0),
        input_stream=target_input,
    )
    return (
        trajectory.sample(
            dt=0.1, include_inputs=True, include_location_label=True
        )
        .rename(
            {"x0": "temperature", "u0": "target", "location_label": "location"}
        )
        .with_row_index("step")
        .with_columns(
            pl.col("step").cast(pl.Int64),
            (pl.col("location") == "heating").cast(pl.Int64).alias("heating"),
        )
        .select(
            ["step", "t", "location", "temperature", "target", "heating"],
        )
    )


def main() -> None:
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    generate_trace_frame().write_csv(DATA_PATH)


if __name__ == "__main__":
    main()
