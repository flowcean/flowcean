"""Hybrid boiler example using the new hybrid system simulator."""

from collections.abc import Mapping

import numpy as np
import polars as pl

from flowcean.cli import initialize
from flowcean.core import evaluate_offline, learn_offline
from flowcean.ode import Guard, HybridSystem, Mode, Transition, simulate
from flowcean.polars import DataFrame, SlidingWindow, TrainTestSplit
from flowcean.sklearn import (
    MeanAbsoluteError,
    MeanSquaredError,
    RegressionTree,
)
from flowcean.utils import initialize_random


def _target_schedule(
    times: np.ndarray,
    *,
    rng: np.random.Generator,
    change_probability: float,
    minimum: float,
    maximum: float,
) -> np.ndarray:
    targets = np.empty_like(times)
    current = rng.uniform(minimum, maximum)
    for idx in range(times.size):
        if rng.random() < change_probability:
            current = rng.uniform(minimum, maximum)
        targets[idx] = current
    return targets


def main() -> None:
    initialize()
    initialize_random(seed=42)

    rng = np.random.default_rng(42)
    sample_dt = 0.01
    steps = 10_000
    times = np.linspace(0.0, sample_dt * (steps - 1), steps)
    targets = _target_schedule(
        times,
        rng=rng,
        change_probability=0.002,
        minimum=30.0,
        maximum=60.0,
    )

    def target_at(t: float) -> float:
        return float(np.interp(t, times, targets))

    def heating_flow(
        _: float,
        __: np.ndarray,
        params: Mapping[str, float],
    ) -> np.ndarray:
        return np.array([params["heating_rate"]], dtype=float)

    def cooling_flow(
        _: float,
        __: np.ndarray,
        params: Mapping[str, float],
    ) -> np.ndarray:
        return np.array([-params["cooling_rate"]], dtype=float)

    def guard_high(
        t: float,
        state: np.ndarray,
        params: Mapping[str, float],
    ) -> float:
        target = target_at(t) + params["hysteresis"] * 0.5
        return state[0] - target

    def guard_low(
        t: float,
        state: np.ndarray,
        params: Mapping[str, float],
    ) -> float:
        target = target_at(t) - params["hysteresis"] * 0.5
        return state[0] - target

    heating = Mode(name="heating", flow=heating_flow)
    cooling = Mode(name="cooling", flow=cooling_flow)
    transitions = [
        Transition(
            source="heating",
            target="cooling",
            guard=Guard(
                name="too_hot",
                fn=guard_high,
                direction=1,
                terminal=True,
            ),
        ),
        Transition(
            source="cooling",
            target="heating",
            guard=Guard(
                name="too_cold",
                fn=guard_low,
                direction=-1,
                terminal=True,
            ),
        ),
    ]

    system = HybridSystem(
        modes={"heating": heating, "cooling": cooling},
        transitions=transitions,
        initial_mode="heating",
        initial_state=np.array([30.0], dtype=float),
        params={"heating_rate": 0.5, "cooling_rate": 0.1, "hysteresis": 1.0},
    )

    trace = simulate(system, t_span=(times[0], times[-1]), sample_times=times)
    data = pl.DataFrame(
        {
            "temperature": trace.x[:, 0],
            "target": targets,
        },
    )

    environment = DataFrame(data)
    train, test = TrainTestSplit(ratio=0.8).split(environment)

    train = train | SlidingWindow(window_size=10)
    test = test | SlidingWindow(window_size=10)

    learner = RegressionTree(max_depth=5)

    inputs = [f"temperature_{i}" for i in range(10)] + [
        f"target_{i}" for i in range(9)
    ]
    outputs = ["temperature_9"]
    model = learn_offline(
        train,
        learner,
        inputs,
        outputs,
    )
    report = evaluate_offline(
        model,
        test,
        inputs,
        outputs,
        [MeanAbsoluteError(), MeanSquaredError()],
    )
    print(report)


if __name__ == "__main__":
    main()
