#!/usr/bin/env python

import random
from collections.abc import Iterator
from typing import Self, override

import numpy as np
import polars as pl
from numpy.typing import NDArray

from flowcean.cli.logging import initialize_logging
from flowcean.environments.hybrid_system import (
    DifferentialMode,
    HybridSystem,
    State,
)
from flowcean.environments.train_test_split import TrainTestSplit
from flowcean.learners.regression_tree import RegressionTree
from flowcean.metrics.regression import MeanAbsoluteError, MeanSquaredError
from flowcean.strategies.offline import evaluate_offline, learn_offline
from flowcean.transforms.sliding_window import SlidingWindow
from flowcean.utils.random import initialize_random


class Temperature(State):
    temperature: float

    def __init__(self, temperature: float) -> None:
        self.temperature = temperature

    @override
    def as_numpy(self) -> NDArray[np.float64]:
        return np.array([self.temperature])

    @override
    @classmethod
    def from_numpy(cls, state: NDArray[np.float64]) -> Self:
        return cls(state[0])


type TargetTemperature = float


class Heating(DifferentialMode[Temperature, TargetTemperature]):
    heating_rate: float = 0.5
    overheat_timeout: float = 1.0

    @override
    def flow(
        self,
        t: float,
        state: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        _ = t
        return np.array([self.heating_rate])

    @override
    def transition(
        self,
        i: TargetTemperature,
    ) -> DifferentialMode[Temperature, TargetTemperature]:
        if self.state.temperature > i or self.t > self.overheat_timeout:
            return Cooling(t=0.0, state=self.state)
        return self


class Cooling(DifferentialMode[Temperature, TargetTemperature]):
    cooling_rate: float = 0.1
    cooldown_timeout: float = 1.0

    @override
    def flow(
        self,
        t: float,
        state: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        _ = t
        return np.array([-self.cooling_rate])

    @override
    def transition(
        self,
        i: TargetTemperature,
    ) -> DifferentialMode[Temperature, TargetTemperature]:
        if self.state.temperature < i and self.t > self.cooldown_timeout:
            return Heating(t=0.0, state=self.state)
        return self


def randomly_changing_values(
    change_probability: float,
    minimum: float,
    maximum: float,
) -> Iterator[float]:
    value = random.uniform(minimum, maximum)
    while True:
        if random.random() < change_probability:
            value = random.uniform(minimum, maximum)
        yield value


def main() -> None:
    initialize_logging()
    initialize_random(seed=42)
    target_temperatures = (
        (0.1 * i, temperature)
        for i, temperature in enumerate(
            randomly_changing_values(
                change_probability=0.002,
                minimum=30.0,
                maximum=60.0,
            )
        )
    )

    environment = HybridSystem(
        initial_mode=Heating(t=0.0, state=Temperature(30)),
        inputs=target_temperatures,
        map_to_dataframe=lambda times, inputs, modes: pl.DataFrame(
            {
                "time": times,
                "target": inputs,
                "temperature": [mode.temperature for mode in modes],
            }
        ),
    ).load()

    data = environment.collect(10_000)
    train, test = TrainTestSplit(ratio=0.8).split(data)

    train = train.with_transform(
        SlidingWindow(window_size=10),
    )
    test = test.with_transform(
        SlidingWindow(window_size=10),
    )

    learner = RegressionTree(max_depth=5, dot_graph_export_path="tree.dot")

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
