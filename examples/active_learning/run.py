#!/usr/bin/env python

from __future__ import annotations

import logging
import random
from math import nan

import polars as pl
from typing_extensions import override

import flowcean.cli
from flowcean.core import (
    ActiveEnvironment,
    ActiveLearner,
    Model,
    StopLearning,
    learn_active,
)

logger = logging.getLogger(__name__)


class MyEnvironment(ActiveEnvironment):
    state: float
    max_value: float
    last_action: pl.DataFrame | None
    max_num_iterations: int

    def __init__(
        self,
        initial_state: float,
        max_value: float,
        max_num_iterations: int,
    ) -> None:
        super().__init__()
        self.state = initial_state
        self.max_value = max_value
        self.last_action = None
        self.max_num_iterations = max_num_iterations

    @override
    def act(self, action: pl.DataFrame) -> None:
        self.last_action = action

    @override
    def step(self) -> None:
        self.state = random.random() * self.max_value
        self.max_num_iterations -= 1
        if self.max_num_iterations < 0:
            raise StopLearning

    @override
    def _observe(self) -> pl.LazyFrame:
        return pl.DataFrame(
            {
                "reward": self._calculate_reward(),
                "sensor": self.state,
            },
        ).lazy()

    def _calculate_reward(self) -> float:
        if self.last_action is None:
            return nan
        return self.max_value - abs(self.state - self.last_action["action"][0])


class MyModel(Model):
    best_action: float

    def __init__(self, best_action: float) -> None:
        self.best_action = best_action

    @override
    def _predict(self, input_features: pl.LazyFrame) -> pl.LazyFrame:
        return pl.DataFrame(
            {
                "action": [
                    self.best_action
                    for _ in range(len(input_features.collect()))
                ],
            },
        ).lazy()


class MyLearner(ActiveLearner):
    model: MyModel
    rewards: list[float]

    def __init__(self) -> None:
        self.model = MyModel(best_action=nan)
        self.rewards = []

    @override
    def learn_active(
        self,
        action: pl.LazyFrame,
        observation: pl.LazyFrame,
    ) -> Model:
        _ = action
        self.model = MyModel(best_action=random.random())
        self.rewards.append(
            observation.select("reward").first().collect().item(),
        )
        return self.model

    @override
    def propose_action(self, observation: pl.LazyFrame) -> pl.DataFrame:
        sensor = observation.select("sensor").first().collect().item()
        sensor_df = pl.DataFrame({"sensor": sensor})
        action = self.model.predict(sensor_df.lazy())
        return pl.concat(
            [sensor_df, action.collect()],
            how="horizontal",
        )


def main() -> None:
    flowcean.cli.initialize()

    environment = MyEnvironment(
        initial_state=0.0,
        max_value=10.0,
        max_num_iterations=1_000,
    )

    learner = MyLearner()

    model = learn_active(
        environment,
        learner,
    )
    print(model)
    print(learner.rewards)


if __name__ == "__main__":
    main()
