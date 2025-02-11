#!/usr/bin/env python
# /// script
# dependencies = [
#     "flowcean",
# ]
#
# [tool.uv.sources]
# flowcean = { path = "../../", editable = true }
# ///

import logging
import random
from math import nan
from pathlib import Path

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
        return self.max_value - abs(self.state - self.last_action["sensor"][0])


class MyModel(Model):
    best_action: float

    def __init__(self, best_action: float) -> None:
        self.best_action = best_action

    @override
    def predict(self, input_features: pl.LazyFrame) -> pl.LazyFrame:
        return pl.DataFrame(
            {
                "action": [
                    self.best_action
                    for _ in range(len(input_features.collect()))
                ],
            },
        ).lazy()

    @override
    def save(self, path: Path) -> None:
        _ = path
        raise NotImplementedError

    @override
    def load(self, path: Path) -> None:
        _ = path
        raise NotImplementedError


class MyLearner(ActiveLearner):
    model: MyModel
    rewards: list[float]

    def __init__(self) -> None:
        self.model = MyModel(best_action=nan)
        self.rewards = []

    @override
    def learn_active(
        self,
        action: pl.DataFrame,
        observation: pl.DataFrame,
    ) -> Model:
        _ = action
        self.model = MyModel(best_action=random.random())
        self.rewards.append(observation["reward"][0])
        return self.model

    @override
    def propose_action(self, observation: pl.DataFrame) -> pl.DataFrame:
        sensor = observation["sensor"]
        action = self.model.predict(pl.DataFrame({"sensor": sensor}).lazy())
        return action.collect()["action"][0]


def main() -> None:
    flowcean.cli.initialize_logging()

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
