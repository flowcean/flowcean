import logging
import random
from dataclasses import dataclass
from math import nan
from pathlib import Path
from typing import Self, override

import polars as pl

import flowcean.cli
from flowcean.core import ActiveEnvironment, ActiveLearner, Model
from flowcean.strategies.active import StopLearning, learn_active

logger = logging.getLogger(__name__)

Action = float


@dataclass
class ReinforcementObservation:
    reward: float
    sensor: float


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
    def load(self) -> Self:
        return self

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
    def observe(self) -> pl.DataFrame:
        return ReinforcementObservation(
            reward=self._calculate_reward(),
            sensor=self.state,
        )

    def _calculate_reward(self) -> float:
        if self.last_action is None:
            return nan
        return self.max_value - abs(self.state - self.last_action)


class MyModel(Model):
    best_action: float

    def __init__(self, best_action: float) -> None:
        self.best_action = best_action

    @override
    def predict(self, input_features: pl.DataFrame) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "action": [
                    self.best_action for _ in range(len(input_features))
                ],
            },
        )

    @override
    def save(self, path: Path) -> None:
        raise NotImplementedError

    @override
    def load(self, path: Path) -> None:
        raise NotImplementedError


class MyLearner(ActiveLearner[Action, ReinforcementObservation]):
    model: MyModel
    rewards: list[float]

    def __init__(self) -> None:
        self.model = MyModel(best_action=nan)
        self.rewards = []

    @override
    def learn_active(
        self,
        action: Action,
        observation: ReinforcementObservation,
    ) -> Model:
        _ = observation.sensor
        self.model = MyModel(best_action=random.random())
        self.rewards.append(observation.reward)
        return self.model

    @override
    def propose_action(self, observation: ReinforcementObservation) -> Action:
        sensor = observation.sensor
        action = self.model.predict(pl.DataFrame({"sensor": [sensor]}))
        return action["action"][0]


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
