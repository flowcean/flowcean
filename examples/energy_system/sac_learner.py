# from harl.sac import SAC
import random
from math import nan
from pathlib import Path
from typing import Self, override

import polars as pl
from environment import Actions, Actuator, Observations, Reward, Sensor

from flowcean.core import ActiveEnvironment, ActiveLearner, Model

#from flowcean.strategies.active import StopLearning, learn_active


class SACModel(Model):
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


class SACLearner(ActiveLearner[Actions, Observations]):
    model: SACModel
    rewards: list[float]

    def __init__(self) -> None:
        self.model = SACModel(best_action=nan)
        self.rewards = []

    @override
    def learn_active(
        self,
        action: Actions,
        observation: Observations,
    ) -> Model:
        _ = observation.sensor
        self.model = SACModel(best_action=random.random())
        self.rewards.append(observation.reward)
        return self.model

    @override
    def propose_action(self, observation: Observations) -> Actions:
        # call propose actions from the muscle
        sensor = observation.sensor
        action = self.model.predict(pl.DataFrame({"sensor": [sensor]}))
        return action["action"][0]
