import logging
from pathlib import Path
from typing import Any, override

import numpy as np
import polars as pl
from environment import Action, Actuator, Observation, Reward, Sensor
from harl.sac.brain import SACBrain
from harl.sac.muscle import SACMuscle
from palaestrai.agent.actuator_information import ActuatorInformation
from palaestrai.agent.objective import Objective
from palaestrai.agent.reward_information import RewardInformation
from palaestrai.agent.sensor_information import SensorInformation
from palaestrai.types.space import Space

from flowcean.core.learner import ActiveLearner
from flowcean.core.model import Model

LOG = logging.getLogger("flowcean.sac_learner")


class SACModel(Model):
    best_action: float

    def __init__(
        self,
        action: Action,
        observation: Observation,
        model: Any,
        start_steps: int = 10000,
    ) -> None:
        self._action = action
        self._observation = observation
        self.muscle = SACMuscle(start_steps)
        self.muscle.update(model)
        self.data_for_brain = {}

    @override
    def predict(self, input_features: pl.DataFrame) -> pl.DataFrame:
        actuators_available = convert_to_actuator_informations(self._action)
        sensors = convert_to_sensor_informations(
            Observation(
                sensors=[
                    Sensor(
                        value=python_literal_to_basic_value(c.max()),
                        uid=c.name,
                        space=get_space_from_uid(
                            c.name, self._observation.sensors
                        ),
                    )
                    for c in input_features.iter_columns()
                ],
                rewards=[],
            )
        )
        actuators, self.data_for_brain = self.muscle.propose_actions(
            sensors, actuators_available
        )

        return pl.DataFrame(
            {
                act.uid: act.value.item()
                for act in actuators
                if (
                    act.value is not None
                    and not isinstance(act.value, int | float)
                )
            },
        )

    def update(self, update: Any) -> None:
        if update is not None:
            self.muscle.update(update)

    @override
    def save(self, path: Path) -> None:
        raise NotImplementedError

    @override
    def load(self, path: Path) -> None:
        raise NotImplementedError

    def summary(self) -> str:
        return str(self.muscle._model)  # noqa: SLF001


class SACLearner(ActiveLearner):
    def __init__(
        self,
        actuator_ids: list[str],
        sensor_ids: list[str],
        objective: Objective,
    ) -> None:
        self.act_ids: list[str] = actuator_ids
        self.sen_ids: list[str] = sensor_ids
        self._objective = objective
        self.model: SACModel
        self.action: Action
        self.observation: Observation
        self.rewards: list[list[Reward]] = []
        self.objectives = []
        self._model_id = (
            "SACModel"  # required by palaestrAI; name is arbitrary
        )

    def load(self, action: Action, observation: Observation) -> None:
        self.action = filter_action(action, self.act_ids)
        self.observation = filter_observation(observation, self.sen_ids)

        LOG.info("Loading SAC brain from palaestrAI.")
        self.brain = SACBrain()
        self.brain._seed = 0  # noqa: SLF001
        self.brain._sensors = convert_to_sensor_informations(  # noqa: SLF001
            self.observation
        )
        self.brain._actuators = convert_to_actuator_informations(self.action)  # noqa: SLF001
        self.brain.setup()
        LOG.info("Loading SAC muscle from palaestrAI.")
        self.model = SACModel(
            self.action,
            self.observation,
            self.brain.thinking(self._model_id, None),
        )

    @override
    def learn_active(
        self,
        action: Action,
        observation: Observation,
    ) -> Model:
        rewards = convert_to_reward_informations(observation)

        self.brain.memory.append(
            muscle_uid=self._model_id,
            sensor_readings=convert_to_sensor_informations(
                filter_observation(observation, self.sen_ids)
            ),
            actuator_setpoints=convert_to_actuator_informations(
                filter_action(action, self.act_ids)
            ),
            rewards=rewards,
            done=False,
        )
        objective = np.array(
            [self._objective.internal_reward(self.brain.memory)]
        )
        self.brain.memory.append(
            self._model_id,
            objective=objective,
        )
        update = self.brain.thinking(self._model_id, self.model.data_for_brain)
        self.model.update(update)

        self.rewards.append(observation.rewards)
        self.objectives.append(objective[0])
        return self.model

    @override
    def propose_action(self, observation: Observation) -> Action:
        filtered = filter_observation(observation, self.sen_ids)

        df = pl.DataFrame({s.uid: s.value for s in filtered.sensors})
        prediction_df = self.model.predict(df)

        return Action(
            actuators=[
                Actuator(
                    value=python_literal_to_basic_value(c.max()),
                    uid=c.name,
                    space=get_space_from_uid(c.name, self.action.actuators),
                )
                for c in prediction_df.iter_columns()
            ]
        )


def python_literal_to_basic_value(value: Any) -> int | float:
    try:
        value = int(value)
    except TypeError:
        pass
    else:
        return value
    try:
        value = float(value)
    except TypeError:
        pass
    else:
        return value

    msg = "Cannot convert value to basic type"
    raise TypeError(msg)


def filter_action(action: Action, available_ids: list[str]) -> Action:
    return Action(
        actuators=[a for a in action.actuators if a.uid in available_ids]
    )


def filter_observation(
    observation: Observation, available_ids: list[str]
) -> Observation:
    return Observation(
        sensors=[s for s in observation.sensors if s.uid in available_ids],
        rewards=observation.rewards,
    )


def convert_to_actuator_informations(
    action: Action,
) -> list[ActuatorInformation]:
    infos = []
    for obj in action.actuators:
        space = Space.from_string(obj.space)
        infos.append(
            ActuatorInformation(
                value=np.array(obj.value, dtype=space.dtype)
                if obj.value is not None
                else None,
                uid=obj.uid,
                space=space,
            )
        )
    return infos


def convert_to_sensor_informations(
    observation: Observation,
) -> list[SensorInformation]:
    infos = []
    for obj in observation.sensors:
        space = Space.from_string(obj.space)
        infos.append(
            SensorInformation(
                value=np.array(obj.value, dtype=space.dtype)
                if obj.value is not None
                else None,
                uid=obj.uid,
                space=space,
            )
        )
    return infos


def convert_to_reward_informations(
    observation: Observation,
) -> list[RewardInformation]:
    infos = []
    for obj in observation.rewards:
        space = Space.from_string(obj.space)
        infos.append(
            RewardInformation(
                value=np.array(obj.value, dtype=space.dtype)
                if obj.value is not None
                else None,
                uid=obj.uid,
                space=space,
            )
        )
    return infos


def get_space_from_uid(
    name: str, entries: list[Actuator] | list[Sensor]
) -> str:
    return next(o.space for o in entries if name == o.uid)
