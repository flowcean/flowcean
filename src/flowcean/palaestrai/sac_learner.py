import numpy as np
from harl.sac.brain import SACBrain
from palaestrai.agent.objective import Objective
from typing_extensions import override

from flowcean.core.learner import ActiveLearner
from flowcean.core.model import Model
from flowcean.core.strategies.active import Action, Interface, Observation
from flowcean.palaestrai.sac_model import SACModel
from flowcean.palaestrai.util import (
    convert_to_actuator_informations,
    convert_to_reward_informations,
    convert_to_sensor_informations,
    filter_action,
    filter_observation,
)

MODEL_ID = "SACModel"


class SACLearner(ActiveLearner):
    model: SACModel
    brain: SACBrain
    objective: Objective
    objectives: list[float]
    rewards: list[list[Interface]]
    actuator_ids: list[str]
    sensor_ids: list[str]
    action: Action
    observation: Observation

    def __init__(self, actuator_ids: list[str], sensor_ids: list[str]) -> None:
        self.actuator_ids = actuator_ids
        self.sensor_ids = sensor_ids
        self.objectives = []
        self.rewards = []

    def load(self, action: Action, observation: Observation) -> None:
        self.action = filter_action(action, self.actuator_ids)
        self.observation = filter_observation(observation, self.sensor_ids)

        self.brain = SACBrain()
        self.brain._seed = 0  # noqa: SLF001
        self.brain._sensors = convert_to_sensor_informations(self.observation)  # noqa: SLF001
        self.brain._actuators = convert_to_actuator_informations(self.action)  # noqa: SLF001
        self.brain.setup()

        self.model = SACModel(
            self.action,
            self.observation,
            self.brain.thinking(MODEL_ID, None),
        )

    @override
    def learn_active(self, action: Action, observation: Observation) -> Model:
        filtered_action = filter_action(action, self.actuator_ids)
        filtered_observation = filter_observation(observation, self.sensor_ids)
        rewards = convert_to_reward_informations(observation)

        self.brain.memory.append(
            muscle_uid=MODEL_ID,
            sensor_readings=convert_to_sensor_informations(
                filtered_observation,
            ),
            actuator_setpoints=convert_to_actuator_informations(
                filtered_action,
            ),
            rewards=rewards,
            done=False,
        )
        objective = np.array(
            [self.objective.internal_reward(self.brain.memory)],
        )
        self.brain.memory.append(MODEL_ID, objective=objective)
        update = self.brain.thinking(MODEL_ID, self.model.data_for_brain)
        self.model.update(update)

        self.rewards.append(observation.rewards)
        self.objectives.append(objective[0])
        return self.model

    @override
    def propose_action(self, observation: Observation) -> Action:
        filtered = filter_observation(observation, self.sensor_ids)
        return self.model.predict(filtered)
