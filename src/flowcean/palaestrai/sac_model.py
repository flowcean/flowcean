from typing import Any

from harl.sac.muscle import SACMuscle
from typing_extensions import override

from flowcean.core.model import Model
from flowcean.core.strategies.active import Action, Observation
from flowcean.palaestrai.util import (
    convert_to_actuator_information,
    convert_to_interface,
    convert_to_sensor_information,
    filter_observation,
)


class SACModel(Model):
    action: Action
    observation: Observation
    muscle: SACMuscle
    data_for_brain: dict

    def __init__(
        self,
        action: Action,
        observation: Observation,
        model: Any,
        start_steps: int = 10000,
    ) -> None:
        self.action = action
        self.observation = observation
        self.sensor_ids = [obj.uid for obj in observation.sensors]
        self.muscle = SACMuscle(start_steps)
        self.muscle.update(model)
        self.data_for_brain = {}

    @override
    def _predict(self, input_features: Observation) -> Action:
        actuators_available = convert_to_actuator_information(self.action)
        sensors = convert_to_sensor_information(
            filter_observation(input_features, self.sensor_ids),
        )

        actuators, self.data_for_brain = self.muscle.propose_actions(
            sensors,
            actuators_available,
        )
        return Action(actuators=convert_to_interface(actuators))

    def update(self, update: Any) -> None:
        if update is not None:
            self.muscle.update(update)
