from typing import Any

from harl.sac.muscle import SACMuscle
from typing_extensions import Self, override

from flowcean.core.model import Model
from flowcean.core.strategies.active import Action, Interface, Observation
from flowcean.palaestrai.util import (
    convert_to_actuator_informations,
    convert_to_interface,
    convert_to_sensor_informations,
)

# from flowcean.palaestrai.sac_learner import convert_to_actuator_informations


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
        self.muscle = SACMuscle(start_steps)
        self.muscle.update(model)
        self.data_for_brain = {}

    @override
    def predict(self, input_features: Observation) -> Action:
        actuators_available = convert_to_actuator_informations(self.action)
        sensors = convert_to_sensor_informations(input_features)

        actuators, self.data_for_brain = self.muscle.propose_actions(
            sensors,
            actuators_available,
        )
        return Action(actuators=convert_to_interface(actuators))

    @override
    def save_state(self) -> dict[str, Any]:
        raise NotImplementedError

    @override
    @classmethod
    def load_from_state(cls, state: dict[str, Any]) -> Self:
        _ = state
        raise NotImplementedError

    def update(self, update: Any) -> None:
        if update is not None:
            self.muscle.update(update)
