import io
from typing import Any

import torch
from harl.sac.muscle import SACMuscle
from typing_extensions import Self, override

from flowcean.core.model import Model
from flowcean.core.strategies.active import (
    Action,
    Observation,
    interface_dict,
    interface_from_dict,
)
from flowcean.palaestrai.util import (
    convert_to_actuator_informations,
    convert_to_interface,
    convert_to_sensor_informations,
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
    def predict(self, input_features: Observation) -> Action:
        actuators_available = convert_to_actuator_informations(self.action)
        sensors = convert_to_sensor_informations(
            filter_observation(input_features, self.sensor_ids),
        )

        actuators, self.data_for_brain = self.muscle.propose_actions(
            sensors,
            actuators_available,
        )
        return Action(actuators=convert_to_interface(actuators))

    @override
    def save_state(self) -> dict[str, Any]:
        bio = io.BytesIO()
        torch.save(self.muscle._model, bio)  # noqa: SLF001
        bio.seek(0)
        action = [interface_dict(a) for a in self.action.actuators]
        observation = [interface_dict(s) for s in self.observation.sensors]
        return {
            "model_bytes": bio.getvalue(),
            "action": action,
            "observation": observation,
        }

    @override
    @classmethod
    def load_from_state(cls, state: dict[str, Any]) -> Self:
        bio = io.BytesIO(state["model_bytes"])

        return cls(
            Action(
                actuators=[interface_from_dict(a) for a in state["action"]],
            ),
            Observation(
                sensors=[interface_from_dict(s) for s in state["observation"]],
                rewards=[],
            ),
            bio,
        )

    def update(self, update: Any) -> None:
        if update is not None:
            self.muscle.update(update)
