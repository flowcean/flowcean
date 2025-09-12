import io
from typing import Any

import torch
from harl.sac.muscle import SACMuscle
from palaestrai.types.mode import Mode
from typing_extensions import Self, override

from flowcean.core import (
    Action,
    Observation,
)
from flowcean.core.model import Model
from flowcean.core.strategies.active import interface_dict, interface_from_dict
from flowcean.palaestrai.util import (
    convert_to_actuator_information,
    convert_to_interface,
    convert_to_sensor_information,
    filter_action,
    filter_observation,
)


class SACModel(Model):
    def __init__(
        self,
        action: Action,
        observation: Observation,
        sensor_ids: list[str],
        actuator_ids: list[str],
        model: Any,
        *,
        start_steps: int = 10000,
        training_mode: bool = False,
    ) -> None:
        self.action: Action = action
        self.observation: Observation = observation
        self.sensor_ids: list[str] = sensor_ids
        self.actuator_ids: list[str] = actuator_ids
        self.muscle: SACMuscle = SACMuscle(start_steps)
        self.muscle.update(model)
        self.data_for_brain: dict[str, Any] = {}

        if training_mode:
            self.muscle._mode = Mode.TRAIN  # noqa: SLF001
        else:
            self.muscle._mode = Mode.TEST  # noqa: SLF001

    @override
    def _predict(self, input_features: Observation) -> Action:
        actuators_available = convert_to_actuator_information(
            filter_action(self.action, self.actuator_ids),
        )
        sensors = convert_to_sensor_information(
            filter_observation(input_features, self.sensor_ids),
        )

        actuators, self.data_for_brain = self.muscle.propose_actions(
            sensors,
            actuators_available,
        )
        return Action(actuators=convert_to_interface(actuators))

    @override
    def predict(self, input_features: Observation) -> Action:
        return self._predict(input_features)

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
            "sensor_ids": self.sensor_ids,
            "actuator_ids": self.actuator_ids,
        }

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
            state["sensor_ids"],
            state["actuator_ids"],
            bio,
        )

    def update(self, update: Any) -> None:
        if update is not None:
            self.muscle.update(update)

    def train(self) -> None:
        self.muscle._mode = Mode.TRAIN  # noqa: SLF001

    def eval(self) -> None:
        self.muscle._mode = Mode.TEST  # noqa: SLF001
