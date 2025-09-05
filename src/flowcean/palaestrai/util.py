from typing import Any

import numpy as np
import torch
from harl.sac.action_type import ActionType
from harl.sac.network import (
    Actor,
    Critic,
    DiscreteMLPActor,
    MLPActorCritic,
    MLPQFunction,
    SquashedGaussianMLPActor,
)
from palaestrai.agent.actuator_information import ActuatorInformation
from palaestrai.agent.reward_information import RewardInformation
from palaestrai.agent.sensor_information import SensorInformation
from palaestrai.types import Box

from flowcean.core.strategies.active import (
    Action,
    ActiveInterface,
    Observation,
)

BOX_SPACE = "Box(low=%s, high=%s, shape=%s, dtype=%s)"

torch.serialization.add_safe_globals(
    [
        Actor,
        Critic,
        SquashedGaussianMLPActor,
        DiscreteMLPActor,
        MLPActorCritic,
        MLPQFunction,
        torch.nn.modules.container.Sequential,
        torch.nn.modules.linear.Linear,
        torch.nn.modules.activation.ReLU,
        torch.nn.modules.linear.Identity,
        ActionType,
    ],
)


def filter_action(action: Action, available_ids: list[str]) -> Action:
    return Action(
        actuators=[a for a in action.actuators if a.uid in available_ids],
    )


def filter_observation(
    observation: Observation,
    available_ids: list[str],
) -> Observation:
    return Observation(
        sensors=[s for s in observation.sensors if s.uid in available_ids],
        rewards=observation.rewards,
    )


def convert_to_actuator_information(
    action: Action,
) -> list[ActuatorInformation]:
    infos = []
    for obj in action.actuators:
        space = Box(
            low=obj.value_min,
            high=obj.value_max,
            shape=obj.shape,
            dtype=obj.dtype,
        )
        infos.append(
            ActuatorInformation(
                value=np.array(obj.value, dtype=space.dtype)
                if obj.value is not None
                else None,
                uid=obj.uid,
                space=space,
            ),
        )
    return infos


def convert_to_sensor_information(
    observation: Observation,
) -> list[SensorInformation]:
    infos = []
    for obj in observation.sensors:
        space = Box(
            low=obj.value_min,
            high=obj.value_max,
            shape=obj.shape,
            dtype=obj.dtype,
        )
        infos.append(
            SensorInformation(
                value=np.array(obj.value, dtype=space.dtype)
                if obj.value is not None
                else None,
                uid=obj.uid,
                space=space,
            ),
        )
    return infos


def convert_to_reward_information(
    observation: Observation,
) -> list[RewardInformation]:
    infos = []
    for obj in observation.rewards:
        space = Box(
            low=obj.value_min,
            high=obj.value_max,
            shape=obj.shape,
            dtype=obj.dtype,
        )
        infos.append(
            RewardInformation(
                value=np.array(obj.value, dtype=space.dtype)
                if obj.value is not None
                else None,
                uid=obj.uid,
                space=space,
            ),
        )
    return infos


def convert_to_interface(
    information_objects: list[SensorInformation]
    | list[ActuatorInformation]
    | list[RewardInformation],
) -> list[ActiveInterface]:
    """Convert a list of information objects to interface objects.

    Takes a list of SensorInformation, ActuatorInformation or
    RewardInformation from the palaestrAI ecosystem and converts them
    to a list of interface objects.
    """
    interfaces = []
    for obj in information_objects:
        space = obj.space
        if isinstance(space, Box):
            interfaces.append(
                ActiveInterface(
                    uid=obj.uid if isinstance(obj.uid, str) else "",
                    value=obj.value,
                    value_min=space.low,
                    value_max=space.high,
                    shape=space.shape,
                    dtype=space.dtype
                    if space.dtype
                    in (type[np.floating[Any]], type[np.integer[Any]])
                    else np.float64,
                ),
            )

    return interfaces
