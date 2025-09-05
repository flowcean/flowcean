import io
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from harl.sac.brain import SACBrain
from palaestrai.agent.objective import Objective
from typing_extensions import override

from flowcean.core.learner import ActiveLearner
from flowcean.core.model import Model
from flowcean.core.strategies.active import (
    Action,
    ActiveInterface,
    Observation,
)
from flowcean.palaestrai.sac_model import SACModel
from flowcean.palaestrai.util import (
    convert_to_actuator_information,
    convert_to_reward_information,
    convert_to_sensor_information,
    filter_action,
    filter_observation,
)

MODEL_ID = "SACModel"


class SACLearner(ActiveLearner):
    """Learner class for the palaestrAI SAC agent."""

    model: SACModel
    brain: SACBrain
    agent_objective: Objective
    objectives: list[float]
    rewards: list[list[ActiveInterface]]
    actuator_ids: list[str]
    sensor_ids: list[str]
    action: Action
    observation: Observation

    def __init__(
        self,
        actuator_ids: list[str],
        sensor_ids: list[str],
        agent_objective: Objective,
        *,
        replay_size: int = int(1e6),
        fc_dims: Sequence[int] = (256, 256),
        activation: str = "torch.nn.ReLU",
        gamma: float = 0.99,
        polyak: float = 0.995,
        lr: float = 1e-3,
        batch_size: int = 100,
        update_after: int = 1000,
        update_every: int = 50,
    ) -> None:
        r"""Initialize the SAC learner.

        Args:
            actuator_ids: The IDs of actuators the learner should use to
                interact with the environment.
            sensor_ids: The IDs of sensors the learner should be able to see
                from the environment.
            agent_objective: The objective function that takes environment
                rewards and converts them to an objective for the agent.
            replay_size: Maximum length of replay buffer.
            fc_dims: Dimensions of the hidden layers of the agent's actor and
                critic networks. "fc" stands for "fully connected".
            activation: Activation function to use
            gamma: Discount factor. (Always between 0 and 1.)
            polyak: Interpolation factor in polyak averaging for target
                networks. Target networks are updated towards main networks
                according to: $$\theta_{\text{targ}} \leftarrow \rho \theta_{
                \text{targ}} + (1-\rho) \theta,$$ where $\rho$ is polyak.
                (Always between 0 and 1, usually close to 1.)
            lr: Learning rate (used for both policy and value learning).
            batch_size: Minibatch size for SGD.
            update_after: Number of env interactions to collect before starting
                to do gradient descent updates. Ensures replay buffer is full
                enough for useful updates.
            update_every: Number of env interactions that should elapse between
                gradient descent updates.
                Note: Regardless of how long you wait between updates, the
                ratio of environment interactions to gradient steps is locked
                to 1.
        """
        self.actuator_ids = actuator_ids
        self.sensor_ids = sensor_ids
        self.agent_objective = agent_objective
        self.objective_values = []
        self.rewards = []
        self.brain_params = {
            "replay_size": replay_size,
            "fc_dims": fc_dims,
            "activation": activation,
            "gamma": gamma,
            "polyak": polyak,
            "lr": lr,
            "batch_size": batch_size,
            "update_after": update_after,
            "update_every": update_every,
        }

    def setup(self, action: Action, observation: Observation) -> None:
        self.action = filter_action(action, self.actuator_ids)
        self.observation = filter_observation(observation, self.sensor_ids)

        self.brain = SACBrain(**self.brain_params)
        self.brain._seed = 0  # noqa: SLF001
        self.brain._sensors = convert_to_sensor_information(self.observation)  # noqa: SLF001
        self.brain._actuators = convert_to_actuator_information(self.action)  # noqa: SLF001
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
        rewards = convert_to_reward_information(observation)

        self.brain.memory.append(
            muscle_uid=MODEL_ID,
            sensor_readings=convert_to_sensor_information(
                filtered_observation,
            ),
            actuator_setpoints=convert_to_actuator_information(
                filtered_action,
            ),
            rewards=rewards,
            done=False,
        )
        objective = np.array(
            [self.agent_objective.internal_reward(self.brain.memory)],
        )
        self.brain.memory.append(MODEL_ID, objective=objective)
        update = self.brain.thinking(MODEL_ID, self.model.data_for_brain)
        self.model.update(update)

        self.rewards.append(observation.rewards)
        self.objective_values.append(objective[0])
        return self.model

    @override
    def propose_action(self, observation: Observation) -> Action:
        filtered = filter_observation(observation, self.sensor_ids)
        return self.model.predict(filtered)

    def save(self, file_path: str) -> None:
        Path(file_path).mkdir(parents=True, exist_ok=True)

        bio = io.BytesIO()
        torch.save(self.brain.actor, bio)
        bio.seek(0)
        with (Path(file_path) / "sac_actor").open("wb") as fp:
            fp.write(bio.read())

        bio.seek(0)
        bio.truncate(0)
        torch.save(self.brain.actor_target, bio)
        bio.seek(0)
        with (Path(file_path) / "sac_actor_target").open("wb") as fp:
            fp.write(bio.read())

        bio.seek(0)
        bio.truncate(0)
        torch.save(self.brain.critic, bio)
        bio.seek(0)
        with (Path(file_path) / "sac_critic").open("wb") as fp:
            fp.write(bio.read())

        bio.seek(0)
        bio.truncate(0)
        torch.save(self.brain.critic_target, bio)
        bio.seek(0)
        with (Path(file_path) / "sac_critic_target").open("wb") as fp:
            fp.write(bio.read())

    def load(self, file_path: str) -> None:
        with (Path(file_path) / "sac_actor").open("rb") as fp:
            bio = io.BytesIO()
            bio.write(fp.read())
        bio.seek(0)
        self.brain.actor = torch.load(bio, map_location=self.brain._device)  # noqa: SLF001

        bio.seek(0)
        self.model.update(bio)

        with (Path(file_path) / "sac_actor_target").open("rb") as fp:
            bio = io.BytesIO()
            bio.write(fp.read())
        bio.seek(0)
        self.brain.actor_target = torch.load(
            bio,
            map_location=self.brain._device,  # noqa: SLF001
        )

        with (Path(file_path) / "sac_critic").open("rb") as fp:
            bio = io.BytesIO()
            bio.write(fp.read())
        bio.seek(0)
        self.brain.critic = torch.load(bio, map_location=self.brain._device)  # noqa: SLF001

        with (Path(file_path) / "sac_critic_target").open("rb") as fp:
            bio = io.BytesIO()
            bio.write(fp.read())
        bio.seek(0)
        self.brain.critic_target = torch.load(
            bio,
            map_location=self.brain._device,  # noqa: SLF001
        )
